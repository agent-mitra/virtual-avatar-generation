from typing import Optional

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
import os
import torch
from typing import Union

smoothnet_dir = os.path.dirname(os.path.abspath(__file__))

def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret

def quaternion_to_axis_angle(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to axis/angle.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotations given as a vector in axis angle form, as a tensor
            of shape (..., 3), where the magnitude is the angle
            turned anticlockwise in radians around the vector's
            direction.
    """
    norms = torch.norm(quaternions[..., 1:], p=2, dim=-1, keepdim=True)
    half_angles = torch.atan2(norms, quaternions[..., :1])
    angles = 2 * half_angles
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    return quaternions[..., 1:] / sin_half_angles_over_angles


def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)

    return quat_candidates[
        F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :
    ].reshape(batch_dim + (4,))

def matrix_to_rotation_6d(matrix: torch.Tensor) -> torch.Tensor:
    """
    Converts rotation matrices to 6D rotation representation by Zhou et al. [1]
    by dropping the last row. Note that 6D representation is not unique.
    Args:
        matrix: batch of rotation matrices of size (*, 3, 3)

    Returns:
        6D rotation representation, of size (*, 6)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """
    batch_dim = matrix.size()[:-2]
    return matrix[..., :2, :].clone().reshape(batch_dim + (6,))

class Compose:

    def __init__(self, transforms: list):
        """Composes several transforms together. This transform does not
        support torchscript.

        Args:
            transforms (list): (list of transform functions)
        """
        self.transforms = transforms

    def __call__(self,
                 rotation: Union[torch.Tensor, np.ndarray],
                 convention: str = 'xyz',
                 **kwargs):
        convention = convention.lower()
        if not (set(convention) == set('xyz') and len(convention) == 3):
            raise ValueError(f'Invalid convention {convention}.')
        if isinstance(rotation, np.ndarray):
            data_type = 'numpy'
            rotation = torch.FloatTensor(rotation)
        elif isinstance(rotation, torch.Tensor):
            data_type = 'tensor'
        else:
            raise TypeError(
                'Type of rotation should be torch.Tensor or numpy.ndarray')
        for t in self.transforms:
            if 'convention' in t.__code__.co_varnames:
                rotation = t(rotation, convention.upper(), **kwargs)
            else:
                rotation = t(rotation, **kwargs)
        if data_type == 'numpy':
            rotation = rotation.detach().cpu().numpy()
        return rotation
    
def rotmat_to_aa(
    matrix: Union[torch.Tensor, np.ndarray]
) -> Union[torch.Tensor, np.ndarray]:
    """Convert rotation matrixs to axis angles.

    Args:
        matrix (Union[torch.Tensor, numpy.ndarray]): input shape
                should be (..., 3, 3). ndim of input is unlimited.
        convention (str, optional): Convention string of three letters
                from {“x”, “y”, and “z”}. Defaults to 'xyz'.

    Returns:
        Union[torch.Tensor, numpy.ndarray]: shape would be (..., 3).
    """
    if matrix.shape[-1] != 3 or matrix.shape[-2] != 3:
        raise ValueError(f'Invalid rotation matrix  shape f{matrix.shape}.')
    t = Compose([matrix_to_quaternion, quaternion_to_axis_angle])
    return t(matrix)

def rotmat_to_rot6d(
    matrix: Union[torch.Tensor, np.ndarray]
) -> Union[torch.Tensor, np.ndarray]:
    """Convert rotation matrixs to rotation 6d representations.

    Args:
        matrix (Union[torch.Tensor, numpy.ndarray]): input shape
                should be (..., 3, 3). ndim of input is unlimited.
    Returns:
        Union[torch.Tensor, numpy.ndarray]: shape would be (..., 6).

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """
    if matrix.shape[-1] != 3 or matrix.shape[-2] != 3:
        raise ValueError(f'Invalid rotation matrix  shape f{matrix.shape}.')
    t = Compose([matrix_to_rotation_6d])
    return t(matrix)

def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """
    Converts 6D rotation representation by Zhou et al. [1] to rotation matrix
    using Gram--Schmidt orthogonalization per Section B of [1].
    Args:
        d6: 6D rotation representation, of size (*, 6)

    Returns:
        batch of rotation matrices of size (*, 3, 3)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """

    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)

def rot6d_to_rotmat(
    rotation_6d: Union[torch.Tensor, np.ndarray]
) -> Union[torch.Tensor, np.ndarray]:
    """Convert rotation 6d representations to rotation matrixs.

    Args:
        rotation_6d (Union[torch.Tensor, numpy.ndarray]): input shape
                should be (..., 6). ndim of input is unlimited.
    Returns:
        Union[torch.Tensor, numpy.ndarray]: shape would be (..., 3, 3).

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """
    if rotation_6d.shape[-1] != 6:
        raise ValueError(f'Invalid input rotation_6d f{rotation_6d.shape}.')
    t = Compose([rotation_6d_to_matrix])
    return t(rotation_6d)

def axis_angle_to_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as axis/angle to rotation matrices.

    Args:
        axis_angle: Rotations given as a vector in axis angle form,
            as a tensor of shape (..., 3), where the magnitude is
            the angle turned anticlockwise in radians around the
            vector's direction.

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    return quaternion_to_matrix(axis_angle_to_quaternion(axis_angle))

def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))

def axis_angle_to_quaternion(axis_angle: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as axis/angle to quaternions.

    Args:
        axis_angle: Rotations given as a vector in axis angle form,
            as a tensor of shape (..., 3), where the magnitude is
            the angle turned anticlockwise in radians around the
            vector's direction.

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    angles = torch.norm(axis_angle, p=2, dim=-1, keepdim=True)
    half_angles = angles * 0.5
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    quaternions = torch.cat(
        [torch.cos(half_angles), axis_angle * sin_half_angles_over_angles], dim=-1
    )
    return quaternions

def aa_to_rotmat(
    axis_angle: Union[torch.Tensor, np.ndarray]
) -> Union[torch.Tensor, np.ndarray]:
    """
    Convert axis_angle to rotation matrixs.
    Args:
        axis_angle (Union[torch.Tensor, numpy.ndarray]): input shape
                should be (..., 3). ndim of input is unlimited.

    Returns:
        Union[torch.Tensor, numpy.ndarray]: shape would be (..., 3, 3).
    """
    if axis_angle.shape[-1] != 3:
        raise ValueError(
            f'Invalid input axis angles shape f{axis_angle.shape}.')
    t = Compose([axis_angle_to_matrix])
    return t(axis_angle)

def load_checkpoint(model, checkpoint_path, map_location='cpu'):
    """
    Load the weights from a checkpoint into the provided model.

    Args:
    - model (torch.nn.Module): The PyTorch model to which the weights should be loaded.
    - checkpoint_path (str): Path to the checkpoint file.
    - map_location (str or torch.device): The device to which the checkpoint weights should be loaded.

    Returns:
    - model with the loaded weights.
    """
    # Load the state dictionary from the checkpoint path
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    
    # Update the model's state dictionary with the loaded state dictionary
    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict)
    
    return model



class SmoothNetResBlock(nn.Module):
    """Residual block module used in SmoothNet.

    Args:
        in_channels (int): Input channel number.
        hidden_channels (int): The hidden feature channel number.
        dropout (float): Dropout probability. Default: 0.5
    Shape:
        Input: (*, in_channels)
        Output: (*, in_channels)
    """

    def __init__(self, in_channels, hidden_channels, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(in_channels, hidden_channels)
        self.linear2 = nn.Linear(hidden_channels, in_channels)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        self.dropout = nn.Dropout(p=dropout, inplace=True)

    def forward(self, x):
        identity = x
        x = self.linear1(x)
        x = self.dropout(x)
        x = self.lrelu(x)
        x = self.linear2(x)
        x = self.dropout(x)
        x = self.lrelu(x)

        out = x + identity
        return out


class SmoothNet(nn.Module):
    """SmoothNet is a plug-and-play temporal-only network to refine human
    poses. It works for 2d/3d/6d pose smoothing.
    "SmoothNet: A Plug-and-Play Network for Refining Human Poses in Videos",
    arXiv'2021. More details can be found in the `paper
    <https://arxiv.org/abs/2112.13715>`__ .
    Note:
        N: The batch size
        T: The temporal length of the pose sequence
        C: The total pose dimension (e.g. keypoint_number * keypoint_dim)
    Args:
        window_size (int): The size of the input window.
        output_size (int): The size of the output window.
        hidden_size (int): The hidden feature dimension in the encoder,
            the decoder and between residual blocks. Default: 512
        res_hidden_size (int): The hidden feature dimension inside the
            residual blocks. Default: 256
        num_blocks (int): The number of residual blocks. Default: 3
        dropout (float): Dropout probability. Default: 0.5
    Shape:
        Input: (N, C, T) the original pose sequence
        Output: (N, C, T) the smoothed pose sequence
    """

    def __init__(self,
                 window_size: int,
                 output_size: int,
                 hidden_size: int = 512,
                 res_hidden_size: int = 512,
                 num_blocks: int = 5,
                 dropout: float = 0.1):
        super().__init__()
        self.window_size = window_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.res_hidden_size = res_hidden_size
        self.num_blocks = num_blocks
        self.dropout = dropout

        assert output_size <= window_size, (
            'The output size should be less than or equal to the window size.',
            f' Got output_size=={output_size} and window_size=={window_size}')

        # Build encoder layers
        self.encoder = nn.Sequential(
            nn.Linear(window_size, hidden_size),
            nn.LeakyReLU(0.1, inplace=True))

        # Build residual blocks
        res_blocks = []
        for _ in range(num_blocks):
            res_blocks.append(
                SmoothNetResBlock(
                    in_channels=hidden_size,
                    hidden_channels=res_hidden_size,
                    dropout=dropout))
        self.res_blocks = nn.Sequential(*res_blocks)

        # Build decoder layers
        self.decoder = nn.Linear(hidden_size, output_size)

    def forward(self, x: Tensor) -> Tensor:
        """Forward function."""
        N, C, T = x.shape
        num_windows = T - self.window_size + 1

        assert T >= self.window_size, (
            'Input sequence length must be no less than the window size. ',
            f'Got x.shape[2]=={T} and window_size=={self.window_size}')

        # Unfold x to obtain input sliding windows
        # [N, C, num_windows, window_size]
        x = x.unfold(2, self.window_size, 1)

        # Forward layers
        x = self.encoder(x)
        x = self.res_blocks(x)
        x = self.decoder(x)  # [N, C, num_windows, output_size]

        # Accumulate output ensembles
        out = x.new_zeros(N, C, T)
        count = x.new_zeros(T)

        for t in range(num_windows):
            out[..., t:t + self.output_size] += x[:, :, t]
            count[t:t + self.output_size] += 1.0

        return out.div(count)


# @POST_PROCESSING.register_module(name=['SmoothNetFilter', 'smoothnet'])
class SmoothNetFilter:
    """Apply SmoothNet filter.
    "SmoothNet: A Plug-and-Play Network for Refining Human Poses in Videos",
    arXiv'2021. More details can be found in the `paper
    <https://arxiv.org/abs/2112.13715>`__ .
    Args:
        window_size (int): The size of the filter window. It's also the
            window_size of SmoothNet model.
        output_size (int): The output window size of SmoothNet model.
        checkpoint (str): The checkpoint file of the pretrained SmoothNet
            model. Please note that `checkpoint` should be matched with
            `window_size` and `output_size`.
        hidden_size (int): SmoothNet argument. See :class:`SmoothNet` for
            details. Default: 512
        hidden_res_size (int): SmoothNet argument. See :class:`SmoothNet`
            for details. Default: 256
        num_blocks (int): SmoothNet argument. See :class:`SmoothNet` for
            details. Default: 3
        device (str): Device for model inference. Default: 'cpu'
        root_index (int, optional): If not None, relative keypoint coordinates
            will be calculated as the SmoothNet input, by centering the
            keypoints around the root point. The model output will be
            converted back to absolute coordinates. Default: None
    """

    def __init__(
        self,
        window_size: int = 8,
        output_size: int = 8,
        checkpoint: Optional[str] = os.path.join(smoothnet_dir, '..', '..', 'trackers', 'smoothnet_windowsize8.pth.tar'),
        hidden_size: int = 512,
        res_hidden_size: int = 512,
        num_blocks: int = 5,
        device: str = 'cuda', #cpu
    ):
        super(SmoothNetFilter, self).__init__()
        self.window_size = window_size
        self.device = device
        self.smoothnet = SmoothNet(window_size, output_size, hidden_size,
                                   res_hidden_size, num_blocks)
        self.smoothnet.to(device)
        if checkpoint:
            load_checkpoint(
                self.smoothnet, checkpoint, map_location=self.device)
        self.smoothnet.eval()

        for p in self.smoothnet.parameters():
            p.requires_grad_(False)

    def __call__(self, x: np.ndarray):
        x_type = 'tensor'
        if not isinstance(x, torch.Tensor):
            x_type = 'array'

        assert x.ndim == 3, ('Input should be an array with shape [T, K, C]'
                             f', but got invalid shape {x.shape}')

        T, K, C = x.shape

        assert C == 3 or C == 6 or C == 9

        if T < self.window_size:
            # Skip smoothing if the input length is less than the window size
            smoothed = x
        else:
            if x_type == 'array':
                dtype = x.dtype

            # Convert to tensor and forward the model
            with torch.no_grad():
                if x_type == 'array':
                    x = torch.tensor(
                        x, dtype=torch.float32, device=self.device)
                if C == 9:
                    input_type = 'matrix'
                    x = rotmat_to_rot6d(x.reshape(-1, 3, 3)).reshape(T, K, -1)
                elif C == 3:
                    input_type = 'axis_angles'
                    x = rotmat_to_rot6d(aa_to_rotmat(x.reshape(-1,3))).reshape(T, K, -1)
                else:
                    input_type = 'rotation_6d'
                x = x.view(1, T, -1).permute(0, 2, 1)  # to [1, KC, T]
                smoothed = self.smoothnet(x)  # in shape [1, KC, T]

            # Convert model output back to input shape and format
            smoothed = smoothed.permute(0, 2, 1).view(T, K, -1)  # to [T, K, C]

            if input_type == 'matrix':
                smoothed = rot6d_to_rotmat(smoothed.reshape(-1, 6)).reshape(
                    T, K, C)
            elif input_type == 'axis_angles':
                smoothed = rotmat_to_aa(
                    rot6d_to_rotmat(smoothed.reshape(-1, 6))).reshape(T, K, C)

            if x_type == 'array':
                smoothed = smoothed.cpu().numpy().astype(
                    dtype)  # to numpy.ndarray

        return smoothed
