import os
import torch
from torch.utils.data import Dataset
import logging
logging.basicConfig(level=logging.INFO)
import torch.distributed as dist
from PIL import Image
import re

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
    output_6d = torch.zeros(batch_dim + (6,), dtype=matrix.dtype, device=matrix.device)
    output_6d[..., :3] = matrix[..., :, 0]  # First column of the rotation matrix
    output_6d[..., 3:] = matrix[..., :, 1] 
    return output_6d

def extract_number(filename):
    numbers = re.findall(r'\d+', filename)
    return int(numbers[0])

def normalize_bboxes(bboxes, img_width, img_height):
    # Expand img_width and img_height to match the shape of bboxes for broadcasting
    img_dimensions = torch.tensor([img_width, img_height, img_width, img_height], dtype=bboxes.dtype, device=bboxes.device)
    normalized_bboxes = bboxes / img_dimensions
    return normalized_bboxes

class VideoDataset(Dataset):
    def __init__(self):
        """
        Dataset class used for loading images and corresponding annotations.
        Args:
            cfg (CfgNode): Model config file.
            video_dir (str): Path to video files.
            pkl_dir (str): Path to pkl files.
            train (bool): Whether it is for training or not (enables data augmentation).
        """
        super(VideoDataset, self).__init__()
        self.seqLen = 1100
        self.file_path = '<TODO>' #TODO: fill in (aka full path)
        self.mocap_path = '<TODO>' #TODO: fill in (aka full path)
        self.contextPath = '<TODO>' #TODO: fill in (aka full path)
        self.inpaintedFramesPath = '<TODO>' #TODO: fill in (aka full path)
        file = open(self.file_path, 'r')
        self.data = file.readlines()
        self.dataLength = len(self.data) 
        file.close()

    def __len__(self):
        return self.dataLength
    
    def __getitem__(self, index):
        # Remember, bboxes are on normal img sizes not the -%DINOv2 patch size size. 
        # IF you choose to normalize, only the bounding boxes need to be normalized by the patch size scale factor.
        file_name = self.data[index].strip()
        smpl_file = f"{self.mocap_path}/{file_name}"
        sample = torch.load(smpl_file, map_location="cpu")
        
        name = f'{file_name.strip()[len("training_data_"):-len(".pt")]}'
        contextFrames_path = f'{self.contextPath}/{name}'
        inpaintedFrames_path = f'{self.inpaintedFramesPath}/{name}'

        img_width, img_height = sample['image_width'], sample['image_height']
        bbox = normalize_bboxes(sample['person_bbox'][:self.seqLen], img_width, img_height)
        pred_camera = sample['camera_3D'][:self.seqLen]
        final_body_pose = torch.cat((sample['smpl']['global_orient'][:self.seqLen], sample['smpl']['body_pose'][:self.seqLen]), dim=1) #(seqLen, 24, 3, 3)
        pose_6d = matrix_to_rotation_6d(final_body_pose).reshape(self.seqLen, -1)
        betas = sample['smpl']['betas'][:self.seqLen].view(self.seqLen, -1)
        assert bbox is not None 
        assert pred_camera is not None
        assert pose_6d is not None 
        assert betas is not None
        x = torch.cat([bbox, pred_camera, pose_6d, betas], dim=-1) #only include this, in loss function, can add additional loss by regressing pred_cmaera_t and keypoints
        x_permuted = x.permute(1, 0) #needs to be features, numFrames
        
        inpainted_image_files = os.listdir(inpaintedFrames_path)
        inpainted_image_files = sorted(inpainted_image_files, key=extract_number)[:self.seqLen]
        context_image_files = os.listdir(contextFrames_path)
        context_image_files = sorted(context_image_files, key=extract_number)[:self.seqLen]
        
        img = Image.open(os.path.join(inpaintedFrames_path, inpainted_image_files[0]))
        W, H = img.size
        if H > W:
            newH = 490 #these numbers preserve aspect ratio as best as possible, while being divisible by DINOv2 patch size, while making it within its pretraining range of 512 max px
            newW = 280
        else:
            newH = 280
            newW = 490
        
        return x_permuted, (inpaintedFrames_path, contextFrames_path, (newH, newW), context_image_files, inpainted_image_files)