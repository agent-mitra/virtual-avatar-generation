import subprocess
import os, time, cv2
from typing import Optional
import logging
import shutil, re
import numpy as np 
import glob
from PIL import Image
from torchvision.utils import make_grid
from torch.utils.data._utils.collate import default_collate
import torch, argparse
from pipeline.track.phalp.utils.smpl_utils import SMPL
import torchvision.transforms as T
from model.backbones import create_backbone
from model.architecture.diffusion_transformer import DiT_action
from model.architecture.diffusion import create_diffusion
from collections import defaultdict
logging.basicConfig(level=logging.INFO)
import pyrootutils, joblib
from pipeline.track.phalp.visualize.py_renderer import Renderer
from torch.nn import functional as F
from torchvision.io import read_image
CACHE_DIR = os.path.join(os.environ.get("HOME"), ".cache")
HOME_DIR = os.path.join(os.environ.get("HOME"), "virtual-avatar-generation")

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

def padTensor(item):
    padLength = 100
    totalPads = len(item.shape) * 2
    padConfig = [0] * totalPads
    seqLengthIdxToAlter = totalPads - 1
    #add to seqLength to the end (right pad)
    padConfig[seqLengthIdxToAlter] = max(0, padLength - item.shape[0])
    padConfig = tuple(padConfig)
    item = F.pad(item, padConfig, 'constant', 0)
    del totalPads, padConfig
    return item


def extract_number(filename):
    numbers = re.findall(r'\d+', filename)
    return int(numbers[0])

def numpy_to_torch_image(ndarray):
    torch_image = torch.from_numpy(ndarray)
    torch_image = torch_image.unsqueeze(0)
    torch_image = torch_image.permute(0, 3, 1, 2)
    torch_image = torch_image[:, [2,1,0], :, :]
    return torch_image

def normalize_bboxes(bboxes, img_width, img_height):
    # Expand img_width and img_height to match the shape of bboxes for broadcasting
    img_dimensions = torch.tensor([img_width, img_height, img_width, img_height], dtype=bboxes.dtype, device=bboxes.device)
    normalized_bboxes = bboxes / img_dimensions
    return normalized_bboxes

def unnormalize_bboxes(bboxes, img_width, img_height):
    # Expand img_width and img_height to match the shape of bboxes for broadcasting
    img_dimensions = torch.tensor([img_width, img_height, img_width, img_height], dtype=bboxes.dtype, device=bboxes.device)
    unnormalized_bboxes = bboxes * img_dimensions
    return unnormalized_bboxes

def cam_crop_to_full(
    pred_cam: torch.Tensor,
    center: torch.Tensor,
    scale: torch.Tensor,
    img_size: torch.Tensor,
    device,
    focal_length: float = 5000.0,
) -> torch.Tensor:
    """
    Compute perspective camera translation, given the weak-persepctive camera, the bounding box and the image dimensions.
    """
    pred_cam = pred_cam.to(device)
    scale = scale.to(device)
    focal_length           = focal_length * torch.ones(pred_cam.shape[0], 2, device=device)
    pred_cam_t         = torch.stack([pred_cam[:,1], pred_cam[:,2], 2*focal_length[:, 0]/(pred_cam[:,0]*torch.tensor(scale[:, 0]) + 1e-9)], dim=1)
    pred_cam_t[:, :2] += torch.tensor(center-img_size/2.) * pred_cam_t[:, [2]] / focal_length
    return pred_cam_t

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

def rot6d_to_rotmat(x: torch.Tensor) -> torch.Tensor:
    """
    Convert 6D rotation representation to 3x3 rotation matrix.
    Based on Zhou et al., "On the Continuity of Rotation Representations in Neural Networks", CVPR 2019
    Args:
        x (torch.Tensor): (B,6) Batch of 6-D rotation representations.
    Returns:
        torch.Tensor: Batch of corresponding rotation matrices with shape (B,3,3).
    """
    x = x.reshape(-1, 2, 3).permute(0, 2, 1).contiguous()
    a1 = x[:, :, 0]
    a2 = x[:, :, 1]
    b1 = F.normalize(a1)
    b2 = F.normalize(a2 - torch.einsum("bi,bi->b", b1, a2).unsqueeze(-1) * b1)
    b3 = torch.linalg.cross(b1, b2)
    return torch.stack((b1, b2, b3), dim=-1)

ai_body_colors = [
    [241.0,91.0,181.0,],
    [0.0,187.0,249.0,],
    [254.0,228.0,64.0,],
    [0.0,245.0,212.0,],
    [222.0,158.0,54.0,],
    [93.0,169.0,233.0,],
    [155.0,93.0,229.0,],
    [24.0,242.0,178.0,],
    [252.0,116.0,186.0,],
    [138.0,225.0,252.0,],
    [178.0,171.0,242.0,],
    [24.0,231.0,98.0,],
    [232.0,93.0,117.0,],
    [207.0,250.0,231.0,],
    [253.0,208.0,242.0,],
    [109.0,157.0,197.0,],
    [166.0,189.0,219.0,],
    [253.0,146.0,207.0,],
    [167.0,201.0,87.0,],
    [117.0,68.0,177.0,],
    [255.0,229.0,50.0,],
    [251.0,202.0,239.0,],
    [58.0,134.0,255.0,],
    [255.0,0.0,110.0,],
    [251.0,86.0,7.0,],
    [188.0,51.0,209.0,],
    [122.0,229.0,130.0,],
    [0.0,48.0,73.0,],
    [214.0,40.0,40.0,],
    [229.0,179.0,179.0,],
    [0.0,187.0,249.0,],
    [255.0,190.0,11.0,],
    [204.0,213.0,174.0,],
    [0.0,245.0,212.0,],
    [255.0,153.0,200.0,],
    [144.0,251.0,146.0,],
    [189.0,211.0,147.0,],
    [230.0,0.0,86.0,],
    [0.0,95.0,57.0,],
    [0.0,174.0,126.0,],
    [255.0,116.0,163.0,],
    [189.0,198.0,255.0,],
    [90.0,219.0,255.0,],
    [158.0,0.0,142.0,],
    [255.0,147.0,126.0,],
    [164.0,36.0,0.0,],
    [0.0,21.0,68.0,],
    [145.0,208.0,203.0,],
    [95.0,173.0,78.0,],
    [107.0,104.0,130.0,],
    [0.0,125.0,181.0,],
    [106.0,130.0,108.0,],
    [252.0,246.0,189.0,],
    [208.0,244.0,222.0,],
    [169.0,222.0,249.0,],
    [228.0,193.0,249.0,],
    [122.0,204.0,174.0,],
    [194.0,140.0,159.0,],
    [0.0,143.0,156.0,],
    [235.0,0.0,0.0,],
    [255.0,2.0,157.0,],
    [104.0,61.0,59.0,],
    [150.0,138.0,232.0,],
    [152.0,255.0,82.0,],
    [167.0,87.0,64.0,],
    [1.0,255.0,254.0,],
    [255.0,238.0,232.0,],
    [254.0,137.0,0.0,],
    [1.0,208.0,255.0,],
    [187.0,136.0,0.0,],
    [165.0,255.0,210.0,],
    [255.0,166.0,254.0,],
    [119.0,77.0,0.0,],
    [122.0,71.0,130.0,],
    [38.0,52.0,0.0,],
    [0.0,71.0,84.0,],
    [67.0,0.0,44.0,],
    [181.0,0.0,255.0,],
    [255.0,177.0,103.0,],
    [255.0,219.0,102.0,],
    [126.0,45.0,210.0,],
    [229.0,111.0,254.0,],
    [222.0,255.0,116.0,],
    [0.0,255.0,120.0,],
    [0.0,155.0,255.0,],
    [0.0,100.0,1.0,],
    [0.0,118.0,255.0,],
    [133.0,169.0,0.0,],
    [0.0,185.0,23.0,],
    [120.0,130.0,49.0,],
    [0.0,255.0,198.0,],
    [255.0,110.0,65.0,],
    [232.0,94.0,190.0,],
    [1.0,0.0,103.0,],
    [149.0,0.0,58.0,],
    [98.0,14.0,0.0,],
    [0.0,0.0,0.0,],
] #blue-ish

def get_colors(pallette="phalp"):  

    try:
        if(pallette=="phalp"):
            colors = ai_body_colors
        else:
            raise ValueError("Invalid pallette")

        RGB_tuples    = np.vstack([colors, np.random.uniform(0, 255, size=(10000, 3)), [[0,0,0]]])
        b             = np.where(RGB_tuples==0)
        RGB_tuples[b] = 1    
    except:
        from colordict import ColorDict
        colormap = np.array(list(ColorDict(norm=255, mode='rgb', palettes_path="", is_grayscale=False, palettes='all').values()))
        RGB_tuples = np.vstack([colormap[1:, :3], np.random.uniform(0, 255, size=(10000, 3)), [[0,0,0]]])
        
    return RGB_tuples

def extract_number(filename):
    numbers = re.findall(r'\d+', filename)
    return int(numbers[0])

class RunPipeline:
    def __init__(self, video_name, video_dir, ckpt, mode):
        self.video_name = video_name
        self.video_dir = video_dir
        self.checkpoint = ckpt
        if mode == 0:
            self.mode = "latent"
        else:
            self.mode = "noise"
        self.device = 'cuda'
        self.seqLen = 1100
        self.savePath = os.path.abspath(f"output/{self.video_name}.mp4")
        data_directory = os.path.join(HOME_DIR, '.cache', '4DHumans', 'data')
        smpl_dict = {
            "DATA_DIR": data_directory,
            "MODEL_PATH": os.path.join(data_directory, 'smpl'),
            "GENDER": "neutral",
            "NUM_BODY_JOINTS": 23,
            "JOINT_REGRESSOR_EXTRA": os.path.join(data_directory, 'SMPL_to_J19.pkl'),
            "MEAN_PARAMS": os.path.join(data_directory, 'smpl_mean_params.npz'),
        }
        smpl_cfg             = {k.lower(): v for k,v in smpl_dict.items()}
        self.smpl            = SMPL(**smpl_cfg, pose2rot=False).to(self.device)
        self.video_path = os.path.abspath(f"{video_dir}/{video_name}.mp4")
        self.mocap_dir = f"{HOME_DIR}/pipeline/track"
        self.inpaint_tool_dir = f"{HOME_DIR}/pipeline/environment/tool"
        self.context_tool_dir = f"{HOME_DIR}/pipeline/context"
        self.output_dir = f"{HOME_DIR}/pipeline/track/outputs"
        self.highDimension = 640
        self.lowDimension = 368
        self.patch_size = 14 #DINOv2 patch size
        self.render = None
        self.colors = get_colors(pallette="phalp")
        loadingFile = os.path.abspath("PHALP/phalp/visualize/smpl_mean_params.npz")
        mean_params = np.load(loadingFile)
        self.init_body_pose = torch.from_numpy(mean_params['pose'].astype('float32')).unsqueeze(0)
        self.init_betas = torch.from_numpy(mean_params['shape'].astype('float32')).unsqueeze(0)
        self.init_cam = torch.from_numpy(mean_params['cam'].astype('float32')).unsqueeze(0)
        print(self.video_path)
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print("Error: Could not open video file.")
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        
    def render_single_frame(self, pred_smpl_params, pred_cam_t, color, img_size = 256, image=None, use_image=False):
                
        pred_smpl_params = default_collate(pred_smpl_params)
        pred_smpl_params['betas'] = self.init_betas #TODO: remove this and see!!!
        smpl_output = self.smpl(**{k: v.float().cuda() for k,v in pred_smpl_params.items()}, pose2rot=False)
        pred_vertices = smpl_output.vertices.cpu()
        
        pred_cam_t = torch.tensor(pred_cam_t, device=self.device) 
        pred_cam_t_bs = pred_cam_t.unsqueeze(1).repeat(1, pred_vertices.size(1), 1)

        rgb_from_pred, validmask = self.render.visualize_all(pred_vertices.numpy(), pred_cam_t_bs.cpu().numpy(), color, image, use_image=use_image)
        
        return rgb_from_pred, validmask
        
    def getFrameTensors(self, path, image_files, newH, newW, device):
        transform = torch.nn.Sequential(
            T.Resize((newH, newW), interpolation=T.InterpolationMode.BICUBIC),
            T.ConvertImageDtype(torch.float),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        )
        imgs_tensor_orig = torch.zeros(len(image_files), 3, newH, newW, device=device)
        for i, image_file in enumerate(image_files):
            img = read_image(os.path.join(path, image_file)).to(device)
            imgs_tensor_orig[i] = transform(img)
        return imgs_tensor_orig

    def inpaint_person(self):
        result_path = f"{HOME_DIR}/pipeline/environment/data/results"
        script_path = f"{self.inpaint_tool_dir}/video_inpainting.py"
        context_path = f"{self.context_tool_dir}/run_pipeline.py"
        if self.height > self.width:
            cmd = [
                "python",
                script_path,
                "--outroot", result_path,
                "--imgH", f"{self.highDimension}",
                "--imgW", f"{self.lowDimension}",
                "--inference_mode",
                "--video_source", self.video_path
                
            ]
        elif self.width > self.height:
            cmd = [
                "python",
                script_path,
                "--outroot", result_path,
                "--imgH", f"{self.lowDimension}",
                "--imgW", f"{self.highDimension}",
                "--inference_mode",
                "--video_source", self.video_path
            ]
        else:
            cmd = [
                "python",
                script_path,
                "--outroot", result_path,
                "--imgH", f"{self.lowDimension}",
                "--imgW", f"{self.lowDimension}",
                "--inference_mode",
                "--video_source", self.video_path
            ]
        subprocess.run(cmd, cwd=self.inpaint_tool_dir, check=True)
        getContextFrames = [
            "python", 
            context_path,
            "--originalVideoDir", self.video_path,
            "--inpaintedVideoDir", f"{result_path}/{self.video_name}.mp4",
            "--processNumber", str(self.processNumber)
        ]
        subprocess.run(getContextFrames, cwd=self.context_tool_dir, check=True)
        return
    
    
    def inference(self):
        #donwload video from firebase here
        render_res = 256
        inpainted_video_path = f"{HOME_DIR}/pipeline/environment/data/results/{self.video_name}.mp4"
        inpaintedFrames_path = os.path.join(f"data/inputInpaintedFrames/{self.video_name}")
        if not os.path.exists(inpaintedFrames_path):
            os.makedirs(inpaintedFrames_path)
        getInpaintedFrames = [
            'ffmpeg',
            '-i', inpainted_video_path,
            '-copyts', 
            '-qscale:v', '2', 
            '-vf', f'fps=24',
            f'{inpaintedFrames_path}/%06d.jpg'
        ]
        subprocess.run(getInpaintedFrames,stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        contextFrames_path = f"{HOME_DIR}/pipeline/context/assets/contextFrames/1/{self.video_name}"
        context_tensor = self.getFrameTensors(contextFrames_path)
        self.initLength = context_tensor.shape[0]
        print(f"Length of context is {self.initLength}")
        shutil.rmtree(contextFrames_path)    
        
        width, height = self.lowDimension, self.lowDimension
        if self.height > self.width:
            height = self.highDimension
            width = self.lowDimension
        elif self.width > self.height:
            height = self.lowDimension
            width = self.highDimension

        #run mocap to get initial body pose and shapes (for conditioning)
        cmd = [
            "python",
            "scripts/demo.py",
            f"video.source={self.video_path}",
            f"video.output_dir={self.output_dir}",
            "video.inference=True",
            f"video.width={width}",
            f"video.height={height}"
        ]
        if width != height and (self.mode == "latent"): #perfectly square videos will not fit in batch size when training the model, so ignore
            subprocess.run(cmd, cwd=self.mocap_dir, check=True)
        else:
            print("Uh oh. Square videos are not allowed")
            return

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
            
        context_tensor = self.getFrameTensors(contextFrames_path, context_image_files, newH, newW, self.device)
        video_tensor = self.getFrameTensors(inpaintedFrames_path, inpainted_image_files, newH, newW, self.device)
        backbone = create_backbone(model="large", device=self.device)
        zero_vector = torch.from_numpy(torch.load(f"{HOME_DIR}/model/backbones/dinov2_zero_vector.pt", map_location=self.device)).to(self.device)
        with torch.no_grad():
            context_feature_vector = backbone.forward_features(context_tensor.to(self.device))['x_norm_patchtokens']    
        vid_vecs = []
        with torch.no_grad():
            for batch in range(0, video_tensor.shape[0], 350):
                video_feature_vector = backbone.forward_features(video_tensor[batch: batch + 350].to(self.device))['x_norm_patchtokens']
                vid_vecs.append(video_feature_vector)
                del video_feature_vector
        video_feature_vectors = torch.cat(vid_vecs, dim=0).to(self.device)
        assert video_feature_vectors.shape[0] == video_tensor.shape[0]
        combined_vector = torch.cat((context_feature_vector, video_feature_vectors), dim=0)[:self.seqLen].to(self.device)
        if combined_vector.shape[0] < self.seqLen:
            toAdd = self.seqLen - combined_vector.shape[0]
            zeros_to_add = zero_vector.repeat(toAdd, 1, 1)
            combined_vector = torch.cat((combined_vector, zeros_to_add), dim=0)
            del zeros_to_add
        assert combined_vector.shape[0] == self.seqLen
        
        torch.manual_seed(0)
        torch.set_grad_enabled(False)
        
        ### RUN DIFFUSION INFERENCE (START)
        if self.mode == "latent":
            videoPklFile = joblib.load(f"{self.output_dir}/results/demo_{self.video_name}.pkl")
            frameKeys = list(videoPklFile.keys()) #total number of frames
            sample = defaultdict(dict)
        
            zeros_tensor = torch.zeros(self.initLength)
            
            ones_tensor = torch.stack([torch.tensor(videoPklFile[frame]['camera_bbox'][0]) for frame in frameKeys])
            values_to_concatenate = torch.cat([zeros_tensor, ones_tensor], dim=0)
            sample['camera_3D'] = self.padTensor(values_to_concatenate)
            del zeros_tensor, ones_tensor, values_to_concatenate           
            
            zeros_tensor = torch.zeros(self.initLength, 4)
            ones_tensor = torch.stack([torch.tensor(videoPklFile[frame]['bbox'][0]) for frame in frameKeys])  #mocap (PHALP) preds are in x0 y0 w h format (should be rendered in that format, too) so so is bbox.
            values_to_concatenate = torch.cat([zeros_tensor, ones_tensor], dim=0)
            sample['person_bbox'] = self.padTensor(values_to_concatenate)
            del zeros_tensor, ones_tensor, values_to_concatenate
            
            zeros_tensor = torch.zeros(self.initLength, 1, 3, 3) 
            ones_tensor = torch.stack([torch.tensor(videoPklFile[frame]['smpl'][0]['global_orient']) for frame in frameKeys])
            values_to_concatenate = torch.cat([zeros_tensor, ones_tensor], dim=0)
            sample['smpl']['global_orient'] = self.padTensor(values_to_concatenate)
            del zeros_tensor, ones_tensor, values_to_concatenate
            
            zeros_tensor = torch.zeros(self.initLength, 23, 3, 3)
            ones_tensor = torch.stack([torch.tensor(videoPklFile[frame]['smpl'][0]['body_pose']) for frame in frameKeys])
            values_to_concatenate = torch.cat([zeros_tensor, ones_tensor], dim=0)
            sample['smpl']['body_pose'] = self.padTensor(values_to_concatenate)
            del zeros_tensor, ones_tensor, values_to_concatenate
            
            zeros_tensor = torch.zeros(self.initLength, 10) 
            ones_tensor = torch.stack([torch.tensor(videoPklFile[frame]['smpl'][0]['betas']) for frame in frameKeys])
            values_to_concatenate = torch.cat([zeros_tensor, ones_tensor], dim=0)
            sample['smpl']['betas'] = self.padTensor(values_to_concatenate)
            del zeros_tensor, ones_tensor, values_to_concatenate

        model = DiT_action().to(self.device)
        self.checkpoint = torch.load(self.checkpoint, map_location=self.device)
        state_dict = self.checkpoint["ema"] 
        model.load_state_dict(state_dict)
        model.eval()
        if self.mode == "latent":
            diffusion = create_diffusion(str(250), predict_xstart=True) #predict original latents
            #START: CONDITION ON USER'S BODY SHAPE AND INITIAL POSES (makes predictions better and customized)
            bbox = normalize_bboxes(sample['person_bbox'][:self.seqLen].unsqueeze(0), width, height)
            pred_camera = sample['camera_3D'][:self.seqLen].unsqueeze(0)
            final_body_pose = torch.cat((sample['smpl']['global_orient'][:self.seqLen], sample['smpl']['body_pose'][:self.seqLen]), dim=1) #(seqLen, 24, 3, 3)
            pose_6d = matrix_to_rotation_6d(final_body_pose).reshape(self.seqLen, -1).unsqueeze(0)
            betas = sample['smpl']['betas'][:self.seqLen].view(self.seqLen, -1).unsqueeze(0)
            overwrite_first_24frames = torch.cat([bbox, pred_camera, pose_6d, betas], dim=-1).to(self.device)[:,:context_tensor.shape[0] + 24] #every video is 24 fps so 24 frames is 1 second + length of context.
            overwrite_first_24frames = overwrite_first_24frames.permute(0,2,1).to(self.device) #optionally, can remove for general body parameters
            #END: CODE to CONDITION ON USER'S BODY SHAPE AND INITIAL POSES (makes predictions better and customized)
        else:
            diffusion = create_diffusion(str(250))
            overwrite_first_24frames = None
        noise = torch.randn(1, 161, self.seqLen, device=self.device)
        y = combined_vector.unsqueeze(0).to(self.device)
        model_kwargs = dict(y=y) #scale is default 4.0
        width, height = self.width, self.height

        with torch.autocast(device_type=self.device):
            samples = diffusion.p_sample_loop( 
                model.forward, noise.shape, overwrite_first_24frames, noise, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=self.device
            )
            samples = samples.permute(0,2,1).to(self.device)
        ### RUN DIFFUSION INFERENCE (END)
        
        ### CONVERT BACK FROM 6D REPRESENTATION (START)
        pred_bbox = unnormalize_bboxes(samples[:,:,0:4].squeeze(0), width, height)
        box_size = torch.stack([pred_bbox[:,2], pred_bbox[:,3]], dim=-1)
        box_center = pred_bbox[:, :2] + 0.5 * box_size
        img_size = torch.tensor([height,width])
        pred_camera = samples[:,:,4:7].squeeze(0)
        pose_6d_vector = samples[:,:,7:151].squeeze(0).reshape(self.seqLen, 24, 6)
        betas_vector = samples[:,:,151:161].squeeze(0)
        pred_pose = rot6d_to_rotmat(pose_6d_vector.reshape(self.seqLen, -1)).view(self.seqLen, 24, 3, 3)
        body_pose_vector = pred_pose[:, 1:]
        global_orient_vector = pred_pose[:, [0]]
        new_image_size = img_size.max()
        top, left                 = (new_image_size - height)//2, (new_image_size - width)//2,
        ratio = (1.0/int(new_image_size))*render_res
        center = (box_center + torch.stack([left, top], dim=-1))*ratio
        scale = ratio * torch.max(box_size, dim=1)[0].unsqueeze(-1)
        pred_camera_t_vector = cam_crop_to_full(pred_camera, center, scale, render_res, self.device)
        ### CONVERT BACK FROM 6D REPRESENTATION (END)
        cutLength = self.seqLen - context_tensor.shape[0]
        list_of_frames = sorted(glob.glob(inpaintedFrames_path + "/*.jpg"))[:cutLength]
        texture_file = np.load(f"{CACHE_DIR}/phalp/3D/texture.npz")
        faces_cpu       = texture_file['smpl_faces'].astype('uint32')
        up_scale = 2
        output_resolution = 1440
        self.video_to_save = None
        initLength = context_tensor.shape[0]
        print(f"Total frames is {len(list_of_frames)}") #there is a slight discrepancy here where we delete last frame in the mocap data collection
                                                        #so, at the end, there is a valid video frame but zero predictions (due to bad mask when video inpainting, so handle that here)
        for t_, frame_name in enumerate(list_of_frames[:-1]): #omit last frame 
            print(f"Visualizing on frame number {t_}")
            
            up_scale = int(output_resolution / render_res)
            image_size = render_res*up_scale
            del self.render
            self.render = None
            self.render = Renderer(focal_length=5000, img_res=render_res*up_scale, faces=faces_cpu, 
                               metallicFactor=0.0, roughnessFactor=0.7)
            self.render_size = image_size 
            image_frame               = cv2.imread(frame_name)
            final_visuals_dic = defaultdict()

            idx = t_ + initLength
            final_visuals_dic['smpl'] = defaultdict()
            final_visuals_dic['smpl']['global_orient'] = global_orient_vector[idx].cpu().numpy()
            final_visuals_dic['smpl']['body_pose'] = body_pose_vector[idx].cpu().numpy()
            final_visuals_dic['smpl']['betas'] = betas_vector[idx].cpu().numpy()
            final_visuals_dic['camera'] = pred_camera_t_vector[idx].cpu().numpy()
            
            cv_image = image_frame
            tracked_smpl = [final_visuals_dic['smpl']]
            tracked_cameras = [final_visuals_dic['camera']]
            
            NUM_PANELS    = 1

            img_height, img_width, _      = cv_image.shape
            new_image_size                = max(img_height, img_width)
            render_image_size             = render_res*up_scale
            ratio_                        = render_res*up_scale/new_image_size
            
            delta_w                       = new_image_size - img_width
            delta_h                       = new_image_size - img_height
            top, bottom, left, right      = delta_h//2, delta_h-(delta_h//2), delta_w//2, delta_w-(delta_w//2)
            top_, bottom_, left_, right_  = int(top*ratio_), int(bottom*ratio_), int(left*ratio_), int(right*ratio_)
            img_height_, img_width_       = int(img_height*ratio_), int(img_width*ratio_)

            image_padded                  = cv2.copyMakeBorder(cv_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
            image_resized                 = cv2.resize(image_padded, (render_res*up_scale, render_res*up_scale))
            scale_                        = output_resolution/img_width
            frame_size                    = (output_resolution*NUM_PANELS, int(img_height*(scale_)))
            image_resized_rgb             = numpy_to_torch_image(np.array(image_resized)/255.)
            
            if(len(tracked_smpl)>0):
                tracked_smpl              = np.array(tracked_smpl)
                tracked_cameras           = np.array(tracked_cameras)
                tracked_cameras[:, 2]     = tracked_cameras[:, 2]/up_scale
                ids_x = [True] 
                tracked_colors            = np.array(self.colors[list([1])])/255.0
                rendered_image_final, valid_mask  = self.render_single_frame(
                                                                        tracked_smpl[ids_x],
                                                                        tracked_cameras[ids_x],
                                                                        tracked_colors, 
                                                                        img_size   = render_image_size, 
                                                                        image      = (0*image_resized)/255.0, 
                                                                        use_image  = True,
                                                                        )

                rendered_image_final = numpy_to_torch_image(np.array(rendered_image_final))

                valid_mask = np.repeat(valid_mask, 3, 2)
                valid_mask = np.array(valid_mask, dtype=float)
                valid_mask = numpy_to_torch_image(np.array(valid_mask))
                
                rendered_image_final = valid_mask*rendered_image_final + (1-valid_mask)*image_resized_rgb
                rendered_image_final = rendered_image_final[:, :, top_:top_+img_height_, left_:left_+img_width_]
            else:
                import copy
                rendered_image_final = copy.deepcopy(image_resized_rgb)
                rendered_image_final = rendered_image_final[:, :, top_:top_+img_height_, left_:left_+img_width_]

            grid_img = make_grid(rendered_image_final, nrow=10)
            grid_img = grid_img[[2,1,0], :, :]
            ndarr    = grid_img.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
            cv_ndarr = cv2.resize(ndarr, frame_size)
            if self.video_to_save is None:
                self.video_to_save = {
                    "video": cv2.VideoWriter(self.savePath, cv2.VideoWriter_fourcc(*'mp4v'), 24, frameSize=frame_size),
                    "path" : self.savePath,
                }
            self.video_to_save["video"].write(cv_ndarr)
        if(self.video_to_save is not None):
            self.video_to_save["video"].release()
            self.video_to_save = None
        print("Done!")
        shutil.rmtree(inpaintedFrames_path)
        return 

def extract_numbers(s):
    return int(''.join(filter(str.isdigit, s)))

def get_all_videos(video_dir="videos"): 
    video_files = [f for f in os.listdir(video_dir) if f.lower().endswith(".mp4")]
    sorted_videos = sorted(video_files, key=extract_numbers)
    return [f.split(".")[0] for f in sorted_videos]

def main() -> Optional[float]:
    parser = argparse.ArgumentParser(description="Process videos from a specified directory.")
    parser.add_argument('--mode', type=int, help='Mode (latent or noise; 0 for latent, 1 for noise)', required=True)
    parser.add_argument('--ckpt', type=str, help='Directory containing checkpoint to do inference on', required=True)
    parser.add_argument('--videoDir', type=str, help='Directory containing videos to run inference on', required=True)
    args = parser.parse_args()
    videoDir = os.path.join(HOME_DIR, args.videoDir)
    ckpt = args.ckpt
    for video_name in os.listdir(videoDir):
        start_time = time.time()
        pipeline = RunPipeline(video_name, videoDir, ckpt, args.mode)   
        pipeline.inpaint_person()
        pipeline.inference()
        end_time = time.time()
        elapsed_time_minutes = (end_time - start_time) / 60.0
        print(f"Done in {elapsed_time_minutes}")
        imgsPath = os.path.abspath("pipeline/environment/data/inference/{}".format(video_name))
        if os.path.exists(imgsPath):
            shutil.rmtree(imgsPath)
    print("Finished.")

if __name__ == "__main__":
    main()