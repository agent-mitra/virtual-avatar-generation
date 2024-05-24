from __future__ import absolute_import, division, print_function
import re
import copy
import datetime
import math
import os
import pickle
from typing import List
import shutil 
import cv2
import subprocess
import dill
import numpy as np
import scipy.stats as stats
import torch
from rich.progress import (BarColumn, Progress, SpinnerColumn,
                           TaskProgressColumn, TextColumn, TimeElapsedColumn,
                           TimeRemainingColumn, TransferSpeedColumn)
from phalp.utils.colors import phalp_colors, slahmr_colors


def get_progress_bar(sequence, total=None, description=None, disable=False):
    columns: List["ProgressColumn"] = (
            [TextColumn("[progress.description]{task.description}")] if description else []
        )
    columns.extend(
        (   
            SpinnerColumn(spinner_name="runner"),
            BarColumn(
                        style="bar.back",
                        complete_style="bar.complete",
                        finished_style="bar.finished",
                        pulse_style="bar.pulse",
                    ),
            TaskProgressColumn(show_speed=True),
            "eta :",
            TimeRemainingColumn(), # elapsed_when_finished=True
            " time elapsed :",
            TimeElapsedColumn(),
        )
    )

    progress_bar = Progress(*columns, 
                            auto_refresh=True,
                            console= None,
                            transient = False,
                            get_time = None, 
                            refresh_per_second = 10.0,
                            disable=disable)
    
    return progress_bar

def progress_bar(sequence, total=None, description=None, disable=False):
    progress_bar = get_progress_bar(sequence, total, description, disable)
    with progress_bar:
        yield from progress_bar.track(
            sequence, total=total, description=description, update_period=0.1
        )

def numpy_to_torch_image(ndarray):
    torch_image = torch.from_numpy(ndarray)
    torch_image = torch_image.unsqueeze(0)
    torch_image = torch_image.permute(0, 3, 1, 2)
    torch_image = torch_image[:, [2,1,0], :, :]
    return torch_image
  
def get_colors(pallette="phalp"):  

    try:
        if(pallette=="phalp"):
            colors = phalp_colors
        elif(pallette=="slahmr"):
            colors = slahmr_colors
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

def task_divider(data, batch_id, num_task):

    batch_length = len(data)//num_task
    start_       = batch_id*(batch_length+1)
    end_         = (batch_id+1)*(batch_length+1)
    if(start_>len(data)): exit()
    if(end_  >len(data)): end_ = len(data)
    data    = data[start_:end_] if batch_id>=0 else data

    return data

def get_prediction_interval(y, y_hat, x, x_hat):
    n     = y.size
    resid = y - y_hat
    s_err = np.sqrt(np.sum(resid**2) / (n-2))                    # standard deviation of the error
    t     = stats.t.ppf(0.975, n - 2)                            # used for CI and PI bands
    pi    = t * s_err * np.sqrt( 1 + 1/n + (x_hat - np.mean(x))**2 / np.sum((x - np.mean(x))**2))
    return pi

def pose_camera_vector_to_smpl(pose_camera_vector):
    # pose_camera_vector: [229,]
    global_orient = pose_camera_vector[:9].reshape(1,3,3)
    body_pose = pose_camera_vector[9:9+207].reshape(23,3,3)
    betas = pose_camera_vector[9+207:9+207+10].reshape(10,)
    camera = pose_camera_vector[9+207+10:9+207+10+3].reshape(1,3)
    camera[:,2] *= 200.0
    return {'global_orient': global_orient, 'body_pose': body_pose, 'betas': betas}, camera[0]

def smpl_to_pose_camera_vector(smpl_params, camera):
    # convert smpl parameters to camera to pose_camera_vector for smoothness.
    global_orient_  = smpl_params['global_orient'].reshape(1, -1) # 1x3x3 -> 9
    body_pose_      = smpl_params['body_pose'].reshape(1, -1) # 23x3x3 -> 207
    shape_          = smpl_params['betas'].reshape(1, -1) # 10 -> 10
    loca_           = copy.deepcopy(camera.view(1, -1)) # 3 -> 3
    loca_[:, 2]     = loca_[:, 2]/200.0
    pose_embedding  = np.concatenate((global_orient_, body_pose_, shape_, loca_.cpu().numpy()), 1)
    return pose_embedding

def convert_pkl(old_pkl):
    # Code adapted from https://github.com/nkolot/ProHMR
    # Convert SMPL pkl file to be compatible with Python 3
    # Script is from https://rebeccabilbro.github.io/convert-py2-pickles-to-py3/

    # Make a name for the new pickle
    new_pkl = os.path.splitext(os.path.basename(old_pkl))[0]+"_p3.pkl"

    # Convert Python 2 "ObjectType" to Python 3 object
    dill._dill._reverse_typemap["ObjectType"] = object

    # Open the pickle using latin1 encoding
    with open(old_pkl, "rb") as f:
        loaded = pickle.load(f, encoding="latin1")

    # Re-save as Python 3 pickle
    with open(new_pkl, "wb") as outfile:
        pickle.dump(loaded, outfile)

class FrameExtractor():
    '''
    Class used for extracting frames from a video file.
    '''
    def __init__(self, video_path):
        self.video_path = video_path
        # self.vid_cap    = cv2.VideoCapture(video_path)
        # self.n_frames   = int(self.vid_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # self.fps        = int(self.vid_cap.get(cv2.CAP_PROP_FPS))

    # def get_video_duration(self):
    #     duration = self.n_frames/self.fps
    #     print(f'Duration: {datetime.timedelta(seconds=duration)}')

    # def get_n_images(self, every_x_frame):
    #     n_images = math.floor(self.n_frames / every_x_frame) + 1
    #     print(f'Extracting every {every_x_frame} (nd/rd/th) frame would result in {n_images} images.')

    def get_modified_parent_directory(self, path):
        # Remove the trailing slash if it exists
        path = path.rstrip('/')
        parent_dir = os.path.dirname(path)
        # Add 'orig/' to the end
        modified_parent_dir = os.path.join(parent_dir, 'orig/')
        return modified_parent_dir

    def extract_frames(self, every_x_frame, img_name, dest_path=None, img_ext = '.jpg', start_frame=1000, end_frame=2000, width=368, height=640, render=False):
        if dest_path is None:
            dest_path = os.getcwd()
        else:
            if not os.path.isdir(dest_path):
                os.mkdir(dest_path); print(f'Created the following directory: {dest_path}')

        orig_dest_path = self.get_modified_parent_directory(dest_path)
        if not os.path.isdir(orig_dest_path):
            os.mkdir(orig_dest_path); print(f'Created the following directory as well: {orig_dest_path}')

        if not render:
            command = [
                'ffmpeg',
                '-i', self.video_path,
                '-copyts', 
                '-qscale:v', '2', 
                '-vf', f'fps=24,scale={width}:{height}',
                f'{dest_path}/%06d.jpg'
            ] #change resolution as well
            command_original = [
                'ffmpeg',
                '-i', self.video_path,
                '-copyts', 
                '-qscale:v', '2', 
                '-vf', f'fps=24',
                f'{orig_dest_path}/%06d.jpg'
            ] #change resolution as well
            print(command)
            subprocess.run(command)
            subprocess.run(command_original)
            first_image_path = f"{dest_path}/000001.jpg"
            # Read the first image
            image = cv2.imread(first_image_path)

            # Get its dimensions
            imgHeight, imgWidth, _ = image.shape
            assert(imgHeight==height)
            assert (imgWidth==width)
        else:
            command = [
                'ffmpeg',
                '-i', self.video_path,
                '-copyts', 
                '-qscale:v', '2', 
                '-vf', f'fps=24',
                f'{dest_path}/%06d.jpg'
            ] #change resolution as well
            # command = [
            #     'ffmpeg',
            #     '-i', self.video_path,
            #     '-copyts', 
            #     '-qscale:v', '2', 
            #     '-vf', f'fps=1',
            #     f'{dest_path}/%06d.jpg'
            # ]
            print(command)
            subprocess.run(command)
            # print("COPYING..")
            # video_frame_count = get_frame_count(self.video_path) #already at 24 fps so
            # duplicate_frames(dest_path, orig_dest_path, video_frame_count)
            # print("Finished copying!")
        

def get_frame_count(video_path):
    try:
        cmd = ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=nb_frames', '-of', 'default=nokey=1:noprint_wrappers=1', video_path]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return int(result.stdout)
    except Exception as e:
        print(f"Error getting frame count: {e}")
        return 0

def duplicate_frames(src_directory, temp_directory, total_frames_24fps):
    # Ensure temporary directory exists
    os.makedirs(temp_directory, exist_ok=True)

    # Get the list of original frames
    extracted_frames = [f for f in os.listdir(src_directory) if f.endswith('.jpg')]
    extracted_frames.sort()
    print(total_frames_24fps)
    print(len(extracted_frames))

    # Calculate how many times each frame should be duplicated
    duplicates_per_frame = total_frames_24fps // len(extracted_frames)

    current_frame_index = 0
    for i in range(total_frames_24fps):
        src_frame = extracted_frames[current_frame_index]
        src_path = os.path.join(src_directory, src_frame)
        dest_frame = f'{i+1:06d}.jpg'  # Adjusted for the naming convention
        temp_dest_path = os.path.join(temp_directory, dest_frame)

        shutil.copy(src_path, temp_dest_path)

        if (i + 1) % duplicates_per_frame == 0 and current_frame_index < len(extracted_frames) - 1:
            current_frame_index += 1

    # Move the duplicated frames from temp directory to source directory
    for frame in os.listdir(temp_directory):
        shutil.move(os.path.join(temp_directory, frame), os.path.join(src_directory, frame))
        

def get_video_duration(video_path):
    try:
        # Run the ffmpeg command and capture stderr
        result = subprocess.run(['ffmpeg', '-i', video_path], stderr=subprocess.PIPE, stdout=subprocess.PIPE)
        output = result.stderr.decode('utf-8')

        # Search for the duration in the stderr output
        match = re.search(r'Duration: (\d{2}):(\d{2}):(\d{2})\.\d+', output)
        if match:
            hours, minutes, seconds = map(int, match.groups())
            return hours * 3600 + minutes * 60 + seconds
    except Exception as e:
        print(f"Error getting video duration: {e}")
    return 0

def perspective_projection(points, rotation, translation,
                           focal_length, camera_center):
    """
    This function computes the perspective projection of a set of points
    Input:
        points (bs, N, 3): 3D points
        rotation (bs, 3, 3): Camera rotation
        translation (bs, 3): Camera translation
        focal_length (bs,) or scalar: Focal length
        camera_center (bs, 2): Camera center
    """
    batch_size = points.shape[0]
    K = torch.zeros([batch_size, 3, 3], device=points.device, dtype=points.dtype)
    K[:,0,0] = focal_length[:,0]
    K[:,1,1] = focal_length[:,1]
    K[:,2,2] = 1.
    K[:,:-1, -1] = camera_center

    # Transform points    
    points = torch.einsum('bij,bkj->bki', rotation, points)
    points = points + translation.unsqueeze(1)

    # Apply perspective distortion
    projected_points = points / (points[:,:,-1].unsqueeze(-1))
    
    # Apply camera intrinsics
    projected_points = torch.einsum('bij,bkj->bki', K, projected_points)

    return projected_points[:, :, :-1]
        
def compute_uvsampler(vt, ft, tex_size=6):
    """
    For this mesh, pre-computes the UV coordinates for
    F x T x T points.
    Returns F x T x T x 2
    """
    uv = obj2nmr_uvmap(ft, vt, tex_size=tex_size)
    uv = uv.reshape(-1, tex_size, tex_size, 2)
    return uv

def obj2nmr_uvmap(ft, vt, tex_size=6):
    """
    Converts obj uv_map to NMR uv_map (F x T x T x 2),
    where tex_size (T) is the sample rate on each face.
    """
    # This is F x 3 x 2
    uv_map_for_verts = vt[ft]

    # obj's y coordinate is [1-0], but image is [0-1]
    uv_map_for_verts[:, :, 1] = 1 - uv_map_for_verts[:, :, 1]

    # range [0, 1] -> [-1, 1]
    uv_map_for_verts = (2 * uv_map_for_verts) - 1

    alpha = np.arange(tex_size, dtype=np.float64) / (tex_size - 1)
    beta = np.arange(tex_size, dtype=np.float64) / (tex_size - 1)
    import itertools

    # Barycentric coordinate values
    coords = np.stack([p for p in itertools.product(*[alpha, beta])])

    # Compute alpha, beta (this is the same order as NMR)
    v2 = uv_map_for_verts[:, 2]
    v0v2 = uv_map_for_verts[:, 0] - uv_map_for_verts[:, 2]
    v1v2 = uv_map_for_verts[:, 1] - uv_map_for_verts[:, 2]
    # Interpolate the vertex uv values: F x 2 x T*2
    uv_map = np.dstack([v0v2, v1v2]).dot(coords.T) + v2.reshape(-1, 2, 1)

    # F x T*2 x 2  -> F x T x T x 2
    uv_map = np.transpose(uv_map, (0, 2, 1)).reshape(-1, tex_size, tex_size, 2)

    return uv_map
