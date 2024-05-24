#Segment and track
import os
import cv2
from SegTracker import SegTracker
from model_args import aot_args,sam_args,segtracker_args
from PIL import Image
from aot_tracker import _palette
import numpy as np
import torch
import imageio
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation
import gc
from collections import defaultdict
# from typing import List, Dict, list, dict
import mmcv, math, tempfile
import subprocess
import time, argparse
import logging, traceback
import shutil, joblib
from pathlib import Path
from model import AppModel
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import numpy as np

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig

from phalp.configs.base import FullConfig
from phalp.models.hmar.hmr import HMR2018Predictor
from phalp.utils import get_pylogger

warnings.filterwarnings('ignore')

log = get_pylogger(__name__)

#Cmd: python3 run_pipeline.py
logging.basicConfig(level=logging.INFO)

log = get_pylogger(__name__)
HOME_DIR = os.path.join(os.environ.get("HOME"), "SABR-Coach")

class HMR2Predictor(HMR2018Predictor):
    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        # Setup our new model
        from hmr2.models import download_models, load_hmr2

        # Download and load checkpoints
        download_models()
        model, _ = load_hmr2()

        self.model = model
        self.model.eval()

    def forward(self, x):
        hmar_out = self.hmar_old(x)
        batch = {
            'img': x[:,:3,:,:],
            'mask': (x[:,3,:,:]).clip(0,1),
        }
        model_out = self.model(batch)
        out = hmar_out | {
            'pose_smpl': model_out['pred_smpl_params'],
            'pred_cam': model_out['pred_cam'],
        }
        return out

class HMR2_4dhuman(AppModel):
    def __init__(self, cfg):
        super().__init__(cfg)

    def setup_hmr(self):
        self.HMAR = HMR2Predictor(self.cfg)

@dataclass
class Human4DConfig(FullConfig):
    # override defaults if needed
    pass

# cs = ConfigStore.instance()
# cs.store(name="config", node=Human4DConfig)

full_config_instance = FullConfig()

def save_prediction(pred_mask,output_dir,file_name):
    white_palette = [255, 255, 255] * 256
    save_mask = Image.fromarray(pred_mask.astype(np.uint8))
    save_mask = save_mask.convert(mode='P')
    #save_mask.putpalette(_palette)
    save_mask.putpalette(white_palette)
    save_mask.save(os.path.join(output_dir,file_name))
    
def colorize_mask(pred_mask, obj_id=None):
    white_palette = [255, 255, 255] * 256
    save_mask = Image.fromarray(pred_mask.astype(np.uint8))
    save_mask = save_mask.convert(mode='P')
    #save_mask.putpalette(_palette)
    save_mask.putpalette(white_palette)
    save_mask = save_mask.convert(mode='RGB')
    return np.array(save_mask)

def draw_mask(img, mask, obj_ids=[], alpha=0.5, intensity_reduction=1): #was 0.5): no need for intensity reduction for visual in context prompting
    # Check if any object IDs are specified
    if obj_ids:
        mask_to_zero_out = ~np.isin(mask, obj_ids)
        mask[mask_to_zero_out] = 0

    # Create a binary mask for the objects
    binary_mask = (mask != 0)
    contours = binary_dilation(binary_mask, iterations=1) ^ binary_mask

    # Colorize only the specified objects
    colorized_objects = img * (1 - alpha) + colorize_mask(mask) * alpha

    # Reduce the intensity of the non-masked areas
    reduced_intensity_img = img * intensity_reduction

    # Combine the colorized objects with the reduced intensity background
    img_mask = np.zeros_like(img)
    img_mask[:] = reduced_intensity_img
    img_mask[binary_mask] = colorized_objects[binary_mask]
    img_mask[contours, :] = 0

    return img_mask.astype(img.dtype)

def visualize_pose_results(framesPerSecond, offset, video_path, oldName,
                               pose_preds_all,
                               kpt_score_threshold, vis_dot_radius,
                               vis_line_thickness, empty_f_i=[],bounding_boxes={}):
        if video_path is None or pose_preds_all is None:
            return
        cap = cv2.VideoCapture(video_path)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        fps = cap.get(cv2.CAP_PROP_FPS)
        print(video_path)
        print(f"FPS here is: {framesPerSecond}, writing to a {int(framesPerSecond)} video")
        frame_idx = 0
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        temp_file = tempfile.NamedTemporaryFile(suffix='.mp4')
        path = oldName
        head, tail = os.path.split(path)
        writer = cv2.VideoWriter(f"{HOME_DIR}/virtual-avatar-generation/utils/collect_transformer_samples/visualized_2d_pose_videos/{tail}", fourcc, framesPerSecond, (width, height))
        idx = 0
        while True:
            ok, frame = cap.read()
            frame_idx += 1
            if not ok:
                break
            if frame_idx in empty_f_i:
                continue
            frameName = f"frame_{frame_idx}"
            pose_preds = pose_preds_all[frameName]
            indicesToPlot = [17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 
                            38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54]
            joints = pose_preds[indicesToPlot]
            for x, y, _ in joints:
                # Adjust coordinates for the new frame size
                x_coord = int(x) # * frame.shape[1])
                y_coord = int(y) # * frame.shape[0])
                cv2.circle(frame, (x_coord, y_coord), 5, (0, 255, 0), -1)
            cv2.rectangle(frame, (bounding_boxes[frameName][0], bounding_boxes[frameName][1]), (bounding_boxes[frameName][2], bounding_boxes[frameName][3]), (255, 0, 0), 2)
            #END
            writer.write(frame)
            idx += 1
        cap.release()
        writer.release()
        out_file = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
        return out_file.name

class HumanPathPipeline:
    def __init__(self, cfg, video_name, video_dir, inpainted_video_dir, processNumber):
        self.video_name = video_name
        self.video_path = video_dir
        self.fgt_tool_dir = os.path.abspath("FGT/tool")
        self.sabr_dir = os.getcwd()  # Store the initial SABR directory
        self.highDimension = 640
        self.lowDimension = 368
        self.pose_model_with_depth = HMR2_4dhuman(cfg)
        self.det_score_threshold=0.35
        self.indicesToPlotLen = 0
        self.process_number = processNumber
        self.io_args = {
            'input_video': inpainted_video_dir,
            'output_mask_dir': f'./assets/{video_name}_masks', # save pred masks
            'output_video': f'./assets/{video_name}_seg.mp4', # mask+frame vizualization, mp4 or avi, else the same as input video
            'output_gif': f'./assets/{video_name}_seg.gif', # mask visualization
        }

        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print("Error: Could not open video file.")
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # Release the video capture object
        cap.release()

    def run_fgt_tool(self):
        result_path = f"{self.sabr_dir}/FGT/data/results"
        script_path = f"{self.fgt_tool_dir}/video_inpainting.py"
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
        subprocess.run(cmd, cwd=self.fgt_tool_dir, check=True)
        return
    
    def tuneSAM(self, offset, empty_fi):
        cap = cv2.VideoCapture(self.io_args['input_video'])
        frame_idx = 0
        segtracker = SegTracker(segtracker_args,sam_args,aot_args)
        segtracker.restart_tracker()
        with torch.cuda.amp.autocast():
            while cap.isOpened():
                if (frame_idx + 1) in empty_fi: #must take care of pose offset here
                    frame_idx += 1
                    continue     
                print(f"Tuning SAM on frame: {frame_idx}")          
                ret, frame = cap.read()
                frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                pred_mask = segtracker.seg(frame)
                #print(len(pred_mask), pred_mask.shape)
                torch.cuda.empty_cache()
                obj_ids = np.unique(pred_mask)
                obj_ids = obj_ids[obj_ids!=0]
                print("processed frame {}, obj_num {}".format(frame_idx,len(obj_ids)),end='\n')
                break
            cap.release()
            del segtracker
            torch.cuda.empty_cache()
            gc.collect()
            
    def runSAMOnVideo(self, relevant_joint_coordinates, offset=1, fps=24, empty_f_i=[], interactionFramesPeriod=24*1):
        # For every sam_gap frames, we use SAM to find new objects and add them for tracking
        # larger sam_gap is faster but may not spot new objects in time
        totalFrames = len(relevant_joint_coordinates)
        print(f"Total joint coordinates length is {totalFrames}")
        segtracker_args = {
            'sam_gap': totalFrames//5, #fps*3, #use *2 or *3, # the interval to run sam to segment new objects #can make it * 5 or 10
            'min_area': 0, # minimal mask area to add a new mask as a new object
            'max_obj_num': 10000, # maximal object number to track in a video
            'min_new_obj_iou': 0.5, # the area of a new object in the background should > 80% 
        }

        # source video to segment
        with tempfile.NamedTemporaryFile(delete=True, suffix=".mp4") as temp_converted_video:
            ffmpeg_command = [
                'ffmpeg', '-y', '-i', self.io_args['input_video'],
                '-vf', f'fps={fps}', 
                '-an', temp_converted_video.name, '-loglevel', 'quiet'
            ]
            if fps != 24:
                print(f"Running at fps {fps}")
                subprocess.run(ffmpeg_command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                output_file = temp_converted_video.name
            else:
                output_file = self.io_args['input_video']
            cap = cv2.VideoCapture(output_file)
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            print(f"Total number of frames: {total_frames}")
            # output masks
            output_dir = self.io_args['output_mask_dir']
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            pred_list = {}
            masked_pred_list = []

            torch.cuda.empty_cache()
            gc.collect()
            sam_gap = segtracker_args['sam_gap']
            frame_idx = 0
            isFirst = True
            contact_counters = defaultdict(lambda: [0, 0])
            segtracker = SegTracker(segtracker_args,sam_args,aot_args)
            segtracker.restart_tracker()
            frameTrack = 0
            empty_f_i = set(empty_f_i)
            with torch.cuda.amp.autocast():
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frameTrack += 1
                    if (frame_idx + 1 in empty_f_i or (f"frame_{frame_idx + 1}" not in relevant_joint_coordinates)): #must take care of pose offset here
                        frame_idx += 1
                        continue
                    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                    if isFirst:
                        isFirst = False
                        print("first frame")
                        pred_mask = segtracker.seg(frame)
                        # torch.cuda.empty_cache()
                        # gc.collect()
                        segtracker.add_reference(frame, pred_mask)
                    elif ((frame_idx % sam_gap) == 0): # or ((totalFrames - frame_idx + 1) <= 7):
                        print("sam gap")
                        seg_mask = segtracker.seg(frame)
                        # torch.cuda.empty_cache()
                        # gc.collect()
                        track_mask = segtracker.track(frame)
                        # find new objects, and update tracker with new objects
                        new_obj_mask = segtracker.find_new_objs(track_mask,seg_mask)
                        #save_prediction(new_obj_mask,output_dir,str(frame_idx)+'_new.png')
                        pred_mask = track_mask + new_obj_mask
                        segtracker.add_reference(frame, pred_mask)
                        pred_mask = segtracker.track(frame, update_memory=True)
                    else:
                        pred_mask = segtracker.track(frame,update_memory=True)

                    joint_coords = relevant_joint_coordinates[f"frame_{frame_idx + 1}"]
                    obj_ids = np.unique(pred_mask)
                    obj_ids = obj_ids[obj_ids != 0]
                    obj_ids_in_frame = set(obj_ids)
                    for obj_id in obj_ids:
                        if self.is_touching(joint_coords, pred_mask, obj_id):
                            obj_ids_in_frame.add(obj_id)
                            contact_counters[obj_id][0] += 1
                            contact_counters[obj_id][1] = max(contact_counters[obj_id][0], contact_counters[obj_id][1])
                        else:
                            contact_counters[obj_id][0] = 0
                    for obj_id in set(contact_counters.keys()) - obj_ids_in_frame:
                        contact_counters[obj_id][0] = 0                       
                    torch.cuda.empty_cache()
                    gc.collect()
                    #save_prediction(pred_mask,output_dir,str(frame_idx)+'.png')
                    pred_list[f"frame_{frame_idx + 1}"] = pred_mask
                    
                    
                    print("processed frame {}, obj_num {}".format(frame_idx,segtracker.get_obj_num()),end='\r')
                    frame_idx += 1
                cap.release()
                print('\nfinished')
            #saving video visualization here
            cap = cv2.VideoCapture(output_file)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_idx = 0
            totalIds = []
            frame_counts = []
            for obj_id, counts in contact_counters.items():
                if counts[1] >= interactionFramesPeriod:
                    totalIds.append(obj_id)
            obj_id_frames = {obj_id: (-1, 0) for obj_id in totalIds}  # Initialize to track max area and frame index
            all_frames = []

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                if (frame_idx + 1 in empty_f_i or (f"frame_{frame_idx + 1}" not in pred_list)):
                    frame_idx += 1
                    continue
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pred_mask = pred_list[f"frame_{frame_idx + 1}"]
                masked_frame = draw_mask(frame, pred_mask, obj_ids=totalIds)
                masked_frame = cv2.cvtColor(masked_frame, cv2.COLOR_RGB2BGR)

                # Calculate segmentation area for present object IDs
                for obj_id in totalIds:
                    mask = pred_mask == obj_id
                    area = np.sum(mask)  # Calculate area
                    if area > obj_id_frames[obj_id][1]:  # Get the largest area
                        obj_id_frames[obj_id] = (frame_idx, area)
                
                all_frames.append((frame_idx, masked_frame))
                print(f'frame {frame_idx} processed', end='\r')
                frame_idx += 1

            cap.release()

            # Extract frame indices with the maximum area for each object ID and sort them
            selected_frames_indices = sorted({frame_data[0] for frame_data in obj_id_frames.values()})

            # Extract and sort the selected frames based on frame indices
            final_frames = [(idx, frame) for idx, frame in all_frames if idx in selected_frames_indices]

            # Ensure final frames are sorted by their indices to maintain chronological order
            top_frames = sorted(final_frames, key=lambda x: x[0])
            
            processDir = f"assets/contextFrames/{self.process_number}"
            if not os.path.exists(processDir):
                os.makedirs(processDir)
            
            videoContextFramesDir = f"assets/contextFrames/{self.process_number}/{self.video_name}"
            if not os.path.exists(videoContextFramesDir):
                os.makedirs(videoContextFramesDir)
                
            for frame_number, frame in top_frames:
                frame_path = os.path.join(f"assets/contextFrames/{self.process_number}/{self.video_name}/frame_{frame_number}.jpg")
                cv2.imwrite(frame_path, frame)
                    
            # print("\n{} saved".format(self.io_args['output_video']))
            print('\nfinished')
            del segtracker, frame_counts
            torch.cuda.empty_cache()
            gc.collect()
            
    def is_touching(self, joint_coords, pred_mask, obj_id,  max_area_threshold=10000, max_relative_size=0.1):
        frame_area = pred_mask.shape[0] * pred_mask.shape[1]
        assert len(joint_coords) == self.indicesToPlotLen
        for coord in joint_coords:
            x, y, confidence_score = coord
            x_coord = int(x) # * frame.shape[1])
            y_coord = int(y) # * frame.shape[0])
            if confidence_score > 0.35:
                x_coord = min(x_coord, pred_mask.shape[1] - 1)
                y_coord = min(y_coord, pred_mask.shape[0] - 1)
                if pred_mask[y_coord, x_coord] == obj_id:
                    obj_area = np.sum(pred_mask == obj_id)
                    relative_size = obj_area / frame_area
                    if relative_size <= max_relative_size:
                        return True
        return False
    
    def get_touched_objects(self, fps):
        #temp file turn video into 24 fps and run on there
        indicesToPlot = [17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 
                            38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54] #hand and foot indices only
        self.indicesToPlotLen = len(indicesToPlot)
        width, height = self.highDimension, self.lowDimension
        if self.height > self.width:
            height, width = self.highDimension, self.lowDimension
        elif self.width > self.height:
            height, width = self.lowDimension, self.highDimension
        else:
            height, width = self.lowDimension, self.lowDimension
        relevant_joint_coordinates = {}
        with tempfile.NamedTemporaryFile(delete=True, suffix=".mp4") as temp_converted_video:
            ffmpeg_command = [
                'ffmpeg', '-y', '-i', self.video_path,
                '-vf', f'scale={width}:{height},fps={fps}', 
                '-an', temp_converted_video.name, '-loglevel', 'quiet'
            ]
            subprocess.run(ffmpeg_command, check=True)
            output_file = temp_converted_video.name
            cap = cv2.VideoCapture(output_file)
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            #get solo preds here maybe
            pose_preds_all, offset, empty_f_i, bounding_boxes = self.pose_model_with_depth.run(output_file, self.det_score_threshold)
            #visualize_pose_results(fps, offset, output_file, self.video_path, pose_preds_all, 0.3, 4, 2, empty_f_i=empty_f_i,bounding_boxes=bounding_boxes)
            for pose_preds in pose_preds_all:
                relevant_joint_coordinates[pose_preds] = pose_preds_all[pose_preds][indicesToPlot]
        assert len(relevant_joint_coordinates) == len(pose_preds_all)
        #so now you have the relevant joints for each frame, now do the SAM track (on FGT-ed video) and do the blob intersection logic here with ttl 
        self.tuneSAM(offset=offset,empty_fi=empty_f_i)
        self.runSAMOnVideo(relevant_joint_coordinates, offset=offset, fps=fps, empty_f_i=empty_f_i, interactionFramesPeriod=fps*3)
        #now since you have the frames with the rock ids, segment those rock ids and gray out rest of the frame and return them (or save them here)
    

def extract_numbers(s):
    return int(''.join(filter(str.isdigit, s)))

def get_all_videos(video_dir="videos"): 
    video_files = [f for f in os.listdir(video_dir) if f.lower().endswith(".mp4")]
    sorted_videos = sorted(video_files, key=extract_numbers)
    return [f.split(".")[0] for f in sorted_videos]   

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process videos to get climbing route frames as context for transformer model.")
    parser.add_argument("--originalVideoDir", type=str, required=True, help="Video directory of original (raw) climbing video.")
    parser.add_argument("--inpaintedVideoDir", type=str, required=True, help="Video directory of inpainted climbing video.")
    parser.add_argument("--processNumber", type=int, required=True, help="Machine running this data collection process.")
    args = parser.parse_args()
    videoDir = args.originalVideoDir
    inpainted_video_dir = args.inpaintedVideoDir
    process_number = args.processNumber
    video_name = os.path.basename(videoDir)
    video_name = f"{video_name[:len(video_name)-4]}"
    start_time = time.time()
    logging.info(f"Processing {video_name}")
    fps = 5
    pipeline = HumanPathPipeline(full_config_instance, video_name, videoDir, inpainted_video_dir, process_number) 
    pipeline.get_touched_objects(fps)  
    end_time = time.time()
    elapsed_time_minutes = (end_time - start_time) / 60.0
    logging.info(f"Completed {video_name} in {elapsed_time_minutes} minutes")
    masksPath = os.path.abspath("assets/{}_masks".format(video_name))
    if os.path.exists(masksPath):
        shutil.rmtree(masksPath)
    print("Finished.")