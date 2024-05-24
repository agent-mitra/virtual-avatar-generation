import subprocess
import os, time, cv2
import logging, traceback
from botocore.exceptions import ClientError
import boto3, io, shutil, argparse, joblib, io
import torch, gc
from collections import defaultdict
from botocore.config import Config
from torch.nn import functional as F

logging.basicConfig(level=logging.INFO)
HOME_DIR = os.path.join(os.environ.get("HOME"), "virtual-avatar-generation")

#CHANGE TO WHOEVER YOUR CLOUD IS
def create_client():
    region_config = Config(
        region_name = os.getenv('CLOUDFLARE_REGION_NAME')
    )
    
    return boto3.client(
        's3',
        aws_access_key_id=os.getenv('CLOUDFLARE_ACCESS_KEY_ID'),
        aws_secret_access_key=os.getenv('CLOUDFLARE_SECRET_ACCESS_KEY'),
        config=region_config,
        endpoint_url=os.getenv('CLOUDFLARE_ENDPOINT_URL')
    )

class RunPipeline:
    def __init__(self, video_name, video_dir, s3_client):
        self.video_name = video_name
        self.video_dir = video_dir
        self.video_path = f"{video_dir}/{video_name}.mp4"
        self.mocap_dir = f"{HOME_DIR}/pipeline/track"
        self.inpaint_tool_dir = f"{HOME_DIR}/pipeline/environment/tool"
        self.context_tool_dir = f"{HOME_DIR}/pipeline/context"
        self.output_dir = f"{HOME_DIR}/pipeline/track/outputs"
        self.highDimension = 640 #max dimension to downsample video to 
        self.lowDimension = 368 #min dimension to downsample video to
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            logging.info("Error: Could not open video file.")
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.processNumber = 1
        self.padLength = 1250
        self.s3_client = s3_client
        cap.release()

    def run_mocap_demo(self, retries=1):
        width = self.lowDimension
        height = self.lowDimension
        if self.height > self.width:
            width = self.lowDimension
            height = self.highDimension
        elif self.width > self.height:
            width = self.highDimension
            height = self.lowDimension
        
        cmd = [
            "python",
            "scripts/demo.py",
            f"video.source={self.video_path}",
            f"video.output_dir={self.output_dir}",
            f"video.width={width}",
            f"video.height={height}"
        ]
        if width != height: #perfectly square videos will not fit in batch size when training the model, so ignore
            subprocess.run(cmd, cwd=self.mocap_dir, check=True)

    def count_images(self):
        image_extensions = ['.jpg', '.png', '.jpeg', '.bmp', '.tiff']
        target_directory = f"{HOME_DIR}/pipeline/track/outputs/_DEMO/{self.video_name}/img"
        if not os.path.exists(target_directory):
            logging.info(f"Directory {target_directory} does not exist.")
            return 0
        all_files = os.listdir(target_directory)
        image_count = sum(1 for file in all_files if any(file.endswith(ext) for ext in image_extensions))
        return image_count

    def run_inpaint_tool(self, retries=1):
        frame_path = f"{HOME_DIR}/pipeline/track/outputs/_DEMO/{self.video_name}/orig"
        mask_path = f"{HOME_DIR}/pipeline/track/outputs/_DEMO/{self.video_name}/mask"
        result_path = f"{HOME_DIR}/pipeline/environment/data/results"
        script_path = f"{self.inpaint_tool_dir}/video_inpainting.py"
        context_path = f"{self.context_tool_dir}/run_pipeline.py"
        image_count = self.count_images()
        logging.info(f"Number of frames is {image_count} , with width: {self.width} and height: {self.height}")
        if self.height > self.width:
            cmd = [
                "python",
                script_path,
                "--path", frame_path,
                "--path_mask", mask_path,
                "--outroot", result_path,
                "--imgH", f"{self.highDimension}",
                "--imgW", f"{self.lowDimension}"
            ]
        elif self.width > self.height:
            cmd = [
                "python",
                script_path,
                "--path", frame_path,
                "--path_mask", mask_path,
                "--outroot", result_path,
                "--imgH", f"{self.lowDimension}",
                "--imgW", f"{self.highDimension}"
            ]
        else:
            cmd = [
                "python",
                script_path,
                "--path", frame_path,
                "--path_mask", mask_path,
                "--outroot", result_path,
                "--imgH", f"{self.lowDimension}",
                "--imgW", f"{self.lowDimension}"
            ]
            return #perfectly square videos will not fit in batch size when training the model, so ignore
        subprocess.run(cmd, cwd=self.inpaint_tool_dir, check=True)
        #get context frames (aka video frames with route annotated) #TODO: can comment this part out if you are using text context or other types
        getContextFrames = [
            "python", 
            context_path,
            "--originalVideoDir", self.video_path,
            "--inpaintedVideoDir", f"{result_path}/{self.video_name}.mp4",
            "--processNumber", str(self.processNumber)
        ]
        subprocess.run(getContextFrames, cwd=self.context_tool_dir, check=True)
        return

    def padTensor(self, item):
        totalPads = len(item.shape) * 2
        padConfig = [0] * totalPads
        seqLengthIdxToAlter = totalPads - 1
        padConfig[seqLengthIdxToAlter] = max(0, self.padLength - item.shape[0])
        padConfig = tuple(padConfig)
        item = F.pad(item, padConfig, 'constant', 0)
        del totalPads, padConfig
        return item

    def upload_tensors(self):
        self.initLength = len(os.listdir(f"{self.context_tool_dir}/assets/contextFrames/{self.processNumber}/{self.video_name}"))
        videoPklFile = joblib.load(f"{self.output_dir}/results/demo_{self.video_name}.pkl")
        frameKeys = list(videoPklFile.keys()) #total number of frames
        item = defaultdict(dict)
    
        zeros_tensor = torch.zeros(self.initLength)
        ones_tensor = torch.ones(len(frameKeys))
        values_to_concatenate = torch.cat([zeros_tensor, ones_tensor], dim=0)
        item['frame_validity'] = self.padTensor(values_to_concatenate.unsqueeze(-1))
        del zeros_tensor, ones_tensor, values_to_concatenate
        
        item['image_width'] = self.width
        item['image_height'] = self.height
        
        zeros_tensor = torch.zeros(self.initLength, 3)
        for frame in frameKeys:
            if len(videoPklFile[frame]['camera_bbox']) == 0:
                videoPklFile.pop(frame)
        frameKeys = list(videoPklFile.keys())
        
        ones_tensor = torch.stack([torch.tensor(videoPklFile[frame]['camera_bbox'][0]) for frame in frameKeys])
        values_to_concatenate = torch.cat([zeros_tensor, ones_tensor], dim=0)
        item['camera_3D'] = self.padTensor(values_to_concatenate)
        del zeros_tensor, ones_tensor, values_to_concatenate           
        
        zeros_tensor = torch.zeros(self.initLength, 3)
        ones_tensor = torch.stack([torch.tensor(videoPklFile[frame]['camera'][0]) for frame in frameKeys])
        values_to_concatenate = torch.cat([zeros_tensor, ones_tensor], dim=0)
        item['camera_for_rendering_on_image'] = self.padTensor(values_to_concatenate) #just in case; do not use in loss, etc. since this should be derived algorithmically
        del zeros_tensor, ones_tensor, values_to_concatenate
        
        zeros_tensor = torch.zeros(self.initLength, 4)
        ones_tensor = torch.stack([torch.tensor(videoPklFile[frame]['bbox'][0]) for frame in frameKeys])  #mocap (PHALP) preds are in x0 y0 w h format (should be rendered in that format, too) so so is bbox.
        values_to_concatenate = torch.cat([zeros_tensor, ones_tensor], dim=0)
        item['person_bbox'] = self.padTensor(values_to_concatenate)
        del zeros_tensor, ones_tensor, values_to_concatenate
        
        zeros_tensor = torch.zeros(self.initLength, 45, 3)
        ones_tensor = torch.stack([torch.tensor(videoPklFile[frame]['3d_joints'][0]) for frame in frameKeys])
        values_to_concatenate = torch.cat([zeros_tensor, ones_tensor], dim=0)
        item['3d_joints'] = self.padTensor(values_to_concatenate)
        del zeros_tensor, ones_tensor, values_to_concatenate
        
        zeros_tensor = torch.zeros(self.initLength, 90)
        ones_tensor = torch.stack([torch.tensor(videoPklFile[frame]['2d_joints'][0]) for frame in frameKeys])
        values_to_concatenate = torch.cat([zeros_tensor, ones_tensor], dim=0)
        item['2d_joints'] = self.padTensor(values_to_concatenate)
        del zeros_tensor, ones_tensor, values_to_concatenate
        
        zeros_tensor = torch.zeros(self.initLength, 1, 3, 3) 
        ones_tensor = torch.stack([torch.tensor(videoPklFile[frame]['smpl'][0]['global_orient']) for frame in frameKeys])
        values_to_concatenate = torch.cat([zeros_tensor, ones_tensor], dim=0)
        item['smpl']['global_orient'] = self.padTensor(values_to_concatenate)
        del zeros_tensor, ones_tensor, values_to_concatenate
        
        zeros_tensor = torch.zeros(self.initLength, 23, 3, 3)
        ones_tensor = torch.stack([torch.tensor(videoPklFile[frame]['smpl'][0]['body_pose']) for frame in frameKeys])
        values_to_concatenate = torch.cat([zeros_tensor, ones_tensor], dim=0)
        print(f"values to concatenate shape is {values_to_concatenate.shape}")
        item['smpl']['body_pose'] = self.padTensor(values_to_concatenate)
        del zeros_tensor, ones_tensor, values_to_concatenate
        
        zeros_tensor = torch.zeros(self.initLength, 10) 
        ones_tensor = torch.stack([torch.tensor(videoPklFile[frame]['smpl'][0]['betas']) for frame in frameKeys])
        values_to_concatenate = torch.cat([zeros_tensor, ones_tensor], dim=0)
        item['smpl']['betas'] = self.padTensor(values_to_concatenate)
        del zeros_tensor, ones_tensor, values_to_concatenate

        #Comment out as necessary (this is just for saving files locally)
        dest_path = os.path.join(HOME_DIR, f"savedTensors/")
        if not os.path.isdir(dest_path):
            os.mkdir(dest_path); print(f'Created the following directory: {dest_path}')
        file_path = f'{dest_path}/training_data_{self.video_name}.pt'
        torch.save(item, file_path)
        print(f"Finished saving training data {self.video_name} locally!")

        #start saving code to cloud here, uncomment as necessary
        # buffer = io.BytesIO()
        # torch.save(item, buffer)
        # buffer.seek(0)
        # self.s3_client.put_object(Bucket=self.bucketName, Key=f'training_data_{self.video_name}.pt', Body=buffer)
        # logging.info(f"Finished saving training data {self.video_name} to Cloudflare R2!")
        del videoPklFile, item
        with torch.no_grad():
            torch.cuda.empty_cache()
        gc.collect()


def extract_numbers(s):
    return int(''.join(filter(str.isdigit, s)))

def get_all_videos(video_dir="videos"): 
    video_files = [f for f in os.listdir(video_dir) if f.lower().endswith(".mp4")]
    sorted_videos = sorted(video_files, key=extract_numbers)
    return [f.split(".")[0] for f in sorted_videos]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process videos from a specified directory.")
    parser.add_argument('--videoDir', type=str, help='Directory containing videos to process', required=True)
    args = parser.parse_args()
    videoDir = os.path.join(HOME_DIR, args.videoDir)
    s3_client = None #create_client() #TODO: uncomment as necessary (if saving in cloud)
    runNumber = 1 #TODO: change this if you want to modify which directory videos are being saved in in your cloud service (AWS or Cloudflare)
    all_videos = get_all_videos(videoDir)
    total_videos = len(all_videos)
    logging.info(f"Total videos is {total_videos}")
    for idx, video_name in enumerate(all_videos):
        curVideoNumber = extract_numbers(video_name)
        start_time = time.time()
        logging.info(f"Processing video {idx+1}/{total_videos}: {video_name}")
        pipeline = RunPipeline(video_name, videoDir, s3_client)
        pipeline.run_mocap_demo()   
        pipeline.run_inpaint_tool()
        pipeline.upload_tensors()
        end_time = time.time()
        elapsed_time_minutes = (end_time - start_time) / 60.0
        logging.info(f"Completed video {idx+1}/{total_videos}: {video_name} in {elapsed_time_minutes} minutes")
        
        #start saving code to cloud here, uncomment as necessary
        # inpaintedVideo = os.path.abspath(f"pipeline/environment/data/results/{video_name}.mp4")
        # mocapPreds = os.path.abspath(f"pipeline/track/outputs/results/preds_{video_name}.pkl")
        # mocapViz = os.path.abspath(f"pipeline/track/outputs/viz_{video_name}.mp4")
        
        # files_to_upload = {
        #     "inpaintedVideo": ("inpainted-videos", f"run-{runNumber}/{video_name}.mp4", os.path.abspath("pipeline/environment/data/results/{}.mp4".format(video_name))),
        #     "mocapPreds": ("mocap-predictions", f"run-{runNumber}/preds_{video_name}.pkl", os.path.abspath("pipeline/track/outputs/results/demo_{}.pkl".format(video_name))),
        #     "mocapViz": ("mocap-visualizations", f"run-{runNumber}/viz_{video_name}.mp4", os.path.abspath("pipeline/track/outputs/PHALP_{}.mp4".format(video_name)))
        # }

        # isError = False
        # for key, (bucket, key_path, local_file_path) in files_to_upload.items():
        #     try:
        #         response = s3_client.upload_file(local_file_path, bucket, key_path)
        #     except ClientError as e:
        #         logging.error(e)
        #         isError = True

        # if isError:
        #     continue
        imgsPath = os.path.abspath("pipeline/track/outputs/_DEMO/{}".format(video_name))
        # for _, _, local_file_path in files_to_upload.values():
        #     if os.path.exists(local_file_path):
        #         os.remove(local_file_path)
        if os.path.exists(imgsPath):
            shutil.rmtree(imgsPath)
    logging.info("Finished.")