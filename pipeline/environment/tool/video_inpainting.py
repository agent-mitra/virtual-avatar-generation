import os
import sys

sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))
sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..", "FGT")))
sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..", "LAFC")))

import warnings

warnings.filterwarnings("ignore")

import argparse
import os
import cv2
import glob
import copy
import time
import torchvision
import numpy as np
import torch
import imageio
from PIL import Image
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.structures import Boxes, Instances
from segment_anything import sam_model_registry, SamPredictor
import scipy.ndimage
from skimage.feature import canny
import torchvision.transforms.functional as F
import torch.nn.functional as F2
from RAFT import utils
from RAFT import RAFT
import imageio
import yaml
from importlib import import_module
from itertools import zip_longest
import utils.region_fill as rf
from utils.Poisson_blend_img import Poisson_blend_img
from get_flowNN_gradient import get_flowNN_gradient
from torchvision.transforms import ToTensor
import cvbase, subprocess

import sys
# Navigate up to the 'home' directory
current_dir = os.path.dirname(os.path.realpath(__file__))
home_dir = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
# Add the 'home' directory to the Python path
sys.path.append(home_dir)
from pipeline.track.phalp.utils import get_pylogger
from pipeline.track.phalp.utils.utils_detectron2 import (DefaultPredictor_Lazy,
                                          DefaultPredictor_with_RPN)

log = get_pylogger(__name__)
current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_file_path))))
class FrameExtractor():
    '''
    Class used for extracting frames from a video file.
    '''
    def __init__(self, video_path):
        self.video_path = video_path
        self.setup_detectron2()
        sam_checkpoint = os.path.join(project_root, "pipeline", "track", "phalp", "trackers", "sam_vit_h_4b8939.pth")
        model_type = "vit_h"
        device = "cuda"
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=device)
        self.person_predictor = SamPredictor(sam)
        
        
    def setup_detectron2(self):
        log.info("Loading Detection model...")
        from detectron2.config import LazyConfig
        from pathlib import Path
        import phalp
        cfg_path = Path(phalp.__file__).parent/'configs'/'cascade_mask_rcnn_vitdet_h_75ep.py'
        self.detectron2_cfg = LazyConfig.load(str(cfg_path))
        self.detectron2_cfg.train.init_checkpoint = 'https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl'
        for i in range(3):
            self.detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.5
        self.detector = DefaultPredictor_Lazy(self.detectron2_cfg)       

        # for predicting masks with only bounding boxes, e.g. for running on ground truth tracks
        self.setup_detectron2_with_RPN()
        
    def setup_detectron2_with_RPN(self):
        self.detectron2_cfg = get_cfg()
        self.detectron2_cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"))   
        self.detectron2_cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        self.detectron2_cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST   = 0.4
        self.detectron2_cfg.MODEL.WEIGHTS   = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml")
        self.detectron2_cfg.MODEL.META_ARCHITECTURE =  "GeneralizedRCNN_with_proposals"
        self.detector_x = DefaultPredictor_with_RPN(self.detectron2_cfg)
        
    def get_detections(self, image):
        
        outputs     = self.detector(image)   
        instances   = outputs['instances']
        instances   = instances[instances.pred_classes==0]

        pred_bbox   = instances.pred_boxes.tensor.cpu().numpy()
        pred_masks  = instances.pred_masks.cpu().numpy()
        pred_scores = instances.scores.cpu().numpy()
        pred_classes= instances.pred_classes.cpu().numpy()
        
        ground_truth_track_id = [1 for i in list(range(len(pred_scores)))]
        ground_truth_annotations = [[] for i in list(range(len(pred_scores)))]

        return pred_bbox, pred_masks, pred_scores, pred_classes, ground_truth_track_id, ground_truth_annotations

    def extract_frames(self, dest_path=None, fps=24):
        if dest_path is None:
            dest_path = os.getcwd()
        else:
            if not os.path.isdir(dest_path):
                os.mkdir(dest_path); print(f'Created the following directory: {dest_path}')
                
        command = [
            'ffmpeg',
            '-i', self.video_path,
            '-copyts', 
            '-qscale:v', '2', 
            '-vf', f'fps={fps}',
            f'{dest_path}/%06d.jpg'
        ] #change resolution as well
        print(f"Running ffmpeg command: {command}")
        subprocess.run(command)
        
@staticmethod
def read_frame(frame_path):
    frame = None
    # frame path can be either a path to an image or a list of [video_path, frame_id in pts]
    if(isinstance(frame_path, tuple)):
        frame = torchvision.io.read_video(frame_path[0], pts_unit='pts', start_pts=frame_path[1], end_pts=frame_path[1]+1)[0][0]
        frame = frame.numpy()[:, :, ::-1]
    elif(isinstance(frame_path, str)):
        frame = cv2.imread(frame_path)
    else:
        raise Exception("Invalid frame path")

    return frame


def to_tensor(img):
    img = Image.fromarray(img)
    img_t = F.to_tensor(img).float()
    return img_t


def diffusion(flows, masks):
    flows_filled = []
    for i in range(flows.shape[0]):
        flow, mask = flows[i], masks[i]
        flow_filled = np.zeros(flow.shape)
        flow_filled[:, :, 0] = rf.regionfill(flow[:, :, 0], mask[:, :, 0])
        flow_filled[:, :, 1] = rf.regionfill(flow[:, :, 1], mask[:, :, 0])
        flows_filled.append(flow_filled)
    return flows_filled


def np2tensor(array, near="c"):
    if isinstance(array, list):
        array = np.stack(array, axis=0)  # [t, h, w, c]
    if near == "c":
        array = (
            torch.from_numpy(np.transpose(array, (3, 0, 1, 2))).unsqueeze(0).float()
        )  # [1, c, t, h, w]
    elif near == "t":
        array = torch.from_numpy(np.transpose(array, (0, 3, 1, 2))).unsqueeze(0).float()
    else:
        raise ValueError(f"Unknown near type: {near}")
    return array


def tensor2np(array):
    array = torch.stack(array, dim=-1).squeeze(0).permute(1, 2, 0, 3).cpu().numpy()
    return array


def gradient_mask(mask):
    gradient_mask = np.logical_or.reduce(
        (
            mask,
            np.concatenate(
                (mask[1:, :], np.zeros((1, mask.shape[1]), dtype=bool)), axis=0
            ),
            np.concatenate(
                (mask[:, 1:], np.zeros((mask.shape[0], 1), dtype=bool)), axis=1
            ),
        )
    )

    return gradient_mask


def indicesGen(pivot, interval, frames, t):
    singleSide = frames // 2
    results = []
    for i in range(-singleSide, singleSide + 1):
        index = pivot + interval * i
        if index < 0:
            index = abs(index)
        if index > t - 1:
            index = 2 * (t - 1) - index
        results.append(index)
    return results


def get_ref_index(f, neighbor_ids, length, ref_length, num_ref):
    ref_index = []
    if num_ref == -1:
        for i in range(0, length, ref_length):
            if i not in neighbor_ids:
                ref_index.append(i)
    else:
        start_idx = max(0, f - ref_length * (num_ref // 2))
        end_idx = min(length, f + ref_length * (num_ref // 2))
        for i in range(start_idx, end_idx + 1, ref_length):
            if i not in neighbor_ids:
                if len(ref_index) > num_ref:
                    break
                ref_index.append(i)
    return ref_index


def save_flows(output, videoFlowF, videoFlowB):
    create_dir(os.path.join(output, "completed_flow", "forward_flo"))
    create_dir(os.path.join(output, "completed_flow", "backward_flo"))
    create_dir(os.path.join(output, "completed_flow", "forward_png"))
    create_dir(os.path.join(output, "completed_flow", "backward_png"))
    N = videoFlowF.shape[-1]
    for i in range(N):
        forward_flow = videoFlowF[..., i]
        backward_flow = videoFlowB[..., i]
        forward_flow_vis = cvbase.flow2rgb(forward_flow)
        backward_flow_vis = cvbase.flow2rgb(backward_flow)
        cvbase.write_flow(
            forward_flow,
            os.path.join(
                output, "completed_flow", "forward_flo", "{:05d}.flo".format(i)
            ),
        )
        cvbase.write_flow(
            backward_flow,
            os.path.join(
                output, "completed_flow", "backward_flo", "{:05d}.flo".format(i)
            ),
        )
        imageio.imwrite(
            os.path.join(
                output, "completed_flow", "forward_png", "{:05d}.png".format(i)
            ),
            forward_flow_vis,
        )
        imageio.imwrite(
            os.path.join(
                output, "completed_flow", "backward_png", "{:05d}.png".format(i)
            ),
            backward_flow_vis,
        )


def save_fgcp(output, frames, masks):
    create_dir(os.path.join(output, "prop_frames"))
    create_dir(os.path.join(output, "masks_left"))
    create_dir(os.path.join(output, "prop_frames_npy"))
    create_dir(os.path.join(output, "masks_left_npy"))

    assert len(frames) == masks.shape[2]
    for i in range(len(frames)):
        cv2.imwrite(
            os.path.join(output, "prop_frames", "%05d.png" % i), frames[i] * 255.0
        )
        cv2.imwrite(
            os.path.join(output, "masks_left", "%05d.png" % i), masks[:, :, i] * 255.0
        )
        np.save(
            os.path.join(output, "prop_frames_npy", "%05d.npy" % i), frames[i] * 255.0
        )
        np.save(
            os.path.join(output, "masks_left_npy", "%05d.npy" % i),
            masks[:, :, i] * 255.0,
        )


def create_dir(dir):
    """Creates a directory if not exist."""
    if not os.path.exists(dir):
        os.makedirs(dir)


def initialize_RAFT(args, device):
    """Initializes the RAFT model."""
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(
        torch.load(args.raft_model, map_location=torch.device(device))
    )

    model = model.module
    model.to(device)
    model.eval()

    return model


def initialize_LAFC(args, device):
    #assert len(os.listdir(args.lafc_ckpts)) == 2
    print(args.lafc_ckpts)
    checkpoint, config_file = (
        glob.glob(os.path.join(args.lafc_ckpts, "*.tar"))[0],
        glob.glob(os.path.join(args.lafc_ckpts, "*.yaml"))[0],
    )
    with open(config_file, "r") as f:
        configs = yaml.full_load(f)
    model = configs["model"]
    pkg = import_module("LAFC.models.{}".format(model))
    model = pkg.Model(configs)
    state = torch.load(checkpoint, map_location=torch.device(device))
    model.load_state_dict(state["model_state_dict"])
    model = model.to(device)
    return model, configs


def initialize_FGT(args, device):
    #assert len(os.listdir(args.fgt_ckpts)) == 2
    checkpoint, config_file = (
        glob.glob(os.path.join(args.fgt_ckpts, "*.tar"))[0],
        glob.glob(os.path.join(args.fgt_ckpts, "*.yaml"))[0],
    )
    with open(config_file, "r") as f:
        configs = yaml.full_load(f)
    model = configs["model"]
    net = import_module("FGT.models.{}".format(model))
    model = net.Model(configs).to(device)
    state = torch.load(checkpoint, map_location=torch.device(device))
    model.load_state_dict(state["model_state_dict"])
    return model, configs


def calculate_flow(args, model, video, mode):
    """Calculates optical flow."""
    if mode not in ["forward", "backward"]:
        raise NotImplementedError

    imgH, imgW = args.imgH, args.imgW
    Flow = np.empty(((imgH, imgW, 2, 0)), dtype=np.float32)

    if args.vis_flows:
        create_dir(os.path.join(args.outroot, "flow", mode + "_flo"))
        create_dir(os.path.join(args.outroot, "flow", mode + "_png"))

    with torch.no_grad():
        for i in range(video.shape[0] - 1):
            print(
                "Calculating {0} flow {1:2d} <---> {2:2d}".format(mode, i, i + 1),
                "\r",
                end="",
            )
            if mode == "forward":
                # Flow i -> i + 1
                image1 = video[i, None]
                image2 = video[i + 1, None]
            elif mode == "backward":
                # Flow i + 1 -> i
                image1 = video[i + 1, None]
                image2 = video[i, None]
            else:
                raise NotImplementedError

            _, flow = model(image1, image2, iters=20, test_mode=True)
            flow = flow[0].permute(1, 2, 0).cpu().numpy()
            # resize optical flows
            h, w = flow.shape[:2]
            if h != imgH or w != imgW:
                flow = cv2.resize(flow, (imgW, imgH), cv2.INTER_LINEAR)
                flow[:, :, 0] *= float(imgW) / float(w)
                flow[:, :, 1] *= float(imgH) / float(h)

            Flow = np.concatenate((Flow, flow[..., None]), axis=-1)

            if args.vis_flows:
                # Flow visualization.
                flow_img = utils.flow_viz.flow_to_image(flow)
                flow_img = Image.fromarray(flow_img)

                # Saves the flow and flow_img.
                flow_img.save(
                    os.path.join(args.outroot, "flow", mode + "_png", "%05d.png" % i)
                )
                utils.frame_utils.writeFlow(
                    os.path.join(args.outroot, "flow", mode + "_flo", "%05d.flo" % i),
                    flow,
                )

    return Flow


def extrapolation(args, video_ori, corrFlowF_ori, corrFlowB_ori):
    """Prepares the data for video extrapolation."""
    imgH, imgW, _, nFrame = video_ori.shape

    # Defines new FOV.
    imgH_extr = int(args.H_scale * imgH)
    imgW_extr = int(args.W_scale * imgW)
    imgH_extr = imgH_extr - imgH_extr % 4
    imgW_extr = imgW_extr - imgW_extr % 4
    H_start = int((imgH_extr - imgH) / 2)
    W_start = int((imgW_extr - imgW) / 2)

    # Generates the mask for missing region.
    flow_mask = np.ones(((imgH_extr, imgW_extr)), dtype=bool)
    flow_mask[H_start : H_start + imgH, W_start : W_start + imgW] = 0

    mask_dilated = gradient_mask(flow_mask)

    # Extrapolates the FOV for video.
    video = np.zeros(((imgH_extr, imgW_extr, 3, nFrame)), dtype=np.float32)
    video[H_start : H_start + imgH, W_start : W_start + imgW, :, :] = video_ori

    for i in range(nFrame):
        print("Preparing frame {0}".format(i), "\r", end="")
        video[:, :, :, i] = (
            cv2.inpaint(
                (video[:, :, :, i] * 255).astype(np.uint8),
                flow_mask.astype(np.uint8),
                3,
                cv2.INPAINT_TELEA,
            ).astype(np.float32)
            / 255.0
        )

    # Extrapolates the FOV for flow.
    corrFlowF = np.zeros(((imgH_extr, imgW_extr, 2, nFrame - 1)), dtype=np.float32)
    corrFlowB = np.zeros(((imgH_extr, imgW_extr, 2, nFrame - 1)), dtype=np.float32)
    corrFlowF[H_start : H_start + imgH, W_start : W_start + imgW, :] = corrFlowF_ori
    corrFlowB[H_start : H_start + imgH, W_start : W_start + imgW, :] = corrFlowB_ori

    return (
        video,
        corrFlowF,
        corrFlowB,
        flow_mask,
        mask_dilated,
        (W_start, H_start),
        (W_start + imgW, H_start + imgH),
    )


def complete_flow(config, flow_model, flows, flow_masks, mode, device):
    if mode not in ["forward", "backward"]:
        raise NotImplementedError(f"Error flow mode {mode}")
    flow_masks = np.moveaxis(flow_masks, -1, 0)  # [N, H, W]
    flows = np.moveaxis(flows, -1, 0)  # [N, H, W, 2]
    if len(flow_masks.shape) == 3:
        flow_masks = flow_masks[:, :, :, np.newaxis]
    if mode == "forward":
        flow_masks = flow_masks[0:-1]
    else:
        flow_masks = flow_masks[1:]

    num_flows, flow_interval = config["num_flows"], config["flow_interval"]

    diffused_flows = diffusion(flows, flow_masks)

    flows = np2tensor(flows)
    flow_masks = np2tensor(flow_masks)
    diffused_flows = np2tensor(diffused_flows)

    flows = flows.to(device)
    flow_masks = flow_masks.to(device)
    diffused_flows = diffused_flows.to(device)

    t = diffused_flows.shape[2]
    filled_flows = [None] * t
    pivot = num_flows // 2
    for i in range(t):
        indices = indicesGen(i, flow_interval, num_flows, t)
        print("Indices: ", indices, "\r", end="")
        cand_flows = flows[:, :, indices]
        cand_masks = flow_masks[:, :, indices]
        inputs = diffused_flows[:, :, indices]
        pivot_mask = cand_masks[:, :, pivot]
        pivot_flow = cand_flows[:, :, pivot]
        with torch.no_grad():
            output_flow = flow_model(inputs, cand_masks)
        if isinstance(output_flow, tuple) or isinstance(output_flow, list):
            output_flow = output_flow[0]
        comp = output_flow * pivot_mask + pivot_flow * (1 - pivot_mask)
        if filled_flows[i] is None:
            filled_flows[i] = comp
    assert None not in filled_flows
    return filled_flows


def read_flow(flow_dir, video):
    nFrame, _, imgH, imgW = video.shape
    Flow = np.empty(((imgH, imgW, 2, 0)), dtype=np.float32)
    flows = sorted(glob.glob(os.path.join(flow_dir, "*.flo")))
    for flow in flows:
        flow_data = cvbase.read_flow(flow)
        h, w = flow_data.shape[:2]
        flow_data = cv2.resize(flow_data, (imgW, imgH), cv2.INTER_LINEAR)
        flow_data[:, :, 0] *= float(imgW) / float(w)
        flow_data[:, :, 1] *= float(imgH) / float(h)
        Flow = np.concatenate((Flow, flow_data[..., None]), axis=-1)
    return Flow


def norm_flows(flows):
    assert len(flows.shape) == 5, "FLow shape: {}".format(flows.shape)
    flattened_flows = flows.flatten(3)
    flow_max = torch.max(flattened_flows, dim=-1, keepdim=True)[0]
    flows = flows / flow_max.unsqueeze(-1)
    return flows


def save_results(outdir, comp_frames):
    out_dir = os.path.join(outdir, "frames")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    for i in range(len(comp_frames)):
        out_path = os.path.join(out_dir, "{:05d}.png".format(i))
        cv2.imwrite(out_path, comp_frames[i][:, :, ::-1])


def video_inpainting(args):
    # device = torch.device('cuda:{}'.format(args.gpu))
    if not args.inference_mode:
        print("RUNNING FGT DATA COLLECTION MODE...")
    if args.inference_mode:
        print("RUNNING FGT INFERENCE MODE...")
        yolo_checkpoint = os.path.abspath("tool/yolov8l-seg.pt")
        from ultralytics import YOLO
        model = YOLO(yolo_checkpoint)
        source_path = args.video_source
        video_name = source_path.split('/')[-1].split('.')[0]
        fgt_base_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
        os.system("rm -rf " + fgt_base_directory + "/data/inference/" + video_name + "/img/")
        os.system("rm -rf " + fgt_base_directory + "/data/inference/" + video_name + "/mask/")
        os.makedirs(fgt_base_directory + "/data/inference/" + video_name, exist_ok=True)
        os.makedirs(fgt_base_directory + "/data/inference/" + video_name + "/img", exist_ok=True)
        os.makedirs(fgt_base_directory + "/data/inference/" + video_name + "/mask", exist_ok=True)
        #frame extractor, self, video_path
        fe = FrameExtractor(source_path)
        outputImgDir = fgt_base_directory + "/data/inference/" + video_name + "/img"
        args.path = outputImgDir
        fe.extract_frames(dest_path= outputImgDir, fps=24)
        #test on original fps then do the frame duplication later.
        list_of_frames = sorted(glob.glob(outputImgDir + "/*.jpg"))
        totalFrames = len(list_of_frames)
        for t_, frame_name in enumerate(list_of_frames):
            #print(f"Running frame number {t_ + 1}/{totalFrames}")
            image_frame               = read_frame(frame_name)
            frame = np.zeros_like(image_frame)
            temp_frame = np.zeros_like(image_frame)
            allMasks = []
            
            results = model(image_frame)
            classes_xy_masks = []
            if results[0].masks:
                xy_masks = results[0].masks.xy
                for i in range(len(results[0].boxes)):
                    classBox = results[0].boxes[i].cls
                    if classBox[0] == 0:
                        classes_xy_masks.append(xy_masks[i])

            #del results
            for boundary in classes_xy_masks:
                imageA_mask = np.zeros_like(image_frame[..., 0])
                boundary = np.array(boundary, dtype=np.int32)
                if len(boundary) > 0:
                    cv2.fillPoly(imageA_mask, [boundary], 255)
                    allMasks.append(imageA_mask)
                del imageA_mask, boundary
            pred_bboxes_into_sam, pred_masks_orig, _, _, _, _ = fe.get_detections(image_frame)
            fe.person_predictor.set_image(image_frame)
            for box in pred_bboxes_into_sam: #detectron2 is fairly imperfect so we double down with SAM as well.
                input_box = np.array(box)
                masks, _, _ = fe.person_predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=input_box[None, :],
                    multimask_output=False,
                )
                if len(masks) > 0:
                    mask = masks[0]
                    allMasks.append(mask)
            
            for segMask in pred_masks_orig: 
                allMasks.append(segMask)
                
            if len(allMasks) > 0:
                combined_mask = np.zeros_like(allMasks[0], dtype=bool)
                for mask in allMasks:
                    combined_mask = np.logical_or(combined_mask, mask)
                kernel_12x12 = np.ones((12, 12), np.uint8)
                dilated_final = cv2.dilate(temp_frame, kernel_12x12, iterations=1)
                frame = cv2.bitwise_or(frame, dilated_final)
                frame[combined_mask] = [255, 255, 255]
            outputMaskDir = fgt_base_directory + "/data/inference/" + video_name + "/mask"
            newFrameName = frame_name.split('/')[-1].split('.')[0]
            cv2.imwrite(f'{outputMaskDir}/{newFrameName}.jpg', frame)
        args.path_mask = outputMaskDir
    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.opt is not None:
        with open(args.opt, "r") as f:
            opts = yaml.full_load(f)

    for k in opts.keys():
        if k in args:
            setattr(args, k, opts[k])

    # Flow model.
    RAFT_model = initialize_RAFT(args, device)
    # LAFC (flow completion)
    LAFC_model, LAFC_config = initialize_LAFC(args, device)
    # FGT
    FGT_model, FGT_config = initialize_FGT(args, device)

    # Loads frames.
    allFrames = glob.glob(os.path.join(args.path, "*.png")) + glob.glob(
        os.path.join(args.path, "*.jpg")
    )
    # Loads masks.
    allMasks = glob.glob(os.path.join(args.path_mask, "*.png")) + glob.glob(
        os.path.join(args.path_mask, "*.jpg")
    )

    all_comp_frames = []
    num_frames = len(allFrames)
    batch_size = num_frames // 10
    num_batches = batch_size
    allFrameIndices = []
    all_included_indices = set()
    for batch_num in range(num_batches+1):
        start_idx = batch_num
        if batch_num == num_batches:
            filename_list = []
            remaining_indices = set(range(num_frames)) - all_included_indices
            if len(remaining_indices) == 0:
                break
            filename_list.extend([allFrames[i] for i in sorted(remaining_indices)])
        else:
            included_indices = set(range(start_idx, num_frames, batch_size))
            all_included_indices.update(included_indices)
            filename_list = allFrames[start_idx::batch_size]
        filename_list = sorted(filename_list, key=extract_number)
        allFrameIndices.append(filename_list)
        print(f"STARTING BATCH {batch_num}/{num_batches}")
        # Obtains imgH, imgW and nFrame.
        imgH, imgW = args.imgH, args.imgW
        nFrame = len(filename_list)

        if imgH < 350:
            flowH, flowW = imgH * 2, imgW * 2
        else:
            flowH, flowW = imgH, imgW

        # Load video.
        video, video_flow = [], []
        for filename in sorted(filename_list):
            frame = (
                torch.from_numpy(np.array(Image.open(filename)).astype(np.uint8))
                .permute(2, 0, 1)
                .float()
                .unsqueeze(0)
            )
            frame = F2.upsample(
                frame, size=(imgH, imgW), mode="bilinear", align_corners=False
            )
            frame_flow = F2.upsample(
                frame, size=(flowH, flowW), mode="bilinear", align_corners=False
            )
            video.append(frame)
            video_flow.append(frame_flow)

        video = torch.cat(video, dim=0)  # [n, c, h, w]
        video_flow = torch.cat(video_flow, dim=0)
        gts = video.clone()
        video = video.to(device)
        video_flow = video_flow.to(device)

        # Calcutes the corrupted flow.
        forward_flows = calculate_flow(
            args, RAFT_model, video_flow, "forward"
        )  # [B, C, 2, N]
        backward_flows = calculate_flow(args, RAFT_model, video_flow, "backward")

        # Makes sure video is in BGR (opencv) format.
        video = (
            video.permute(2, 3, 1, 0).cpu().numpy()[:, :, ::-1, :] / 255.0
        )  # np array -> [h, w, c, N] (0~1)
        if batch_num == num_batches:
            filename_list = []
            filename_list.extend([allMasks[i] for i in sorted(remaining_indices)])
        else:
            filename_list = allMasks[start_idx::batch_size]
        filename_list = sorted(filename_list, key=extract_number)
        mask = []
        mask_dilated = []
        flow_mask = []
        for filename in sorted(filename_list):
            mask_img = np.array(Image.open(filename).convert("L"))
            mask_img = cv2.resize(
                mask_img, dsize=(imgW, imgH), interpolation=cv2.INTER_NEAREST
            )

            if args.flow_mask_dilates > 0:
                flow_mask_img = scipy.ndimage.binary_dilation(
                    mask_img, iterations=args.flow_mask_dilates
                )
            else:
                flow_mask_img = mask_img
            flow_mask.append(flow_mask_img)

            if args.frame_dilates > 0:
                mask_img = scipy.ndimage.binary_dilation(
                    mask_img, iterations=args.frame_dilates
                )
            mask.append(mask_img)
            mask_dilated.append(gradient_mask(mask_img))

        # mask indicating the missing region in the video.
        mask = np.stack(mask, -1).astype(bool)  # [H, W, C, N]
        mask_dilated = np.stack(mask_dilated, -1).astype(bool)
        flow_mask = np.stack(flow_mask, -1).astype(bool)

        # Completes the flow.
        videoFlowF = complete_flow(
            LAFC_config, LAFC_model, forward_flows, flow_mask, "forward", device
        )
        videoFlowB = complete_flow(
            LAFC_config, LAFC_model, backward_flows, flow_mask, "backward", device
        )
        videoFlowF = tensor2np(videoFlowF)
        videoFlowB = tensor2np(videoFlowB)
        print("\nFinish flow completion.")

        if args.vis_completed_flows:
            save_flows(args.outroot, videoFlowF, videoFlowB)

        # Prepare gradients
        gradient_x = np.empty(((imgH, imgW, 3, 0)), dtype=np.float32)
        gradient_y = np.empty(((imgH, imgW, 3, 0)), dtype=np.float32)

        for indFrame in range(nFrame):
            img = video[:, :, :, indFrame]
            img[mask[:, :, indFrame], :] = 0
            img = (
                cv2.inpaint(
                    (img * 255).astype(np.uint8),
                    mask[:, :, indFrame].astype(np.uint8),
                    3,
                    cv2.INPAINT_TELEA,
                ).astype(np.float32)
                / 255.0
            )

            gradient_x_ = np.concatenate(
                (np.diff(img, axis=1), np.zeros((imgH, 1, 3), dtype=np.float32)), axis=1
            )
            gradient_y_ = np.concatenate(
                (np.diff(img, axis=0), np.zeros((1, imgW, 3), dtype=np.float32)), axis=0
            )
            gradient_x = np.concatenate(
                (gradient_x, gradient_x_.reshape(imgH, imgW, 3, 1)), axis=-1
            )
            gradient_y = np.concatenate(
                (gradient_y, gradient_y_.reshape(imgH, imgW, 3, 1)), axis=-1
            )

            gradient_x[mask_dilated[:, :, indFrame], :, indFrame] = 0
            gradient_y[mask_dilated[:, :, indFrame], :, indFrame] = 0

        gradient_x_filled = gradient_x
        gradient_y_filled = gradient_y
        mask_gradient = mask_dilated
        video_comp = video

        # Gradient propagation.
        gradient_x_filled, gradient_y_filled, mask_gradient = get_flowNN_gradient(
            args,
            gradient_x_filled,
            gradient_y_filled,
            mask,
            mask_gradient,
            videoFlowF,
            videoFlowB,
            None,
            None,
        )

        # if there exist holes in mask, Poisson blending will fail. So I did this trick. I sacrifice some value. Another solution is to modify Poisson blending.
        for indFrame in range(nFrame):
            mask_gradient[:, :, indFrame] = scipy.ndimage.binary_fill_holes(
                mask_gradient[:, :, indFrame]
            ).astype(bool)

        # After one gradient propagation iteration
        # gradient --> RGB
        frameBlends = []
        for indFrame in range(nFrame):
            print("Poisson blending frame {0:3d}".format(indFrame))

            if mask[:, :, indFrame].sum() > 0:
                try:
                    frameBlend, UnfilledMask = Poisson_blend_img(
                        video_comp[:, :, :, indFrame],
                        gradient_x_filled[:, 0 : imgW - 1, :, indFrame],
                        gradient_y_filled[0 : imgH - 1, :, :, indFrame],
                        mask[:, :, indFrame],
                        mask_gradient[:, :, indFrame],
                    )
                except:
                    frameBlend, UnfilledMask = (
                        video_comp[:, :, :, indFrame],
                        mask[:, :, indFrame],
                    )

                frameBlend = np.clip(frameBlend, 0, 1.0)
                tmp = (
                    cv2.inpaint(
                        (frameBlend * 255).astype(np.uint8),
                        UnfilledMask.astype(np.uint8),
                        3,
                        cv2.INPAINT_TELEA,
                    ).astype(np.float32)
                    / 255.0
                )
                frameBlend[UnfilledMask, :] = tmp[UnfilledMask, :]

                video_comp[:, :, :, indFrame] = frameBlend
                mask[:, :, indFrame] = UnfilledMask

                frameBlend_ = copy.deepcopy(frameBlend)
                # Green indicates the regions that are not filled yet.
                frameBlend_[mask[:, :, indFrame], :] = [0, 1.0, 0]
            else:
                frameBlend_ = video_comp[:, :, :, indFrame]
            frameBlends.append(frameBlend_)

        if args.vis_prop:
            save_fgcp(args.outroot, frameBlends, mask)

        video_length = len(frameBlends)

        for i in range(len(frameBlends)):
            frameBlends[i] = frameBlends[i][:, :, ::-1]

        frames_first = np2tensor(frameBlends, near="t").to(device)
        mask = np.moveaxis(mask, -1, 0)
        mask = mask[:, :, :, np.newaxis]
        masks = np2tensor(mask, near="t").to(device)
        normed_frames = frames_first * 2 - 1
        comp_frames = [None] * video_length

        ref_length = args.step
        ref_length = 1 #10 #batch_size #int(len(filename_list)//10), decrease more??
        args.step = batch_size
        num_ref = args.num_ref
        neighbor_stride = args.neighbor_stride
        neighbor_stride = batch_size

        videoFlowF = np.moveaxis(videoFlowF, -1, 0)

        videoFlowF = np.concatenate([videoFlowF, videoFlowF[-1:, ...]], axis=0)

        flows = np2tensor(videoFlowF, near="t")
        flows = norm_flows(flows).to(device)

        #for f in range(0, video_length, neighbor_stride):
        for f in range(0, video_length):
            neighbor_ids = [
                i
                for i in range(
                    max(0, f - neighbor_stride), min(video_length, f + neighbor_stride + 1)
                )
            ]
            ref_ids = get_ref_index(f, neighbor_ids, video_length, ref_length, num_ref)
            selected_frames = normed_frames[:, neighbor_ids + ref_ids]
            selected_masks = masks[:, neighbor_ids + ref_ids]
            masked_frames = selected_frames * (1 - selected_masks)
            selected_flows = flows[:, neighbor_ids + ref_ids]
            with torch.no_grad():
                filled_frames = FGT_model(masked_frames, selected_flows, selected_masks)
            filled_frames = (filled_frames + 1) / 2
            filled_frames = filled_frames.cpu().permute(0, 2, 3, 1).numpy() * 255
            for i in range(len(neighbor_ids)):
                idx = neighbor_ids[i]
                valid_frame = frames_first[0, idx].cpu().permute(1, 2, 0).numpy() * 255.0
                valid_mask = masks[0, idx].cpu().permute(1, 2, 0).numpy()
                comp = np.array(filled_frames[i]).astype(np.uint8) * valid_mask + np.array(
                    valid_frame
                ).astype(np.uint8) * (1 - valid_mask)
                if comp_frames[idx] is None:
                    comp_frames[idx] = comp
                else:
                    comp_frames[idx] = (
                        comp_frames[idx].astype(np.float32) * 0.5
                        + comp.astype(np.float32) * 0.5
                    )
        if args.vis_frame:
            save_results(args.outroot, comp_frames)
        create_dir(args.outroot)
        for i in range(len(comp_frames)):
            comp_frames[i] = comp_frames[i].astype(np.uint8)
        all_comp_frames.append(comp_frames)
    #continue here
    flat_allFrameIndices = [filename for sublist in allFrameIndices for filename in sublist]
    flat_matrix = [image for sublist in all_comp_frames for image in sublist]
    combined_list = list(zip(flat_allFrameIndices, flat_matrix))
    sorted_combined_list = sorted(combined_list, key=lambda x: extract_number(x[0]))
    final_comp_frames = [item[1] for item in sorted_combined_list]
    if not args.inference_mode:
        video_name = extract_video_name(args.path)
    imageio.mimwrite(
        os.path.join(args.outroot, f"{video_name}.mp4"), final_comp_frames, fps=24, quality=8
    )
    end_time = time.time()
    elapsed_time_seconds = end_time - start_time
    elapsed_time_minutes = elapsed_time_seconds / 60.0
    print(f"Inpainting finished in {elapsed_time_minutes} minutes with step: {args.step}, stride: {args.neighbor_stride}, h,w: {args.imgH, args.imgW}")
    print(f"Done, please check your result in {args.outroot} ")


import random

def sample(population, k):
  return random.sample(population, k)

import re

def extract_video_name(s):
    # Define the regular expression pattern to match the video_name
    pattern = r'/track/outputs/_DEMO/(.*?)/orig'
    match = re.search(pattern, s)
    
    if match:
        return match.group(1)
    else:
        return "Not found"

import re
def extract_number(filename):
    # Use regex to find the number in the filename
    match = re.search(r'(\d+)\.jpg$', filename)
    # If a number is found, return it as an integer, otherwise return 0
    return int(match.group(1)) if match else 0


def main(args):
    print(f"CUDA Availability: {torch.cuda.is_available()}")
    assert args.mode in (
        "object_removal",
        "video_extrapolation",
        "watermark_removal",
    ), (
        "Accepted modes: 'object_removal', 'video_extrapolation', and 'watermark_removal', but input is %s"
    ) % args.mode
    video_inpainting(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--opt",
        default= os.path.join(project_root, "pipeline", "inpaint", "tool", "configs", "object_removal.yaml"), #"configs/object_removal.yaml",
        help="Please select your config file for inference",
    )
    # video completion
    parser.add_argument(
        "--mode",
        default="object_removal",
        choices=["object_removal", "watermark_removal", "video_extrapolation"],
        help="modes: object_removal / video_extrapolation",
    )
    parser.add_argument(
        "--path", default="/data", help="dataset for evaluation"
    )
    parser.add_argument(
        "--path_mask",
        default="data",
        help="mask for object removal",
    )
    parser.add_argument(
        "--outroot", default="quick_start/walking3", help="output directory"
    )
    parser.add_argument(
        "--consistencyThres",
        dest="consistencyThres",
        default=5,
        type=float,
        help="flow consistency error threshold",
    )
    parser.add_argument("--alpha", dest="alpha", default=0.1, type=float)
    parser.add_argument("--Nonlocal", dest="Nonlocal", default=False, type=bool)

    # RAFT
    parser.add_argument(
        "--raft_model",
        default=os.path.join(project_root, "pipeline", "inpaint", "LAFC", "flowCheckPoint", "raft-things.pth"), #"../LAFC/flowCheckPoint/raft-things.pth",
        help="restore checkpoint",
    )
    parser.add_argument("--small", action="store_true", help="use small model")
    parser.add_argument(
        "--mixed_precision", action="store_true", help="use mixed precision"
    )
    parser.add_argument(
        "--alternate_corr",
        action="store_true",
        help="use efficent correlation implementation",
    )

    # LAFC
    parser.add_argument("--lafc_ckpts", type=str, default=os.path.join(project_root, "pipeline", "inpaint", "LAFC", "checkpoint")) #"../LAFC/checkpoint")

    # FGT
    parser.add_argument("--fgt_ckpts", type=str, default=os.path.join(project_root, "pipeline", "inpaint", "FGT", "checkpoint")) #"../FGT/checkpoint")

    # extrapolation
    parser.add_argument(
        "--H_scale", dest="H_scale", default=2, type=float, help="H extrapolation scale"
    )
    parser.add_argument(
        "--W_scale", dest="W_scale", default=2, type=float, help="W extrapolation scale"
    )

    # Image basic information
    parser.add_argument("--imgH", type=int, default=1280) #640, 720, 800, divisible by 4 [360p]
    parser.add_argument("--imgW", type=int, default=888) #368, 400, 400, divisible by 4 [360p]
    parser.add_argument("--flow_mask_dilates", type=int, default=11) #8 prev
    parser.add_argument("--frame_dilates", type=int, default=11)

    parser.add_argument("--gpu", type=int, default=0)

    # FGT inference parameters
    parser.add_argument("--step", type=int, default=5) #100/200/400
    parser.add_argument("--num_ref", type=int, default=-1)
    parser.add_argument("--neighbor_stride", type=int, default=20) #5

    # visualization
    parser.add_argument(
        "--vis_flows", action="store_true", help="Visualize the initialized flows"
    )
    parser.add_argument(
        "--vis_completed_flows",
        action="store_true",
        help="Visualize the completed flows",
    )
    parser.add_argument(
        "--vis_prop",
        action="store_true",
        help="Visualize the frames after stage-I filling (flow guided content propagation)",
    )
    parser.add_argument("--vis_frame", action="store_true", help="Visualize frames")
    parser.add_argument("--inference_mode", action="store_true", help="Enable inference mode")
    parser.add_argument("--video_source", type=str, default="data/results/")

    args = parser.parse_args()

    main(args)
