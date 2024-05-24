from __future__ import annotations
from pathlib import Path
import os
import subprocess
import sys
import tempfile
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline, interp1d
from PIL import Image
import math
import random
import cv2
import copy
import numpy as np
import re
import torch
import mmcv
import itertools
import uuid
import torch.nn as nn
from collections import deque
from skimage.feature import hog #pip install scikit-image
from skimage import color
from postprocessing.postprocessing import *
import numpy as np
import gc
from dwpose import DWposeDetector
import torch
import torch.nn as nn
from pycocotools import mask as mask_utils
from scenedetect import AdaptiveDetector, detect
from sklearn.linear_model import Ridge
import traceback
import warnings

warnings.filterwarnings('ignore')

import cv2
import copy
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.structures import Boxes, Instances
from pycocotools import mask as mask_utils
from sklearn.linear_model import Ridge

from phalp.configs.base import CACHE_DIR
from phalp.external.deep_sort_ import nn_matching
from phalp.external.deep_sort_.detection import Detection
from phalp.external.deep_sort_.tracker import Tracker
from phalp.models.hmar import HMAR
from phalp.models.predictor import Pose_transformer_v2
from phalp.utils import get_pylogger
from phalp.utils.io import IO_Manager
from phalp.utils.utils import (convert_pkl, get_prediction_interval,
                               progress_bar, smpl_to_pose_camera_vector)
from phalp.utils.utils_dataset import process_image, process_mask
from phalp.utils.utils_detectron2 import (DefaultPredictor_Lazy,
                                          DefaultPredictor_with_RPN)
from phalp.utils.utils_download import cache_url
from phalp.visualize.postprocessor import Postprocessor
from phalp.visualize.visualizer import Visualizer
import signal

log = get_pylogger(__name__)

selected_keypoints = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, # COCO Body Keypoints which includes eyes (17 keypoints)
    17, 18, 19, 20, 21, 22, # COCO Foot Keypoints (6 keypoints)

    112, 114, 115, 117, 118, 121, 122, 125, 126, 129, 130,  # Right Wrist and MCP joints
    116, 120, 124, 128, 132, # Right DIP joints

    91, 93, 94, 96, 97, 100, 101, 104, 105, 108, 109, # Left Wrist and MCP joints ( l to r)
    95, 99, 103, 107, 111, # Left DIP joints
    60, 61, 66, 67 # Eye keypoints
]

def distance_between_boxes(box1, box2):
    # Calculate the center of box1
    center_x1 = (box1[0] + box1[2]) / 2
    center_y1 = (box1[1] + box1[3]) / 2
    
    # Calculate the center of box2
    center_x2 = (box2[0] + box2[2]) / 2
    center_y2 = (box2[1] + box2[3]) / 2
    
    # Calculate the distance between the two centers
    distance = math.sqrt((center_x2 - center_x1) ** 2 + (center_y2 - center_y1) ** 2)
    
    return distance


class AppModel(nn.Module):
    def __init__(self, cfg):
        super(AppModel, self).__init__()
        self.cfg = cfg
        self.device = torch.device(self.cfg.device)
        # download wights and configs from Google Drive
        self.cached_download_from_drive()
        
        # setup HMR, and pose_predictor. Override this function to use your own model
        self.setup_hmr()
        
        # setup temporal pose predictor
        self.setup_predictor()
        
        # setup Detectron2, override this function to use your own model
        self.setup_detectron2()
        
        # move to device
        self.to(self.device)
        
        # train or eval
        self.train() if(self.cfg.train) else self.eval()
        self.pose_model = DWposeDetector()
        
    def setup_hmr(self):
        log.info("Loading HMAR model...")
        self.HMAR = HMAR(self.cfg)
        self.HMAR.load_weights(self.cfg.hmr.hmar_path)
        
    def setup_predictor(self):
        log.info("Loading Predictor model...")
        self.pose_predictor = Pose_transformer_v2(self.cfg, self)
        self.pose_predictor.load_weights(self.cfg.pose_predictor.weights_path)
        
    def setup_detectron2(self):
        log.info("Loading Detection model...")
        if self.cfg.phalp.detector == 'maskrcnn':
            self.detectron2_cfg = model_zoo.get_config('new_baselines/mask_rcnn_regnety_4gf_dds_FPN_400ep_LSJ.py', trained=True)
            self.detectron2_cfg.model.roi_heads.box_predictor.test_score_thresh = 0.5
            self.detectron2_cfg.model.roi_heads.box_predictor.test_nms_thresh   = 0.4
            self.detector       = DefaultPredictor_Lazy(self.detectron2_cfg)
            self.class_names    = self.detector.metadata.get('thing_classes')
        elif self.cfg.phalp.detector == 'vitdet':
            from detectron2.config import LazyConfig
            import phalp
            cfg_path = Path(phalp.__file__).parent/'configs'/'cascade_mask_rcnn_vitdet_h_75ep.py'
            self.detectron2_cfg = LazyConfig.load(str(cfg_path))
            self.detectron2_cfg.train.init_checkpoint = 'https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl'
            for i in range(3):
                self.detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.5
            self.detector = DefaultPredictor_Lazy(self.detectron2_cfg)
        else:
            raise ValueError(f"Detector {self.cfg.phalp.detector} not supported")        

        # for predicting masks with only bounding boxes, e.g. for running on ground truth tracks
        self.setup_detectron2_with_RPN()
        # TODO: make this work with DefaultPredictor_Lazy
        
    def setup_detectron2_with_RPN(self):
        self.detectron2_cfg = get_cfg()
        self.detectron2_cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"))   
        self.detectron2_cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        self.detectron2_cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST   = 0.4
        self.detectron2_cfg.MODEL.WEIGHTS   = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml")
        self.detectron2_cfg.MODEL.META_ARCHITECTURE =  "GeneralizedRCNN_with_proposals"
        self.detector_x = DefaultPredictor_with_RPN(self.detectron2_cfg)
        
    def setup_deepsort(self):
        log.info("Setting up DeepSort...")
        metric  = nn_matching.NearestNeighborDistanceMetric(self.cfg, self.cfg.phalp.hungarian_th, self.cfg.phalp.past_lookback)
        self.tracker = Tracker(self.cfg, metric, max_age=self.cfg.phalp.max_age_track, n_init=self.cfg.phalp.n_init, phalp_tracker=self, dims=[4096, 4096, 99])
   
    def run(
        self, video_path, box_score_threshold):
        cap = cv2.VideoCapture(video_path)
        # eval mode
        self.eval()
        # setup rendering, deep sort and directory structure
        self.setup_deepsort()
        print(f"VIDEO PATH IS {video_path}")
        eval_keys       = ['tracked_ids', 'tracked_bbox', 'tid', 'bbox', 'tracked_time']
        history_keys    = ['appe', 'loca', 'pose', 'uv'] if self.cfg.render.enable else []
        prediction_keys = ['prediction_uv', 'prediction_pose', 'prediction_loca'] if self.cfg.render.enable else []
        extra_keys_1    = ['center', 'scale', 'size', 'img_path', 'img_name', 'class_name', 'conf', 'annotations']
        extra_keys_2    = ['smpl', 'camera', 'camera_bbox', '3d_joints', '2d_joints', 'mask', 'extra_data']
        history_keys    = history_keys + extra_keys_1 + extra_keys_2
        visual_store_   = eval_keys + history_keys + prediction_keys
        tmp_keys_       = ['uv', 'prediction_uv', 'prediction_pose', 'prediction_loca']
        mainPersonTrack = None
        preds_all, self.empty_f_i, offset, madeOffset = [], [], 1, False
        mainPersonTrack = None
        tracked_frames = []
        final_visuals_dic = {}
        f_i = 0
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break
            f_i += 1
            rgb_frame = frame[:, :, ::-1]
            img_height, img_width, _ = rgb_frame.shape
            new_image_size            = max(img_height, img_width)
            center_of_image = (img_width / 2, img_height / 2)
            top, left                 = (new_image_size - img_height)//2, (new_image_size - img_width)//2,
            measurments               = [img_height, img_width, new_image_size, left, top]
            self.cfg.phalp.shot       =  0
            t_ = f_i
            frame_name = f"frame_{f_i}"
            final_visuals_dic.setdefault(frame_name, {'time': t_, 'shot': self.cfg.phalp.shot, 'frame_path': frame_name})
            for key_ in visual_store_: final_visuals_dic[frame_name][key_] = []
            pred_bbox, pred_masks, pred_scores, pred_classes, gt_tids, gt_annots = self.get_detections(rgb_frame)
            preds = self.pose_model(rgb_frame, pred_bbox)
            extra_data = preds
            ############ HMAR ##############
            detections = self.get_human_features(rgb_frame, pred_masks, pred_bbox, pred_scores, rgb_frame, pred_classes, t_, measurments, gt_tids, gt_annots, extra_data)
            ############ tracking ##############
            self.tracker.predict()
            self.tracker.update(detections, t_, frame_name, self.cfg.phalp.shot)
            
            if (len(preds) == 0 or len(self.tracker.tracks) == 0 or len(detections) == 0):
                print(f"Empty frame at {f_i}!")
                self.empty_f_i.append(f_i)
                continue
            if not madeOffset:
                offset = f_i
                print(f"Okay! Offset of {offset} created.")
                madeOffset = True
            if mainPersonTrack is None:
                closestTrack = self.tracker.tracks[0]
                min_distance = float('inf')
                for track in self.tracker.tracks:
                    point = track.track_data['history'][-1]['center']
                    distance = ((point[0] - center_of_image[0]) ** 2 + (point[1] - center_of_image[1]) ** 2) ** 0.5
                    if distance < min_distance:
                        min_distance = distance
                        closestTrack = track
                tracks_ = closestTrack
                mainPersonTrack = tracks_.track_id
            else:
                tracks_ = None
                for track in self.tracker.tracks:
                    if track.track_id == mainPersonTrack:
                        tracks_ = track
                        break
                if (tracks_ == None) or (tracks_ != None and tracks_.time_since_update > 0):
                    closestTrack = self.tracker.tracks[0]
                    min_distance = float('inf')

                    for track in self.tracker.tracks:
                        point = track.track_data['history'][-1]['center']
                        distance = ((point[0] - center_of_image[0]) ** 2 + (point[1] - center_of_image[1]) ** 2) ** 0.5
                        if distance < min_distance:
                            min_distance = distance
                            closestTrack = track
                    tracks_ = closestTrack
                    mainPersonTrack = tracks_.track_id

            if(frame_name not in tracked_frames): tracked_frames.append(frame_name)
            if(not(tracks_.is_confirmed())): continue
            
            track_id        = tracks_.track_id
            track_data_hist = tracks_.track_data['history'][-1]
            track_data_pred = tracks_.track_data['prediction']
            final_visuals_dic[frame_name]['tid'].append(track_id)
            final_visuals_dic[frame_name]['bbox'].append(track_data_hist['bbox'])
            final_visuals_dic[frame_name]['tracked_time'].append(tracks_.time_since_update)

            for hkey_ in history_keys:     final_visuals_dic[frame_name][hkey_].append(track_data_hist[hkey_])
            for pkey_ in prediction_keys:  final_visuals_dic[frame_name][pkey_].append(track_data_pred[pkey_.split('_')[1]][-1])
            #print(f"Last updated was {tracks_.time_since_update}")
            if(tracks_.time_since_update==0): #aka if that track was seen in this frame..bc if main person track was not seen
                final_visuals_dic[frame_name]['tracked_ids'].append(track_id)
                final_visuals_dic[frame_name]['tracked_bbox'].append(track_data_hist['bbox'])
                if(tracks_.hits==self.cfg.phalp.n_init):
                    for pt in range(self.cfg.phalp.n_init-1):
                        track_data_hist_ = tracks_.track_data['history'][-2-pt]
                        track_data_pred_ = tracks_.track_data['prediction']
                        frame_name_      = tracked_frames[-2-pt] #TODO: bug: if valid frame after unseen tid removals and it goes here then it fails fyi
                        if (frame_name_ not in final_visuals_dic):
                            continue
                        final_visuals_dic[frame_name_]['tid'].append(track_id)
                        final_visuals_dic[frame_name_]['bbox'].append(track_data_hist_['bbox'])
                        final_visuals_dic[frame_name_]['tracked_ids'].append(track_id)
                        final_visuals_dic[frame_name_]['tracked_bbox'].append(track_data_hist_['bbox'])
                        final_visuals_dic[frame_name_]['tracked_time'].append(0)
                        for hkey_ in history_keys:    final_visuals_dic[frame_name_][hkey_].append(track_data_hist_[hkey_])
                        for pkey_ in prediction_keys: final_visuals_dic[frame_name_][pkey_].append(track_data_pred_[pkey_.split('_')[1]][-1])

            ############ save the video ##############
            #Ahhh, this is why the stuck poses from prev frame were still showing before.. because it does not remove unseen tid
            if len(final_visuals_dic[frame_name]['tracked_ids']) == 0:
                print("Removing unseen tid")
                final_visuals_dic.pop(frame_name)
                match = re.search(r"frame_(\d+)", frame_name)
                fi = int(match.group(1))
                self.empty_f_i.append(fi)
                continue
        cap.release()
        preds_all = []
        bounding_boxes = []
        allFrames = list(final_visuals_dic.keys())
        for frameName in allFrames:
            if len(final_visuals_dic[frameName]['bbox']) >= 1 and len(final_visuals_dic[frameName]['extra_data']) >= 1:
                bounding_boxes.append(final_visuals_dic[frameName]['bbox'][0])
                preds_all.append(final_visuals_dic[frameName]['extra_data'][0])
            else:
                match = re.search(r"frame_(\d+)", frameName)
                print("Empty frame.. second")
                final_visuals_dic.pop(frameName)
                fi = int(match.group(1))
                self.empty_f_i.append(fi)
        bounding_boxes = [[int(box[0]), int(box[1]), int(box[0] + box[2]), int(box[1] + box[3])] for box in bounding_boxes] #[x_min, y_min, x_max, y_max] format for cv2 drawing
        preds_all_2 = modifyToSelectedKeypoints(preds_all, selected_keypoints)
        return_preds_with_frame_names = {}
        bboxes = {}
        allFrames = list(final_visuals_dic.keys())
        for idx in range(len(allFrames)):
            frame = allFrames[idx]
            return_preds_with_frame_names[frame] = preds_all_2[idx]
            bboxes[frame] = bounding_boxes[idx]
        return return_preds_with_frame_names, offset, self.empty_f_i, bboxes    
    
    def get_detections(self, image):
        outputs     = self.detector(image)   
        instances   = outputs['instances']
        instances   = instances[instances.pred_classes==0]
        instances   = instances[instances.scores>self.cfg.phalp.low_th_c]

        pred_bbox   = instances.pred_boxes.tensor.cpu().numpy()
        pred_masks  = instances.pred_masks.cpu().numpy()
        pred_scores = instances.scores.cpu().numpy()
        pred_classes= instances.pred_classes.cpu().numpy()
        
        ground_truth_track_id = [1 for i in list(range(len(pred_scores)))]
        ground_truth_annotations = [[] for i in list(range(len(pred_scores)))]

        return pred_bbox, pred_masks, pred_scores, pred_classes, ground_truth_track_id, ground_truth_annotations
    
    def forward_for_tracking(self, vectors, attibute="A", time=1):
        
        if(attibute=="P"):

            vectors_pose         = vectors[0]
            vectors_data         = vectors[1]
            vectors_time         = vectors[2]
        
            en_pose              = torch.from_numpy(vectors_pose)
            en_data              = torch.from_numpy(vectors_data)
            en_time              = torch.from_numpy(vectors_time)
            
            if(len(en_pose.shape)!=3):
                en_pose          = en_pose.unsqueeze(0) # (BS, 7, pose_dim)
                en_time          = en_time.unsqueeze(0) # (BS, 7)
                en_data          = en_data.unsqueeze(0) # (BS, 7, 6)
            
            with torch.no_grad():
                pose_pred = self.pose_predictor.predict_next(en_pose, en_data, en_time, time)
            
            return pose_pred.cpu()


        if(attibute=="L"):
            vectors_loca         = vectors[0]
            vectors_time         = vectors[1]
            vectors_conf         = vectors[2]

            en_loca              = torch.from_numpy(vectors_loca)
            en_time              = torch.from_numpy(vectors_time)
            en_conf              = torch.from_numpy(vectors_conf)
            time                 = torch.from_numpy(time)

            if(len(en_loca.shape)!=3):
                en_loca          = en_loca.unsqueeze(0)             
                en_time          = en_time.unsqueeze(0)             
            else:
                en_loca          = en_loca.permute(0, 1, 2)         

            BS = en_loca.size(0)
            t_ = en_loca.size(1)

            en_loca_xy           = en_loca[:, :, :90]
            en_loca_xy           = en_loca_xy.view(BS, t_, 45, 2)
            en_loca_n            = en_loca[:, :, 90:]
            en_loca_n            = en_loca_n.view(BS, t_, 3, 3)

            new_en_loca_n = []
            for bs in range(BS):
                x0_                  = np.array(en_loca_xy[bs, :, 44, 0])
                y0_                  = np.array(en_loca_xy[bs, :, 44, 1])
                n_                   = np.log(np.array(en_loca_n[bs, :, 0, 2]))
                t_                   = np.array(en_time[bs, :])

                loc_                 = torch.diff(en_time[bs, :], dim=0)!=0
                if(self.cfg.phalp.distance_type=="EQ_020" or self.cfg.phalp.distance_type=="EQ_021"):
                    loc_                 = 1
                else:
                    loc_                 = loc_.shape[0] - torch.sum(loc_)+1

                M = t_[:, np.newaxis]**[0, 1]
                time_ = 48 if time[bs]>48 else time[bs]

                clf = Ridge(alpha=5.0)
                clf.fit(M, n_)
                n_p = clf.predict(np.array([1, time_+1+t_[-1]]).reshape(1, -1))
                n_p = n_p[0]
                n_hat = clf.predict(np.hstack((np.ones((t_.size, 1)), t_.reshape((-1, 1)))))
                n_pi  = get_prediction_interval(n_, n_hat, t_, time_+1+t_[-1])

                clf  = Ridge(alpha=1.2)
                clf.fit(M, x0_)
                x_p  = clf.predict(np.array([1, time_+1+t_[-1]]).reshape(1, -1))
                x_p  = x_p[0]
                x_p_ = (x_p-0.5)*np.exp(n_p)/5000.0*256.0
                x_hat = clf.predict(np.hstack((np.ones((t_.size, 1)), t_.reshape((-1, 1)))))
                x_pi  = get_prediction_interval(x0_, x_hat, t_, time_+1+t_[-1])

                clf  = Ridge(alpha=2.0)
                clf.fit(M, y0_)
                y_p  = clf.predict(np.array([1, time_+1+t_[-1]]).reshape(1, -1))
                y_p  = y_p[0]
                y_p_ = (y_p-0.5)*np.exp(n_p)/5000.0*256.0
                y_hat = clf.predict(np.hstack((np.ones((t_.size, 1)), t_.reshape((-1, 1)))))
                y_pi  = get_prediction_interval(y0_, y_hat, t_, time_+1+t_[-1])
                
                new_en_loca_n.append([x_p_, y_p_, np.exp(n_p), x_pi/loc_, y_pi/loc_, np.exp(n_pi)/loc_, 1, 1, 0])
                en_loca_xy[bs, -1, 44, 0] = x_p
                en_loca_xy[bs, -1, 44, 1] = y_p
                
            new_en_loca_n        = torch.from_numpy(np.array(new_en_loca_n))
            xt                   = torch.cat((en_loca_xy[:, -1, :, :].view(BS, 90), (new_en_loca_n.float()).view(BS, 9)), 1)

        return xt
    
    def get_pose_distance(self, track_pose, detect_pose):
        """Compute pair-wise squared l2 distances between points in `track_pose` and `detect_pose`.""" 
        track_pose, detect_pose = np.asarray(track_pose), np.asarray(detect_pose)

        if(self.cfg.phalp.pose_distance=="smpl"):
            # remove additional dimension used for encoding location (last 3 elements)
            track_pose = track_pose[:, :-3]
            detect_pose = detect_pose[:, :-3]

        if len(track_pose) == 0 or len(detect_pose) == 0:
            return np.zeros((len(track_pose), len(detect_pose)))
        track_pose2, detect_pose2 = np.square(track_pose).sum(axis=1), np.square(detect_pose).sum(axis=1)
        r2 = -2. * np.dot(track_pose, detect_pose.T) + track_pose2[:, None] + detect_pose2[None, :]
        r2 = np.clip(r2, 0., float(np.inf))

        return r2
    
    def get_uv_distance(self, t_uv, d_uv):
        t_uv         = torch.from_numpy(t_uv).cuda().float()
        d_uv         = torch.from_numpy(d_uv).cuda().float()
        d_mask       = d_uv[3:, :, :]>0.5
        t_mask       = t_uv[3:, :, :]>0.5
        
        mask_dt      = torch.logical_and(d_mask, t_mask)
        mask_dt      = mask_dt.repeat(4, 1, 1)
        mask_        = torch.logical_not(mask_dt)
        
        t_uv[mask_]  = 0.0
        d_uv[mask_]  = 0.0

        with torch.no_grad():
            t_emb    = self.HMAR.autoencoder_hmar(t_uv.unsqueeze(0), en=True)
            d_emb    = self.HMAR.autoencoder_hmar(d_uv.unsqueeze(0), en=True)
        t_emb        = t_emb.view(-1)/10**3
        d_emb        = d_emb.view(-1)/10**3
        return t_emb.cpu().numpy(), d_emb.cpu().numpy(), torch.sum(mask_dt).cpu().numpy()/4/256/256/2
    
    def calculate_iou(self, box1, box2):
        """
        Calculate the intersection over union (IoU) between two bounding boxes.

        Args:
            box1: A tuple (x1, y1, x2, y2) representing the coordinates of the first bounding box.
            box2: A tuple (x1, y1, x2, y2) representing the coordinates of the second bounding box.

        Returns:
            The IoU score between the two bounding boxes.
        """
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        # Calculate the area of intersection
        intersection_area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)

        # Calculate the area of each bounding box
        box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
        box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

        # Calculate the area of union
        union_area = box1_area + box2_area - intersection_area

        # Calculate the IoU score
        iou = intersection_area / union_area

        # Add a size penalty term (fixes bug where if box2 is massive [when someone steps in front of camera or whole area is predicted], it is penalized as 
        # iou in this case would obviously be high)
        size_penalty_factor = 0.3
        size_difference = abs(box1_area - box2_area) / max(box1_area, box2_area)
        size_penalty = size_penalty_factor * size_difference

        iou_with_penalty = iou - size_penalty

        return iou
    
    def process_bbox(self,result):
        bbox_result = []
        bbox_labels = []
        pose_result = []
        bboxArray = []
        bboxArrayCoordinates = []
        for res in result:
            if 'bbox' in res:
                bbox_result.append(res['bbox'])
                bbox_labels.append(res.get('label', None))
        bboxes = np.vstack(bbox_result)
        bboxes = np.split(
        bboxes, bboxes.shape[0], axis=0) if bboxes.shape[0] > 0 else []
        if isinstance(bboxes, np.ndarray):
            bboxes = [bboxes]
        for i, _bboxes in enumerate(bboxes):
            _bboxes = _bboxes.astype(np.int32)
            _top_k = _bboxes.shape[0]
            for j in range(_top_k):
                left_top = (_bboxes[j, 0], _bboxes[j, 1])
                right_bottom = (_bboxes[j, 2], _bboxes[j, 3])
                x1, y1 = left_top
                x2, y2 = right_bottom
                xmin = min(x1, x2)
                ymin = min(y1, y2)
                box_width = max(x1, x2) - xmin
                box_height = max(y1, y2) - ymin
                bbx = (xmin, ymin, box_width, box_height)
                bboxArray.append(bbx)
                bboxArrayCoordinates.append((_bboxes[j, 0], _bboxes[j, 1], _bboxes[j, 2], _bboxes[j, 3]))
        return bboxArray, bboxArrayCoordinates

    def get_croped_image(self, image, bbox, seg_mask):
        
        # Encode the mask for storing, borrowed from tao dataset
        # https://github.com/TAO-Dataset/tao/blob/master/scripts/detectors/detectron2_infer.py
        masks_decoded = np.array(np.expand_dims(seg_mask, 2), order='F', dtype=np.uint8)
        rles = mask_utils.encode(masks_decoded)
        for rle in rles: 
            rle["counts"] = rle["counts"].decode("utf-8")
            
        seg_mask = seg_mask.astype(int)*255
        if(len(seg_mask.shape)==2):
            seg_mask = np.expand_dims(seg_mask, 2)
            seg_mask = np.repeat(seg_mask, 3, 2)
        
        center_      = np.array([(bbox[2] + bbox[0])/2, (bbox[3] + bbox[1])/2])
        scale_       = np.array([(bbox[2] - bbox[0]), (bbox[3] - bbox[1])])
        mask_tmp     = process_mask(seg_mask.astype(np.uint8), center_, 1.0*np.max(scale_))
        image_tmp    = process_image(image, center_, 1.0*np.max(scale_))
        masked_image = torch.cat((image_tmp, mask_tmp[:1, :, :]), 0)
        
        return masked_image, center_, scale_, rles
    

    def get_human_features(self, image, seg_mask, bbox, score, frame_name, cls_id, t_, measurments, gt=1, ann=None, extra_data=[]):
        NPEOPLE = len(score)

        if(NPEOPLE==0): return []

        img_height, img_width, new_image_size, left, top = measurments                
        ratio = 1.0/int(new_image_size)*self.cfg.render.res
        masked_image_list = []
        center_list = []
        scale_list = []
        rles_list = []
        selected_ids = []
        for p_ in range(NPEOPLE):
            masked_image, center_, scale_, rles = self.get_croped_image(image, bbox[p_], seg_mask[p_])
            masked_image_list.append(masked_image)
            center_list.append(center_)
            scale_list.append(scale_)
            rles_list.append(rles)
            selected_ids.append(p_)

        masked_image_list = torch.stack(masked_image_list, dim=0)
        BS = masked_image_list.size(0)
        
        with torch.no_grad():
            extra_args      = {}
            hmar_out        = self.HMAR(masked_image_list.cuda(), **extra_args) 
            uv_vector       = hmar_out['uv_vector']
            appe_embedding  = self.HMAR.autoencoder_hmar(uv_vector, en=True)
            appe_embedding  = appe_embedding.view(appe_embedding.shape[0], -1)
            pred_smpl_params, pred_joints_2d, pred_joints, pred_cam  = self.HMAR.get_3d_parameters(hmar_out['pose_smpl'], hmar_out['pred_cam'],
                                                                                               center=(np.array(center_list) + np.array([left, top]))*ratio,
                                                                                               img_size=self.cfg.render.res,
                                                                                               scale=np.max(np.array(scale_list), axis=1, keepdims=True)*ratio)
            pred_smpl_params = [{k:v[i].cpu().numpy() for k,v in pred_smpl_params.items()} for i in range(BS)]
            
            if(self.cfg.phalp.pose_distance=="joints"):
                pose_embedding  = pred_joints.cpu().view(BS, -1)
            elif(self.cfg.phalp.pose_distance=="smpl"):
                pose_embedding = []
                for i in range(BS):
                    pose_embedding_  = smpl_to_pose_camera_vector(pred_smpl_params[i], pred_cam[i])
                    pose_embedding.append(torch.from_numpy(pose_embedding_[0]))
                pose_embedding = torch.stack(pose_embedding, dim=0)
            else:
                raise ValueError("Unknown pose distance")
            pred_joints_2d_ = pred_joints_2d.reshape(BS,-1)/self.cfg.render.res
            pred_cam_ = pred_cam.view(BS, -1)
            pred_joints_2d_.contiguous()
            pred_cam_.contiguous()
            loca_embedding  = torch.cat((pred_joints_2d_, pred_cam_, pred_cam_, pred_cam_), 1)
        
        # keeping it here for legacy reasons (T3DP), but it is not used.
        full_embedding    = torch.cat((appe_embedding.cpu(), pose_embedding, loca_embedding.cpu()), 1)
        
        detection_data_list = []
        for i, p_ in enumerate(selected_ids):
            detection_data = {
                                "bbox"            : np.array([bbox[p_][0], bbox[p_][1], (bbox[p_][2] - bbox[p_][0]), (bbox[p_][3] - bbox[p_][1])]),
                                "mask"            : rles_list[i],
                                "conf"            : score[p_], 
                                
                                "appe"            : appe_embedding[i].cpu().numpy(), 
                                "pose"            : pose_embedding[i].numpy(), 
                                "loca"            : loca_embedding[i].cpu().numpy(), 
                                "uv"              : uv_vector[i].cpu().numpy(), 
                                
                                "embedding"       : full_embedding[i], 
                                "center"          : center_list[i],
                                "scale"           : scale_list[i],
                                "smpl"            : pred_smpl_params[i],
                                "camera"          : pred_cam_[i].cpu().numpy(),
                                "camera_bbox"     : hmar_out['pred_cam'][i].cpu().numpy(),
                                "3d_joints"       : pred_joints[i].cpu().numpy(),
                                "2d_joints"       : pred_joints_2d_[i].cpu().numpy(),
                                "size"            : [img_height, img_width],
                                "img_path"        : frame_name,
                                "img_name"        : frame_name.split('/')[-1] if isinstance(frame_name, str) else None,
                                "class_name"      : cls_id[p_],
                                "time"            : t_,

                                "ground_truth"    : gt[p_],
                                "annotations"     : ann[p_],
                                "extra_data"      : extra_data[p_]
                            }
            detection_data_list.append(Detection(detection_data)) 

        return detection_data_list

    def cached_download_from_drive(self, additional_urls=None):
        """Download a file from Google Drive if it doesn't exist yet.
        :param url: the URL of the file to download
        :param path: the path to save the file to
        """
        
        os.makedirs(os.path.join(CACHE_DIR, "phalp"), exist_ok=True)
        os.makedirs(os.path.join(CACHE_DIR, "phalp/3D"), exist_ok=True)
        os.makedirs(os.path.join(CACHE_DIR, "phalp/weights"), exist_ok=True)
        os.makedirs(os.path.join(CACHE_DIR, "phalp/ava"), exist_ok=True)

        smpl_path = os.path.join(CACHE_DIR, "phalp/3D/models/smpl/SMPL_NEUTRAL.pkl")

        if not os.path.exists(smpl_path):
            # We are downloading the SMPL model here for convenience. Please accept the license
            # agreement on the SMPL website: https://smpl.is.tue.mpg.
            os.makedirs(os.path.join(CACHE_DIR, "phalp/3D/models/smpl"), exist_ok=True)
            os.system('wget https://github.com/classner/up/raw/master/models/3D/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl')

            convert_pkl('basicModel_neutral_lbs_10_207_0_v1.0.0.pkl')
            os.system('rm basicModel_neutral_lbs_10_207_0_v1.0.0.pkl')
            os.system('mv basicModel_neutral_lbs_10_207_0_v1.0.0_p3.pkl ' + smpl_path)

        additional_urls = additional_urls if additional_urls is not None else {}
        download_files = {
            "head_faces.npy"           : ["https://people.eecs.berkeley.edu/~jathushan/projects/phalp/3D/head_faces.npy", os.path.join(CACHE_DIR, "phalp/3D")],
            "mean_std.npy"             : ["https://people.eecs.berkeley.edu/~jathushan/projects/phalp/3D/mean_std.npy", os.path.join(CACHE_DIR, "phalp/3D")],
            "smpl_mean_params.npz"     : ["https://people.eecs.berkeley.edu/~jathushan/projects/phalp/3D/smpl_mean_params.npz", os.path.join(CACHE_DIR, "phalp/3D")],
            "SMPL_to_J19.pkl"          : ["https://people.eecs.berkeley.edu/~jathushan/projects/phalp/3D/SMPL_to_J19.pkl", os.path.join(CACHE_DIR, "phalp/3D")],
            "texture.npz"              : ["https://people.eecs.berkeley.edu/~jathushan/projects/phalp/3D/texture.npz", os.path.join(CACHE_DIR, "phalp/3D")],

            "hmar_v2_weights.pth"      : ["https://people.eecs.berkeley.edu/~jathushan/projects/phalp/weights/hmar_v2_weights.pth", os.path.join(CACHE_DIR, "phalp/weights")],
            "pose_predictor.pth"       : ["https://people.eecs.berkeley.edu/~jathushan/projects/phalp/weights/pose_predictor_40006.ckpt", os.path.join(CACHE_DIR, "phalp/weights")],
            "pose_predictor.yaml"      : ["https://people.eecs.berkeley.edu/~jathushan/projects/phalp/weights/config_40006.yaml", os.path.join(CACHE_DIR, "phalp/weights")],
            
            # data for ava dataset
            #"ava_labels.pkl"           : ["https://people.eecs.berkeley.edu/~jathushan/projects/phalp/ava/ava_labels.pkl", os.path.join(CACHE_DIR, "phalp/ava")],
            #"ava_class_mapping.pkl"   : ["https://people.eecs.berkeley.edu/~jathushan/projects/phalp/ava/ava_class_mappping.pkl", os.path.join(CACHE_DIR, "phalp/ava")],
    
        } | additional_urls # type: ignore
        
        for file_name, url in download_files.items():
            if not os.path.exists(os.path.join(url[1], file_name)):
                print("Downloading file: " + file_name)
                # output = gdown.cached_download(url[0], os.path.join(url[1], file_name), fuzzy=True)
                output = cache_url(url[0], os.path.join(url[1], file_name))
                assert os.path.exists(os.path.join(url[1], file_name)), f"{output} does not exist"