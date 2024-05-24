# Openpose
# Original from CMU https://github.com/CMU-Perceptual-Computing-Lab/openpose
# 2nd Edited by https://github.com/Hzzone/pytorch-openpose
# 3rd Edited by ControlNet
# 4th Edited by ControlNet (added face and correct hands)

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import torch
import numpy as np
from . import util
from .wholebody import Wholebody

def draw_pose(pose, H, W):
    bodies = pose['bodies']
    faces = pose['faces']
    hands = pose['hands']
    candidate = bodies['candidate']
    subset = bodies['subset']
    canvas = np.zeros(shape=(H, W, 3), dtype=np.uint8)

    canvas = util.draw_bodypose(canvas, candidate, subset)

    canvas = util.draw_handpose(canvas, hands)

    canvas = util.draw_facepose(canvas, faces)

    return canvas


class DWposeDetector:
    def __init__(self):

        self.pose_estimation = Wholebody()

    def __call__(self, oriImg, pred_bbox):
        oriImg = oriImg.copy()
        H, W, C = oriImg.shape
        with torch.no_grad():
            candidate, subset = self.pose_estimation(oriImg, pred_bbox)
            nums, keys, locs = candidate.shape
            candidate[..., 0] #/= float(W)
            candidate[..., 1] #/= float(H)
            #print(f"Candidate is {candidate}")
            # print(f"Subset is {subset}")
            body = candidate[:,:18].copy()
            body = body.reshape(nums*18, locs)
            score = subset[:,:18]
            # for i in range(len(score)):
            #     for j in range(len(score[i])):
            #         if score[i][j] > 0.3:
            #             score[i][j] = int(18*i+j)
            #         else:
            #             score[i][j] = -1

            un_visible = subset<0.3
            #candidate[un_visible] = -1

            foot = candidate[:,18:24]

            faces = candidate[:,24:92]

            hands = candidate[:,92:113]
            hands = np.vstack([hands, candidate[:,113:]])
            
            bodies = dict(candidate=body, subset=score)
            pose = dict(bodies=bodies, hands=hands, faces=faces)
            # candidate_withScores = []
            
            # for candidate_list, subset_list in zip(candidate, subset):
            #     frameList = []
            #     for i in range(len(candidate_list)):
            #         frameList.append([candidate_list[i][0], candidate_list[i][1], subset_list[i]])
            #     candidate_withScores.append(frameList)

            # candidate_withScores = np.array(candidate_withScores)
            # Convert candidate to numpy array and subset to numpy array for proper broadcasting
            candidate_np = np.array(candidate)
            subset_np = np.array(subset)[:,:,None] # Add an extra dimension to subset for broadcasting

            # Combine candidate and subset into one array
            candidate_withScores = np.concatenate((candidate_np, subset_np), axis=2)
            return candidate_withScores
