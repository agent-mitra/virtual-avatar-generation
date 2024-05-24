import numpy as np
import math
import copy
from scipy.interpolate import CubicSpline


selected_keypoints = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, # COCO Body Keypoints which includes eyes (17 keypoints)
    17, 18, 19, 20, 21, 22, # COCO Foot Keypoints (6 keypoints)

    112, 114, 115, 117, 118, 121, 122, 125, 126, 129, 130,  # Right Wrist and MCP joints
    116, 120, 124, 128, 132, # Right DIP joints

    91, 93, 94, 96, 97, 100, 101, 104, 105, 108, 109, # Left Wrist and MCP joints ( l to r)
    95, 99, 103, 107, 111, # Left DIP joints
    60, 61, 66, 67 # Eye keypoints
]

def distance(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def modifyToSelectedKeypoints(preds_all, selected_keypoints):
    num_keypoints = len(selected_keypoints)
    predsArr = []
    for key in preds_all:
        # Select only the keypoints with indices in keypointsArr
        keyPArr = {}
        keyPArr['keypoints'] = key[selected_keypoints]
        assert len(keyPArr['keypoints']) == num_keypoints
        predsArr.append(keyPArr)
    predsArr = np.array(predsArr)
    numFrames = len(preds_all)
    preds_all_2 = []
    for pred in predsArr:
        values = pred['keypoints']
        assert len(values) == num_keypoints
        preds_all_2.append(values)
    assert len(preds_all_2) == numFrames
    preds_all_2 = np.array(preds_all_2)
    if (len(preds_all_2) > 0):
        N, k = preds_all_2.shape[0], preds_all_2.shape[1]
        assert k == num_keypoints
        assert len(preds_all_2) == numFrames
    preds_all_2 = np.array(preds_all_2)
    return preds_all_2

def initalizeConfidenceMap(preds):
    lastConfident = {}
    keypoints = getKeypoints(preds)
    lastConfident['left_upper_wrist'] = copy.deepcopy(keypoints['left_upper_wrist'])
    lastConfident['right_upper_wrist'] = copy.deepcopy(keypoints['right_upper_wrist'])

    lastConfident['left_lower_wrist'] = copy.deepcopy(keypoints['left_lower_wrist'])
    lastConfident['right_lower_wrist'] = copy.deepcopy(keypoints['right_lower_wrist'])

    lastConfident['left_shoulder'] = copy.deepcopy(keypoints['left_shoulder'])
    lastConfident['right_shoulder'] = copy.deepcopy(keypoints['right_shoulder'])
    lastConfident['left_elbow'] = copy.deepcopy(keypoints['left_elbow'])
    lastConfident['right_elbow'] = copy.deepcopy(keypoints['right_elbow'])

    lastConfident['left_hip'] = copy.deepcopy(keypoints['left_hip'])
    lastConfident['right_hip'] = copy.deepcopy(keypoints['right_hip'])

    lastConfident['left_knee'] = copy.deepcopy(keypoints['left_knee'])
    lastConfident['right_knee'] = copy.deepcopy(keypoints['right_knee'])

    lastConfident['left_ankle'] = copy.deepcopy(keypoints['left_ankle'])
    lastConfident['right_ankle'] = copy.deepcopy(keypoints['right_ankle'])

    lastConfident['left_heel'] = copy.deepcopy(keypoints['left_heel'])
    lastConfident['left_toe'] = copy.deepcopy(keypoints['left_toe'])
    lastConfident['left_smalltoe'] = copy.deepcopy(keypoints['left_smalltoe'])
    lastConfident['right_toe'] = copy.deepcopy(keypoints['right_toe'])
    lastConfident['right_smalltoe'] = copy.deepcopy(keypoints['right_smalltoe'])
    lastConfident['right_heel'] = copy.deepcopy(keypoints['right_heel'])

    lastConfident['left_mcp_1'] = copy.deepcopy(keypoints['left_mcp_1'])
    lastConfident['left_mcp_2'] = copy.deepcopy(keypoints['left_mcp_2'])
    lastConfident['left_mcp_3'] = copy.deepcopy(keypoints['left_mcp_3'])
    lastConfident['left_mcp_4'] = copy.deepcopy(keypoints['left_mcp_4'])
    lastConfident['left_mcp_5'] = copy.deepcopy(keypoints['left_mcp_5'])
    
    lastConfident['left_pip_1'] = copy.deepcopy(keypoints['left_pip_1'])
    lastConfident['left_pip_2'] = copy.deepcopy(keypoints['left_pip_2'])
    lastConfident['left_pip_3'] = copy.deepcopy(keypoints['left_pip_3'])
    lastConfident['left_pip_4'] = copy.deepcopy(keypoints['left_pip_4'])
    lastConfident['left_pip_5'] = copy.deepcopy(keypoints['left_pip_5'])

    lastConfident['left_dip_1'] = copy.deepcopy(keypoints['left_dip_1'])
    lastConfident['left_dip_2'] = copy.deepcopy(keypoints['left_dip_2'])
    lastConfident['left_dip_3'] = copy.deepcopy(keypoints['left_dip_3'])
    lastConfident['left_dip_4'] = copy.deepcopy(keypoints['left_dip_4'])
    lastConfident['left_dip_5'] = copy.deepcopy(keypoints['left_dip_5'])

    lastConfident['right_mcp_1'] = copy.deepcopy(keypoints['right_mcp_1'])
    lastConfident['right_mcp_2'] = copy.deepcopy(keypoints['right_mcp_2'])
    lastConfident['right_mcp_3'] = copy.deepcopy(keypoints['right_mcp_3'])
    lastConfident['right_mcp_4'] = copy.deepcopy(keypoints['right_mcp_4'])
    lastConfident['right_mcp_5'] = copy.deepcopy(keypoints['right_mcp_5'])
    
    lastConfident['right_pip_1'] = copy.deepcopy(keypoints['right_pip_1'])
    lastConfident['right_pip_2'] = copy.deepcopy(keypoints['right_pip_2'])
    lastConfident['right_pip_3'] = copy.deepcopy(keypoints['right_pip_3'])
    lastConfident['right_pip_4'] = copy.deepcopy(keypoints['right_pip_4'])
    lastConfident['right_pip_5'] = copy.deepcopy(keypoints['right_pip_5'])

    lastConfident['right_dip_1'] = copy.deepcopy(keypoints['right_dip_1'])
    lastConfident['right_dip_2'] = copy.deepcopy(keypoints['right_dip_2'])
    lastConfident['right_dip_3'] = copy.deepcopy(keypoints['right_dip_3'])
    lastConfident['right_dip_4'] = copy.deepcopy(keypoints['right_dip_4'])
    lastConfident['right_dip_5'] = copy.deepcopy(keypoints['right_dip_5'])

    lastConfident['left_eye_main'] = copy.deepcopy(keypoints['left_eye_main'])
    lastConfident['right_eye_main'] = copy.deepcopy(keypoints['right_eye_main'])
    lastConfident['left_eye_1'] = copy.deepcopy(keypoints['left_eye_1'])
    lastConfident['left_eye_2'] = copy.deepcopy(keypoints['left_eye_2'])
    lastConfident['right_eye_1'] = copy.deepcopy(keypoints['right_eye_1'])
    lastConfident['right_eye_2'] = copy.deepcopy(keypoints['right_eye_2'])

    return lastConfident

def getKeypoints(preds):
    keypoints = {}
#     selected_keypoints = [
#         0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, # COCO Body Keypoints which includes eyes (17 keypoints)
#         17, 18, 19, 20, 21, 22, # COCO Foot Keypoints (6 keypoints)

#         112, 114, 115, 117, 118, 121, 122, 125, 126, 129, 130,  # Right Wrist and MCP joints
#         116, 120, 124, 128, 132, # Right DIP joints

#         91, 93, 94, 96, 97, 100, 101, 104, 105, 108, 109, # Left Wrist and MCP joints ( l to r)
#         95, 99, 103, 107, 111, # Left DIP joints
#         60, 61, 66, 67 # Eye keypoints
#     ]
    keypoints['left_upper_wrist'] = copy.deepcopy(preds[9])
    keypoints['right_upper_wrist'] = copy.deepcopy(preds[10])

    keypoints['left_lower_wrist'] = copy.deepcopy(preds[91])
    keypoints['right_lower_wrist'] = copy.deepcopy(preds[112])

    keypoints['left_shoulder'] = copy.deepcopy(preds[5])
    keypoints['right_shoulder'] = copy.deepcopy(preds[6])
    keypoints['left_elbow'] = copy.deepcopy(preds[7])
    keypoints['right_elbow'] = copy.deepcopy(preds[8])

    keypoints['left_hip'] = copy.deepcopy(preds[11])
    keypoints['right_hip'] = copy.deepcopy(preds[12])

    keypoints['left_knee'] = copy.deepcopy(preds[13])
    keypoints['right_knee'] = copy.deepcopy(preds[14])

    keypoints['left_ankle'] = copy.deepcopy(preds[15])
    keypoints['right_ankle'] = copy.deepcopy(preds[16])

    keypoints['left_toe'] = copy.deepcopy(preds[17])
    keypoints['left_smalltoe'] = copy.deepcopy(preds[18])
    keypoints['left_heel'] = copy.deepcopy(preds[19])
    
    keypoints['right_toe'] = copy.deepcopy(preds[20])
    keypoints['right_smalltoe'] = copy.deepcopy(preds[21])
    keypoints['right_heel'] = copy.deepcopy(preds[22])

    keypoints['left_mcp_1'] = copy.deepcopy(preds[93])
    keypoints['left_mcp_2'] = copy.deepcopy(preds[96])
    keypoints['left_mcp_3'] = copy.deepcopy(preds[100])
    keypoints['left_mcp_4'] = copy.deepcopy(preds[104])
    keypoints['left_mcp_5'] = copy.deepcopy(preds[108])
    
    keypoints['left_pip_1'] = copy.deepcopy(preds[94])
    keypoints['left_pip_2'] = copy.deepcopy(preds[97])
    keypoints['left_pip_3'] = copy.deepcopy(preds[101])
    keypoints['left_pip_4'] = copy.deepcopy(preds[105])
    keypoints['left_pip_5'] = copy.deepcopy(preds[109])

    keypoints['left_dip_1'] = copy.deepcopy(preds[95])
    keypoints['left_dip_2'] = copy.deepcopy(preds[99])
    keypoints['left_dip_3'] = copy.deepcopy(preds[103])
    keypoints['left_dip_4'] = copy.deepcopy(preds[107])
    keypoints['left_dip_5'] = copy.deepcopy(preds[111])

    keypoints['right_mcp_1'] = copy.deepcopy(preds[114])
    keypoints['right_mcp_2'] = copy.deepcopy(preds[117])
    keypoints['right_mcp_3'] = copy.deepcopy(preds[121])
    keypoints['right_mcp_4'] = copy.deepcopy(preds[125])
    keypoints['right_mcp_5'] = copy.deepcopy(preds[129])
    
    keypoints['right_pip_1'] = copy.deepcopy(preds[115])
    keypoints['right_pip_2'] = copy.deepcopy(preds[118])
    keypoints['right_pip_3'] = copy.deepcopy(preds[122])
    keypoints['right_pip_4'] = copy.deepcopy(preds[126])
    keypoints['right_pip_5'] = copy.deepcopy(preds[130])

    keypoints['right_dip_1'] = copy.deepcopy(preds[116])
    keypoints['right_dip_2'] = copy.deepcopy(preds[120])
    keypoints['right_dip_3'] = copy.deepcopy(preds[124])
    keypoints['right_dip_4'] = copy.deepcopy(preds[128])
    keypoints['right_dip_5'] = copy.deepcopy(preds[132])

    keypoints['left_eye_main'] = copy.deepcopy(preds[1])
    keypoints['right_eye_main'] = copy.deepcopy(preds[2])
    keypoints['left_eye_1'] = copy.deepcopy(preds[60])
    keypoints['left_eye_2'] = copy.deepcopy(preds[61])
    keypoints['right_eye_1'] = copy.deepcopy(preds[66])
    keypoints['right_eye_2'] = copy.deepcopy(preds[67])
    return keypoints


def getPosePreds(preds, keypoints, chance=0.0):
    preds[9] = (keypoints['left_upper_wrist'][0], keypoints['left_upper_wrist'][1],  keypoints['left_upper_wrist'][2] + chance)
    preds[10] = (keypoints['right_upper_wrist'][0], keypoints['right_upper_wrist'][1], keypoints['right_upper_wrist'][2] + chance)

    preds[91] = (keypoints['left_lower_wrist'][0], keypoints['left_lower_wrist'][1], keypoints['left_lower_wrist'][2] + chance)
    preds[112] = (keypoints['right_lower_wrist'][0], keypoints['right_lower_wrist'][1], keypoints['right_lower_wrist'][2] + chance)

    preds[11] = (keypoints['left_hip'][0], keypoints['left_hip'][1], keypoints['left_hip'][2] + chance)
    preds[12] = (keypoints['right_hip'][0], keypoints['right_hip'][1], keypoints['right_hip'][2] + chance)
    preds[5] = (keypoints['left_shoulder'][0], keypoints['left_shoulder'][1], keypoints['left_shoulder'][2] + chance)
    preds[6] = (keypoints['right_shoulder'][0], keypoints['right_shoulder'][1], keypoints['right_shoulder'][2] + chance)
    preds[7] = (keypoints['left_elbow'][0], keypoints['left_elbow'][1], keypoints['left_elbow'][2] + chance)
    preds[8] = (keypoints['right_elbow'][0], keypoints['right_elbow'][1], keypoints['right_elbow'][2] + chance)

    preds[13] = (keypoints['left_knee'][0], keypoints['left_knee'][1], keypoints['left_knee'][2] + chance)
    preds[14] = (keypoints['right_knee'][0], keypoints['right_knee'][1], keypoints['right_knee'][2] + chance)

    preds[15] = (keypoints['left_ankle'][0], keypoints['left_ankle'][1], keypoints['left_ankle'][2] + chance)
    preds[16] = (keypoints['right_ankle'][0], keypoints['right_ankle'][1], keypoints['right_ankle'][2] + chance)

    preds[19] = (keypoints['left_heel'][0], keypoints['left_heel'][1], keypoints['left_heel'][2] + chance)
    preds[17] = (keypoints['left_toe'][0], keypoints['left_toe'][1], keypoints['left_toe'][2] + chance)
    preds[18] = (keypoints['left_smalltoe'][0], keypoints['left_smalltoe'][1], keypoints['left_smalltoe'][2] + chance)
    
    preds[20] = (keypoints['right_toe'][0], keypoints['right_toe'][1], keypoints['right_toe'][2] + chance)
    preds[21] = (keypoints['right_smalltoe'][0], keypoints['right_smalltoe'][1], keypoints['right_smalltoe'][2] + chance)
    preds[22] = (keypoints['right_heel'][0], keypoints['right_heel'][1], keypoints['right_heel'][2] + chance)

    preds[93] = (keypoints['left_mcp_1'][0], keypoints['left_mcp_1'][1], keypoints['left_mcp_1'][2] + chance)
    preds[96] = (keypoints['left_mcp_2'][0], keypoints['left_mcp_2'][1], keypoints['left_mcp_2'][2] + chance)
    preds[100] = (keypoints['left_mcp_3'][0], keypoints['left_mcp_3'][1], keypoints['left_mcp_3'][2] + chance)
    preds[104] = (keypoints['left_mcp_4'][0], keypoints['left_mcp_4'][1], keypoints['left_mcp_4'][2] + chance)
    preds[108] = (keypoints['left_mcp_5'][0], keypoints['left_mcp_5'][1], keypoints['left_mcp_5'][2] + chance)
    
    preds[94] = (keypoints['left_pip_1'][0], keypoints['left_pip_1'][1], keypoints['left_pip_1'][2] + chance)
    preds[97] = (keypoints['left_pip_2'][0], keypoints['left_pip_2'][1], keypoints['left_pip_2'][2] + chance)
    preds[101] = (keypoints['left_pip_3'][0], keypoints['left_pip_3'][1], keypoints['left_pip_3'][2] + chance)
    preds[105] = (keypoints['left_pip_4'][0], keypoints['left_pip_4'][1], keypoints['left_pip_4'][2] + chance)
    preds[109] = (keypoints['left_pip_5'][0], keypoints['left_pip_5'][1], keypoints['left_pip_5'][2] + chance)

    preds[95] = (keypoints['left_dip_1'][0], keypoints['left_dip_1'][1], keypoints['left_dip_1'][2] + chance)
    preds[99] = (keypoints['left_dip_2'][0], keypoints['left_dip_2'][1], keypoints['left_dip_2'][2] + chance)
    preds[103] = (keypoints['left_dip_3'][0], keypoints['left_dip_3'][1], keypoints['left_dip_3'][2] + chance)
    preds[107] = (keypoints['left_dip_4'][0], keypoints['left_dip_4'][1], keypoints['left_dip_4'][2] + chance)
    preds[111] = (keypoints['left_dip_5'][0], keypoints['left_dip_5'][1], keypoints['left_dip_5'][2] + chance)

    preds[114] = (keypoints['right_mcp_1'][0], keypoints['right_mcp_1'][1], keypoints['right_mcp_1'][2] + chance)
    preds[117] = (keypoints['right_mcp_2'][0], keypoints['right_mcp_2'][1], keypoints['right_mcp_2'][2] + chance)
    preds[121] = (keypoints['right_mcp_3'][0], keypoints['right_mcp_3'][1], keypoints['right_mcp_3'][2] + chance)
    preds[125] = (keypoints['right_mcp_4'][0], keypoints['right_mcp_4'][1], keypoints['right_mcp_4'][2] + chance)
    preds[129] = (keypoints['right_mcp_5'][0], keypoints['right_mcp_5'][1], keypoints['right_mcp_5'][2] + chance)
    
    preds[115] = (keypoints['right_pip_1'][0], keypoints['right_pip_1'][1], keypoints['right_pip_1'][2] + chance)
    preds[118] = (keypoints['right_pip_2'][0], keypoints['right_pip_2'][1], keypoints['right_pip_2'][2] + chance)
    preds[122] = (keypoints['right_pip_3'][0], keypoints['right_pip_3'][1], keypoints['right_pip_3'][2] + chance)
    preds[126] = (keypoints['right_pip_4'][0], keypoints['right_pip_4'][1], keypoints['right_pip_4'][2] + chance)
    preds[130] = (keypoints['right_pip_5'][0], keypoints['right_pip_5'][1], keypoints['right_pip_5'][2] + chance)

    preds[116] = (keypoints['right_dip_1'][0], keypoints['right_dip_1'][1], keypoints['right_dip_1'][2] + chance)
    preds[120] = (keypoints['right_dip_2'][0], keypoints['right_dip_2'][1], keypoints['right_dip_2'][2] + chance)
    preds[124] = (keypoints['right_dip_3'][0], keypoints['right_dip_3'][1], keypoints['right_dip_3'][2] + chance)
    preds[128] = (keypoints['right_dip_4'][0], keypoints['right_dip_4'][1], keypoints['right_dip_4'][2] + chance)
    preds[132] = (keypoints['right_dip_5'][0], keypoints['right_dip_5'][1], keypoints['right_dip_5'][2] + chance)

    preds[1] = (keypoints['left_eye_main'][0], keypoints['left_eye_main'][1], keypoints['left_eye_main'][2] + chance)
    preds[2] = (keypoints['right_eye_main'][0], keypoints['right_eye_main'][1], keypoints['right_eye_main'][2] + chance)
    preds[60] = (keypoints['left_eye_1'][0], keypoints['left_eye_1'][1], keypoints['left_eye_1'][2] + chance)
    preds[61] = (keypoints['left_eye_2'][0], keypoints['left_eye_2'][1], keypoints['left_eye_2'][2] + chance)
    preds[66] = (keypoints['right_eye_1'][0], keypoints['right_eye_1'][1], keypoints['right_eye_1'][2] + chance)
    preds[67] = (keypoints['right_eye_2'][0], keypoints['right_eye_2'][1], keypoints['right_eye_2'][2] + chance)
    return copy.deepcopy(preds)

def getKeypointBoundaries(preds):
    # Compute lower leg length
    good_keypoints = getKeypoints(preds)
    lower_leg_lengths = []
    for side in ['left', 'right']:
        knee_key = f'{side}_knee'
        ankle_key = f'{side}_ankle'
        lower_leg_lengths.append(distance(good_keypoints[knee_key], good_keypoints[ankle_key]))
    assert len(lower_leg_lengths) > 0

    # Compute upper leg length
    upper_leg_lengths = []
    for side in ['left', 'right']:
        hip_key = f'{side}_hip'
        knee_key = f'{side}_knee'
        upper_leg_lengths.append(distance(good_keypoints[hip_key], good_keypoints[knee_key]))
    assert len(upper_leg_lengths) > 0

    # Compute hand length
    hand_lengths = []
    for side in ['left', 'right']:
        wrist_key = f'{side}_lower_wrist'
        mcp_key = f'{side}_mcp_1'  # You can use any of the MCP keypoints, assuming equal length
        dist = distance(good_keypoints[wrist_key], good_keypoints[mcp_key])
        hand_lengths.append(dist)
    assert len(hand_lengths) > 0

    # Compute finger length1
    finger_lengths1 = []
    for side in ['left', 'right']:
        mcp_key = f'{side}_mcp_1'  # You can use any of the MCP keypoints, assuming equal length
        pip_key = f'{side}_pip_1'  # You can use any of the DIP keypoints, assuming equal length
        finger_lengths1.append(distance(good_keypoints[mcp_key], good_keypoints[pip_key]))
    assert len(finger_lengths1) > 0
    
    finger_lengths2 = []
    for side in ['left', 'right']:
        pip_key = f'{side}_pip_1'  # You can use any of the MCP keypoints, assuming equal length
        dip_key = f'{side}_dip_1'  # You can use any of the DIP keypoints, assuming equal length
        finger_lengths2.append(distance(good_keypoints[pip_key], good_keypoints[dip_key]))
    assert len(finger_lengths2) > 0

    # Compute eye width
    eye_widths = []
    for side in ['left', 'right']:
        main_eye_key = f'{side}_eye_main'
        eye1_key = f'{side}_eye_1'
        eye2_key = f'{side}_eye_2'
        eye_widths.append(distance(good_keypoints[eye1_key], good_keypoints[eye2_key]))
    assert len(eye_widths) > 0

    # Compute foot length
    foot_lengths = []
    for side in ['left', 'right']:
        ankle_key = f'{side}_toe'
        foot_key = f'{side}_heel' #heel, not foot but same thing
        foot_lengths.append(distance(good_keypoints[ankle_key], good_keypoints[foot_key]))
    assert len(foot_lengths) > 0

    # Compute forearm length
    forearm_lengths = []
    for side in ['left', 'right']:
        elbow_key = f'{side}_elbow'
        wrist_key = f'{side}_lower_wrist' #heel, not foot but same thing
        forearm_lengths.append(distance(good_keypoints[elbow_key], good_keypoints[wrist_key]))
    assert len(forearm_lengths) > 0

    # Compute forearm length
    upper_arm_lengths = []
    for side in ['left', 'right']:
        shoulder_key = f'{side}_shoulder'
        elbow_key = f'{side}_elbow' 
        upper_arm_lengths.append(distance(good_keypoints[shoulder_key], good_keypoints[elbow_key]))
    assert len(upper_arm_lengths) > 0

    # return [max_lower_leg_length, max_upper_leg_length, max_hand_length, max_finger_length, max_eye_width, 
    #         max_foot_length, max_forearm_length, max_upper_arm_lengths]
    return [lower_leg_lengths, upper_leg_lengths, hand_lengths, finger_lengths1, finger_lengths2, eye_widths, foot_lengths, forearm_lengths, upper_arm_lengths]


def adjust_keypoints(posePreds, lengthCache, confidenceMap, threshold=0.6):
    # Adjust ankle and knee keypoints
    keypoints = getKeypoints(posePreds)
    lower_leg_lengths, upper_leg_lengths, hand_lengths, finger_lengths1, finger_lengths2, eye_widths, foot_lengths, forearm_lengths, upper_arm_lengths = lengthCache
    lower_leg_lengths_left, lower_leg_lengths_right = lower_leg_lengths
    upper_leg_lengths_left, upper_leg_lengths_right = upper_leg_lengths
    hand_lengths_left, hand_lengths_right = hand_lengths
    
    finger_lengths_left1, finger_lengths_right1 = finger_lengths1
    finger_lengths_left2, finger_lengths_right2 = finger_lengths2
    
    foot_lengths_left, foot_lengths_right = foot_lengths
    forearm_lengths_left, forearm_lengths_right = forearm_lengths
    upper_arm_lengths_left, upper_arm_lengths_right = upper_arm_lengths
    
#     if keypoints['left_hip'][2] < threshold:
#         if keypoints['right_hip'][2] >= threshold:
#             opposingCache = ('right_hip', 'right_knee')
#             keypoints = copy.deepcopy(adjust_hip_to_knee(keypoints, 'left_hip', 'left_knee', upper_leg_lengths_left, confidenceMap, True, opposingCache))
#         else:
#             keypoints = copy.deepcopy(adjust_hip_to_knee(keypoints, 'left_hip', 'left_knee', upper_leg_lengths_left, confidenceMap))
#     confidenceMap['left_hip'] = copy.deepcopy(keypoints['left_hip'])

#     if keypoints['right_hip'][2] < threshold:
#         keypoints = copy.deepcopy(adjust_hip_to_knee(keypoints, 'right_hip', 'right_knee', upper_leg_lengths_right, confidenceMap))
#     confidenceMap['right_hip'] = copy.deepcopy(keypoints['right_hip'])

#     if keypoints['left_upper_wrist'][2] < threshold:
#         if keypoints['right_upper_wrist'][2] < threshold:
#             opposingCache = ('right_upper_wrist', 'right_elbow')
#             keypoints = copy.deepcopy(adjust_wrist_to_elbow(keypoints, 'left_upper_wrist', 'left_elbow', forearm_lengths_left, confidenceMap, True, opposingCache))
#         else:
#             keypoints = copy.deepcopy(adjust_wrist_to_elbow(keypoints, 'left_upper_wrist', 'left_elbow', forearm_lengths_left, confidenceMap))
#     confidenceMap['left_upper_wrist'] = copy.deepcopy(keypoints['left_upper_wrist'])

#     if keypoints['right_upper_wrist'][2] < threshold:
#         if keypoints['left_upper_wrist'][2] < threshold:
#             opposingCache = ('left_upper_wrist', 'left_elbow')
#             keypoints = copy.deepcopy(adjust_wrist_to_elbow(keypoints, 'right_upper_wrist', 'right_elbow', forearm_lengths_right, confidenceMap, True, opposingCache))
#         else:
#             keypoints = copy.deepcopy(adjust_wrist_to_elbow(keypoints, 'right_upper_wrist', 'right_elbow', forearm_lengths_right, confidenceMap))
#     confidenceMap['right_upper_wrist'] = copy.deepcopy(keypoints['right_upper_wrist'])

    if keypoints['left_ankle'][2] < threshold:
        if keypoints['right_ankle'][2] < threshold:
            opposingCache = ('right_knee', 'right_ankle')
            keypoints = copy.deepcopy(adjust_ankle_to_knee(keypoints, 'left_knee', 'left_ankle', lower_leg_lengths_left, confidenceMap, True, opposingCache))
        else:
            keypoints = copy.deepcopy(adjust_ankle_to_knee(keypoints, 'left_knee', 'left_ankle', lower_leg_lengths_left, confidenceMap))
    #can also try with a threshold range (if >= 0.3 for example)
    confidenceMap['left_ankle'] = copy.deepcopy(keypoints['left_ankle'])

    if keypoints['right_ankle'][2] < threshold:
        if keypoints['left_ankle'][2] < threshold:
            opposingCache = ('left_knee', 'left_ankle')
            keypoints = copy.deepcopy(adjust_ankle_to_knee(keypoints, 'right_knee', 'right_ankle', lower_leg_lengths_right, confidenceMap, True, opposingCache))
        else:
            keypoints = copy.deepcopy(adjust_ankle_to_knee(keypoints, 'right_knee', 'right_ankle', lower_leg_lengths_right, confidenceMap))
    confidenceMap['right_ankle'] = copy.deepcopy(keypoints['right_ankle'])

    # Adjust wrists
    for i in range(1, 6):
        if keypoints[f'left_mcp_{i}'][2] < threshold:
            if keypoints[f'right_mcp_{i}'][2] >= threshold:
                opposingCache = ('right_lower_wrist', f'right_mcp_{i}')
                keypoints = copy.deepcopy(adjust_mcp_to_wrist(keypoints, 'left_lower_wrist', f'left_mcp_{i}', 'left_shoulder', 'left_elbow', hand_lengths_left, forearm_lengths_left, confidenceMap, i, True, opposingCache))
            else:
                keypoints = copy.deepcopy(adjust_mcp_to_wrist(keypoints, 'left_lower_wrist', f'left_mcp_{i}', 'left_shoulder', 'left_elbow', hand_lengths_left, forearm_lengths_left, confidenceMap, i))
        confidenceMap[f'left_mcp_{i}'] = copy.deepcopy(keypoints[f'left_mcp_{i}'])

        if keypoints[f'right_mcp_{i}'][2] < threshold:
            if keypoints[f'left_mcp_{i}'][2] >= threshold:
                opposingCache = ('left_lower_wrist', f'left_mcp_{i}')
                keypoints = copy.deepcopy(adjust_mcp_to_wrist(keypoints, 'right_lower_wrist', f'right_mcp_{i}', 'right_shoulder', 'right_elbow', hand_lengths_right, forearm_lengths_right, confidenceMap, i, True, opposingCache))
            else:
                keypoints = copy.deepcopy(adjust_mcp_to_wrist(keypoints, 'right_lower_wrist', f'right_mcp_{i}', 'right_shoulder', 'right_elbow', hand_lengths_right, forearm_lengths_right, confidenceMap, i))
        confidenceMap[f'right_mcp_{i}'] = copy.deepcopy(keypoints[f'right_mcp_{i}'])
        
        
        
    # Adjust PIP joints (can use same function as the dip one)
    for i in range(1, 6):
        if keypoints[f'left_pip_{i}'][2] < threshold:
            if keypoints[f'right_pip_{i}'][2] >= threshold:
                opposingCache = (f'right_mcp_{i}', f'right_pip_{i}')
                keypoints = copy.deepcopy(adjust_dip_to_mcp(keypoints, f'left_mcp_{i}', f'left_pip_{i}', finger_lengths_left1, confidenceMap, i, True, opposingCache))
            else:
                keypoints = copy.deepcopy(adjust_dip_to_mcp(keypoints, f'left_mcp_{i}', f'left_pip_{i}', finger_lengths_left1, confidenceMap, i))
        confidenceMap[f'left_pip_{i}'] = copy.deepcopy(keypoints[f'left_pip_{i}'])

        if keypoints[f'right_pip_{i}'][2] < threshold:
            if keypoints[f'left_pip_{i}'][2] >= threshold:
                opposingCache = (f'left_mcp_{i}', f'left_pip_{i}')
                keypoints = copy.deepcopy(adjust_dip_to_mcp(keypoints, f'right_mcp_{i}', f'right_pip_{i}', finger_lengths_right1, confidenceMap, i, True, opposingCache))
            else:
                keypoints = copy.deepcopy(adjust_dip_to_mcp(keypoints, f'right_mcp_{i}', f'right_pip_{i}', finger_lengths_right1, confidenceMap, i))
        confidenceMap[f'right_pip_{i}'] = copy.deepcopy(keypoints[f'right_pip_{i}'])

    # Adjust DIP joints
    for i in range(1, 6):
        if keypoints[f'left_dip_{i}'][2] < threshold:
            if keypoints[f'right_dip_{i}'][2] >= threshold:
                opposingCache = (f'right_pip_{i}', f'right_dip_{i}')
                keypoints = copy.deepcopy(adjust_dip_to_mcp(keypoints, f'left_pip_{i}', f'left_dip_{i}', finger_lengths_left2, confidenceMap, i, True, opposingCache))
            else:
                keypoints = copy.deepcopy(adjust_dip_to_mcp(keypoints, f'left_pip_{i}', f'left_dip_{i}', finger_lengths_left2, confidenceMap, i))
        confidenceMap[f'left_dip_{i}'] = copy.deepcopy(keypoints[f'left_dip_{i}'])

        if keypoints[f'right_dip_{i}'][2] < threshold:
            if keypoints[f'left_dip_{i}'][2] >= threshold:
                opposingCache = (f'left_pip_{i}', f'left_dip_{i}')
                keypoints = copy.deepcopy(adjust_dip_to_mcp(keypoints, f'right_pip_{i}', f'right_dip_{i}', finger_lengths_right2, confidenceMap, i, True, opposingCache))
            else:
                keypoints = copy.deepcopy(adjust_dip_to_mcp(keypoints, f'right_pip_{i}', f'right_dip_{i}', finger_lengths_right2, confidenceMap, i))
        confidenceMap[f'right_dip_{i}'] = copy.deepcopy(keypoints[f'right_dip_{i}'])

    # Adjust foot
    if keypoints['left_toe'][2] < threshold:
        if keypoints['right_toe'][2] >= threshold:
            opposingCache = ('right_toe', 'right_ankle', 'right_knee')
            keypoints = copy.deepcopy(adjust_toe_to_ankle(keypoints, 'left_toe', 'left_ankle', 'left_knee', foot_lengths_left, confidenceMap, True, opposingCache))
        else:
            keypoints = copy.deepcopy(adjust_toe_to_ankle(keypoints, 'left_toe', 'left_ankle', 'left_knee', foot_lengths_left, confidenceMap))
    confidenceMap['left_toe'] = copy.deepcopy(keypoints['left_toe'])

    if keypoints['right_toe'][2] < threshold:
        if keypoints['left_toe'][2] >= threshold:
            opposingCache = ('left_toe', 'left_ankle', 'left_knee')
            keypoints = copy.deepcopy(adjust_toe_to_ankle(keypoints, 'right_toe', 'right_ankle', 'right_knee', foot_lengths_right, confidenceMap, True, opposingCache))
        else:
            keypoints = copy.deepcopy(adjust_toe_to_ankle(keypoints, 'right_toe', 'right_ankle', 'right_knee', foot_lengths_right, confidenceMap))
    confidenceMap['right_toe'] = copy.deepcopy(keypoints['right_toe'])
        
    # Adjust small toe
    if keypoints['left_smalltoe'][2] < threshold:
        if keypoints['right_smalltoe'][2] >= threshold:
            opposingCache = ('right_smalltoe', 'right_ankle', 'right_knee')
            keypoints = copy.deepcopy(adjust_toe_to_ankle(keypoints, 'left_smalltoe', 'left_ankle', 'left_knee', foot_lengths_left, confidenceMap, True, opposingCache))
        else:
            keypoints = copy.deepcopy(adjust_toe_to_ankle(keypoints, 'left_smalltoe', 'left_ankle', 'left_knee', foot_lengths_left, confidenceMap))
    confidenceMap['left_smalltoe'] = copy.deepcopy(keypoints['left_smalltoe'])

    if keypoints['right_smalltoe'][2] < threshold:
        if keypoints['left_smalltoe'][2] >= threshold:
            opposingCache = ('left_smalltoe', 'left_ankle', 'left_knee')
            keypoints = copy.deepcopy(adjust_toe_to_ankle(keypoints, 'right_smalltoe', 'right_ankle', 'right_knee', foot_lengths_right, confidenceMap, True, opposingCache))
        else:
            keypoints = copy.deepcopy(adjust_toe_to_ankle(keypoints, 'right_smalltoe', 'right_ankle', 'right_knee', foot_lengths_right, confidenceMap))
    confidenceMap['right_smalltoe'] = copy.deepcopy(keypoints['right_smalltoe'])
    
    # Adjust heels
    if keypoints['left_heel'][2] < threshold:
        if keypoints['right_heel'][2] >= threshold:
            opposingCache = ( 'right_heel', 'right_toe', 'right_ankle')
            keypoints = copy.deepcopy(adjust_heel_to_toe(keypoints, 'left_heel', 'left_toe', 'left_ankle', foot_lengths_left, confidenceMap, True, opposingCache))
        else:
            keypoints = copy.deepcopy(adjust_heel_to_toe(keypoints, 'left_heel', 'left_toe', 'left_ankle', foot_lengths_left, confidenceMap))
    confidenceMap['left_heel'] = copy.deepcopy(keypoints['left_heel'])

    if keypoints['right_heel'][2] < threshold:
        if keypoints['left_heel'][2] >= threshold:
            opposingCache = ('left_heel', 'left_toe', 'left_ankle')
            keypoints = copy.deepcopy(adjust_heel_to_toe(keypoints, 'right_heel', 'right_toe', 'right_ankle', foot_lengths_right, confidenceMap, True, opposingCache))
        else:
            keypoints = copy.deepcopy(adjust_heel_to_toe(keypoints, 'right_heel', 'right_toe', 'right_ankle', foot_lengths_right, confidenceMap))
    confidenceMap['right_heel'] = copy.deepcopy(keypoints['right_heel'])

    # Adjust toes - already handled within adjust_heel_to_toes function
    posePreds = getPosePreds(posePreds, keypoints)
    return copy.deepcopy(posePreds), copy.deepcopy(confidenceMap)


def adjust_ankle_to_knee(keypoints, knee_key, ankle_key, max_lower_leg_length, confidenceMap, mirror=False, opposingCache=None):
    # x_delta = confidenceMap[ankle_key][0] - confidenceMap[knee_key][0]
    # y_delta = confidenceMap[ankle_key][1] - confidenceMap[knee_key][1]
    # keypoints[ankle_key] = (keypoints[knee_key][0] + x_delta, keypoints[knee_key][1] + y_delta, keypoints[ankle_key][2])
    if mirror is True:
        opp_knee, opp_ankle = opposingCache
        x_delta = keypoints[opp_ankle][0] - keypoints[opp_knee][0]
        y_delta = keypoints[opp_ankle][1] - keypoints[opp_knee][1]
        keypoints[ankle_key] = (keypoints[knee_key][0] + x_delta, keypoints[knee_key][1] + y_delta, keypoints[ankle_key][2])
    else:
        v = keypoints[ankle_key] - keypoints[knee_key]
        v_star = np.linalg.norm(v)
        if v_star == 0:
            keypoints[ankle_key] = confidenceMap[ankle_key]
            return keypoints
        assert v_star > 0
        v = v / v_star
        assert max_lower_leg_length is not None
        keypoints[ankle_key] = keypoints[knee_key] + v * max_lower_leg_length
    return keypoints

def adjust_toe_to_ankle(keypoints, toe_key, ankle_key, knee_key, max_foot_length, confidenceMap, mirror = False, opposingCache=None):
    # x_delta = confidenceMap[toe_key][0] - confidenceMap[ankle_key][0]
    # y_delta = confidenceMap[toe_key][1] - confidenceMap[ankle_key][1]
    # keypoints[toe_key] = (keypoints[ankle_key][0] + x_delta, keypoints[ankle_key][1] + y_delta, keypoints[toe_key][2])
    if mirror is True:
        #mirror opposite end
        opp_toe, opp_ankle, opp_knee = opposingCache
        x_delta = keypoints[opp_toe][0] - keypoints[opp_ankle][0]
        y_delta = keypoints[opp_toe][1] - keypoints[opp_ankle][1]
        keypoints[toe_key] = (keypoints[ankle_key][0] + x_delta, keypoints[ankle_key][1] + y_delta, keypoints[toe_key][2])
    else:
        v = keypoints[ankle_key] - keypoints[knee_key]
        v_star = np.linalg.norm(v)
        if v_star == 0:
            keypoints[toe_key] = confidenceMap[toe_key]
            return keypoints
        assert v_star > 0
        v = v / v_star
        keypoints[toe_key] = keypoints[ankle_key] + v * 0.5 * max_foot_length
        assert v is not None
        assert max_foot_length is not None
    return keypoints

def adjust_heel_to_toe(keypoints, heel_key, toe_key, ankle_key, max_foot_length, confidenceMap, mirror=False, opposingCache=None):
    # x_delta = confidenceMap[heel_key][0] - confidenceMap[toe_key][0]
    # y_delta = confidenceMap[heel_key][1] - confidenceMap[toe_key][1]
    # keypoints[heel_key] = (keypoints[toe_key][0] + x_delta, keypoints[toe_key][1] + y_delta, keypoints[heel_key][2])
    #mirror opposite end
    if mirror is True:
        opp_heel, opp_toe, opp_ankle = opposingCache
        x_delta = keypoints[opp_heel][0] - keypoints[opp_toe][0]
        y_delta = keypoints[opp_heel][1] - keypoints[opp_toe][1]
        keypoints[heel_key] = (keypoints[toe_key][0] + x_delta, keypoints[toe_key][1] + y_delta, keypoints[heel_key][2])
    else:
        v = keypoints[toe_key] - keypoints[heel_key]
        v_star = np.linalg.norm(v)
        if v_star == 0:
            keypoints[heel_key] = confidenceMap[heel_key]
            return keypoints
        assert v_star > 0
        v = v / v_star
        keypoints[heel_key] = keypoints[toe_key] - v * max_foot_length
        assert v is not None
        assert max_foot_length is not None 
    return keypoints

def adjust_hip_to_knee(keypoints, hip_key, knee_key, upper_leg_length, confidenceMap, mirror=False, opposingCache=None):

    return keypoints

def adjust_wrist_to_elbow(keypoints, wrist_key, elbow_key, max_forearm_length, confidenceMap, mirror=False, opposingCache=None):
    # x_delta = confidenceMap[wrist_key][0] - confidenceMap[elbow_key][0]
    # y_delta = confidenceMap[wrist_key][1] - confidenceMap[elbow_key][1]
    # keypoints[wrist_key] = (keypoints[elbow_key][0] + x_delta, keypoints[elbow_key][1] + y_delta, keypoints[wrist_key][2])
    if mirror is True:
        opp_wrist, opp_elbow = opposingCache
        x_delta = keypoints[opp_wrist][0] - keypoints[opp_elbow][0]
        y_delta = keypoints[opp_wrist][1] - keypoints[opp_elbow][1]
        keypoints[wrist_key] = (keypoints[elbow_key][0] + x_delta, keypoints[elbow_key][1] + y_delta, keypoints[wrist_key][2])
    else:
        v = keypoints[elbow_key] - keypoints[wrist_key]
        v_star = np.linalg.norm(v)
        if v_star == 0:
            keypoints[wrist_key] = confidenceMap[wrist_key]
            return keypoints
        assert v_star > 0
        v = v / v_star
        keypoints[wrist_key] = keypoints[elbow_key] + v * max_forearm_length
        assert v is not None
        assert max_forearm_length is not None
    return keypoints

# Replace the following adjust_* functions with your actual helper functions that apply the biomechanics constraints
def adjust_mcp_to_wrist(keypoints, wrist_key, mcp_key, shoulder_key, elbow_key, max_hand_length, max_forearm_length, confidenceMap, mcp_index=1, mirror=False, opposingCache=None):
    # x_delta = confidenceMap[mcp_key][0] - confidenceMap[wrist_key][0]
    # y_delta = confidenceMap[mcp_key][1] - confidenceMap[wrist_key][1]
    # keypoints[mcp_key] = (keypoints[wrist_key][0] + x_delta, keypoints[wrist_key][1] + y_delta, keypoints[mcp_key][2])
    if mirror is True:
        opp_wrist, opp_mcp_i = opposingCache
        x_delta = keypoints[opp_mcp_i][0] - keypoints[opp_wrist][0]
        y_delta = keypoints[opp_mcp_i][1] - keypoints[opp_wrist][1]
        keypoints[mcp_key] = (keypoints[wrist_key][0] + x_delta, keypoints[wrist_key][1] + y_delta, keypoints[mcp_key][2])
        
#         wrist_pt = keypoints[wrist_key]
#         mcp_pt = keypoints[mcp_key]
#         wrist_mcp_vec = np.array(mcp_pt) - np.array(wrist_pt)
#         wrist_mcp_vec /= np.linalg.norm(wrist_mcp_vec)
#         opp_wrist_pt = keypoints[opp_wrist]
#         opp_mcp_pt = keypoints[opp_mcp_i]
#         opp_wrist_mcp_vec = np.array(opp_mcp_pt) - np.array(opp_wrist_pt)
#         opp_wrist_mcp_vec /= np.linalg.norm(opp_wrist_mcp_vec)
        
#         mcp_dir_vec = wrist_mcp_vec + opp_wrist_mcp_vec
#         mcp_dir_vec /= np.linalg.norm(mcp_dir_vec)
#         x_delta = keypoints[opp_mcp_i][0] - keypoints[opp_wrist][0]
#         y_delta = keypoints[opp_mcp_i][1] - keypoints[opp_wrist][1]
#         z_delta = keypoints[opp_mcp_i][2] - keypoints[opp_wrist][2]
#         keypoints[mcp_key] =  keypoints[wrist_key] + mcp_dir_vec * [x_delta, y_delta, z_delta]
#         print(f"Keypoints mcp is {keypoints[mcp_key]}")
    else:
        v = keypoints[wrist_key] - keypoints[elbow_key] #change all this back to keypoints!
        v_star = np.linalg.norm(v)
        if v_star == 0:
            keypoints[mcp_key] = confidenceMap[mcp_key]
            return keypoints
        assert v_star > 0
        v = v / v_star
        keypoints[mcp_key] = keypoints[wrist_key] + v * max_hand_length

        assert v is not None
        assert max_hand_length is not None
    return keypoints

def adjust_dip_to_mcp(keypoints, mcp_key, dip_key, max_finger_length, confidenceMap, dip_index=1, mirror=False, opposingCache=None):
    # x_delta = confidenceMap[dip_key][0] - confidenceMap[mcp_key][0]
    # y_delta = confidenceMap[dip_key][1] - confidenceMap[mcp_key][1]
    # keypoints[dip_key] = (keypoints[mcp_key][0] + x_delta, keypoints[mcp_key][1] + y_delta, keypoints[dip_key][2])
    if mirror is True:
        opp_mcp_i, opp_dip_i= opposingCache
        x_delta = keypoints[opp_dip_i][0] - keypoints[opp_mcp_i][0]
        y_delta = keypoints[opp_dip_i][1] - keypoints[opp_mcp_i][1]
        keypoints[dip_key] = (keypoints[mcp_key][0] + x_delta, keypoints[mcp_key][1] + y_delta, keypoints[dip_key][2])
    else:
        v = keypoints[mcp_key] - keypoints[dip_key]
        v_star = np.linalg.norm(v)
        if v_star == 0:
            keypoints[dip_key] = confidenceMap[dip_key]
            return keypoints
        assert v_star > 0
        v = v / v_star
        keypoints[dip_key] = keypoints[mcp_key] + v * max_finger_length

        assert v is not None
        assert max_finger_length is not None
    return keypoints


def interpolate_keypoints(emptyFrames, empty_f_i, preds, prevPreds, preds_all):
    nextPreds = preds
    count = len(emptyFrames)
    t = np.linspace(0, 1, count + 2)
    t = t[1:-1]
    splines = []
    for dim in range(3):
        spline = CubicSpline([0, 1], [prevPreds[0]['keypoints'][:,dim], nextPreds[0]['keypoints'][:,dim]])
        splines.append(spline) 
    for i in range(len(emptyFrames)):
        frame = emptyFrames[i]
        fi = empty_f_i[i]
        print(f"Processing a prev empty frame: {fi}" )
        interpolate_preds_base = copy.deepcopy(prevPreds)
        assert len(interpolate_preds_base) == 1
        interpolate_preds_base[0]['keypoints'] = np.vstack([spline(t[i]) for spline in splines]).T 
        preds_all.append(copy.deepcopy(interpolate_preds_base))
    return copy.deepcopy(interpolate_preds_base), copy.deepcopy(preds_all)

def rotate(point, origin, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.
    """
    ox, oy, conf = origin
    px, py, _ = point

    qx = ox + np.cos(angle) * (px - ox) - np.sin(angle) * (py - oy)
    qy = oy + np.sin(angle) * (px - ox) + np.cos(angle) * (py - oy)

    return qx, qy, conf

import json, os
def importToJSONPost(pose_preds, video_path, offset):
    data = {}
    _, tail = os.path.split(video_path)
    data["offset"] = offset
    for i, frame in enumerate(pose_preds):
        keypoints = {}
        for j, keypoint in enumerate(frame):
            keypoint_data = {
                "x": float(keypoint[0]),
                "y": float(keypoint[1]),
                "confidence_score": float(keypoint[2])
            }
            keypoints[f"keypoint_{j+1}"] = keypoint_data
        data[f"frame_{i+1}"] = keypoints

    with open(f"labels/{tail[:len(tail)-4]}.json", "w") as outfile:
        json.dump(data, outfile)



def area_difference(bbox1, bbox2):
    # Calculate the area of bbox1
    area_bbox1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])

    # Calculate the area of bbox2
    area_bbox2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])

    # Calculate the absolute difference between the areas
    area_diff = abs(area_bbox1 - area_bbox2)

    return area_diff



def parseJSON(jsonPath):
    with open(jsonPath, "r") as infile:
        data = json.load(infile)

    frameCoords = []
    for frame in data:
        if frame == "offset":
            offset = data[frame]
            continue
        keypoints = []
        for keypoint in data[frame]:
            coords = (data[frame][keypoint]['x'], data[frame][keypoint]['y'], data[frame][keypoint]['confidence_score'])
            keypoints.append(coords)
        frameCoords.append(keypoints)

    return np.array(frameCoords), offset


def apply_smoothing_filter(filters, preds, joint_idx):
    return filters[joint_idx](preds)
    # smoothed_coords = []
    # print(f"Coords is {coords}")
    # for i, coord in enumerate(coords):
    #     print(i,coord)
    #     smoothed_joint = [filters[i][j](c) for j, c in enumerate(coord)]
    #     smoothed_coords.append(smoothed_joint)
    # return smoothed_coords



class PoseSmoother:
    def __init__(self, window_size):
        self.window_size = window_size
        self.buffer = []

    def smooth(self, joint_predictions):
        self.buffer.append(joint_predictions)

        # Remove the oldest frame if buffer size exceeds window size
        if len(self.buffer) > self.window_size:
            self.buffer.pop(0)

        # Calculate the moving average
        smoothed_joint_predictions = np.mean(self.buffer, axis=0)

        return copy.deepcopy(smoothed_joint_predictions)
    