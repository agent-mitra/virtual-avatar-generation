#!/usr/bin/env bash
unzip weights_download.zip
mkdir pipeline/environment/FGT/checkpoint
mkdir pipeline/environment/FGT/flowCheckPoint
mkdir pipeline/environment/LAFC/checkpoint
mkdir pipeline/track/data
mv weights/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl pipeline/track/data
mv weights/dw-ll_ucoco_384.onnx pipeline/context/dwpose
mv weights/sam_vit_h_4b8939.pth pipeline/track/phalp/trackers
mv weights/yolov8l-seg.pt pipeline/track/phalp/trackers
mv weights/dinov2_zero_vector.pt model/backbones
mv weights/dinov2_vit_base.pth model/backbones
mv weights/dinov2_vit_large.pth model/backbones
mv weights/fgt/* pipeline/environment/FGT/checkpoint
mv weights/lafc/* pipeline/environment/LAFC/checkpoint
mv weights/lafc_single/* pipeline/environment/FGT/flowCheckPoint
rm -r weights
rm weights_download.zip

