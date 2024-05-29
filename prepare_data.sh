#!/usr/bin/env bash
unzip weights_download.zip
mkdir pipeline/environment/FGT/checkpoint
mkdir pipeline/environment/FGT/flowCheckPoint
mkdir pipeline/environment/LAFC/checkpoint
mkdir pipeline/track/data
mv weights_download/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl pipeline/track/data
mv weights_download/dw-ll_ucoco_384.onnx pipeline/context/dwpose
mv weights_download/sam_vit_h_4b8939.pth pipeline/track/phalp/trackers
mv weights_download/yolov8l-seg.pt pipeline/track/phalp/trackers
mv weights_download/dinov2_zero_vector.pt model/backbones
mv weights_download/dinov2_vit_base.pth model/backbones
mv weights_download/dinov2_vit_large.pth model/backbones
mv weights_download/fgt/* pipeline/environment/FGT/checkpoint
mv weights_download/lafc/* pipeline/environment/LAFC/checkpoint
mv weights_download/lafc_single/* pipeline/environment/FGT/flowCheckPoint
rm -r weights_download
rm weights_download.zip

