# 3D Reconstruction from two views using OpenCV and LoFTR
Implementation of a full two-view 3D reconstruction pipeline using OpenCV and LoFTR feature matching network.

## Overview
This project includes the full pipeline for reconstructing 3D scene from images captured by the same camera in 2 different viewpoints. The pipeline includes 4 stages: 
1. Camera calibration
2. Feature matching
3. Pose recovery
4. Triangulation (3D Reconstruction)

## Calibration
The camera is calibrated by taking multiple images of checkerboard pattern.

The calibration process includes:
- Detecting corners
- Estimate camera parameters using OpenCV calibration methods
- Computing the intrinsic matrix $K$
The instrinsic matrix $K$ is later used for Pose Recovery

## Matching features
Feature correspondances between two images is obtain using LoFTR (Local Feature Transformer). Unlike conventional mehthods (e.g. SIFT, ORB) which rely on detecting keypoints and description before matching, LoFTR skips explicit keypoint detection, learn to match pixels or regions directly using deep features and transformers. 

The output of this stage is two set of matched point pairs $(x_1, y_1) - (x_2, y_2)$, which is later used for Pose Recovery

## Pose recovery
From intrinsic matrix K and correspondances pairs $(x_1, y_1) - (x_2, y_2)$
The following information is obtained:
- Essential Matrix $E$
- Rotational Matrix $R$
- Translation Vector $t$

After that, two camera matrices can be formulated as: 

$$
P_1 = K[I \mid 0]
$$

$$
P_2 = K[R \mid t]
$$

## Triangulation
Once the camera pose is recovered, the matched feature points are triangulated to reconstruct the corresponding 3D coordinates.

## Dependencies
The project uses the following libraries:
- Python 3.9+
- OpenCV
- Numpy
- Pytorch
- Kornia (for LoFTR)
- Matplotlib




