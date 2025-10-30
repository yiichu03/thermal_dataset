import os
import numpy as np
import cv2
import open3d as o3d
import shutil 
import argparse

import glob
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R, Slerp

import torch

def read_poses(file):
    poses = []
    with open(file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split()  
                poses.append([float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]),
                             float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])]) 
    return np.array(poses)

def linear_interpol(pose_data, time):
    times = pose_data[:, 0]
    poses = pose_data[:, 1:]

    interpolated_translation = np.array([np.interp(time, times, poses[:, i]) for i in range(3)])
    quaternions = R.from_quat(poses[:, 3:7])
    if time <= times[0]:
        interpolated_rotation = quaternions[0]
    elif time >= times[-1]:
        interpolated_rotation = quaternions[-1]
    else:
        slerp = Slerp(times, quaternions)
        interpolated_rotation = slerp(time)

    interpolated_quat = interpolated_rotation.as_quat()  # qx qy qz qw

    return np.array([*interpolated_translation, *interpolated_quat])

dataset_path = "/mnt/Data/homebrew_thermal/Nus"

scenes = [
    "run_20250902_161637", 
    "run_20250902_162619",
    "run_20250902_163806", 
    "run_20250902_164805",
    "run_20250902_170456", 
    "run_20250902_172254", 
    "run_20250910_152028",
    "run_20250910_153427",
    "run_20250911_200532",
    "run_20250911_202017",
    "run_20250911_203739",
    "run_20250912_180027",
    "run_20250912_181939",
    "run_20250912_180927",
]


device='cuda:1'

K_l = np.array([[344.08059125, 0.0, 320.20695996], 
                        [0.0, 343.71871889, 271.73937497], 
                        [0.0, 0.0, 1.0]], dtype=np.float64)    
   
K_r = np.array([[342.69763215, 0.0, 323.56293972], 
            [0.0, 342.45604858, 267.49213891], 
            [0.0, 0.0, 1.0]], dtype=np.float64)

D_l = np.array([-0.21823207, 0.04657087, 0.0, 0.0, 0.0], dtype=np.float64) 

D_r = np.array([-0.22042839, 0.04895111, 0.0, 0.0, 0.0], dtype=np.float64)


K_rgbd = np.array([[386.4121093750, 0.0, 321.5436096191], 
            [0.0, 385.8505249023, 243.3150024414], 
            [0.0, 0.0, 1.0]], dtype=np.float64)

D_rgbd = np.array([-0.0550349094, 0.0627269447, -0.0008203980, 0.0003794413, -0.0196892563], dtype=np.float64)

# Left to Right Thermal Cam
R_thl2thr = np.array([[0.9998928479139975, 0.006664201062753905, 0.013033844967273217],
            [-0.0065988424903717065, 0.999965470127153, -0.005051121773369857],
            [-0.01306705660135398, 0.004964572245152228, 0.9999022977542356]], dtype=np.float64) 

# Left to Right Thermal Cam
t_thl2thr = np.array([[-0.1216520307054718], [0.00037876701143810795], [-0.0015966275775746094]], dtype=np.float64)

# Left to RGBD
R_thl2rgbd = np.array([[0.999979691664164, 0.006359021898875782, -0.0004232017641123605],
        [-0.006357600546184011, 0.9999744135762715, 0.0032791932067709082],
        [0.00044404339730520217, -0.003276436064047327, 0.9999945338811506]], dtype=np.float64) 
# Left to RGBD
t_thl2rgbd = np.array([[-0.07309151021484832], [0.06182557268473313], [-0.02263297651799525]], dtype=np.float64) 

R_rgbd2lidar = R.from_quat([-0.0025930314846405633, 0.7233645800633655, -0.6904201829671065, -0.007545293177736468]).as_matrix()

t_rgbd2lidar = np.array([[-0.026], [-0.07986985425426218], [-0.07194136971184459]], dtype=np.float64)

# R_rgbd2thl = R_thl2rgbd.T
# t_rgbd2thl = -R_rgbd2thl @ t_thl2rgbd

R_thl2lidar = R_rgbd2lidar @ R_thl2rgbd   
t_thl2lidar = R_rgbd2lidar @ t_thl2rgbd + t_rgbd2lidar

R_thl2lidar_torch = torch.tensor(R_thl2lidar, dtype=torch.float64, device=device)
t_thl2lidar_torch = torch.tensor(t_thl2lidar, dtype=torch.float64, device=device)

h, w = 512, 640

R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(K_l, D_l, K_r, D_r, (w, h), 
                            R_thl2thr, t_thl2thr, flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)

map1L, map2L = cv2.initUndistortRectifyMap(K_l, D_l, R1, P1, (w, h), cv2.CV_32FC1)
map1R, map2R = cv2.initUndistortRectifyMap(K_r, D_r, R2, P2, (w, h), cv2.CV_32FC1)

# new_K_rgbd, _ = cv2.getOptimalNewCameraMatrix(K_rgbd, D_rgbd, (w, ), alpha=0, newImgSize=(w, h))

fx_thl, fy_thl = P1[0, 0], P1[1, 1]
cx_thl, cy_thl = P1[0, 2], P1[1, 2]

print(f"[INFO] New bf(baseline * focal_x): {-P2[0, 3]}")

for scene in scenes:
    scene_path = f"{dataset_path}/{scene}"
    print(f"[INFO] Processing {scene_path}")
    images_left = sorted(glob.glob(os.path.join(scene_path, 'left_thermal/left_motion/*.png')))
    images_right = sorted(glob.glob(os.path.join(scene_path, 'right_thermal/right_motion/*.png')))
    left_rectified_dir = os.path.join(scene_path, "left_thermal/rectified_left_lidar")
    right_rectified_dir = os.path.join(scene_path, "right_thermal/rectified_right_lidar")
    depth_reproject_dir = os.path.join(scene_path, "realsense/depth_reproject_lidar")
    depth_reproject_vis_dir = os.path.join(scene_path, "realsense/depth_reproject_lidar_vis")
    os.makedirs(left_rectified_dir, exist_ok=True)
    os.makedirs(right_rectified_dir, exist_ok=True)
    os.makedirs(depth_reproject_dir, exist_ok=True)
    os.makedirs(depth_reproject_vis_dir, exist_ok=True)

    poses_lidar = read_poses(os.path.join(scene_path, "rosbag/fastlio2/scan_states_odom.txt"))

    pcd = o3d.io.read_point_cloud(os.path.join(scene_path, "rosbag/fastlio2/scans.pcd"))
    points_w = np.asarray(pcd.points, dtype=np.float64)
    print(f"[PCD] N={points_w.shape[0]}")

    for left, right in tqdm(zip(images_left, images_right), total=len(images_left)):
        left_img = cv2.imread(left, cv2.IMREAD_UNCHANGED)
        right_img = cv2.imread(right, cv2.IMREAD_UNCHANGED)
        timestamp = os.path.splitext(os.path.basename(left))[0]
        pose_lidar = linear_interpol(poses_lidar, float(timestamp))

        R_lidar2w = torch.tensor(R.from_quat([pose_lidar[3], pose_lidar[4], pose_lidar[5], pose_lidar[6]]).as_matrix(),
                                dtype=torch.float64, device=device)
        t_lidar2w = torch.tensor(pose_lidar[0:3].reshape(3,1), dtype=torch.float64, device=device)
        
        points_w = torch.tensor(np.asarray(pcd.points), dtype=torch.float64, device=device)

        R_thl2w = R_lidar2w @ R_thl2lidar_torch
        t_thl2w = R_lidar2w @ t_thl2lidar_torch + t_lidar2w

        points_thl = (R_thl2w.T @ (points_w.T - t_thl2w)).T

        Z = points_thl[:, 2]
        mask_front = (Z > 1e-6) & torch.isfinite(Z)
        points_thl = points_thl[mask_front]
        Z = points_thl[:, 2]

        u = fx_thl * (points_thl[:, 0] / Z) + cx_thl
        v = fy_thl * (points_thl[:, 1] / Z) + cy_thl

        mask_inside = (u >= 0) & (u < w) & (v >= 0) & (v < h) & torch.isfinite(u) & torch.isfinite(v)
        u = u[mask_inside].long()
        v = v[mask_inside].long()
        z = Z[mask_inside].float()

        Depth_thl = torch.full((h, w), float('inf'), dtype=torch.float32, device=device)
        idx = v * w + u

        Depth_flat = Depth_thl.view(-1)
        Depth_flat.scatter_reduce_(0, idx, z, reduce='amin')

        Depth_thl = Depth_flat.view(h, w)
        Depth_thl[~torch.isfinite(Depth_thl)] = 0.0

        depth_reproject = Depth_thl.cpu().numpy()
        depth_uint8 = cv2.normalize(depth_reproject, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        base_l = os.path.basename(left)
        base_r = os.path.basename(right)

        rectified_left = cv2.remap(left_img, map1L, map2L, cv2.INTER_NEAREST)
        rectified_right = cv2.remap(right_img, map1R, map2R, cv2.INTER_NEAREST)

        cv2.imwrite(f"{left_rectified_dir}/{base_l}", rectified_left)
        cv2.imwrite(f"{right_rectified_dir}/{base_r}", rectified_right)
        np.save(f"{depth_reproject_dir}/{timestamp}.npy", depth_reproject.astype(np.float32))
        cv2.imwrite(f"{depth_reproject_vis_dir}/{base_l}", depth_uint8)
