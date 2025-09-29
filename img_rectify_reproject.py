
import cv2
import numpy as np
import glob
import os
import torch
from tqdm import tqdm
import pyrealsense2 as rs

dataset_path = "path/to/Nus_thermal_dataset"

scenes = [
    "run_20250902_161637", 
    # "run_20250902_162619",
    # "run_20250902_163806", 
    # "run_20250902_164805",
    # "run_20250902_170456", 
    # "run_20250902_172254", 
    # "run_20250910_152028",
    # "run_20250910_153427",
    # "run_20250911_200532",
    # "run_20250911_202017",
    # "run_20250911_203739",
    # "run_20250912_180027",
    # "run_20250912_181939",
    # "run_20250912_180927",
]


device='cuda:0'


def invert_brown_conrady(xd, yd, D, iters=8, eps=1e-9):
    k1, k2, p1, p2, k3 = [float(v) for v in D]
    xu = xd.clone()
    yu = yd.clone()
    for _ in range(iters):
        r2 = xu*xu + yu*yu
        r4 = r2*r2
        r6 = r4*r2
        radial = 1.0 + k1*r2 + k2*r4 + k3*r6
        x_tang = 2.0*p1*xu*yu + p2*(r2 + 2.0*xu*xu)
        y_tang = p1*(r2 + 2.0*yu*yu) + 2.0*p2*xu*yu
        xd_hat = xu*radial + x_tang
        yd_hat = yu*radial + y_tang
        ex = xd_hat - xd
        ey = yd_hat - yd
        xu = xu - ex
        yu = yu - ey
        if (ex.abs().max() < eps) and (ey.abs().max() < eps):
            break
    return xu, yu

def depth_to_points_inverse_bc(depth, K, D, min_z=0.6, max_z=6.0, iters=8):
    device = depth.device
    H, W = depth.shape
    fx, fy = K[0,0], K[1,1]
    cx, cy = K[0,2], K[1,2]

    u = torch.arange(W, device=device, dtype=torch.float32)
    v = torch.arange(H, device=device, dtype=torch.float32)
    uu, vv = torch.meshgrid(u, v, indexing='xy')

    xd = (uu - cx) / fx
    yd = (vv - cy) / fy

    m = (depth > min_z) & (depth < max_z) & torch.isfinite(depth)

    xd_v = xd[m]
    yd_v = yd[m]
    z = depth[m]

    xu, yu = invert_brown_conrady(xd_v, yd_v, D, iters=iters)

    X = xu * z
    Y = yu * z
    Z = z
    pts = torch.stack([X, Y, Z], dim=1)

    return pts
   
def depth_reproject(img_depth, R, t, K_d, D_d, K_l, D_l, min_threshold=0.6, max_threshold=6.0):
    Hl, Wl = 512, 640
    if img_depth is None:
        print(f"[WARN] None type")
        return None

    depth_img = torch.from_numpy(img_depth).to(device).float() * 0.001 # m

    pts3d_d = depth_to_points_inverse_bc(depth_img, K_d, D_d)

    R_ts = torch.from_numpy(R).to(device).float()
    t_ts = torch.from_numpy(t).to(device).float()
    pts3d_t = (R_ts @ pts3d_d.T + t_ts).T

    if pts3d_t is None or pts3d_t.size == 0:
        return None

    rvec = np.zeros((3,1), dtype=np.float64) 
    tvec = np.zeros((3,1), dtype=np.float64)

    pts3d_t_np = pts3d_t.cpu().numpy().astype(np.float64) 

    imgpts, _ = cv2.projectPoints(pts3d_t_np.reshape(-1,1,3), rvec, tvec, K_l, D_l)
    imgpts = imgpts.reshape(-1,2)

    proj_u = torch.from_numpy(imgpts[:,0].astype(np.float32)).to(device)
    proj_v = torch.from_numpy(imgpts[:,1].astype(np.float32)).to(device)
    Zl = torch.from_numpy(pts3d_t_np[:,2].astype(np.float32)).to(device)

    u = torch.round(proj_u).long()
    v = torch.round(proj_v).long()

    mask_valid = (u >= 0) & (u < Wl) & (v >= 0) & (v < Hl)
    u, v, Zl = u[mask_valid], v[mask_valid], Zl[mask_valid]

    depth_thermal_np = np.full((Hl, Wl), np.inf, dtype=np.float32)
    depth_thermal = torch.from_numpy(depth_thermal_np).to(device)

    idx = v * Wl + u
    flat_depth = depth_thermal.view(-1)
    flat_depth = flat_depth.scatter_reduce(0, idx, Zl, reduce="amin", include_self=True)

    depth_thermal = flat_depth.view(Hl, Wl)
    depth_thermal[depth_thermal == float('inf')] = 0.0

    depth_th = depth_thermal.cpu().numpy().astype(np.float32) 

    return depth_th

def depth_to_disp(depth, bf):
    depth_ts = torch.from_numpy(depth).float() 
    disp = torch.zeros_like(depth_ts)
    valid_depth_mask = depth_ts > 0
    disp[valid_depth_mask] = bf / depth_ts[valid_depth_mask]
    disp_th_float32 = disp.cpu().numpy().astype(np.float32)
    return disp_th_float32

K_l = np.array([[344.08059125, 0.0, 320.20695996], 
                        [0.0, 343.71871889, 271.73937497], 
                        [0.0, 0.0, 1.0]], dtype=np.float64)
K_r = np.array([[342.69763215, 0.0, 323.56293972], 
            [0.0, 342.45604858, 267.49213891], 
            [0.0, 0.0, 1.0]], dtype=np.float64)
K_rgb = np.array([[387.043, 0.0, 321.544],
        [0.0, 386.48, 243.315],
        [0.0, 0.0, 1.0]])   
K_rgb_m = np.array([[387.22243591, 0.0, 323.99310727],
        [0.0, 387.07049416, 244.82031200],
        [0.0, 0.0, 1.0]])  

D_l = np.array([-0.21823207, 0.04657087, 0.0, 0.0, 0.0], dtype=np.float64) 
D_r = np.array([-0.22042839, 0.04895111, 0.0, 0.0, 0.0], dtype=np.float64)
D_rgb = np.array([-0.0550349, 0.0627269, -0.000820398, 0.000379441, -0.0196893], dtype=np.float64)
D_rgb_m = np.array([-0.05085226, 0.04278795, 0, 0, 0], dtype=np.float64)
 
# Left to Right Thermal Cam
R_l2r = np.array([[0.9998928479139975, 0.006664201062753905, 0.013033844967273217],
            [-0.0065988424903717065, 0.999965470127153, -0.005051121773369857],
            [-0.01306705660135398, 0.004964572245152228, 0.9999022977542356]], dtype=np.float64) 

# Left to Right Thermal Cam
t_l2r = np.array([[-0.1216520307054718], [0.00037876701143810795], [-0.0015966275775746094]], dtype=np.float64)

# Left to RGBD
R_l2d = np.array([[0.999979691664164, 0.006359021898875782, -0.0004232017641123605],
        [-0.006357600546184011, 0.9999744135762715, 0.0032791932067709082],
        [0.00044404339730520217, -0.003276436064047327, 0.9999945338811506]], dtype=np.float64) 
# Left to RGBD
t_l2d = np.array([[-0.07309151021484832], [0.06182557268473313], [-0.02263297651799525]], dtype=np.float64) 

R_d2l = R_l2d.T
t_d2l = -R_d2l @ t_l2d

h, w = 512, 640

R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(K_l, D_l, K_r, D_r, (w, h), 
                            R_l2r, t_l2r, flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)

map1L, map2L = cv2.initUndistortRectifyMap(K_l, D_l, R1, P1, (w, h), cv2.CV_32FC1)
map1R, map2R = cv2.initUndistortRectifyMap(K_r, D_r, R2, P2, (w, h), cv2.CV_32FC1)

K_l_new = P1[:3, :3]
K_r_new = P2[:3, :3]

bf = 36.807810080911636 * 1000.0 # baseline(mm) * focal_x

print(K_l_new)
print(K_r_new)
print(f"[INFO] New bf(baseline * focal_x): {-P2[0, 3]}")

for scene in scenes:
    scene_path = f"{dataset_path}/{scene}"
    print(f"[INFO] Processing {scene_path}")
    images_left = sorted(glob.glob(os.path.join(scene_path, 'left_thermal/left_associated_depth/*.png')))
    images_right = sorted(glob.glob(os.path.join(scene_path, 'right_thermal/right_associated_depth/*.png')))
    image_depth = sorted(glob.glob(os.path.join(scene_path, 'realsense/depth_associated/*.png')))
    left_rectified_dir = os.path.join(scene_path, "left_thermal/rectified_left")
    right_rectified_dir = os.path.join(scene_path, "right_thermal/rectified_right")
    depth_rectified_dir = os.path.join(scene_path, "realsense/rectified_depth")
    # disp_rectified_dir = os.path.join(scene_path, "realsense/rectified_disp")
    os.makedirs(left_rectified_dir, exist_ok=True)
    os.makedirs(right_rectified_dir, exist_ok=True)
    os.makedirs(depth_rectified_dir, exist_ok=True)
    # os.makedirs(disp_rectified_dir, exist_ok=True)

    for left, right, depth in tqdm(zip(images_left, images_right, image_depth), total=len(images_left)):
        left_img = cv2.imread(left, cv2.IMREAD_UNCHANGED)
        right_img = cv2.imread(right, cv2.IMREAD_UNCHANGED)
        depth_img = cv2.imread(depth, cv2.IMREAD_UNCHANGED)

        base_l = os.path.basename(left)
        base_r = os.path.basename(right)
        base_d = os.path.basename(depth)

        depth_img_reproject = depth_reproject(depth_img, R_d2l, t_d2l, K_rgb_m, D_rgb_m, K_l, D_l)
        if depth_img is None:
            continue

        rectified_left = cv2.remap(left_img, map1L, map2L, cv2.INTER_NEAREST) # Attention: do not use cv2.INTER_LINEAR
        rectified_depth = cv2.remap(depth_img_reproject, map1L, map2L, cv2.INTER_NEAREST)
        rectified_right = cv2.remap(right_img, map1R, map2R, cv2.INTER_NEAREST)

        # rectified_disp = depth_to_disp(rectified_depth, bf)

        cv2.imwrite(f"{left_rectified_dir}/{base_l}", rectified_left)      
        cv2.imwrite(f"{right_rectified_dir}/{base_r}", rectified_right)
        cv2.imwrite(f"{depth_rectified_dir}/{base_d}", (rectified_depth * 1000.0).astype(np.uint16))
        # np.save(f"{disp_rectified_dir}/{os.path.splitext(base_d)[0]}.npy", rectified_disp)

    