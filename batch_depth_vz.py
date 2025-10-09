# batch_depth_color_only.py
import os
import glob
import numpy as np
import cv2
import open3d as o3d

# 依赖：与本脚本同目录的 Lidarpoints_to_rgb.py（含标定与函数）
import lidarpoints_to_rgb as lp

# ========== 配置 ==========
RGB_DIR   = r"D:/run_20250912_180027/realsense/rgb"   # 输入 RGB 文件夹
PCD_PATH  = r"D:/run_20250912_180027/rosbag/fastlio2/scans.pcd"
TRAJ_PATH = r"D:/run_20250912_180027/rosbag/fastlio2/scan_states_odom.txt"
OUT_DIR   = r"D:/run_20250912_180027/output"          # 单一输出文件夹（自动创建）

VIEW_MODE   = "Tl"          # "Tl" = Tl视角伪彩（默认），"C" = RGB视角伪彩
PCTS        = (2, 98)       # 伪彩拉伸分位数
FIXED_RANGE = None          # (zmin,zmax) 固定范围(米)；None=自适应
NEAR_IS_RED = True          # 近处红色

# Tl 内参（与 Lidarpoints_to_rgb.py 保持一致）
TL_W, TL_H = 640, 512
K_tl = np.array([[302.53882139, 0.0,         305.2902832 ],
                 [0.0,          302.53882139,273.44169617],
                 [0.0,          0.0,         1.0        ]], dtype=np.float64)
R1 = np.array([[ 0.99965096,  0.00361561,  0.02617045],
               [-0.00355090,  0.99999052, -0.00251879],
               [-0.02617931,  0.00242499,  0.99965432]], dtype=np.float64)
# ========== 工具 ==========
def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def list_rgb_images(folder):
    exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp")
    files = []
    for ext in exts:
        files += glob.glob(os.path.join(folder, ext))
    files.sort()
    return files

def build_camera_maps_and_newK():
    K = np.array([[lp.fx, 0, lp.cx],
                  [0, lp.fy, lp.cy],
                  [0,    0,    1]], dtype=np.float64)
    new_K, _ = cv2.getOptimalNewCameraMatrix(
        K, lp.DIST, (lp.W, lp.H), alpha=0, newImgSize=(lp.W, lp.H)
    )
    map1, map2 = cv2.initUndistortRectifyMap(K, lp.DIST, np.eye(3), new_K,
                                             (lp.W, lp.H), cv2.CV_32FC1)
    return map1, map2, new_K

def colorize_depth_meters(D_m, z_range=None, percentiles=(2,98), near_is_red=True):
    if hasattr(lp, "colorize_depth_meters"):
        return lp.colorize_depth_meters(D_m, z_range, percentiles, near_is_red)
    # 备用实现（与 lp 中一致）
    D = np.asarray(D_m).astype(np.float32)
    D[~np.isfinite(D)] = 0.0
    valid = D > 0
    H, W = D.shape[:2]
    color = np.zeros((H, W, 3), np.uint8)
    if not np.any(valid):
        return color
    if z_range is None:
        lo = float(np.percentile(D[valid], percentiles[0]))
        hi = float(np.percentile(D[valid], percentiles[1]))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo, hi = float(np.min(D[valid])), float(np.max(D[valid]))
    else:
        lo, hi = map(float, z_range)
        if hi <= lo:
            hi = lo + 1e-6
    N = np.zeros_like(D, np.float32)
    N[valid] = np.clip((D[valid] - lo) / max(1e-9, (hi - lo)), 0, 1)
    if near_is_red:
        N[valid] = 1.0 - N[valid]
    U8 = np.zeros_like(D, np.uint8)
    U8[valid] = np.round(255.0 * N[valid]).astype(np.uint8)
    colormap = getattr(cv2, "COLORMAP_TURBO", cv2.COLORMAP_JET)
    color = cv2.applyColorMap(U8, colormap)
    color[~valid] = (0, 0, 0)
    return color

# ========== 主流程 ==========
def main():
    ensure_dir(OUT_DIR)

    # 预加载：点云与轨迹（只读一次）
    print("[LOAD] point cloud:", PCD_PATH)
    pcd = o3d.io.read_point_cloud(PCD_PATH)
    P_W_all = np.asarray(pcd.points, dtype=np.float64)
    if P_W_all.size == 0:
        raise RuntimeError("Empty PCD.")
    print(f"[PCD] N={P_W_all.shape[0]}")

    print("[LOAD] trajectory:", TRAJ_PATH)
    traj = lp.load_traj(TRAJ_PATH)

    # 去畸变映射
    map1, map2, new_K = build_camera_maps_and_newK()
    fx_u, fy_u = float(new_K[0,0]), float(new_K[1,1])
    cx_u, cy_u = float(new_K[0,2]), float(new_K[1,2])

    # 列出 RGB
    images = list_rgb_images(RGB_DIR)
    if not images:
        raise RuntimeError(f"No RGB found in {RGB_DIR}")
    print(f"[RGB] found {len(images)} files.")

    for idx, rgb_path in enumerate(images, 1):
        stem = os.path.splitext(os.path.basename(rgb_path))[0]  # 时间戳名
        print(f"\n[{idx}/{len(images)}] {rgb_path}")

        img = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
        if img is None:
            print("  [SKIP] cannot read image."); continue
        if (img.shape[1], img.shape[0]) != (lp.W, lp.H):
            print(f"  [SKIP] unexpected size {img.shape[1]}x{img.shape[0]}."); continue

        # 时间戳 -> 最近姿态
        try:
            t_img = float(stem)
        except:
            print("  [SKIP] filename not a timestamp."); continue
        R_W_L, p_W_L, _ = lp.nearest_pose(traj, t_img)

        # 世界<-相机
        R_W_C, p_W_C = lp.compose_T_W_C(R_W_L, p_W_L, t_sel=t_img)

        # 去畸变 RGB
        img_u = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        # 点云 -> 相机系
        # P_C = (P_W_all - p_W_C) @ R_W_C.T
        P_C = (P_W_all - p_W_C) @ R_W_C
        Z = P_C[:, 2]
        front = (Z > 1e-6) & np.isfinite(Z)
        if not np.any(front):
            print("  [WARN] no points in front."); continue
        P_C = P_C[front]; Z = Z[front]
        u = fx_u * (P_C[:,0]/Z) + cx_u
        v = fy_u * (P_C[:,1]/Z) + cy_u
        inside = (u>=0)&(u<lp.W)&(v>=0)&(v<lp.H)&np.isfinite(u)&np.isfinite(v)
        if not np.any(inside):
            print("  [WARN] nothing projects inside image."); continue
        u = u[inside].astype(np.int32)
        v = v[inside].astype(np.int32)
        z = Z[inside].astype(np.float32)

        # Z-buffer 得到 RGB视角深度（米；0=空洞）
        D_C = np.full((lp.H, lp.W), np.inf, np.float32)
        np.minimum.at(D_C, (v, u), z)
        D_C[~np.isfinite(D_C)] = 0.0

        if VIEW_MODE.upper() == "C":
            color = colorize_depth_meters(D_C, z_range=FIXED_RANGE,
                                          percentiles=PCTS, near_is_red=NEAR_IS_RED)
            out_png = os.path.join(OUT_DIR, f"{stem}_depth_color.png")

        elif VIEW_MODE.upper() == "TL":
            # 重投影到 Tl 视角，再伪彩
            img_tl_rgb, depth_tl = lp.warp_rgb_to_Tl_simple(
                img_u=img_u, D_c=D_C, K_c=new_K, K_tl=K_tl,
                B_R_C=lp.B_R_C, B_p_C=lp.B_p_C, B_R_Tl=lp.B_R_Tl, B_p_Tl=lp.B_p_Tl,R_rect_Tl=R1,
                TL_W=TL_W, TL_H=TL_H
            )
            color = colorize_depth_meters(depth_tl, z_range=FIXED_RANGE,
                                          percentiles=PCTS, near_is_red=NEAR_IS_RED)
            out_png = os.path.join(OUT_DIR, f"{stem}_Tl_depth_color.png")
        else:
            raise ValueError("VIEW_MODE must be 'C' or 'Tl'.")

        cv2.imwrite(out_png, color)
        print(f"  [OK] {out_png}")

    print("\n[DONE] saved pseudo-color depth PNGs only.")

if __name__ == "__main__":
    main()
