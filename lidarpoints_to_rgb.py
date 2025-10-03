import os
import numpy as np
import cv2
import open3d as o3d
import shutil 
# ========= 配置 =========


# RGB_PATH = "path/to/run_20250912_180027/realsense/rgb/1757671348.973000000.png"
# PCD_PATH = "path/to/run_20250912_180027/rosbag/fastlio2/scans.pcd"
# TRAJ_PATH = "path/to/run_20250912_180027/rosbag/fastlio2/scan_states_odom.txt"
RGB_PATH = r"D:/run_20250902_170456/run_20250902_170456/realsense/rgb/1756804138.609000000.png"
PCD_PATH = r"D:/run_20250902_170456/run_20250902_170456/rosbag/fastlio2/scans.pcd"
TRAJ_PATH = r"D:/run_20250902_170456/run_20250902_170456/rosbag/fastlio2/scan_states_odom.txt"
TL_RECT_DIR = r"D:/run_20250902_170456/run_20250902_170456/left_thermal/rectified_left"

OUT_DIR = "1756804138.609000000"
OUT_TIMING_TXT = "time.txt"
os.makedirs(OUT_DIR, exist_ok=True)

def _out(name: str) -> str:
    return os.path.join(OUT_DIR, name)

# YAML: model: Inverse Brown Conrady
DIST = np.array([-0.0550349094, 0.0627269447, -0.0008203980, 0.0003794413, -0.0196892563],
                dtype=np.float64)

W, H = 640, 480
fx, fy = 385.8865966797, 385.3258056641
cx, cy = 321.5436096191, 243.3150024414

# B->L, B->C
B_R_L = np.array([[0., -1., 0.],
                  [1.,  0., 0.],
                  [0.,  0., 1.]], dtype=np.float64)
B_p_L = np.array([0., 0., 0.], dtype=np.float64)

B_R_C = np.array([[ 0., 0., 1.],
                  [-1., 0., 0.],
                  [ 0.,-1., 0.]], dtype=np.float64)
B_p_C = np.array([50., 12., -80. ], dtype=np.float64) / 1000.0  # m
# B_R_C从相机坐标系到机体坐标系的旋转矩阵 B_p_C相机原点在机体系下的位置
B_R_Tl = np.array([[ 0., 0., 1.],
                  [-1., 0., 0.],
                  [ 0.,-1., 0.]], dtype=np.float64)
B_p_Tl = np.array([75, 62, -145], dtype=np.float64) / 1000.0  # m

OUT_DEPTH_PNG = "depth_mm.png"
OUT_DEPTH_NPY = "depth_m.npy"
OUT_OVERLAY   = "debug_proj_points3.png"

# ========= 工具 =========
def parse_rgb_time_from_name(path):
    # 文件名形如 1757671253.550000000.png
    name = os.path.splitext(os.path.basename(path))[0]
    return float(name)

def load_traj(path):
    arr = []
    with open(path, 'r') as f:
        for ln in f:
            p = ln.strip().split()
            if len(p) < 8: 
                continue
            t = float(p[0])
            x, y, z = map(float, p[1:4]) # xyz
            qx, qy, qz, qw = map(float, p[4:8]) # 旋转四元数
            q = np.array([qx, qy, qz, qw], np.float64)
            n = np.linalg.norm(q)
            if n == 0: 
                continue
            q /= n
            arr.append((t, np.array([x,y,z],np.float64), q))
    arr.sort(key=lambda x: x[0])
    if not arr: 
        raise RuntimeError("空轨迹")
    return arr

def quat_to_R(q):
    qx,qy,qz,qw = q
    xx,yy,zz = qx*qx, qy*qy, qz*qz
    xy,xz,yz = qx*qy, qx*qz, qy*qz
    wx,wy,wz = qw*qx, qw*qy, qw*qz
    return np.array([
        [1-2*(yy+zz), 2*(xy-wz),   2*(xz+wy)],
        [2*(xy+wz),   1-2*(xx+zz), 2*(yz-wx)],
        [2*(xz-wy),   2*(yz+wx),   1-2*(xx+yy)]
    ], np.float64)

def nearest_pose(traj, t_query):
    ts = np.array([t for t,_,_ in traj], np.float64)
    idx = int(np.argmin(np.abs(ts - t_query)))
    t, W_p_L, W_q_L = traj[idx] # W_p_L：雷达在世界系下的位置向量（xyz） q_W_L：雷达在世界系下的姿态四元数（4,），顺序为 (qx,qy,qz,qw)  
    dms = (t - t_query)*1e3 # 选用的姿态时间 t 以及与图像时间 t_query 的误差 dt
    print(f"[POSE] use t={t:.9f}, dt={dms:+.2f} ms")
    return quat_to_R(W_q_L), W_p_L, t

# ========= 变换：公式直译 =========
def compose_W_T_C(W_R_L, W_p_L, t_sel=None):
    '''
    计算相机在世界坐标系 (W) 下的位姿
    '''
    # 1. 计算 T^L_B = (T^B_L)^-1, 即从机体(B)到LiDAR(L)的变换
    L_R_B = B_R_L.T
    L_p_B = - L_R_B @ B_p_L

    # W_T_B = W_T_L * L_T_B
    W_R_B = W_R_L @ L_R_B
    W_p_B = W_p_L + W_R_L @ L_p_B

    # T^W_C = T^W_B * T^B_C 相机在世界坐标系下
    W_R_C = W_R_B @ B_R_C
    W_p_C = W_p_B + W_R_B @ B_p_C

    # ====== 调试打印 ======
    def _fmt_mat(M):
        return np.array2string(M, formatter={'float_kind':lambda x: f"{x: .6f}"})
    def _fmt_vec(v):
        return np.array2string(v, formatter={'float_kind':lambda x: f"{x: .6f}"})

    ts = f"{t_sel:.9f}" if t_sel is not None else "N/A"
    print("\n[DEBUG|compose_W_T_C]")
    print(f"  time (s): {ts}")
    print("  LiDAR pose in world  (W<-L):")
    print("    W_R_L =\n" + _fmt_mat(W_R_L))
    print("    W_p_L = " + _fmt_vec(W_p_L))
    print("  Camera pose in world (W<-C):")
    print("    W_R_C =\n" + _fmt_mat(W_R_C))
    print("    W_p_C = " + _fmt_vec(W_p_C))
    print("")

    return W_R_C, W_p_C

def overlay_points_colormap(img_u,
                            u, v, z,
                            max_pts=15000,
                            z_range=None,                 # None -> 用分位数; 或者传 (z_min, z_max)
                            z_percentiles=(2.0, 98.0),    # 用于自动范围
                            near_is_red=True,             # 近处是否映射为红色
                            draw_circles=False,           # True 时用小圆点绘制（更醒目但更慢）
                            radius=2, thickness=-1,       # 仅在 draw_circles=True 时生效
                            seed=None,                    # 固定随机子采样
                            out_path=None):
    """
    在去畸变 RGB (img_u) 上按深度 z 给像素 (u,v) 着色并叠加。
    约定: u,v 为已过滤到图像范围内的 int 索引; z 为对应的深度(米)。
    返回: vis(BGR)
    """
    assert img_u.ndim == 3 and img_u.shape[2] == 3, "img_u 必须是 BGR 彩图"
    H, W = img_u.shape[:2]
    assert u.shape == v.shape == z.shape, "u,v,z 尺寸需一致"
    assert u.dtype.kind in "iu" and v.dtype.kind in "iu", "u,v 需为整数像素索引"

    vis = img_u.copy()

    # 下采样
    n = u.size
    if max_pts is not None and n > max_pts:
        if seed is not None:
            rs = np.random.RandomState(seed)
            sel = rs.choice(n, max_pts, replace=False)
        else:
            sel = np.random.choice(n, max_pts, replace=False)
        uu, vv, zz = u[sel], v[sel], z[sel]
    else:
        uu, vv, zz = u, v, z

    if uu.size == 0:
        if out_path:
            cv2.imwrite(out_path, vis)
        return vis

    # 确定色标范围
    if z_range is None:
        z_min = float(np.percentile(zz, z_percentiles[0]))
        z_max = float(np.percentile(zz, z_percentiles[1]))
        if not np.isfinite(z_min) or not np.isfinite(z_max) or z_max <= z_min:
            z_min, z_max = float(np.nanmin(zz)), float(np.nanmax(zz))
    else:
        z_min, z_max = map(float, z_range)
        if z_max <= z_min:
            z_max = z_min + 1e-6

    # 归一化到 [0,255]
    zn = (np.clip(zz, z_min, z_max) - z_min) / max(1e-9, (z_max - z_min))
    if near_is_red:
        zn = 1.0 - zn  # 近->大
    zn_u8 = np.round(255.0 * zn).astype(np.uint8)

    # 生成 BGR 颜色
    colors = cv2.applyColorMap(zn_u8.reshape(-1, 1), cv2.COLORMAP_JET).reshape(-1, 3)

    # 绘制
    if draw_circles:
        # 圆点绘制（醒目但逐点慢）
        for (px, py, c) in zip(uu, vv, colors):
            cv2.circle(vis, (int(px), int(py)), radius, tuple(int(x) for x in c), thickness)
    else:
        # 矢量化着色（快；1 像素散点）
        vis[vv, uu] = colors

    if out_path:
        cv2.imwrite(_out(out_path), vis)
    return vis

def warp_rgb_to_Tl_simple(img_u, D_c, K_c, K_tl,
                          B_R_C, B_p_C, B_R_Tl, B_p_Tl,R_rect_Tl,
                          TL_W=640, TL_H=512):
    """
    输入:
      img_u : 去畸变后的 RGB (Hc,Wc,3)
      D_c   : 对应的深度(米; 0=无效) (Hc,Wc)
      K_c   : RGB 的去畸变内参 (new_K) 3x3
      K_tl  : Tl 的内参 (去畸变后) 3x3  取的 P1[:3, :3]，即 左红外 rectified 相机的新内参
      B_R_C, B_p_C: 机体<-RGB 的外参 (米)
      B_R_Tl,B_p_Tl: 机体<-Tl  的外参 (米)
      R_rect_Tl : cv2.stereoRectify 给左红外相机的矫正旋转 P_Tl_rect = R1 · P_Tl_orig
    返回:
      img_tl_rgb : 在 Tl 视角下的彩色图 (TL_H,TL_W,3)
      depth_tl   : Tl 视角下的深度(沿 Tl-Z; 米; 0=空洞) (TL_H,TL_W)
    """
    Hc, Wc = img_u.shape[:2]
    mask = D_c > 0 # 只用有深度的像素参与重投影
    if not np.any(mask):
        return np.zeros((TL_H, TL_W, 3), np.uint8), np.zeros((TL_H, TL_W), np.float32)

    # --- C 像素+深度 -> C 相机坐标 ---
    vc, uc = np.nonzero(mask)             # 行= v, 列= u
    zc = D_c[vc, uc].astype(np.float64)
    fx_c, fy_c = K_c[0,0], K_c[1,1]
    cx_c, cy_c = K_c[0,2], K_c[1,2]
    Xc = (uc - cx_c) / fx_c * zc
    Yc = (vc - cy_c) / fy_c * zc
    Zc = zc # 相机坐标系下的3D坐标
    P_C = np.stack([Xc, Yc, Zc], axis=1)  # (N,3)
    colors = img_u[vc, uc].copy()         # 颜色数组 (N,3) BGR

    # --- 计算 Tl<-C 外参: T^Tl_C = T^Tl_B * T^B_C ---
    Tl_R_B = B_R_Tl.T #  Tl_R_B 从机体坐标系到TL坐标系的旋转矩阵
    Tl_p_B = - Tl_R_B @ B_p_Tl # 平移部分是 -R_inv * p
    Tl_R_C = Tl_R_B @ B_R_C # 旋转部分: Tl_R_C = Tl_R_B * B_R_C
    Tl_p_C = Tl_p_B + Tl_R_B @ B_p_C # 平移部分: Tl_p_C = Tl_R_B * B_p_C + Tl_p_B

    # --- C -> Tl ---
    # 3D 点在“左目热像（Tl）原始相机坐标系”下的坐标。 行向量右乘转置
    P_Tl = P_C @ Tl_R_C.T + Tl_p_C

    P_Tl = P_Tl @ R_rect_Tl.T      # Tl -> Tl_rect

    Ztl = P_Tl[:,2]
    front = Ztl > 0 # 只保留在相机前方的点
    if not np.any(front):
        return np.zeros((TL_H, TL_W, 3), np.uint8), np.zeros((TL_H, TL_W), np.float32)
    P_Tl = P_Tl[front]; Ztl = Ztl[front]; colors = colors[front]

    # --- Tl 投影 ---
    # 从 rectified 的 Tl 的内参矩阵中提取 fx, fy, cx, cy
    fx_tl, fy_tl = K_tl[0,0], K_tl[1,1]
    cx_tl, cy_tl = K_tl[0,2], K_tl[1,2]
    # 把 rectified Tl 相机坐标系下的 3D 点用内参投到像素平面
    utl = fx_tl * (P_Tl[:,0] / Ztl) + cx_tl
    vtl = fy_tl * (P_Tl[:,1] / Ztl) + cy_tl

    # 只保留落在图像边界内且数值有效（非 NaN/Inf）的投影
    inside = (utl >= 0) & (utl < TL_W) & (vtl >= 0) & (vtl < TL_H) \
             & np.isfinite(utl) & np.isfinite(vtl)
    if not np.any(inside): # 若没有任何点落进图像，直接返回空结果
        print("WARN: no points project inside Tl image")
        return np.zeros((TL_H, TL_W, 3), np.uint8), np.zeros((TL_H, TL_W), np.float32)
    # 用掩码筛选 u,v,深度,颜色
    utl = utl[inside].astype(np.int32)
    vtl = vtl[inside].astype(np.int32)
    Ztl = Ztl[inside]
    colors = colors[inside] 

    # --- Z-buffer (最近者胜) ---
    # 颜色图全黑
    img_tl = np.zeros((TL_H, TL_W, 3), np.uint8) 
    # 深度全设为 +inf 表示“还没有击中”
    depth_tl = np.full((TL_H, TL_W), np.inf, np.float32)

    # 把 2D 像素坐标 (vtl, utl) 映射成线性索引 lin（行主序，索引 = 行*宽 + 列），
    # 便于后面一次性在扁平数组上操作。
    lin = (vtl.astype(np.int64) * TL_W + utl.astype(np.int64))

    # np.argsort(Ztl)：按深度从近到远排序点的下标。
    order = np.argsort(Ztl)                      # 近 -> 远
    # 把线性索引按这个“近→远”的顺序排成 lin_sorted
    lin_sorted = lin[order]
    # 在近→远的顺序里，找出每个像素第一次出现的位置
    # 同一像素的多个候选点里，最先出现的就是最近的
    uniq_lin, first_pos = np.unique(lin_sorted, return_index=True)
    # “胜出”的那些点在原数组中的下标（用于同步取颜色、深度）
    sel = order[first_pos]

    # 把 img_tl 和 depth_tl 展平成一维后，
    # 用 uniq_lin 作为位置，一次性写入中标点的颜色与深度。
    img_tl.reshape(-1,3)[uniq_lin] = colors[sel]
    depth_tl.reshape(-1)[uniq_lin] = Ztl[sel]

    # 仍为 inf 的像素表示无人命中（空洞），置为 0.0 作为无效标记。
    depth_tl[~np.isfinite(depth_tl)] = 0.0

    return img_tl, depth_tl # 返回 Tl 视角的彩色图和对应深度图

def _iter_timestamp_pngs(dir_path):
    """
    产出 (filename, t)；其中 filename 为文件名（含扩展名），
    t 为由文件名（不含扩展名）解析得到的 float 时间戳。
    非法文件名会被跳过。
    """
    if not os.path.isdir(dir_path):
        raise FileNotFoundError(f"目录不存在: {dir_path}")
    for fn in os.listdir(dir_path):
        if not fn.lower().endswith(".png"):
            continue
        name = os.path.splitext(fn)[0]
        try:
            t = float(name)
        except ValueError:
            continue
        yield fn, t

def find_nearest_rectified_left_thermal(dir_path, t_ref):
    """
    返回: (filename, t_sel, dt) 其中 dt = t_sel - t_ref (秒；可正可负)
    """
    best = None  # (fn, t, dt)
    for fn, t in _iter_timestamp_pngs(dir_path):
        dt = t - t_ref
        if (best is None) or (abs(dt) < abs(best[2])):
            best = (fn, t, dt)
    if best is None:
        raise FileNotFoundError(f"未在目录中找到任何时间戳命名的 PNG: {dir_path}")
    fn, t_sel, dt = best
    print(f"[TL] use t={t_sel:.9f}, dt={dt*1e3:+.2f} ms  file={fn}")
    return best

def write_time_sync_report(out_dir, t_rgb, t_traj_sel, t_tl=None):
    """
    将时间戳与时间差写入 txt。
    t_tl 允许为 None（表示未找到热像图）。
    """
    lines = []
    lines.append(f"RGB_path: {RGB_PATH}")
    lines.append(f"TL_rectified_dir: {TL_RECT_DIR}")
    lines.append(f"TRAJ_path: {TRAJ_PATH}")
    lines.append("")

    lines.append(f"t_RGB (s): {t_rgb:.9f}")
    lines.append(f"t_traj_selected (s): {t_traj_sel:.9f}")
    dt_traj_ms = (t_traj_sel - t_rgb) * 1e3
    lines.append(f"Δt(traj - RGB) (ms): {dt_traj_ms:+.2f}")

    if t_tl is not None:
        lines.append(f"t_left_thermal_rectified (s): {t_tl:.9f}")
        dt_tl_ms = (t_tl - t_rgb) * 1e3
        lines.append(f"Δt(thermal - RGB) (ms): {dt_tl_ms:+.2f}")
    else:
        lines.append("t_left_thermal_rectified (s): NOT_FOUND")
        lines.append("Δt(thermal - RGB) (ms): N/A")

    out_txt_path = os.path.join(out_dir, OUT_TIMING_TXT)
    with open(out_txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[SAVE] {out_txt_path}")

# ========= 主流程 =========
def main():
    # 0) 读图与检查尺寸
    img = cv2.imread(RGB_PATH, cv2.IMREAD_COLOR)
    assert img is not None, f"Image not found: {RGB_PATH}"
    assert (img.shape[1], img.shape[0]) == (W, H), \
        f"Image size {img.shape[1]}x{img.shape[0]} != expected {W}x{H}"

    t_img = parse_rgb_time_from_name(RGB_PATH)
    print(f"[RGB] t={t_img:.9f}")

    # ---------- 0.5) 去畸变 -> 得到 img_u 与 new_K ----------
    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0,  0,  1]], dtype=np.float64)

    # alpha=0：尽量去黑边（轻微裁切）；alpha=1：保留视场（边缘可能有黑边）
    new_K, _ = cv2.getOptimalNewCameraMatrix(K, DIST, (W, H), alpha=0, newImgSize=(W, H))
    map1, map2 = cv2.initUndistortRectifyMap(K, DIST, np.eye(3), new_K, (W, H), cv2.CV_32FC1)
    img_u = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR,
                      borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    # 用去畸变后的新内参做投影
    fx_u = float(new_K[0, 0]); fy_u = float(new_K[1, 1])
    cx_u = float(new_K[0, 2]); cy_u = float(new_K[1, 2])

    # 1) 读取轨迹并取最近姿态（雷达在世界）
    traj = load_traj(TRAJ_PATH)
    W_R_L, W_p_L, t_sel = nearest_pose(traj, t_img)

    # 2) 得到相机位姿
    W_R_C, W_p_C = compose_W_T_C(W_R_L, W_p_L, t_sel=t_sel)
    print("[TF] W_p_C (m):", W_p_C)

    


    # 3) 点云（世界）
    pcd = o3d.io.read_point_cloud(PCD_PATH)
    P_W = np.asarray(pcd.points, dtype=np.float64)
    if P_W.size == 0:
        print("[PCD] empty pcd"); return
    print(f"[PCD] N={P_W.shape[0]}")
    # 平移到相机原点 再把向量从世界轴转到相机轴（行向量，右乘旋转）；列向量写法p_C = W_R_C.T (p_W - W_p_C)
    P_C = (P_W - W_p_C) @ W_R_C

    Z = P_C[:, 2]
    # front = Z > 0
    front = (Z > 1e-6) & np.isfinite(Z) # 避免极近/NaN
    if not np.any(front):
        print("[WARN] no points in front"); return
    
    # === 4.1) 将点云保存到相机系下的局部区域 ===
    def _save_pcd(points_xyz, out_path):
        if points_xyz.size == 0:
            print(f"[SAVE] {out_path}: no points to save")
            return
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(points_xyz.astype(np.float64))
        # 二进制 PCD，写入更快更小
        ok = o3d.io.write_point_cloud(out_path, pc, write_ascii=False, compressed=False)
        print(f"[SAVE] {out_path}: {points_xyz.shape[0]} pts, ok={ok}")

    # P_C 为世界->相机后的点（单位: 米）
    Xc, Yc, Zc = P_C[:,0], P_C[:,1], P_C[:,2]

    # 用于debug # A) “左右前后各 2m”的立方体：|X|≤2, |Y|≤2, |Z|≤2（用于整体对齐检查）
    # box_mask = (np.abs(Xc) <= 4.0) & (np.abs(Yc) <= 4.0) & (np.abs(Zc) <= 2.0)
    # P_box = P_C[box_mask]
    # _save_pcd(P_box, "crop_rgb_frame_cube_pm2m.pcd")

    # # B) 只看“正前方”的可视区域：|X|≤2, |Y|≤2, 0<Z≤2（避免相机身后点影响判断）
    # front_mask = (np.abs(Xc) <= 4.0) & (np.abs(Yc) <= 4.0) & (Zc > 0.0) & (Zc <= 2.0)
    # P_front = P_C[front_mask]
    # _save_pcd(P_front, "crop_rgb_frame_front_0to2m.pcd")

    # # （可选）如果点非常多再做一个更大的前向盒：0<Z≤4 m
    # front4_mask = (np.abs(Xc) <= 4.0) & (np.abs(Yc) <= 4.0) & (Zc > 0.0) & (Zc <= 4.0)
    # P_front4 = P_C[front4_mask]
    # _save_pcd(P_front4, "crop_rgb_frame_front_0to4m.pcd")


    P_C = P_C[front]; Z = Z[front] # 只保留相机前方的点
    
    u = fx_u * (P_C[:,0]/Z) + cx_u
    v = fy_u * (P_C[:,1]/Z) + cy_u

    # 像素坐标在图像边界内0 ≤ u < W、0 ≤ v < H，np.isfinite(...)：排除 NaN/Inf
    inside = (u>=0)&(u<W)&(v>=0)&(v<H)&np.isfinite(u)&np.isfinite(v)
    u = u[inside].astype(np.int32)
    v = v[inside].astype(np.int32)
    z = Z[inside].astype(np.float32)
    print(f"[PROJ] inside={z.size}")

    if z.size == 0:
        print("[WARN] nothing inside image"); return

    
    # 5) Z-buffer（米；空洞=0）
    # 创建一个深度图 D，大小与图像相同，初始用 +inf 表示“还没有任何点投到这里”。
    D = np.full((H,W), np.inf, dtype=np.float32)
    # D[v_i, u_i] = min(D[v_i, u_i], z_i) 如果多个点落到同一个像素，更近的那个（更小的 z）会留下
    np.minimum.at(D, (v, u), z) 
    # 仍为 inf 的像素表示从未被任何点命中，改成 0.0 作为“空洞/无效”标记。
    D[~np.isfinite(D)] = 0.0

    # 6) 保存深度与叠加可视化
    # 把米→毫米（乘 1000），并裁到 uint16 范围再保存 PNG
    # cv2.imwrite(_out(OUT_DEPTH_PNG), np.clip(D*1000.0, 0, 65535).astype(np.uint16))
    # 保存原始 float32（米）的深度 NPY
    np.save(_out(OUT_DEPTH_NPY), D)

    # vis = overlay_points_colormap(
    #     img_u, u, v, z,
    #     max_pts=9000000,
    #     z_range=None,               # 或者 (0.3, 20.0)
    #     z_percentiles=(2.0, 50.0),
    #     near_is_red=True,
    #     draw_circles=False,      
    #     radius=2, thickness=-1,
    #     seed=0,
    #     out_path=OUT_OVERLAY      
    # )
    cv2.imwrite(_out("undist.png"), img_u)
    print(f"[SAVE] {OUT_DEPTH_PNG}, {OUT_DEPTH_NPY}")

    # === 6.5) 把 RGB(含深度) 重投影到左目热像 Tl 视角 ===
    # stereoRectify 得到的参数
    R1 = np.array([[ 0.99965096,  0.00361561,  0.02617045],
               [-0.00355090,  0.99999052, -0.00251879],
               [-0.02617931,  0.00242499,  0.99965432]], dtype=np.float64)
    
    TL_W, TL_H = 640, 512
    K_tl = np.array([[302.53882139, 0.0,         305.2902832 ],
                     [0.0,          302.53882139,273.44169617],
                     [0.0,          0.0,         1.0        ]], dtype=np.float64)

    # 注意: B_p_* 已经是“米”
    img_tl_rgb, depth_tl = warp_rgb_to_Tl_simple(
        img_u=img_u,
        D_c=D,               # RGB 深度 (米)
        K_c=new_K,           # RGB 的去畸变内参 (getOptimalNewCameraMatrix 的 new_K)
        K_tl=K_tl,           # Tl 内参 (640x512)
        B_R_C=B_R_C, B_p_C=B_p_C,
        B_R_Tl=B_R_Tl, B_p_Tl=B_p_Tl,
        R_rect_Tl=R1,       # Tl->Tl_rect
        TL_W=TL_W, TL_H=TL_H
    )

    cv2.imwrite(_out("rgb_warped_to_Tl.png"), img_tl_rgb)
    # cv2.imwrite(_out("depth_in_Tl_mm.png"), np.clip(depth_tl*1000.0, 0, 65535).astype(np.uint16))
    np.save(_out("depth_in_Tl_m.npy"), depth_tl)

    # === 7) 寻找与 RGB 最近的去畸变左热像图，并复制到 OUT_DIR ===
    t_left = None  # 如果未找到，则保留为 None
    try:
        tl_fn, t_left, dt_left = find_nearest_rectified_left_thermal(TL_RECT_DIR, t_img)
        src_tl = os.path.join(TL_RECT_DIR, tl_fn)
        dst_tl = os.path.join(OUT_DIR, tl_fn)  # 直接用原时间戳文件名保存
        shutil.copy2(src_tl, dst_tl)
        print(f"[TL] copied: {src_tl} -> {dst_tl}")
    except Exception as e:
        print(f"[TL] WARN: {e}")

    # === 8) 写时间（RGB vs 轨迹、RGB vs 左热像） ===
    write_time_sync_report(OUT_DIR, t_rgb=t_img, t_traj_sel=t_sel, t_tl=t_left)

if __name__ == "__main__":
    main()