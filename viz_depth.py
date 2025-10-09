# viz_depth.py
""" 
把米单位的深度图按分位数（或给的固定范围）线性归一到 [0,1]，
可选“近处更红”，再映射成 0–255 并用伪彩色（Turbo/Jet）生成一张彩色深度图。
若提供底图（RGB 或热像），
代码在确保尺寸一致后用 cv2.addWeighted 按给定透明度
把彩色深度图半透明叠加到底图上并保存。 
"""

import numpy as np
import cv2
import os

# ========= 配置：给两组输入，各包含 深度NPY + 可选底图 =========
# 热红外照片
BASE_IMG = r"1757671242.589000000_2\1757671242.574924707.png"
ROOT = os.path.dirname(BASE_IMG)
OUT_DIR = ROOT
PAIRS = [
    # 1) RGB 相机视角
    dict(
        name="rgb",
        depth_npy=os.path.join(ROOT, "depth_m.npy"),            # 你的 RGB 视角深度
        base_img =os.path.join(ROOT, "undist.png"),             # 去畸变后的 RGB（和 depth_m.npy 同视角）
        alpha=0.40
    ),
    # 2) Tl rect 视角
    dict(
        name="tl_rect",
        depth_npy=os.path.join(ROOT, "depth_in_Tl_m.npy"),      # 你的 Tl rect 视角深度
        base_img =BASE_IMG,                     # 如有对应的 rectified thermal，就填路径；没有可先置 None
        alpha=0.40
    ),
]

PCTS         = (2, 98)       # 颜色映射分位数
FIXED_RANGE  = None          # 如需固定深度范围(米)，填 (near, far)，否则 None
NEAR_IS_RED  = True

def ensure_dir(d):
    os.makedirs(d, exist_ok=True)

def colorize_depth(D_m, z_range=None, percentiles=(2,98), near_is_red=True):
    """
    将“以米为单位的深度图 D_m (float32/float64)”映射为伪彩色 BGR 图像（uint8）。
    - 无效深度（<=0 或 非有限）显示为黑色。
    - 颜色范围默认用分位数自适应，也可用固定范围 z_range=(near, far)。
    - near_is_red=True 表示“近处更红、远处更蓝”。
    """
    # 1) 规范为 float32，先把非有限值置 0（作为无效）
    D = np.asarray(D_m).astype(np.float32)
    D[~np.isfinite(D)] = 0.0
    # 2) 有效掩码：深度 > 0 认为是有效
    valid = D > 0
    # 3) 预建一个全黑的 BGR 输出（作为无效像素默认颜色）
    H, W = D.shape[:2]
    color = np.zeros((H, W, 3), np.uint8)
    if not np.any(valid):
        # 整张图都无效，直接返回全黑
        return color
    
    # 4) 确定可视化的深度范围 [lo, hi]
    if z_range is None:
        # 自动范围：用分位数抑制极端值（例如 2% 与 98%）
        lo = float(np.percentile(D[valid], percentiles[0]))
        hi = float(np.percentile(D[valid], percentiles[1]))
        # 如果分位数异常（例如所有值相同），退化为 min/max
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo, hi = float(np.min(D[valid])), float(np.max(D[valid]))
    else:
        # 固定范围：直接使用传入的 near/far
        lo, hi = map(float, z_range)
        if hi <= lo: hi = lo + 1e-6  # 防止除零
    
    # 5) 把有效深度线性归一化到 [0, 1]
    N = np.zeros_like(D, np.float32)
    N[valid] = np.clip((D[valid] - lo)/max(1e-9, (hi-lo)), 0, 1)
    # 6) 近处是否映射为更热（红色）：反转归一化值
    if near_is_red: N[valid] = 1.0 - N[valid]
    # 7) 映射到 0..255 的灰度，并应用伪彩色映射（TURBO 可用则优先，否则用 JET）
    U8 = np.zeros_like(D, np.uint8)
    U8[valid] = np.round(255.0 * N[valid]).astype(np.uint8)
    color = cv2.applyColorMap(U8, getattr(cv2, "COLORMAP_TURBO", cv2.COLORMAP_JET))
    # 8) 无效像素涂黑
    color[~valid] = (0,0,0)
    return color

def overlay(base_bgr, overlay_bgr, alpha):
    """
    将 overlay_bgr 半透明叠加到 base_bgr 上并返回结果。
    要求两张图尺寸完全一致，否则抛出 ValueError。
    """
    # 任一输入缺失则无法叠加
    if base_bgr is None or overlay_bgr is None:
        return None
    # 尺寸必须一致
    if base_bgr.shape[:2] != overlay_bgr.shape[:2]:
        raise ValueError(
            f"overlay() size mismatch: base={base_bgr.shape[:2]} vs overlay={overlay_bgr.shape[:2]}"
        )
    return cv2.addWeighted(base_bgr, 1.0 - alpha, overlay_bgr, alpha, 0)

def read_image_safe(path):
    """
    直接读取 BGR 图像。
    - 期望输入为标准 3 通道 8-bit（例如 undist.png）
    - 读取失败返回 None
    """
    if not path or not os.path.exists(path):
        return None
    # 强制按 BGR 彩图读取
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"read_image_safe: failed to read image: {path}")
    return img

def main():
    ensure_dir(OUT_DIR)
    for p in PAIRS:
        name = p["name"]
        depth_npy = p["depth_npy"]
        base_img_path = p.get("base_img", None)
        alpha = float(p.get("alpha", 0.4))

        assert os.path.exists(depth_npy), f"[{name}] depth npy not found: {depth_npy}"
        D = np.load(depth_npy)
        color = colorize_depth(D, z_range=FIXED_RANGE, percentiles=PCTS, near_is_red=NEAR_IS_RED)

        cv2.imwrite(os.path.join(OUT_DIR, f"{name}_depth_color.png"), color)

        base = read_image_safe(base_img_path)
        if base is not None:
            mix = overlay(base, color, alpha)
            if mix is not None:
                cv2.imwrite(os.path.join(OUT_DIR, f"{name}_overlay.png"), mix)
        print(f"[OK] {name}")

    print("[DONE] outputs in:", OUT_DIR)

if __name__ == "__main__":
    main()
