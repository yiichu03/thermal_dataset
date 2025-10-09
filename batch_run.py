# -*- coding: utf-8 -*-
import os
import shutil
from datetime import datetime

# 请改成你的核心脚本文件名
import lidarpoints_to_rgb as core


def _find_thermal_png_in_outdir(out_dir: str):
    """
    在结果目录里找到“被复制过来的热像图”（文件名是时间戳 .png）。
    规则：挑选文件名能转成 float 的 .png；若有多个，取最靠近 RGB 时间戳的一个也行，
    但通常只有一个。
    """
    cands = []
    for fn in os.listdir(out_dir):
        if (not fn.lower().endswith(".png")):
            continue
        name, _ = os.path.splitext(fn)
        try:
            float(name)  # 能转成时间戳
            cands.append(fn)
        except Exception:
            pass
    if not cands:
        return None
    # 一般只有 1 个；多于 1 个就取排序后的第 1 个
    cands.sort()
    return os.path.join(out_dir, cands[0])


def _infer_run_root_from_rgb(rgb_path: str) -> str:
    """与核心里同逻辑，冗余实现一份，避免 import 内部函数。"""
    p = os.path.normpath(rgb_path)
    parts = p.split(os.sep)
    if "realsense" in parts:
        i = parts.index("realsense")
        return os.sep.join(parts[:i])
    return os.path.dirname(os.path.dirname(os.path.dirname(rgb_path)))


def _set_core_paths_for_rgb(rgb_path: str):
    """
    给核心模块设置 RGB/PCD/TRAJ/TL 路径（按你现有目录结构推断）。
    同时 RGB_PATH 也在这里设定。
    """
    root = _infer_run_root_from_rgb(rgb_path)
    core.RGB_PATH = rgb_path
    core.PCD_PATH = os.path.join(root, "rosbag", "fastlio2", "scans.pcd")
    core.TRAJ_PATH = os.path.join(root, "rosbag", "fastlio2", "scan_states_odom.txt")
    core.TL_RECT_DIR = os.path.join(root, "left_thermal", "rectified_left")


def main():
    # === 需要处理的 4 张图片（按你给的顺序）===
    image_list = [
        r"D:/run_20250902_170456/run_20250902_170456/realsense/rgb/1756804103.365000000.png",
        r"D:/run_20250902_170456/run_20250902_170456/realsense/rgb/1756803983.612000000.png",
        r"D:/run_20250902_170456/run_20250902_170456/realsense/rgb/1756803983.646000000.png",
        r"D:/run_20250902_170456/run_20250902_170456/realsense/rgb/1756803983.662000000.png",
        r"D:/run_20250902_170456/run_20250902_170456/realsense/rgb/1756803995.795000000.png",
        r"D:/run_20250912_180027/realsense/rgb/1757671258.621000000.png",
        r"D:/run_20250912_180027/realsense/rgb/1757671258.672000000.png",
        r"D:/run_20250912_180027/realsense/rgb/1757671258.688000000.png",
        r"D:/run_20250912_180027/realsense/rgb/1757671263.226000000.png",
        r"D:/run_20250912_180027/realsense/rgb/1757671272.814000000.png",
    ]

    # === 新增：为“本次运行”建立父目录，如 20251003_153012 ===
    now_tag = datetime.now().strftime("%Y%m%d_%H%M%S")   # <<< NEW
    run_root = now_tag                                    # <<< NEW (名字就是 20251003_XXXXXX)
    os.makedirs(run_root, exist_ok=True)                  # <<< NEW
    os.chdir(run_root)                                    # <<< NEW：切换到该目录，后续输出全在里面

    # Img_当前时间 目录放在 run_root 里
    img_dir = f"Img_{now_tag}"
    os.makedirs(img_dir, exist_ok=True)

    # 写个索引文件：记录每对 left/right 的来源
    index_lines = []
    pair_idx = 0

    # 四种 choose 的顺序固定为 00,01,10,11
    choose_pairs = [(0, 0), (0, 1), (1, 0), (1, 1)]

    for rgb_idx, rgb_path in enumerate(image_list, start=1):
        # 先让核心模块知道它要处理哪套数据
        _set_core_paths_for_rgb(rgb_path)

        # 额外：根据 run 自动换 fx/fy（也可以不做，因为核心 main 里已经会自动做了）
        # 这里保留注释说明：
        # core._auto_set_paths_and_intrinsics_from_rgb(core.RGB_PATH)

        for (c_tl_rc, c_l_rc) in choose_pairs:
            pair_idx += 1

            # 把 choose 写入核心模块的全局变量（最小侵入）
            core.CHOOSE_TL_R_C = c_tl_rc
            core.CHOOSE_L_R_C = c_l_rc

            print(f"\n==== RUN image#{rgb_idx} choose={c_tl_rc}{c_l_rc} ====")
            out_dir = core.main()  # 返回类似 '1756804103.365000000_00'

            # 找热像图（left）与彩色扭到热像后的图（right）
            tl_png = _find_thermal_png_in_outdir(out_dir)
            warped_png = os.path.join(out_dir, "rgb_warped_to_Tl.png")

            left_name = f"left{pair_idx}.png"
            right_name = f"right{pair_idx}.png"
            left_dst = os.path.join(img_dir, left_name)
            right_dst = os.path.join(img_dir, right_name)

            # 复制（若某一张不存在，就在索引里标注 MISSING）
            if tl_png and os.path.isfile(tl_png):
                shutil.copy2(tl_png, left_dst)
            else:
                left_dst = "(MISSING)"
                print(f"[WARN] thermal png not found in {out_dir}")

            if os.path.isfile(warped_png):
                shutil.copy2(warped_png, right_dst)
            else:
                right_dst = "(MISSING)"
                print(f"[WARN] rgb_warped_to_Tl.png not found in {out_dir}")

            # 记录索引行
            index_lines.append(
                f"{pair_idx:02d}\tchoose={c_tl_rc}{c_l_rc}\trgb={rgb_path}\tout_dir={os.path.abspath(out_dir)}"
                f"\tleft={left_name if left_dst!='(MISSING)' else left_dst}"
                f"\tright={right_name if right_dst!='(MISSING)' else right_dst}"
            )

    # 写入索引
    with open(os.path.join(img_dir, "index.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(index_lines) + "\n")

    print(f"\n[OK] 合计 {pair_idx} 对图片已复制到：{img_dir}")
    print(f"[OK] 详细映射见：{os.path.join(img_dir, 'index.txt')}")


if __name__ == "__main__":
    main()
