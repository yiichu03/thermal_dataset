*.png（mm）是原始深度的存档格式，*.npy（m）便于进一步处理；debug_proj_points3.png 是好看的可视化叠加；rgb_warped_to_Tl.png / depth_in_Tl_mm.png 是在 Tl 视角下的结果。


undist.png
去畸变后的 RGB 图。内参用 getOptimalNewCameraMatrix 得到的 new_K，所以后续投影都以这张图与 new_K 为准。

debug_proj_points3.png（由 overlay_points_colormap 生成）
把点云投到去畸变 RGB 上，并按“相机系深度 Z（米）”上色的叠加图。

颜色是 COLORMAP_JET，近处更红（因为 near_is_red=True 时做了 1-zn 的反转）。

只是一张“可视化用”的彩色叠加图；不保存真实的深度值。

depth_m.npy
和 undist.png 同分辨率的深度阵列（float32，单位米）。

值为沿相机 Z 轴的深度；

Z-buffer 后每个像素取最近点；

空洞/未命中处为 0.0。

depth_mm.png
把 depth_m.npy 乘 1000 变成毫米后保存为 16 位 PNG（uint16）。

仍然是物理深度（mm）的“原值存档”，不是伽马/拉伸后的可视化图；

很多看图软件不会自动拉伸 16 位灰度，看起来容易发灰/全黑，但数值在里面。

rgb_warped_to_Tl.png
把 RGB（带深度）重投影到 Tl 相机视角 下的彩色图（640×512）。

使用 K_tl、B_R_Tl/B_p_Tl 与 B_R_C/B_p_C 做了几何外参拼接，

Z-buffer 确保每个像素取最近的颜色样本。

depth_in_Tl_mm.png
对应 rgb_warped_to_Tl.png 的 Tl 视角深度图（16 位 PNG，毫米）。

深度定义是 沿 Tl 相机 Z 轴 的距离；

未命中像素为 0。


读取 depth_m.npy（单位米），按分位数自动拉伸后用伪彩色（TURBO 优先，退回 JET）渲染，并在同目录有 undist.png 时顺便做一张半透明叠加