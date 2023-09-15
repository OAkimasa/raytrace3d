import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

from raytrace3d import VectorFunctions

start = time.time()


N_air = 1.0
N_Fused_Silica = 1.458  # Refractive index (Fused Silica at 589nm)

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_box_aspect((1, 1, 1))
ax.set_xlim(0, 150)
ax.set_ylim(-75, 75)
ax.set_zlim(-75, 75)


# ---- 単位は mm ----

# param = [pos, nV, R, Lens_R(axis+-)]
surface_1 = [[10.000, 0., 0.], [1., 0., 0.], 25.4/2, -510.9]  # air->lens1
surface_2 = [[13.000, 0., 0.], [1., 0., 0.], 25.4/2, 467.0]  # lens1->air
surface_3 = [[20.000, 0., 0.], [1., 0., 0.], 25.4/2, -51.09]  # air->lens2
surface_4 = [[23.000, 0., 0.], [1., 0., 0.], 25.4/2, 46.70]  # lens2->air
evaluation_plane = [[200.000, 0., 0.], [1., 0., 0.], 0., 0.]  # air->air
surface_list = [surface_1, surface_2, surface_3, surface_4, evaluation_plane]


## インスタンス生成, 光線追跡開始
VF = VectorFunctions()
VF.set_ax(ax)  # axをVFに登録


for surface in surface_list:
    VF.plot_lens(surface)  # レンズ描画


# 始点を生成する
width = 200
space = 40
rayDensity = 1
rayCenterX = -3200
rayCenterY = 0
rayCenterZ = 0
size = len(np.arange(-width+rayCenterY, 1+width+rayCenterY, space))**2
pointsY, pointsZ = np.meshgrid(
    np.arange(-width+rayCenterY, 1+width+rayCenterY, space),
    np.arange(-width+rayCenterY, 1+width+rayCenterY, space))
pointsX = np.array([rayCenterX]*size)
pointsY = pointsY.reshape(size)*rayDensity
pointsZ = pointsZ.reshape(size)*rayDensity
raySPoint0 = VF.make_points(pointsX, pointsY, pointsZ, size, 3)


focus_pos = np.array([0., 0., 0.])  # センサーの焦点位置
# focus_pos = np.array([18.80, -3.7, -3.7])  # センサーの焦点位置
focal_length = 3200  # [mm] 焦点距離
F_number = 8.0  # F値

lens_unit_aperture_R = focal_length/F_number/2  # 絞り半径
print("lens_unit_aperture_R: {0:.3f} mm".format(lens_unit_aperture_R))
evaluate_plane_pos = np.array(evaluation_plane[0])  # 評価面の位置
# aperture_pos = evaluate_plane_pos - np.array([focal_length, 0., 0.])
aperture_pos = np.array([-3200., 0., 0.])
aperture_plane = np.array(
    [aperture_pos, [1., 0., 0.], lens_unit_aperture_R, np.inf], dtype=object)  # 絞り面
ray_start_pos_init = raySPoint0  # 初期値
ray_start_dir_init = np.array([[1.0, 0.0, 0.0]]*len(raySPoint0))  # 初期値
VF.ray_start_pos = ray_start_pos_init  # 初期値の生成のため
VF.ray_start_dir = ray_start_dir_init  # 初期値の生成のため
VF.set_surface(aperture_plane)
VF.plot_plane(aperture_plane)  # 絞り面描画
VF.raytrace_plane()
VF.plot_line_red(alpha=0.1)  # 絞り面までの平行光を描画
ray_start_dir_init = [focus_pos - pos for pos in VF.ray_end_pos]

# 初期値
# ray_start_pos_init = np.array([-30.0, 3.0, 3.0])  # 初期値
ray_start_pos_init = raySPoint0  # 初期値
# ray_start_dir_init = np.array([[1.0, 0.0, 0.0]]*len(raySPoint0))  # 初期値

# surface_1
VF.ray_start_pos = ray_start_pos_init  # 初期値
VF.ray_start_dir = ray_start_dir_init  # 初期値
VF.set_surface(surface_1,
               refractive_index_before=N_air,
               refractive_index_after=N_Fused_Silica)  # surface_1をVFに登録
VF.raytrace_sphere()  # 光線追跡
VF.refract()  # 空気からレンズ1の屈折
VF.plot_line_red()  # 光線描画

# surface_2
VF.ray_start_pos = VF.ray_end_pos  # surface_1の終点をsurface_2の始点に
VF.ray_start_dir = VF.ray_end_dir  # surface_1の終点をsurface_2の始点に
VF.set_surface(surface_2,
               refractive_index_before=N_Fused_Silica,
               refractive_index_after=N_air)  # surface_2をVFに登録
VF.raytrace_sphere()  # 光線追跡
VF.refract()  # レンズ1から空気の屈折
VF.plot_line_red()  # 光線描画

# surface_3
VF.ray_start_pos = VF.ray_end_pos  # surface_2の終点をsurface_3の始点に
VF.ray_start_dir = VF.ray_end_dir  # surface_2の終点をsurface_3の始点に
VF.set_surface(surface_3,
               refractive_index_before=N_air,
               refractive_index_after=N_Fused_Silica)  # surface_3をVFに登録
VF.raytrace_sphere()  # 光線追跡
VF.refract()  # 空気からレンズ2の屈折
VF.plot_line_red()  # 光線描画

# surface_4
VF.ray_start_pos = VF.ray_end_pos  # surface_3の終点をsurface_4の始点に
VF.ray_start_dir = VF.ray_end_dir  # surface_3の終点をsurface_4の始点に
VF.set_surface(surface_4,
               refractive_index_before=N_Fused_Silica,
               refractive_index_after=N_air)  # surface_4をVFに登録
VF.raytrace_sphere()  # 光線追跡
VF.refract()  # レンズ2から空気の屈折
VF.plot_line_red()  # 光線描画


# evaluation plane
VF.ray_start_pos = VF.ray_end_pos  # surface_6の終点を評価面の始点に
VF.ray_start_dir = VF.ray_end_dir  # surface_6の終点を評価面の始点に
VF.set_surface(evaluation_plane)  # 評価面をVFに登録
VF.raytrace_plane()  # 光線追跡
VF.plot_line_red()  # 光線描画

# focal_length = VF.calc_focal_length(ray_start_pos_init)  # 計算した焦点距離を取得
# print("焦点距離 = " + str(focal_length) + " mm")


print("run time: {0:.3f} sec".format(time.time() - start))
plt.show()

# %% -------------- レンズの焦点距離を計算する ----------------
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_box_aspect((1, 1, 1))
ax.set_xlim(0, 150)
ax.set_ylim(-75, 75)
ax.set_zlim(-75, 75)

## インスタンス生成, 光線追跡開始
VF = VectorFunctions()
VF.set_ax(ax)  # axをVFに登録

# 始点を生成する
width = 5
space = 2
rayDensity = 1
rayCenterX = -100
rayCenterY = 0
rayCenterZ = 0
size = len(np.arange(-width+rayCenterY, 1+width+rayCenterY, space))**2
pointsY, pointsZ = np.meshgrid(
    np.arange(-width+rayCenterY, 1+width+rayCenterY, space),
    np.arange(-width+rayCenterY, 1+width+rayCenterY, space))
pointsX = np.array([rayCenterX]*size)
pointsY = pointsY.reshape(size)*rayDensity
pointsZ = pointsZ.reshape(size)*rayDensity
raySPoint0 = VF.make_points(pointsX, pointsY, pointsZ, size, 3)

# 初期値
# ray_start_pos_init = np.array([-30.0, 3.0, 3.0])  # 初期値
ray_start_pos_init = raySPoint0  # 初期値
ray_start_dir_init = np.array([[1.0, 0.0, 0.0]]*len(raySPoint0))  # 初期値

# surface_1
VF.ray_start_pos = ray_start_pos_init  # 初期値
VF.ray_start_dir = ray_start_dir_init  # 初期値
VF.set_surface(surface_1,
               refractive_index_before=N_air,
               refractive_index_after=N_Fused_Silica)  # surface_1をVFに登録
VF.raytrace_sphere()  # 光線追跡
VF.refract()  # 空気からレンズ1の屈折
VF.plot_line_red()  # 光線描画

# surface_2
VF.ray_start_pos = VF.ray_end_pos  # surface_1の終点をsurface_2の始点に
VF.ray_start_dir = VF.ray_end_dir  # surface_1の終点をsurface_2の始点に
VF.set_surface(surface_2,
               refractive_index_before=N_Fused_Silica,
               refractive_index_after=N_air)  # surface_2をVFに登録
VF.raytrace_sphere()  # 光線追跡
VF.refract()  # レンズ1から空気の屈折
VF.plot_line_red()  # 光線描画

# surface_3
VF.ray_start_pos = VF.ray_end_pos  # surface_2の終点をsurface_3の始点に
VF.ray_start_dir = VF.ray_end_dir  # surface_2の終点をsurface_3の始点に
VF.set_surface(surface_3,
               refractive_index_before=N_air,
               refractive_index_after=N_Fused_Silica)  # surface_3をVFに登録
VF.raytrace_sphere()  # 光線追跡
VF.refract()  # 空気からレンズ2の屈折
VF.plot_line_red()  # 光線描画

# surface_4
VF.ray_start_pos = VF.ray_end_pos  # surface_3の終点をsurface_4の始点に
VF.ray_start_dir = VF.ray_end_dir  # surface_3の終点をsurface_4の始点に
VF.set_surface(surface_4,
               refractive_index_before=N_Fused_Silica,
               refractive_index_after=N_air)  # surface_4をVFに登録
VF.raytrace_sphere()  # 光線追跡
VF.refract()  # レンズ2から空気の屈折
VF.plot_line_red()  # 光線描画


# evaluation plane
VF.ray_start_pos = VF.ray_end_pos  # surface_6の終点を評価面の始点に
VF.ray_start_dir = VF.ray_end_dir  # surface_6の終点を評価面の始点に
VF.set_surface(evaluation_plane)  # 評価面をVFに登録
VF.raytrace_plane()  # 光線追跡
VF.plot_line_red()  # 光線描画

focal_length = VF.calc_focal_length(ray_start_pos_init)  # 計算した焦点距離を取得
print("焦点距離 = " + str(focal_length) + " mm")
