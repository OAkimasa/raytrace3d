# %% -------------------- IMPORT --------------------
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

import raytrace3d as rt3d

# %% -------------------- RAYTRACE --------------------
# ---- raytrace start ----
start = time.time()
print("\nraytrace start")


N_air = 1.0

fig = plt.figure(figsize=(9, 9))
ax = fig.add_subplot(111, projection='3d')
ax.set_box_aspect((1, 1, 1))
ax.set_xlim(0, 50)
ax.set_ylim(-25, 25)
ax.set_zlim(-25, 25)


# ---- 単位は mm ----

# param = [pos, nV, R, Lens_R(axis+-)]
surface_1 = [[0., 0., 0.], [1., 0., 0.], 17.91, -47.8]  # air->lens1
surface_2 = [[7.56, 0., 0.], [1., 0., 0.], 17.91, 93.0]  # lens1->lens2
surface_3 = [[10.26, 0., 0.], [1., 0., 0.], 17.88, -262.5]  # lens2->air
surface_4 = [[18.86, 0., 0.], [1., 0., 0.], 12.60, 112.7]  # air->lens3
surface_5 = [[24.0, 0., 0.], [1., 0., 0.], 12.60, -43.0]  # lens3->air
surface_6 = [[36.76, 0., 0.], [1., 0., 0.], 13.95, 10000000]  # air->lens4
surface_7 = [[38.76, 0., 0.], [1., 0., 0.], 13.95, -65.8]  # lens4->lens5
surface_8 = [[44.56, 0., 0.], [1., 0., 0.], 13.95, 93.05]  # lens5->air
evaluation_plane = [[112.39, 0., 0.], [1., 0., 0.], 0., 0.]  # air->air
surface_list = [surface_1, surface_2, surface_3, surface_4,
                surface_5, surface_6, surface_7, surface_8, evaluation_plane]

# glass_list
N_lens_1 = 1.8
N_lens_2 = 1.6
N_lens_3 = 1.6
N_lens_4 = 1.5
N_lens_5 = 1.7
N_list = [N_air, N_lens_1, N_lens_2, N_air,
          N_lens_3, N_air, N_lens_4, N_lens_5, N_air]


# インスタンス生成
VF = rt3d.VectorFunctions()
VF.set_ax(ax)  # axをVFに登録


# レンズ描画
for surface in surface_list:
    VF.plot_lens(surface)  # レンズ描画


# 始点を生成する
width = 10
space = 1
rayDensity = 1
rayCenterX = -30
rayCenterY = 0
rayCenterZ = 0
size = len(np.arange(-width+rayCenterY, space+width+rayCenterY, space))**2
pointsy, pointsz = np.meshgrid(
    np.arange(-width+rayCenterY, space+width+rayCenterY, space),
    np.arange(-width+rayCenterY, space+width+rayCenterY, space))
pointsX = np.array([rayCenterX]*size)
pointsY = pointsy.reshape(size)*rayDensity
pointsZ = pointsz.reshape(size)*rayDensity
raySPoints = VF.make_points(pointsX, pointsY, pointsZ, size, 3)

# 光軸の光路長を計算
chief_ray_optical_path = 0
before_surface_pos = rayCenterX
chief_ray_optical_path_list = []
for i in range(len(surface_list)):
    chief_ray_optical_path += (surface_list[i]
                               [0][0] - before_surface_pos) * N_list[i]
    chief_ray_optical_path_list.append(chief_ray_optical_path)
    before_surface_pos = surface_list[i][0][0]
print("chief_ray_optical_path: ", chief_ray_optical_path_list)

# 光線追跡
# 初期値
# ray_start_pos_init = np.array([-30.0, 3.0, 3.0])  # 初期値
ray_start_pos_init = raySPoints  # 初期値
ray_start_dir_init = np.array([[1.0, 0.0, 0.0]]*len(raySPoints))  # 初期値

# surface_1
VF.ray_start_pos = ray_start_pos_init  # 初期値
VF.ray_start_dir = ray_start_dir_init  # 初期値
VF.set_surface(surface_1,
               refractive_index_before=N_air,
               refractive_index_after=N_lens_1,
               surface_name='surface_1')  # surface_1を登録
VF.raytrace_sphere()  # 光線追跡
VF.refract()  # 空気からレンズ1の屈折
VF.plot_line_red()  # 光線描画
VF.get_surface()  # surface_1の情報を取得

# surface_2
VF.ray_start_pos = VF.ray_end_pos  # surface_1の終点をsurface_2の始点に
VF.ray_start_dir = VF.ray_end_dir  # surface_1の終点をsurface_2の始点に
VF.set_surface(surface_2,
               refractive_index_before=N_lens_1,
               refractive_index_after=N_lens_2,
               surface_name='surface_2')  # surface_2を登録
VF.raytrace_sphere()  # 光線追跡
VF.refract()  # レンズ1からレンズ2の屈折
VF.plot_line_red()  # 光線描画
VF.get_surface()  # surface_2の情報を取得

# surface_3
VF.ray_start_pos = VF.ray_end_pos  # surface_2の終点をsurface_3の始点に
VF.ray_start_dir = VF.ray_end_dir  # surface_2の終点をsurface_3の始点に
VF.set_surface(surface_3,
               refractive_index_before=N_lens_2,
               refractive_index_after=N_air,
               surface_name='surface_3')  # surface_3を登録
VF.raytrace_sphere()  # 光線追跡
VF.refract()  # レンズ2から空気の屈折
VF.plot_line_red()  # 光線描画
VF.get_surface()  # surface_3の情報を取得

# surface_4
VF.ray_start_pos = VF.ray_end_pos  # surface_3の終点をsurface_4の始点に
VF.ray_start_dir = VF.ray_end_dir  # surface_3の終点をsurface_4の始点に
VF.set_surface(surface_4,
               refractive_index_before=N_air,
               refractive_index_after=N_lens_3,
               surface_name='surface_4')  # surface_4を登録
VF.raytrace_sphere()  # 光線追跡
VF.refract()  # 空気からレンズ3の屈折
VF.plot_line_red()  # 光線描画
VF.get_surface()  # surface_4の情報を取得

# surface_5
VF.ray_start_pos = VF.ray_end_pos  # surface_4の終点をsurface_5の始点に
VF.ray_start_dir = VF.ray_end_dir  # surface_4の終点をsurface_5の始点に
VF.set_surface(surface_5,
               refractive_index_before=N_lens_3,
               refractive_index_after=N_air,
               surface_name='surface_5')  # surface_5を登録
VF.raytrace_sphere()  # 光線追跡
VF.refract()  # レンズ3から空気の屈折
VF.plot_line_red()  # 光線描画
VF.get_surface()  # surface_5の情報を取得

# surface_6
VF.ray_start_pos = VF.ray_end_pos  # surface_5の終点をsurface_6の始点に
VF.ray_start_dir = VF.ray_end_dir  # surface_5の終点をsurface_6の始点に
VF.set_surface(surface_6,
               refractive_index_before=N_air,
               refractive_index_after=N_lens_4,
               surface_name='surface_6')  # surface_6を登録
VF.raytrace_sphere()  # 光線追跡
VF.refract()  # 空気からレンズ4の屈折
VF.plot_line_red()  # 光線描画
VF.get_surface()  # surface_6の情報を取得

# surface_7
VF.ray_start_pos = VF.ray_end_pos  # surface_6の終点をsurface_7の始点に
VF.ray_start_dir = VF.ray_end_dir  # surface_6の終点をsurface_7の始点に
VF.set_surface(surface_7,
               refractive_index_before=N_lens_4,
               refractive_index_after=N_lens_5,
               surface_name='surface_7')  # surface_7を登録
VF.raytrace_sphere()  # 光線追跡
VF.refract()  # レンズ4からレンズ5の屈折
VF.plot_line_red()  # 光線描画
VF.get_surface()  # surface_7の情報を取得

# surface_8
VF.ray_start_pos = VF.ray_end_pos  # surface_7の終点をsurface_8の始点に
VF.ray_start_dir = VF.ray_end_dir  # surface_7の終点をsurface_8の始点に
VF.set_surface(surface_8,
               refractive_index_before=N_lens_5,
               refractive_index_after=N_air,
               surface_name='surface_8')  # surface_8を登録
VF.raytrace_sphere()  # 光線追跡
VF.refract()  # レンズ5から空気の屈折
VF.plot_line_red()  # 光線描画
VF.get_surface()  # surface_8の情報を取得

# 焦点距離の計算
focal_length = VF.calc_focal_length(ray_start_pos_init)  # 計算した焦点距離を取得
print("焦点距離 = " + str(focal_length) + " mm")

# 焦点位置の計算
focal_pos = VF.calc_focal_pos(ray_start_pos_init)  # 計算した焦点位置を取得
print("焦点位置 = " + str(focal_pos) + " mm")

# evaluation plane
VF.ray_start_pos = VF.ray_end_pos  # surface_8の終点を評価面の始点に
VF.ray_start_dir = VF.ray_end_dir  # surface_8の終点を評価面の始点に
VF.set_surface(evaluation_plane)  # 評価面を登録
VF.raytrace_plane()  # 光線追跡
VF.refractive_index_before = N_air  # 入射側の屈折率
VF.refractive_index_after = N_air  # 出射側の屈折率
VF.plot_line_red()  # 光線描画
# plt.show()
plt.close()


# %% -------------------- OPD --------------------
# 光路長の計算
optical_path_length = VF.optical_path_length  # 計算した光路長を取得
#print("at evaluation plane: chief_ray_optical_path = " + str(chief_ray_optical_path_list[8]) + " mm")
#print("at evaluation plane: 光路長 = " + str(optical_path) + " mm")
#print("at evaluation plane: OPD = " + str(optical_path - chief_ray_optical_path_list[8]) + " mm")
fig_2 = plt.figure(figsize=(9, 9))
ax_2 = fig_2.add_subplot(111, projection="3d")

ax_2.view_init(elev=0, azim=45)
pupil_radius_max = np.max(np.sqrt(pointsy**2 + pointsz**2))
OPD = optical_path_length - chief_ray_optical_path_list[-1]  # [mm]
print("OPD max: ", np.max(OPD))
print("OPD min: ", np.min(OPD))
wave_length = 700*1e-6 # [mm]
wavefront_func = OPD/wave_length  # [waves]
wavefront_func_max = np.max(wavefront_func)
wavefront_func_min = np.min(wavefront_func)
print("wavefront_func max: ", wavefront_func_max)
print("wavefront_func min: ", wavefront_func_min)
ax_2.plot(raySPoints[:, 1]/pupil_radius_max, raySPoints[:, 2]/pupil_radius_max,
          wavefront_func, "o", color="b", markersize=3, alpha=0.5)
ax_2.plot_surface(pointsy/pupil_radius_max, pointsz/pupil_radius_max,
                    wavefront_func.reshape(pointsy.shape), cmap="jet", alpha=0.5)
cont = ax_2.contourf(pointsy/pupil_radius_max, pointsz/pupil_radius_max,
                wavefront_func.reshape(pointsy.shape),
                np.arange(wavefront_func_min,wavefront_func_max,1), cmap="jet", alpha=0.5, offset=0)
ax_2.clabel(cont, inline=True, fontsize=10, fmt='%1.1f')
plt.colorbar(cont)
ax_2.set_xlim(-1.1, 1.1)
ax_2.set_ylim(-1.1, 1.1)
ax_2.set_zlim(-4, 4)
ax_2.set_xlabel("pupil y [Rel. Units]")
ax_2.set_ylabel("pupil z [Rel. Units]")
ax_2.set_zlabel("Wavefront Function [waves]")

print("run time: {0:.3f} sec".format(time.time() - start))
plt.show()

# %%
