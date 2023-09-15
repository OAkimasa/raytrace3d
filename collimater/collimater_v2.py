import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

from raytrace3d import VectorFunctions

start = time.time()


N_air = 1.0
N_Fused_Silica = 1.458  # Refractive index (Fused Silica at 589nm)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
aspect_X = 3
aspect_Y = 1
aspect_Z = 1
ax.set_box_aspect((aspect_X, aspect_Y, aspect_Z))
scale = 70  # [mm]
box_center_X = -0
box_center_Y = 0
box_center_Z = 0
ax.set_xlim(-scale*aspect_X+box_center_X, scale*aspect_X+box_center_X)
ax.set_ylim(-scale*aspect_Y+box_center_Y, scale*aspect_Y+box_center_Y)
ax.set_zlim(-scale*aspect_Z+box_center_Z, scale*aspect_Z+box_center_Z)
ax.set_xlabel("[mm]")
ax.set_ylabel("[mm]")
ax.set_zlabel("[mm]")
# 余白の調整
fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)


# ---- 単位は mm ----

# param = [pos, nV, R, Lens_R(axis+-)]
lens_f = "50mm"
if lens_f == "35mm":
    lens_pos = 35.16931  # [mm] レンズの位置
    surface_1 = [[lens_pos+0.000, 0., 0.],
                 [1., 0., 0.], 25.4/2, -30.09]  # air->lens1
    surface_2 = [[lens_pos+5.000, 0., 0.],
                 [1., 0., 0.], 25.4/2, 35.00]  # lens1->air
elif lens_f == "40mm":
    lens_pos = 40.00000
    surface_1 = [[lens_pos+0.000, 0., 0.],
                 [1., 0., 0.], 25.4/2, -35.09]  # air->lens1
    surface_2 = [[lens_pos+5.000, 0., 0.],
                 [1., 0., 0.], 25.4/2, 40.00]  # lens1->air
elif lens_f == "45mm":
    lens_pos = 45.00000
    surface_1 = [[lens_pos+0.000, 0., 0.],
                 [1., 0., 0.], 25.4/2, -40.09]  # air->lens1
    surface_2 = [[lens_pos+5.000, 0., 0.],
                 [1., 0., 0.], 25.4/2, 45.00]  # lens1->air
elif lens_f == "50mm":
    lens_pos = 50.00000
    surface_1 = [[lens_pos+0.000, 0., 0.],
                 [1., 0., 0.], 25.4/2, -43.12]  # air->lens1
    surface_2 = [[lens_pos+5.000, 0., 0.],
                 [1., 0., 0.], 25.4/2, 50.00]  # lens1->air
evaluation_plane = [[200.000, 0., 0.], [1., 0., 0.], 10., 0.]  # air->air
# check_plane = [[35.000, 0., 0.], [1., 0., 0.], 30., 0.]  # air->air
check_plane = [[lens_pos, 0., 0.], [1., 0., 0.], 30., 0.]  # air->air
lens_list = [surface_1, surface_2]
plane_list = [evaluation_plane, check_plane]


## インスタンス生成, 光線追跡開始
VF = VectorFunctions()
VF.set_ax(ax)  # axをVFに登録


for lens in lens_list:
    VF.plot_lens(lens)  # レンズ描画

for plane in plane_list:
    VF.plot_plane(plane)  # 平面描画


focus_pos = np.array([0., 0., 0.])  # センサーの焦点位置
# focus_pos = np.array([18.80, -3.7, -3.7])  # センサーの焦点位置
focal_length = 300  # [mm] 焦点距離
F_number = 8.0  # F値

lens_unit_aperture_R = focal_length/F_number/2  # 絞り半径
print("lens_unit_aperture_R: {0:.3f} mm".format(lens_unit_aperture_R))

# 始点を生成する
width = lens_unit_aperture_R
space = lens_unit_aperture_R//4
rayDensity = 1
rayCenterX = -focal_length
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

bool_list = []
for i in range(len(raySPoint0)):  # 入射瞳に含まれるかどうかの判定
    if np.sqrt(raySPoint0[i][1]**2 + raySPoint0[i][2]**2) <= lens_unit_aperture_R:
        bool_list.append(True)
    else:
        bool_list.append(False)
tmp_list = []
for i, bool in enumerate(bool_list):  # 入射瞳に含まれない光線の初期値を削除
    if bool:
        tmp_list.append(raySPoint0[i])
    else:
        continue
raySPoint0 = np.array(tmp_list)

evaluate_plane_pos = np.array(evaluation_plane[0])  # 評価面の位置
# aperture_pos = evaluate_plane_pos - np.array([focal_length, 0., 0.])
aperture_pos = np.array([rayCenterX, rayCenterY, rayCenterZ])
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


# check plane
VF_check = VectorFunctions()
VF_check.set_ax(ax)  # axをVFに登録
VF_check.ray_start_pos = ray_start_pos_init  # 直前surfaceの終点を評価面の始点に
VF_check.ray_start_dir = ray_start_dir_init  # 直前surfaceの終点を評価面の始点に
VF_check.set_surface(check_plane)  # 評価面をVFに登録
VF_check.raytrace_plane()  # 光線追跡
ax.scatter(VF_check.ray_end_pos[:, 0], VF_check.ray_end_pos[:, 1],
           VF_check.ray_end_pos[:, 2], s=0.1, c='black', alpha=0.5)  # 光線描画
plane_X = np.mean(VF_check.ray_end_pos[:, 0])
max_Y = np.max(VF_check.ray_end_pos[:, 1])
max_Z = np.max(VF_check.ray_end_pos[:, 2])
print("\ncheck_plane: X, ray_diameter: {0:.3f}, {1:.3f}".format(
    plane_X, max_Y*2), "\n")

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

# evaluation plane
VF.ray_start_pos = VF.ray_end_pos  # surface_6の終点を評価面の始点に
VF.ray_start_dir = VF.ray_end_dir  # surface_6の終点を評価面の始点に
VF.set_surface(evaluation_plane)  # 評価面をVFに登録
VF.raytrace_plane()  # 光線追跡
VF.plot_line_red()  # 光線描画


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

# evaluation plane
VF.ray_start_pos = VF.ray_end_pos  # surface_6の終点を評価面の始点に
VF.ray_start_dir = VF.ray_end_dir  # surface_6の終点を評価面の始点に
VF.set_surface(evaluation_plane)  # 評価面をVFに登録
VF.raytrace_plane()  # 光線追跡
VF.plot_line_red()  # 光線描画

focal_length = VF.calc_focal_length(ray_start_pos_init)  # 計算した焦点距離を取得
# print("焦点距離 = " + str(focal_length) + " mm")
