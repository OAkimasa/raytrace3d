import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

from raytrace3d import VectorFunctions
from zernike_fit import zernike_fitting

start = time.time()


N_air = 1.0
# N_Fused_Silica = 1.458  # Refractive index (Fused Silica at 589nm)
N_NSF6HT_C = 1.79608  # Refractive index (SF6HT at 656.3nm)
N_NSF6HT_d = 1.80518  # Refractive index (SF6HT at 587.6nm)
N_NSF6HT_F = 1.82783  # Refractive index (SF6HT at 486.1nm)
N_NSF6HT_list = [N_NSF6HT_C, N_NSF6HT_d, N_NSF6HT_F]
N_NBAF10_C = 1.66578  # Refractive index (N-BAF10 at 656.3nm)
N_NBAF10_d = 1.67003  # Refractive index (N-BAF10 at 587.6nm)
N_NBAF10_F = 1.68000  # Refractive index (N-BAF10 at 486.1nm)
N_NBAF10_list = [N_NBAF10_C, N_NBAF10_d, N_NBAF10_F]

wave_list = [656.3*1e-6, 587.6*1e-6, 486.1*1e-6]  # 波長 [mm]

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams["font.size"] = 12
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams["legend.fancybox"] = False
plt.rcParams["legend.framealpha"] = 1.0
plt.rcParams["legend.edgecolor"] = 'black'
plt.rcParams["legend.handlelength"] = 1.5
plt.rcParams["legend.labelspacing"] = 0.0
plt.rcParams["legend.handletextpad"] = 0.5
plt.rcParams["legend.markerscale"] = 1.0
fig = plt.figure(figsize=(8, 8))
# 余白の調整
fig.subplots_adjust(left=0.05, right=0.95, bottom=0.2, top=0.95)


# ---- 単位は mm ----


# https://www.thorlabs.co.jp/newgrouppage9.cfm?objectgroup_id=2696&pn=AC254-045-A-ML
# 通常の射出瞳を光源へ向ける (マウントのネジを光源へ向ける)
f_b = 40.2  # mm
R_1 = 31.2  # mm
R_2 = -25.9  # mm
R_3 = -130.6  # mm
t_c1 = 7.0  # mm
t_c2 = 2.0  # mm

# %% -------------- F2の光を直径9mm以下の平行光にする光学系 ----------------
ax = fig.add_subplot(221, projection='3d')
ax.set_title(
    "Converting F2 Light to ≤9mm Parallel Beam")
aspect_X = 3
aspect_Y = 1
aspect_Z = 1
ax.set_box_aspect((aspect_X, aspect_Y, aspect_Z))
scale = 70  # [mm]
box_center_X = -100
box_center_Y = 0
box_center_Z = 0
ax.set_xlim(-scale*aspect_X+box_center_X, scale*aspect_X+box_center_X)
ax.set_ylim(-scale*aspect_Y+box_center_Y, scale*aspect_Y+box_center_Y)
ax.set_zlim(-scale*aspect_Z+box_center_Z, scale*aspect_Z+box_center_Z)
ax.set_xlabel("[mm]")
ax.set_ylabel("[mm]")
ax.set_zlabel("[mm]")
ax.view_init(elev=0, azim=-90)

lens_pos = f_b
# param = [pos, nV, R, Lens_R(axis+-)]
lens_shift = np.array([0., 0., 0.])
surface_1 = [[lens_pos+0.00, 0., 0.]+lens_shift,
             [1., 0., 0.], 24/2, R_3]  # air->lens2
surface_2 = [[lens_pos+t_c2, 0., 0.]+lens_shift,
             [1., 0., 0.], 24/2, R_2]  # lens2->lens1
surface_3 = [[lens_pos+t_c1+t_c2, 0., 0.]+lens_shift,
             [1., 0., 0.], 24/2, R_1]  # lens1->air


evaluation_plane = [[200.000, 0., 0.], [1., 0., 0.], 24/2, 0.]  # air->air
check_plane = [[lens_pos, 0., 0.], [1., 0., 0.], 24/2., 0.]  # air->air
lens_list = [surface_1, surface_2, surface_3]
plane_list = [evaluation_plane, check_plane]


## インスタンス生成, 光線追跡開始
VF_C = VectorFunctions()
VF_d = VectorFunctions()
VF_F = VectorFunctions()
VF_list = [VF_C, VF_d, VF_F]
for VF in VF_list:
    VF.set_ax(ax)  # axをVFに登録


for lens in lens_list:
    VF.plot_lens(lens)  # レンズ描画

for plane in plane_list:
    VF.plot_plane(plane)  # 平面描画


focus_pos = np.array([0., 0., 0.])  # センサーの焦点位置
# focus_pos = np.array([18.80, -3.7, -3.7])  # センサーの焦点位置
focal_length = 70  # [mm] 焦点距離
F_number = 2.0  # F値

lens_unit_aperture_R = focal_length/F_number/2  # 絞り半径
print("lens_unit_aperture_R: {0:.3f} mm".format(lens_unit_aperture_R))

# 始点を生成する
width = lens_unit_aperture_R
space = lens_unit_aperture_R//4
rayDensity = 1
rayCenterX = -focal_length
rayCenterY = 0
rayCenterZ = 0
size = len(np.arange(-width+rayCenterY, width+rayCenterY+space, space))**2
pointsy, pointsz = np.meshgrid(
    np.arange(-width+rayCenterY, width+rayCenterY+space, space),
    np.arange(-width+rayCenterY, width+rayCenterY+space, space))
pointsX = np.array([rayCenterX]*size)
pointsY = pointsy.reshape(size)*rayDensity
pointsZ = pointsz.reshape(size)*rayDensity
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
for VF in VF_list:
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


def plot_line_CdF():
    VF_list[0].plot_line_red(alpha=0.1, ms=1)  # フラウンホーファーC線
    VF_list[1].plot_line_orange(alpha=0.1, ms=1)  # フラウンホーファーd線
    VF_list[2].plot_line_blue(alpha=0.1, ms=1)  # フラウンホーファーF線


# surface_1
for i, VF in enumerate(VF_list):
    VF.ray_start_pos = ray_start_pos_init  # 初期値
    VF.ray_start_dir = ray_start_dir_init  # 初期値
    VF.set_surface(surface_1,
                   refractive_index_before=N_air,
                   refractive_index_after=N_NSF6HT_list[i])  # surface_1をVFに登録
    VF.raytrace_sphere()  # 光線追跡
    VF.refract()  # 空気からレンズ1の屈折
plot_line_CdF()  # 光線描画

# surface_2
for i, VF in enumerate(VF_list):
    VF.ray_start_pos = VF.ray_end_pos  # surface_1の終点をsurface_2の始点に
    VF.ray_start_dir = VF.ray_end_dir  # surface_1の終点をsurface_2の始点に
    VF.set_surface(surface_2,
                   refractive_index_before=N_NSF6HT_list[i],
                   refractive_index_after=N_NBAF10_list[i])  # surface_2をVFに登録
    VF.raytrace_sphere()  # 光線追跡
    VF.refract()  # レンズ1から空気の屈折
plot_line_CdF()  # 光線描画

# surface_3
for i, VF in enumerate(VF_list):
    VF.ray_start_pos = VF.ray_end_pos  # surface_2の終点をsurface_3の始点に
    VF.ray_start_dir = VF.ray_end_dir  # surface_2の終点をsurface_3の始点に
    VF.set_surface(surface_3,
                   refractive_index_before=N_NBAF10_list[i],
                   refractive_index_after=N_air)  # surface_3をVFに登録
    VF.raytrace_sphere()  # 光線追跡
    VF.refract()  # 空気からレンズ2の屈折
plot_line_CdF()  # 光線描画

# evaluation plane
for i, VF in enumerate(VF_list):
    VF.ray_start_pos = VF.ray_end_pos  # surface_6の終点を評価面の始点に
    VF.ray_start_dir = VF.ray_end_dir  # surface_6の終点を評価面の始点に
    VF.set_surface(evaluation_plane)  # 評価面をVFに登録
    VF.raytrace_plane()  # 光線追跡
plot_line_CdF()  # 光線描画

for VF in VF_list:
    plane_X = np.mean(VF.ray_end_pos[:, 0])
    max_Y = np.max(VF.ray_end_pos[:, 1])
    max_Z = np.max(VF.ray_end_pos[:, 2])
    print("evaluation_plane: X, ray_diameter: {0:.3f}, {1:.3f}".format(
        plane_X, max_Y*2))

# print("run time: {0:.3f} sec".format(time.time() - start))
# plt.show()


# %% -------------- レンズの焦点距離を計算する ----------------
ax = fig.add_subplot(223, projection='3d')
ax.set_title("Calculating Focal Length of the Lens")
aspect_X = 2
aspect_Y = 1
aspect_Z = 1
ax.set_box_aspect((aspect_X, aspect_Y, aspect_Z))
scale = 20  # [mm]
box_center_X = 20
box_center_Y = 0
box_center_Z = 0
ax.set_xlim(-scale*aspect_X+box_center_X, scale*aspect_X+box_center_X)
ax.set_ylim(-scale*aspect_Y+box_center_Y, scale*aspect_Y+box_center_Y)
ax.set_zlim(-scale*aspect_Z+box_center_Z, scale*aspect_Z+box_center_Z)
ax.set_xlabel("[mm]")
ax.set_ylabel("[mm]")
ax.set_zlabel("[mm]")
ax.view_init(elev=0, azim=-90)

## インスタンス生成, 光線追跡開始
VF_C = VectorFunctions()
VF_d = VectorFunctions()
VF_F = VectorFunctions()
VF_list = [VF_C, VF_d, VF_F]
for VF in VF_list:
    VF.set_ax(ax)  # axをVFに登録

# param = [pos, nV, R, Lens_R(axis+-)]
# lens_pos = 0.  # best focus: width = 4.5/2
lens_pos = 0.165  # best focus: width = 25/2
mode = 1  # 運用時、0: 通常の射出瞳をセンサーへ向ける, 1: 通常の射出瞳を光源へ向ける
if mode:
    surface_1 = [[lens_pos+0.00, 0., 0.]+lens_shift,
                 [1., 0., 0.], 24/2, -R_1]  # air->lens1
    surface_2 = [[lens_pos+t_c1, 0., 0.]+lens_shift,
                 [1., 0., 0.], 24/2, -R_2]  # lens1->lens2
    surface_3 = [[lens_pos+t_c1+t_c2, 0., 0.]+lens_shift,
                 [1., 0., 0.], 24/2, -R_3]  # lens2->air
    evaluation_plane = [[f_b+t_c1+t_c2-0.128, 0., 0.],
                        [1., 0., 0.], 24, 0.]  # air->air
else:
    surface_1 = [[lens_pos+0.00, 0., 0.]+lens_shift,
                [1., 0., 0.], 24/2, R_3]  # air->lens2
    surface_2 = [[lens_pos+t_c2, 0., 0.]+lens_shift,
                [1., 0., 0.], 24/2, R_2]  # lens2->lens1
    surface_3 = [[lens_pos+t_c1+t_c2, 0., 0.]+lens_shift,
                [1., 0., 0.], 24/2, R_1]  # lens1->air
    evaluation_plane = [[35.902, 0., 0.],
                        [1., 0., 0.], 24, 0.]  # air->air
check_plane = [[f_b+t_c1+t_c2, 0., 0.], [1., 0., 0.], 24/2., 0.]  # air->air
lens_list = [surface_1, surface_2, surface_3]
plane_list = [evaluation_plane, check_plane]

for lens in lens_list:
    VF.plot_lens(lens)  # レンズ描画

for plane in plane_list:
    VF.plot_plane(plane)  # 平面描画

# 始点を生成する
# width = 4.5/2
width = 25/2
#space = 2
rayDensity = 1
rayCenterX = -100
rayCenterY = 0
rayCenterZ = 0
# size = len(np.arange(-width+rayCenterY, width+rayCenterY, space))**2
# pointsY, pointsZ = np.meshgrid(
#     np.arange(-width+rayCenterY, width+rayCenterY, space),
#     np.arange(-width+rayCenterY, width+rayCenterY, space))
# pointsX = np.array([rayCenterX]*size)
# pointsY = pointsY.reshape(size)*rayDensity
# pointsZ = pointsZ.reshape(size)*rayDensity
# raySPoint0 = VF.make_points(pointsX, pointsY, pointsZ, size, 3)
size = 20
pointsy, pointsz = np.meshgrid(
    np.linspace(-width+rayCenterY, width+rayCenterY, size),
    np.linspace(-width+rayCenterY, width+rayCenterY, size))
pointsX = np.array([rayCenterX]*size**2)
pointsY = pointsy.reshape(size**2)*rayDensity
pointsZ = pointsz.reshape(size**2)*rayDensity
raySPoint0 = VF.make_points(pointsX, pointsY, pointsZ, size**2, 3)

tmp_y_list = []
tmp_z_list = []
for y_row, z_row in zip(pointsy, pointsz):
    tmp_y_row = []
    tmp_z_row = []
    for y, z in zip(y_row, z_row):
        if np.sqrt(y**2 + z**2) <= width:
            tmp_y_row.append(y)
            tmp_z_row.append(z)
        else:
            tmp_y_row.append(np.nan)
            tmp_z_row.append(np.nan)
    tmp_y_list.append(tmp_y_row)
    tmp_z_list.append(tmp_z_row)
pointsy = np.array(tmp_y_list)
pointsz = np.array(tmp_z_list)
#print("pointsy: ", pointsy)

tmp_pos_list = []
for pos in raySPoint0:
    if np.sqrt(pos[1]**2 + pos[2]**2) <= width:
        tmp_pos_list.append(pos)
    else:
        continue
raySPoint0 = np.array(tmp_pos_list)
# print("raySPoint0: ", raySPoint0)

# 初期値
# ray_start_pos_init = np.array([-30.0, 3.0, 3.0])  # 初期値
ray_start_pos_init = raySPoint0  # 初期値
ray_start_dir_init = np.array([[1.0, 0.0, 0.0]]*len(raySPoint0))  # 初期値

# surface_1
for i, VF in enumerate(VF_list):
    VF.ray_start_pos = ray_start_pos_init  # 初期値
    VF.ray_start_dir = ray_start_dir_init  # 初期値
    VF.set_surface(surface_1,
                   refractive_index_before=N_air,
                   refractive_index_after=N_NBAF10_list[i])  # surface_1をVFに登録
    VF.raytrace_sphere()  # 光線追跡
    VF.refract()  # 空気からレンズ1の屈折
plot_line_CdF()  # 光線描画

# surface_2
for i, VF in enumerate(VF_list):
    VF.ray_start_pos = VF.ray_end_pos  # surface_1の終点をsurface_2の始点に
    VF.ray_start_dir = VF.ray_end_dir  # surface_1の終点をsurface_2の始点に
    VF.set_surface(surface_2,
                   refractive_index_before=N_NBAF10_list[i],
                   refractive_index_after=N_NSF6HT_list[i])  # surface_2をVFに登録
    VF.raytrace_sphere()  # 光線追跡
    VF.refract()  # レンズ1から空気の屈折
plot_line_CdF()  # 光線描画

# surface_3
for i, VF in enumerate(VF_list):
    VF.ray_start_pos = VF.ray_end_pos  # surface_2の終点をsurface_3の始点に
    VF.ray_start_dir = VF.ray_end_dir  # surface_2の終点をsurface_3の始点に
    VF.set_surface(surface_3,
                   refractive_index_before=N_NSF6HT_list[i],
                   refractive_index_after=N_air)  # surface_3をVFに登録
    VF.raytrace_sphere()  # 光線追跡
    VF.refract()  # 空気からレンズ2の屈折
plot_line_CdF()  # 光線描画

# evaluation plane
for i, VF in enumerate(VF_list):
    VF.ray_start_pos = VF.ray_end_pos  # surface_6の終点を評価面の始点に
    VF.ray_start_dir = VF.ray_end_dir  # surface_6の終点を評価面の始点に
    VF.set_surface(evaluation_plane)  # 評価面をVFに登録
    VF.raytrace_plane()  # 光線追跡
plot_line_CdF()  # 光線描画

for VF in VF_list:
    #print("VF: ", VF)
    focal_length = VF.calc_focal_length(ray_start_pos_init)  # 計算した焦点距離を取得
# print("焦点距離 = " + str(focal_length) + " mm")

# %% -------------- 波面収差の表示 ----------------
ax = fig.add_subplot(222, projection='3d')
ax.set_title("Wavefront Error")
aspect_X = 1
aspect_Y = 1
aspect_Z = 1
ax.set_box_aspect((aspect_X, aspect_Y, aspect_Z))
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_zlim(-1, 1)
ax.set_zlim(-15, 15)
ax.set_xlabel("Pupil X")
ax.set_ylabel("Pupil Y")
ax.set_zlabel("wavefront error [waves]")
ax.view_init(elev=0, azim=-65)

# OPDの取得
OPD_list = []
for VF in VF_list:
    OPD = VF.optical_path_length
    OPD_list.append(OPD)

chief_ray_OPD_list = []
for n_1, n_2 in zip(N_NBAF10_list, N_NSF6HT_list):
    chief_ray_OPD = abs(rayCenterX) + t_c1*n_1 + t_c2*n_2 + \
        abs(evaluation_plane[0][0]-surface_3[0][0])
    chief_ray_OPD_list.append(chief_ray_OPD)

pupil_radius_max = np.max(np.sqrt(raySPoint0[:, 1]**2 + raySPoint0[:, 2]**2))
print("pupil_radius_max: {0:.3f} mm".format(pupil_radius_max))

wavefront_func_list = []
for i, OPD in enumerate(OPD_list):
    # OPD = OPD - np.sqrt(raySPoint0[:, 0]**2 +
    #                     raySPoint0[:, 1]**2 +
    #                     raySPoint0[:, 2]**2)
    OPD = OPD - chief_ray_OPD_list[i]
    wavefront_func = OPD/wave_list[i]  # [waves]
    wavefront_func_list.append(wavefront_func)
    print("RMS wavefront error: {0:.3f} waves".format(
        np.sqrt(np.mean(wavefront_func**2))))

wavefront_list = []
for wavefront_func in wavefront_func_list:
    tmp_wavefront = []
    count = 0
    for y_row, z_row in zip(pointsy, pointsz):
        tmp_wavefront_row = []
        for y, z in zip(y_row, z_row):
            if np.sqrt(y**2 + z**2) <= width:
                tmp_wavefront_row.append(wavefront_func[count])
                count += 1
            else:
                tmp_wavefront_row.append(np.nan)
        tmp_wavefront.append(tmp_wavefront_row)
    wavefront = np.array(tmp_wavefront)
    #print("wavefront.shape: ", wavefront.shape)
    wavefront -= np.nanmean(wavefront)
    wavefront_list.append(wavefront)
wavefront_list = np.array(wavefront_list)
#print("wavefront_list.shape: ", wavefront_list.shape)

for i, wavefront_func in enumerate(wavefront_func_list):
    alpha = 0.1
    markersize = 1
    if i == 0:
        ax.plot(raySPoint0[:, 1]/pupil_radius_max, raySPoint0[:, 2]/pupil_radius_max,
                wavefront_func, "o", color="r", markersize=markersize, alpha=alpha)
        ax.plot_surface(pointsy/pupil_radius_max, pointsz/pupil_radius_max,
                        wavefront_list[i], color="r", alpha=alpha)
    elif i == 1:
        ax.plot(raySPoint0[:, 1]/pupil_radius_max, raySPoint0[:, 2]/pupil_radius_max,
                wavefront_func, "o", color="orange", markersize=markersize, alpha=alpha)
        ax.plot_surface(pointsy/pupil_radius_max, pointsz/pupil_radius_max,
                        wavefront_list[i], color="orange", alpha=alpha)
    elif i == 2:
        ax.plot(raySPoint0[:, 1]/pupil_radius_max, raySPoint0[:, 2]/pupil_radius_max,
                wavefront_func, "o", color="b", markersize=markersize, alpha=alpha)
        ax.plot_surface(pointsy/pupil_radius_max, pointsz/pupil_radius_max,
                        wavefront_list[i], color="b", alpha=alpha)


# %% -------------- Zernike fitting ----------------
ax = fig.add_subplot(224)
ax.set_title("Fringe Zernike Mode")
ax.set_ylabel("wavefront error [waves]")

fringe_zernike_mode_name = ["Piston",                           # fringe index: 1
                            "Tip",                              # fringe index: 2
                            "Tilt",                             # fringe index: 3
                            "Defocus",                          # fringe index: 4
                            "Vertical Astigmatism",             # fringe index: 5
                            "Oblique Astigmatism",              # fringe index: 6
                            "Horizontal Coma",                  # fringe index: 7
                            "Vertical Coma",                    # fringe index: 8
                            "Primary Spherical",                # fringe index: 9
                            "Oblique Trefoil",                  # fringe index: 10
                            "Vertical Trefoil",                 # fringe index: 11
                            "Vertical Secondary Astigmatism",   # fringe index: 12
                            "Oblique Secondary Astigmatism",    # fringe index: 13
                            "14",                               # fringe index: 14
                            "15",                               # fringe index: 15
                            "16",                               # fringe index: 16
                            "Vertical Quadrafoil",              # fringe index: 17
                            "Oblique Quadrafoil",               # fringe index: 18
                            "19",                               # fringe index: 19
                            "20",                               # fringe index: 20
                            "21",                               # fringe index: 21
                            "22",                               # fringe index: 22
                            "23",                               # fringe index: 23
                            "24",                               # fringe index: 24
                            "25",                               # fringe index: 25
                            "26",                               # fringe index: 26
                            "27",                               # fringe index: 27
                            "28",                               # fringe index: 28
                            "29",                               # fringe index: 29
                            "30",                               # fringe index: 30
                            "31",                               # fringe index: 31
                            "32",                               # fringe index: 32
                            "33",                               # fringe index: 33
                            "34",                               # fringe index: 34
                            "35",                               # fringe index: 35
                            "36",                               # fringe index: 36
                            "37",                               # fringe index: 37
                            ]
ax.set_xticks(np.arange(0, 37, 1),
              fringe_zernike_mode_name, rotation=-90, fontsize=8)
ax.grid()
ax.set_xlim(0, 21)
# ax.set_ylim(-0.5, 0.5)
ax.set_ylim(-15, 15)

for i, wavefront_func in enumerate(wavefront_func_list):
    alpha = 0.3
    markersize = 1
    if i == 0:
        zrn = zernike_fitting(raySPoint0[:, 1]/pupil_radius_max,
                              raySPoint0[:, 2]/pupil_radius_max,
                              wavefront_func)
        ax.plot(zrn, color="r", linewidth=3,
                linestyle="dotted", alpha=0.8, label="656.3 nm")
    elif i == 1:
        zrn = zernike_fitting(raySPoint0[:, 1]/pupil_radius_max,
                              raySPoint0[:, 2]/pupil_radius_max,
                              wavefront_func)
        ax.plot(zrn, color="orange", linewidth=1.5, alpha=1, label="587.6 nm")
        #print("zrn: ", zrn)
        print("zrn defocus: ", zrn[3], "waves")
        print("zrn Primary Spherical: ", zrn[8], "waves")
    elif i == 2:
        zrn = zernike_fitting(raySPoint0[:, 1]/pupil_radius_max,
                              raySPoint0[:, 2]/pupil_radius_max,
                              wavefront_func)
        ax.plot(zrn, color="b", linewidth=1,
                linestyle="dashed", alpha=0.8, label="486.1 nm")
        #print("len(zrn): ", len(zrn))
ax.legend()

# %% -------------- END ----------------
print("\nrun time: {0:.3f} sec".format(time.time() - start))

plt.show()
