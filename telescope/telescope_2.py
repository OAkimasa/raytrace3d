from weakref import ref
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.optimize as opt
from scipy.interpolate import griddata
from astropy.modeling.models import Gaussian2D
from astropy.convolution import convolve as astroconv
import os
import time
import tqdm

import raytrace3d as rt3d

start = time.time()
print("\nraytrace start")


def gauss(pos, a, x0, y0, sigma, offset):  # 同心円状に広がると仮定してsigmaは１つ
    x = pos[0]
    y = pos[1]
    xc = x - x0
    yc = y - y0
    r = np.sqrt(xc ** 2 + yc ** 2)
    g = a * np.exp(-r ** 2 / (2 * sigma ** 2)) + offset
    return g.ravel()


fig = plt.figure(figsize=(9, 9))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(-800+700, 700+800)
ax.set_ylim(-800, 800)
ax.set_zlim(-800, 800)
ax.view_init(elev=0, azim=-90)
ax.set_box_aspect((1, 1, 1))


# ---- 単位は mm ----
# インスタンス生成
VF_1 = rt3d.VectorFunctions()
VF_1.set_ax(ax)  # axをVF_1に登録, メインの光線追跡計算に使用
VF_2 = rt3d.VectorFunctions()
VF_2.set_ax(ax)  # axをVF_2に登録, 副鏡遮蔽の計算に使用
VF_3 = rt3d.VectorFunctions()
VF_3.set_ax(ax)  # axをVF_3に登録, スパイダー遮蔽の計算に使用
VF_4 = rt3d.VectorFunctions()
VF_4.set_ax(ax)  # axをVF_4に登録, 主鏡（瞳）遮蔽の計算に使用

ray_alpha = 0.1  # 光線の表示透明度

# param = [pos, nV, aperture_R, curvature_R(axis+-)]
shift_pos_primary_mirror = np.array([0., 0, 0])
shift_pos_secondary_mirror=np.array([0., 0, 0])
primary_mirror = [[900+shift_pos_primary_mirror[0], 0.+shift_pos_primary_mirror[1], 0.+shift_pos_primary_mirror[2]],
                    [1., 0., 0.],
                    203, 3200, -1.3402]
secondary_mirror = [[0.+shift_pos_secondary_mirror[0], 0.+shift_pos_secondary_mirror[1], 0.+shift_pos_secondary_mirror[2]],
                    [1, 0, 0],
                    95, 2599.1878, -14.9438]

# M3  45deg reflector
M3 = [[1530., 0., 0.], [np.cos(np.pi/4), 0., np.cos(np.pi/4)], 25.4, np.inf]

# AC508_080_AB_ML: f=80mm, d=50.8mm(2inch)
N_AC508_080_AB_ML_G1_550nm = 1.65361794505  # N-LAK22
N_AC508_080_AB_ML_G2_550nm = 1.81186626679  # N-SF6
collimate_lens_pos = np.array([1530., 0., 40.])
collimate_lens_1 = [collimate_lens_pos+np.array([0., 0., -72.73]), [0., 0., -1.], 25.4, 181.7]  # AC508_080_AB_ML G2b
collimate_lens_2 = [collimate_lens_pos+np.array([0., 0., -72.73-3]), [0., 0., -1.], 25.4, 80.6]  # AC508_080_AB_ML G1bG2f
collimate_lens_3 = [collimate_lens_pos+np.array([0., 0., -72.73-3-12]), [0., 0., -1.], 25.4, -63.6]  # AC508_080_AB_ML G1f

# M4  45deg reflector
M4 = [[1530., 0., -100.], [np.cos(np.pi/4), 0., -np.cos(np.pi/4)], 25.4, np.inf]

# AC508_080_AB_ML: f=80mm, d=50.8mm(2inch)
N_AC508_080_AB_ML_G1_550nm = 1.65361794505  # N-LAK22
N_AC508_080_AB_ML_G2_550nm = 1.81186626679  # N-SF6
field_lens_pos = np.array([1390., 0., -100.])
field_lens_1 = [field_lens_pos+np.array([3+12, 0., 0.]), [-1., 0., 0.], 25.4, +63.6]  # AC508_080_AB_ML G1f
field_lens_2 = [field_lens_pos+np.array([3, 0., 0.]), [-1., 0., 0.], 25.4, -80.6]  # AC508_080_AB_ML G1bG2f
field_lens_3 = [field_lens_pos+np.array([0., 0., 0]), [-1., 0., 0.], 25.4, -181.7]  # AC508_080_AB_ML G2b

N_air = 1.0

# 光線の終点に注目し、光学素子の中心からaperture_Rの範囲内にあるかどうかを判定する関数
def is_in_spider(ray_end_point):
    """
    光線がスパイダーにぶつかるかどうかを判定する。ぶつかるならばTrueを返す。

    Parameters
    ----------
    ray_end_point : ndarray
        光線の終点の座標を格納したndarray。

    Returns
    -------
    bool
        光線がスパイダーにぶつかるかどうか。
        光線の数だけbool値を格納したリストを返す。
    """

    # スパイダークロスの幅
    spider_width = 5  # mm
    # 光線の終点の座標がスパイダークロスの幅をもった十字に入っているかどうか判定
    bool_list = []
    for i in range(len(ray_end_point)):
        if np.abs(ray_end_point[i][1]) <= spider_width/2 or np.abs(ray_end_point[i][2]) <= spider_width/2:
            bool_list.append(True)
            # bool_list.append(1)
        else:
            bool_list.append(False)
            # bool_list.append(0)
    #print("spider bool_list", bool_list)
    # length_ray_end_point = len(ray_end_point)
    # sqrt = int(np.sqrt(length_ray_end_point))
    # test_img = np.array(bool_list).reshape(sqrt, sqrt)
    # #print("test_img", test_img)
    # fig = plt.figure(figsize=(9, 9))
    # plt.imshow(test_img)
    # plt.show()
    return bool_list

# 始点を生成する ===============================================================
# angle_incident = 0.  # deg
# angle_incident = 0.1  # deg
angle_incident = 0.2  # deg
# angle_incident = 0.25  # deg
width = 200
space = 20
rayDensity = 1
rayCenterX = -200
rayCenterY = 0
rayCenterZ = 0
size = len(np.arange(-width+rayCenterY, space+width+rayCenterZ, space))**2
pointsY, pointsZ = np.meshgrid(
    np.arange(-width+rayCenterY, space+width+rayCenterY, space),
    np.arange(-width+rayCenterZ, space+width+rayCenterZ, space))
pointsX = np.array([rayCenterX]*size)
pointsY = pointsY.reshape(size)*rayDensity
pointsZ = pointsZ.reshape(size)*rayDensity
raySPoints = VF_1.make_points(pointsX, pointsY, pointsZ, size, 3)
# =============================================================================

# 初期値生成 ===============================================================
# # 入射角1つを計算
# ray_start_dir_init = np.array([[np.cos(np.deg2rad(angle_incident)),
#                                 0.0,
#                                 np.sin(np.deg2rad(angle_incident))]]*len(raySPoints))  # 初期値
# 入射角3つを同時に計算
ray_start_dir_init_1 = [[1.0, 0.0, 0.0]]*len(raySPoints)  # 初期値
ray_start_dir_init_2 = [[np.cos(np.deg2rad(angle_incident)),
                                0.,
                                np.sin(np.deg2rad(angle_incident))]]*len(raySPoints)  # 初期値
ray_start_dir_init_3 = [[np.cos(np.deg2rad(angle_incident)),
                                0.,
                                np.sin(np.deg2rad(-angle_incident))]]*len(raySPoints)  # 初期値
# 結合
ray_start_dir_init = ray_start_dir_init_1 + ray_start_dir_init_2 + ray_start_dir_init_3
ray_start_dir_init = np.array(ray_start_dir_init)
raySPoints = np.array(list(raySPoints)+list(raySPoints)+list(raySPoints))

print("ray_start_dir_init.shape: ", ray_start_dir_init.shape)
print("raySPoints.shape: ", raySPoints.shape)
# =============================================================================

# 遮蔽計算 ===============================================================
# 副鏡遮蔽を計算
VF_2.ray_start_pos = raySPoints  # 初期値
VF_2.ray_start_dir = ray_start_dir_init  # 初期値
# secondary_mirrorを登録
VF_2.set_surface(secondary_mirror, surface_name='secondary_mirror')
VF_2.raytrace_aspherical()  # 光線追跡
mirror_shield_bool_list = VF_2._is_in_aperture(
    surface=secondary_mirror, ray_end_point=VF_2.ray_end_pos)  # 光線が絞りに入っているか判定
tmp_pos_list = []
tmp_dir_list = []
for i, bool in enumerate(mirror_shield_bool_list):
    if ~bool:
        tmp_pos_list.append(raySPoints[i])
        tmp_dir_list.append(ray_start_dir_init[i])
    else:
        continue
raySPoints = np.array(tmp_pos_list)
ray_start_dir_init = np.array(tmp_dir_list)

# スパイダー遮蔽を計算
VF_3.ray_start_pos = raySPoints  # 初期値
VF_3.ray_start_dir = ray_start_dir_init  # 初期値
# secondary_mirrorを登録
VF_3.set_surface(secondary_mirror, surface_name='secondary_mirror')
VF_3.raytrace_plane()  # 光線追跡
spider_shield_bool_list = is_in_spider(
    ray_end_point=VF_3.ray_end_pos)  # 光線がスパイダーに入っているか判定
tmp_pos_list = []
tmp_dir_list = []
for i, bool in enumerate(spider_shield_bool_list):
    if ~bool:
        tmp_pos_list.append(raySPoints[i])
        tmp_dir_list.append(ray_start_dir_init[i])
    else:
        continue
raySPoints = np.array(tmp_pos_list)
ray_start_dir_init = np.array(tmp_dir_list)

# 主鏡遮蔽を計算
VF_4.ray_start_pos = raySPoints  # 初期値
VF_4.ray_start_dir = ray_start_dir_init  # 初期値
# primary_mirror
VF_4.set_surface(primary_mirror, surface_name='primary_mirror')
VF_4.raytrace_plane()  # 光線追跡
primary_mirror_shield_bool_list = VF_4._is_in_aperture(
    surface=primary_mirror, ray_end_point=VF_4.ray_end_pos)  # 光線が絞りに入っているか判定
tmp_pos_list = []
tmp_dir_list = []
for i, bool in enumerate(primary_mirror_shield_bool_list):
    if bool:
        tmp_pos_list.append(raySPoints[i])
        tmp_dir_list.append(ray_start_dir_init[i])
    else:
        continue
raySPoints = np.array(tmp_pos_list)
ray_start_dir_init = np.array(tmp_dir_list)

ray_start_pos_init = raySPoints  # 初期値
ray_start_dir_init = ray_start_dir_init  # 初期値
# =============================================================================

def KGT40_mirror():
    # レンズ描画
    VF_1.plot_conic(primary_mirror)  # surface描画
    VF_1.plot_conic(secondary_mirror)  # surface描画

    # 光線追跡 ===============================================================
    # primary_mirror
    VF_1.ray_start_pos = ray_start_pos_init  # 初期値
    VF_1.ray_start_dir = ray_start_dir_init  # 初期値
    print("ray_start_dir.shape: ", VF_1.ray_start_dir.shape)
    # main_mirrorを登録
    VF_1.set_surface(primary_mirror, surface_name='primary_mirror')
    VF_1.raytrace_aspherical()  # 光線追跡
    VF_1.reflect()  # 反射
    VF_1.plot_line_red(alpha=ray_alpha)  # 光線描画

    # secondary_mirror
    VF_1.ray_start_pos = VF_1.ray_end_pos  # main_mirrorの終点をsecondary_mirrorの始点に
    VF_1.ray_start_dir = VF_1.ray_end_dir  # main_mirrorの終点をsecondary_mirrorの始点に
    # secondary_mirrorを登録
    VF_1.set_surface(secondary_mirror, surface_name='secondary_mirror')
    VF_1.raytrace_aspherical()  # 光線追跡
    VF_1.reflect()  # 反射
    VF_1.plot_line_red(alpha=ray_alpha)  # 光線描画

    # calculate focal length
    # print(VF_1.ray_end_dir)
    argmin_index = 1
    focal_length = []
    for i in range(len(VF_1.ray_end_dir)):
        tmp_V = -1. * \
            VF_1.ray_end_dir[i]*raySPoints[i][argmin_index] / \
            VF_1.ray_end_dir[i][argmin_index]
        argmax_index = 0
        focal_length.append(tmp_V[argmax_index])
    # 並び替え
    focal_length.sort()
    # print("focal_length = ", focal_length)
    # ignore nan
    focal_length_mean = np.nanmean(focal_length)
    focal_length_std = np.nanstd(focal_length)
    focal_length_max = np.nanmax(focal_length)
    focal_length_min = np.nanmin(focal_length)
    print("focal_length_mean: ", focal_length_mean,
            "std: ", focal_length_std,
            "max: ", focal_length_max, "min: ", focal_length_min)


# evaluate_plane, 焦点の前後で動かす
KGT40_mirror()  # 副鏡までの光線追跡
KGT40_last_ray_pos = VF_1.ray_end_pos  # secondary_mirrorの終点を保存
KGT40_last_ray_dir = VF_1.ray_end_dir  # secondary_mirrorの終点を保存
# 主焦点を計算
evaluate_plane = [[1498.75, 0., 0.], [1., 0., 0.], 100, np.inf]
VF_1.plot_plane(evaluate_plane)  # surface描画
# secondary_mirrorの終点をevaluate_planeの始点に
VF_1.ray_start_pos = KGT40_last_ray_pos
# secondary_mirrorの終点をevaluate_planeの始点に
VF_1.ray_start_dir = KGT40_last_ray_dir
# evaluate_planeを登録
VF_1.set_surface(evaluate_plane, surface_name='evaluate_plane')
VF_1.raytrace_plane()  # 光線追跡
VF_1.plot_line_red(alpha=ray_alpha)  # 光線描画

y_list = VF_1.ray_end_pos[:, 1]
z_list = VF_1.ray_end_pos[:, 2]
pos_RMS = np.sqrt(np.nanmean(y_list**2 + z_list**2))
print('pos_RMS = ', pos_RMS, 'mm')
print('pos_r_max = ', np.nanmax(np.sqrt(y_list**2 + z_list**2)), 'mm')
print('pos_x mean = ', np.nanmean(VF_1.ray_end_pos[:, 0]), 'mm')
print('pos_y mean = ', np.nanmean(VF_1.ray_end_pos[:, 1]), 'mm')
print('pos_z mean = ', np.nanmean(VF_1.ray_end_pos[:, 2]), 'mm')

# 光路差を計算
OPD = VF_1.optical_path_length - np.mean(VF_1.optical_path_length)  # mm
pupil_coord_1 = ray_start_pos_init[:, 1]/203
pupil_coord_2 = ray_start_pos_init[:, 2]/203
# # OPDのプロット
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.set_xlabel("puil_coord_1")
# ax.set_ylabel("puil_coord_2")
# ax.set_zlabel("OPD [mm]")
# ax.set_title("OPD")
# ax.scatter(pupil_coord_1, pupil_coord_2, OPD)

# PSFの計算
psf_nres = 100
base_grid = np.ones((psf_nres, psf_nres))
# 単位円の外は0にする
base_grid[np.sqrt((np.arange(psf_nres)-50)**2 +
                  (np.arange(psf_nres)[:, np.newaxis]-50)**2) > 50] = 0
# 副鏡遮蔽
base_grid[np.sqrt((np.arange(psf_nres)-50)**2 +
                  (np.arange(psf_nres)[:, np.newaxis]-50)**2) < 50*(95/203)] = 0
flux_map = base_grid.copy()

pupil_coord_1_grid, pupil_coord_2_grid = np.meshgrid(
    np.linspace(-1, 1, psf_nres), np.linspace(-1, 1, psf_nres))
OPD_grid = griddata((pupil_coord_1, pupil_coord_2), OPD, (pupil_coord_1_grid, pupil_coord_2_grid), method='cubic')
print("OPD RMS: {0:.5f} nm".format(np.sqrt(np.nanmean(OPD_grid**2))*1e6))
wavelength_mm = 550e-6  # 550nm
phase = 2*np.pi*OPD_grid/wavelength_mm
flux_map = base_grid*np.exp(1j*phase)
phase = np.where(np.isnan(phase), 0, phase)
flux_map = np.where(np.isnan(flux_map), 0, flux_map)

# 周辺を0で埋める 解像度を上げるため
pad_0 = 3
phase = np.pad(phase, [(psf_nres, psf_nres), (psf_nres, psf_nres)], 'constant')
flux_map = np.pad(flux_map, [(psf_nres, psf_nres), (psf_nres, psf_nres)], 'constant')

wavefront = flux_map*np.exp(1j*phase)
PSF = np.fft.fftshift(np.abs(np.fft.fft2(wavefront)**2))
print("PSF peak: {0:.5f}".format(np.max(PSF)))
pixel_scale = np.rad2deg(wavelength_mm/(2*203*pad_0))*3600  # arcsec/pixel
print("pixel_scale: {0:.3f} arcsec/pixel".format(pixel_scale))
extent = pixel_scale*pad_0*psf_nres/2

# PSFのプロット
fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(111)
ax.set_xlabel("x [arcsec]")
ax.set_ylabel("y [arcsec]")
ax.set_title("Prime Focus PSF 550nm")
mappable=ax.imshow(PSF, cmap="hot", extent=[-extent, extent, -extent, extent])
fig.colorbar(mappable, ax=ax)

y_list = VF_1.ray_end_pos[:, 1]
z_list = VF_1.ray_end_pos[:, 2]
pos_RMS = np.sqrt(np.nanmean(y_list**2 + z_list**2))
print('pos_RMS = ', pos_RMS, 'mm')
print('pos_r_max = ', np.nanmax(np.sqrt(y_list**2 + z_list**2)), 'mm')
# print('pos_x mean = ', np.nanmean(VF_1.ray_end_pos[:, 0]), 'mm')
# print('pos_y mean = ', np.nanmean(VF_1.ray_end_pos[:, 1]), 'mm')
# print('pos_z mean = ', np.nanmean(VF_1.ray_end_pos[:, 2]), 'mm')

# M3  45deg reflector
VF_1.plot_mirror(M3)  # surface描画
VF_1.ray_start_pos = KGT40_last_ray_pos
VF_1.ray_start_dir = KGT40_last_ray_dir
VF_1.set_surface(M3, surface_name='M3')
VF_1.raytrace_plane()  # 光線追跡
VF_1.reflect()  # 反射
VF_1.plot_line_red(alpha=ray_alpha)  # 光線描画

# collimate_lens_1
VF_1.plot_lens(collimate_lens_1)  # surface描画
VF_1.ray_start_pos = VF_1.ray_end_pos
VF_1.ray_start_dir = VF_1.ray_end_dir
VF_1.set_surface(collimate_lens_1,
                refractive_index_before=N_air,
                refractive_index_after=N_AC508_080_AB_ML_G2_550nm,
                surface_name='collimate_lens_1')
VF_1.raytrace_sphere()  # 光線追跡
VF_1.refract()  # 屈折
VF_1.plot_line_red(alpha=ray_alpha)  # 光線描画

# collimate_lens_2
VF_1.plot_lens(collimate_lens_2)  # surface描画
VF_1.ray_start_pos = VF_1.ray_end_pos
VF_1.ray_start_dir = VF_1.ray_end_dir
VF_1.set_surface(collimate_lens_2,
                refractive_index_before=N_AC508_080_AB_ML_G2_550nm,
                refractive_index_after=N_AC508_080_AB_ML_G1_550nm,
                surface_name='collimate_lens_2')
VF_1.raytrace_sphere()  # 光線追跡
VF_1.refract()  # 屈折
VF_1.plot_line_red(alpha=ray_alpha)  # 光線描画

# collimate_lens_3
VF_1.plot_lens(collimate_lens_3)  # surface描画
VF_1.ray_start_pos = VF_1.ray_end_pos
VF_1.ray_start_dir = VF_1.ray_end_dir
VF_1.set_surface(collimate_lens_3,
                refractive_index_before=N_AC508_080_AB_ML_G1_550nm,
                refractive_index_after=N_air,
                surface_name='collimate_lens_3')
VF_1.raytrace_sphere()  # 光線追跡
VF_1.refract()  # 屈折
VF_1.plot_line_red(alpha=ray_alpha)  # 光線描画

# M4  45deg reflector
VF_1.plot_mirror(M4)  # surface描画
VF_1.ray_start_pos = VF_1.ray_end_pos
VF_1.ray_start_dir = VF_1.ray_end_dir
VF_1.set_surface(M4, surface_name='M4')
VF_1.raytrace_plane()  # 光線追跡
VF_1.reflect()  # 反射
VF_1.plot_line_red(alpha=ray_alpha)  # 光線描画

# field_lens_1
VF_1.plot_lens(field_lens_1)  # surface描画
VF_1.ray_start_pos = VF_1.ray_end_pos
VF_1.ray_start_dir = VF_1.ray_end_dir
VF_1.set_surface(field_lens_1,
                refractive_index_before=N_air,
                refractive_index_after=N_AC508_080_AB_ML_G1_550nm,
                surface_name='field_lens_1')
VF_1.raytrace_sphere()  # 光線追跡
VF_1.refract()  # 屈折
VF_1.plot_line_red(alpha=ray_alpha)  # 光線描画

# field_lens_2
VF_1.plot_lens(field_lens_2)  # surface描画
VF_1.ray_start_pos = VF_1.ray_end_pos
VF_1.ray_start_dir = VF_1.ray_end_dir
VF_1.set_surface(field_lens_2,
                refractive_index_before=N_AC508_080_AB_ML_G1_550nm,
                refractive_index_after=N_AC508_080_AB_ML_G2_550nm,
                surface_name='field_lens_2')
VF_1.raytrace_sphere()  # 光線追跡
VF_1.refract()  # 屈折
VF_1.plot_line_red(alpha=ray_alpha)  # 光線描画

# field_lens_3
VF_1.plot_lens(field_lens_3)  # surface描画
VF_1.ray_start_pos = VF_1.ray_end_pos
VF_1.ray_start_dir = VF_1.ray_end_dir
VF_1.set_surface(field_lens_3,
                refractive_index_before=N_AC508_080_AB_ML_G2_550nm,
                refractive_index_after=N_air,
                surface_name='field_lens_3')
VF_1.raytrace_sphere()  # 光線追跡
VF_1.refract()  # 屈折
VF_1.plot_line_red(alpha=ray_alpha)  # 光線描画

evaluate_plane = [M4[0]+np.array([-500,0,0]), [1., 0., 0.], 100, np.inf]
VF_1.plot_plane(evaluate_plane)  # surface描画
VF_1.ray_start_pos = VF_1.ray_end_pos
VF_1.ray_start_dir = VF_1.ray_end_dir
VF_1.set_surface(evaluate_plane, surface_name='evaluate_plane')
VF_1.raytrace_plane()  # 光線追跡
VF_1.plot_line_red(alpha=ray_alpha)  # 光線描画

y_list = VF_1.ray_end_pos[:, 1]
z_list = VF_1.ray_end_pos[:, 2]
pos_RMS = np.sqrt(np.nanmean(y_list**2 + z_list**2))
print('pos_RMS = ', pos_RMS, 'mm')
print('pos_r_max = ', np.nanmax(np.sqrt(y_list**2 + z_list**2)), 'mm')
# print('pos_x mean = ', np.nanmean(VF_1.ray_end_pos[:, 0]), 'mm')
# print('pos_y mean = ', np.nanmean(VF_1.ray_end_pos[:, 1]), 'mm')
# print('pos_z mean = ', np.nanmean(VF_1.ray_end_pos[:, 2]), 'mm')


print("run time: {0:.3f} sec".format(time.time() - start))
plt.show()
