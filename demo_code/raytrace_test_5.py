import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

import raytrace3d as rt3d

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
surface_1 = [[50., 0., 0.], [1., 0., 0.], 17.91, -20, -0.3]
surface_2 = [[17.56, 0., 0.], [1., 0., 0.], 17.91, np.inf]
surface_list = [surface_1, surface_2]

"""# glass_list
N_lens_1 = 1.8
N_list = [N_air, N_lens_1, N_air]"""


# インスタンス生成
VF_1 = rt3d.VectorFunctions()  # conic
VF_1.set_ax(ax)  # axをVF_1に登録

VF_2 = rt3d.VectorFunctions()  # parabola
VF_2.set_ax(ax)  # axをVF_2に登録

VF_3 = rt3d.VectorFunctions()  # lens
VF_3.set_ax(ax)  # axをVF_3に登録


# レンズ描画
VF_1.plot_conic(surface_1)  # surface描画
VF_2.plot_parabola(surface_1)  # surface描画  比較!!!!!!!!!!!!!!!!
VF_3.plot_lens(surface_1)  # surface描画  比較!!!!!!!!!!!!!!!!
VF_1.plot_plane(surface_2)  # surface描画


# 始点を生成する
width = 10
space = 3
rayDensity = 1
rayCenterX = -30
rayCenterY = 0
rayCenterZ = 0
size = len(np.arange(-width+rayCenterY, space+width+rayCenterY, space))**2
pointsY, pointsZ = np.meshgrid(
    np.arange(-width+rayCenterY, space+width+rayCenterY, space),
    np.arange(-width+rayCenterY, space+width+rayCenterY, space))
pointsX = np.array([rayCenterX]*size)
pointsY = pointsY.reshape(size)*rayDensity
pointsZ = pointsZ.reshape(size)*rayDensity
raySPoints = VF_1.make_points(pointsX, pointsY, pointsZ, size, 3)


# 光線追跡
# 初期値
ray_start_pos_init = np.array([-30.0, 0.0, 15.0])  # 初期値
# ray_start_pos_init = np.array([-30.0, 0.0, 0.0])  # 初期値
# ray_start_dir_init = np.array([1.0, 0.0, 0.1])  # 初期値
ray_start_dir_init = np.array([1.0, 0.0, 0.0])  # 初期値
ray_start_dir_init = ray_start_dir_init / np.linalg.norm(ray_start_dir_init)
"""ray_start_pos_init = raySPoints  # 初期値
ray_start_dir_init = np.array([[1.0, 0.0, 0.0]]*len(raySPoints))  # 初期値"""

# surface_1
VF_1.ray_start_pos = ray_start_pos_init  # 初期値
VF_1.ray_start_dir = ray_start_dir_init  # 初期値
VF_2.ray_start_pos = ray_start_pos_init  # 初期値
VF_2.ray_start_dir = ray_start_dir_init  # 初期値
VF_3.ray_start_pos = ray_start_pos_init  # 初期値
VF_3.ray_start_dir = ray_start_dir_init  # 初期値
VF_1.set_surface(surface_1,
                 surface_name='surface_1')  # surface_1を登録
VF_2.set_surface(surface_1,
                 surface_name='surface_1')  # surface_1を登録
VF_3.set_surface(surface_1,
                 surface_name='surface_1')  # surface_1を登録
VF_1.raytrace_aspherical()  # 光線追跡
VF_2.raytrace_parabola()  # 光線追跡
VF_3.raytrace_sphere()  # 光線追跡
VF_1.reflect()  # surface_1の反射
VF_2.reflect()  # surface_1の反射
VF_3.reflect()  # surface_1の反射
print("VF_1 : ray_end_pos: ", VF_1.ray_end_pos)
print("VF_2 : ray_end_pos: ", VF_2.ray_end_pos)
print("VF_3 : ray_end_pos: ", VF_3.ray_end_pos)
VF_1.plot_line_red()  # 光線描画
VF_2.plot_line_green()  # 光線描画
VF_3.plot_line_blue()  # 光線描画

# surface_2  evalution_plane
VF_1.ray_start_pos = VF_1.ray_end_pos  # surface_1の終点をsurface_2の始点に
VF_1.ray_start_dir = VF_1.ray_end_dir  # surface_1の終点をsurface_2の始点に
VF_2.ray_start_pos = VF_2.ray_end_pos  # surface_1の終点をsurface_2の始点に
VF_2.ray_start_dir = VF_2.ray_end_dir  # surface_1の終点をsurface_2の始点に
VF_3.ray_start_pos = VF_3.ray_end_pos  # surface_1の終点をsurface_2の始点に
VF_3.ray_start_dir = VF_3.ray_end_dir  # surface_1の終点をsurface_2の始点に
VF_1.set_surface(surface_2,
                 surface_name='surface_2')  # surface_2を登録
VF_2.set_surface(surface_2,
                 surface_name='surface_2')  # surface_2を登録
VF_3.set_surface(surface_2,
                 surface_name='surface_2')  # surface_2を登録
VF_1.raytrace_plane()  # 光線追跡
VF_2.raytrace_plane()  # 光線追跡
VF_3.raytrace_plane()  # 光線追跡
VF_1.plot_line_red()  # 光線描画
VF_2.plot_line_green()  # 光線描画
VF_3.plot_line_blue()  # 光線描画


print("run time: {0:.3f} sec".format(time.time() - start))
plt.show()
