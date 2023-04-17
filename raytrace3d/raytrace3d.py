import numpy as np

"""光学計算や光学系の描画を行うクラス。"""


class VectorFunctions:
    """
    光学計算や光学系の描画を行うクラス。

    Attributes
    ----------
    _ax : Axes3D
        3次元グラフを描画するためのAxes3Dオブジェクト。
    ray_start_pos : list or ndarray
        光線の始点の位置ベクトル。
    ray_start_dir : list or ndarray
        光線の始点の方向ベクトル。
    ray_end_pos : list or ndarray
        光線の終点の位置ベクトル。
    ray_end_dir : list or ndarray
        光線の終点の方向ベクトル。
    _surface_pos : list or ndarray
        光学素子面中心の位置ベクトル。
    _limit_R : float
        光学素子の境界を制限する半径。
    _lens_or_parabola_R : float
        レンズの曲率半径またはパラボラの焦点距離。
    _refractive_index_before : float
        光線の入射側の屈折率。
    _refractive_index_after : float
        光線の出射側の屈折率。
    _refractive_index_calc_optical_path_length : float
        光路長計算用の屈折率。
    _normalV_optical_element : list or ndarray
        光学素子中心の法線ベクトル。
    _normalV_refract_or_reflect : list or ndarray
        屈折または反射計算に用いる法線ベクトル。
    optical_path_length : float
        光路長。
    """

    def __init__(self):
        self._ax = None  # plotline関数のため
        self.ray_start_pos = np.array([0, 0, 0])  # 光線の始点
        self.ray_start_dir = np.array([0, 0, 0])  # 光線の方向ベクトル
        self.ray_end_pos = np.array([0, 0, 0])  # 光線の終点
        self.ray_end_dir = np.array([0, 0, 0])  # 光線の方向ベクトル
        self._surface_pos = np.array([0, 0, 0])  # 光学素子面中心の位置ベクトル
        self._limit_R = 0  # 光学素子の境界を制限する半径
        self._lens_or_parabola_R = 0  # レンズの曲率半径またはパラボラの焦点距離
        self._conic_K = 0  # コーニック定数
        self._surface_name = 'surface'  # 光学素子の名前
        self._refractive_index_before = 1  # 光線の入射側の屈折率
        self._refractive_index_after = 1  # 光線の出射側の屈折率
        self._refractive_index_calc_optical_path_length = 1  # 光路長計算用の屈折率
        self._normalV_optical_element = np.array([0, 0, 0])  # 光学素子中心の法線ベクトル
        self._normalV_refract_or_reflect = np.array(
            [1, 0, 0])  # 屈折または反射計算に用いる法線ベクトル
        self.optical_path_length = 0  # 光路長

    def set_ax(self, ax):
        """
        描画関数のためのAxes3Dオブジェクトをセットする。

        Parameters
        ----------
        ax : Axes3D
            3次元グラフを描画するためのAxes3Dオブジェクト。

        Returns
        -------
        None
        """
        self._ax = ax

    def get_ax(self):
        """
        描画関数のためのAxes3Dオブジェクトを取得する。

        Parameters
        ----------
        None

        Returns
        -------
        Axes3D
            3次元グラフを描画するためのAxes3Dオブジェクト。
        """
        return self._ax

    # 光学素子描画関数
    # ミラー描画
    def plot_mirror(self, params, color='gray'):
        """
        ミラー(plane)を描画する。

        Parameters
        ----------
        params : list or ndarray
            光学素子(plane)のパラメータを格納したリストまたはndarray。
            [[pos_x, pos_y, pos_z], [normalV_x, normalV_y, normalV_z], R, np.inf]
        color : str
            光学素子の色。デフォルトは'gray'。

        Returns
        -------
        None
        """
        X_center = params[0][0]
        Y_center = params[0][1]
        Z_center = params[0][2]
        normalV = params[1]
        R = params[2]

        argmax_index = np.argmax(np.abs(params[1]))

        if argmax_index == 0:
            # 円盤生成
            geneNum = 200
            y = np.linspace(Y_center-R, Y_center+R, geneNum)
            z = np.linspace(Z_center-R, Z_center+R, geneNum)
            Y, Z = np.meshgrid(y, z)
            if normalV[0] == 0:
                X = X_center - (normalV[1]*(Y-Y_center) +
                                normalV[2]*(Z-Z_center)) / 0.01
            else:
                X = X_center - (normalV[1]*(Y-Y_center) +
                                normalV[2]*(Z-Z_center)) / normalV[0]
            for i in range(geneNum):
                for j in range(geneNum):
                    if (X[i][j]-X_center)**2 + (Y[i][j]-Y_center)**2 + (Z[i][j]-Z_center)**2 > R**2:
                        Z[i][j] = np.nan
            #self._ax.quiver(X_center, Y_center, Z_center, normalV[0], normalV[1], normalV[2], color='black', length=50)
            self._ax.plot_wireframe(X, Y, Z, color=color, linewidth=0.3)
        elif argmax_index == 1:
            # 円盤生成
            geneNum = 200
            x = np.linspace(X_center-R, X_center+R, geneNum)
            z = np.linspace(Z_center-R, Z_center+R, geneNum)
            X, Z = np.meshgrid(x, z)
            if normalV[1] == 0:
                Y = Y_center - (normalV[0]*(X-X_center) +
                                normalV[2]*(Z-Z_center)) / 0.01
            else:
                Y = Y_center - (normalV[0]*(X-X_center) +
                                normalV[2]*(Z-Z_center)) / normalV[1]
            for i in range(geneNum):
                for j in range(geneNum):
                    if (X[i][j]-X_center)**2 + (Y[i][j]-Y_center)**2 + (Z[i][j]-Z_center)**2 > R**2:
                        Z[i][j] = np.nan
            #self._ax.quiver(X_center, Y_center, Z_center, normalV[0], normalV[1], normalV[2], color='black', length=50)
            self._ax.plot_wireframe(X, Y, Z, color=color, linewidth=0.3)
        elif argmax_index == 2:
            # 円盤生成
            geneNum = 200
            x = np.linspace(X_center-R, X_center+R, geneNum)
            y = np.linspace(Y_center-R, Y_center+R, geneNum)
            X, Y = np.meshgrid(x, y)
            if normalV[2] == 0:
                Z = Z_center - (normalV[0]*(X-X_center) +
                                normalV[1]*(Y-Y_center)) / 0.01
            else:
                Z = Z_center - (normalV[0]*(X-X_center) +
                                normalV[1]*(Y-Y_center)) / normalV[2]
            for i in range(geneNum):
                for j in range(geneNum):
                    if (X[i][j]-X_center)**2 + (Y[i][j]-Y_center)**2 + (Z[i][j]-Z_center)**2 > R**2:
                        Z[i][j] = np.nan
            #self._ax.quiver(X_center, Y_center, Z_center, normalV[0], normalV[1], normalV[2], color='black', length=50)
            self._ax.plot_wireframe(X, Y, Z, color=color, linewidth=0.3)

    def plot_window(self, params):
        """
        ウィンドウ(plane)を描画する。関数"plot_mirror"の派生型。

        Parameters
        ----------
        params : list or ndarray
            光学素子(plane)のパラメータを格納したリストまたはndarray。
            [[pos_x, pos_y, pos_z], [normalV_x, normalV_y, normalV_z], R, np.inf]

        Returns
        -------
        None
        """
        X_center = params[0][0]
        Y_center = params[0][1]
        Z_center = params[0][2]
        normalV = params[1]
        R = params[2]

        argmax_index = np.argmax(np.abs(params[1]))

        if argmax_index == 0:
            # 円盤生成
            geneNum = 200
            y = np.linspace(Y_center-R, Y_center+R, geneNum)
            z = np.linspace(Z_center-R, Z_center+R, geneNum)
            Y, Z = np.meshgrid(y, z)
            if normalV[0] == 0:
                X = X_center - (normalV[1]*(Y-Y_center) +
                                normalV[2]*(Z-Z_center)) / 0.01
            else:
                X = X_center - (normalV[1]*(Y-Y_center) +
                                normalV[2]*(Z-Z_center)) / normalV[0]
            for i in range(geneNum):
                for j in range(geneNum):
                    if (X[i][j]-X_center)**2 + (Y[i][j]-Y_center)**2 + (Z[i][j]-Z_center)**2 > R**2:
                        Z[i][j] = np.nan
            #self._ax.quiver(X_center, Y_center, Z_center, normalV[0], normalV[1], normalV[2], color='black', length=50)
            self._ax.plot_wireframe(X, Y, Z, color='lightcyan', linewidth=0.3)
            theta = np.linspace(0, 2*np.pi, 100)
            x = np.zeros_like(theta) + X_center
            y = np.cos(theta)*R + Y_center
            z = np.sin(theta)*R + Z_center
            self._ax.plot(x, y, z, color='black', linewidth=0.3)
        elif argmax_index == 1:
            # 円盤生成
            geneNum = 200
            x = np.linspace(X_center-R, X_center+R, geneNum)
            z = np.linspace(Z_center-R, Z_center+R, geneNum)
            X, Z = np.meshgrid(x, z)
            if normalV[1] == 0:
                Y = Y_center - (normalV[0]*(X-X_center) +
                                normalV[2]*(Z-Z_center)) / 0.01
            else:
                Y = Y_center - (normalV[0]*(X-X_center) +
                                normalV[2]*(Z-Z_center)) / normalV[1]
            for i in range(geneNum):
                for j in range(geneNum):
                    if (X[i][j]-X_center)**2 + (Y[i][j]-Y_center)**2 + (Z[i][j]-Z_center)**2 > R**2:
                        Y[i][j] = np.nan
            #self._ax.quiver(X_center, Y_center, Z_center, normalV[0], normalV[1], normalV[2], color='black', length=50)
            self._ax.plot_wireframe(X, Y, Z, color='lightcyan', linewidth=0.3)
            theta = np.linspace(0, 2*np.pi, 100)
            x = R*np.cos(theta) + X_center
            y = np.zeros_like(theta) + Y_center
            z = R*np.sin(theta) + Z_center
            self._ax.plot(x, y, z, color='k', linewidth=0.3)
        elif argmax_index == 2:
            # 円盤生成
            geneNum = 200
            x = np.linspace(X_center-R, X_center+R, geneNum)
            y = np.linspace(Y_center-R, Y_center+R, geneNum)
            X, Y = np.meshgrid(x, y)
            if normalV[2] == 0:
                Z = Z_center - (normalV[0]*(X-X_center) +
                                normalV[1]*(Y-Y_center)) / 0.01
            else:
                Z = Z_center - (normalV[0]*(X-X_center) +
                                normalV[1]*(Y-Y_center)) / normalV[2]
            for i in range(geneNum):
                for j in range(geneNum):
                    if (X[i][j]-X_center)**2 + (Y[i][j]-Y_center)**2 + (Z[i][j]-Z_center)**2 > R**2:
                        Z[i][j] = np.nan
            #self._ax.quiver(X_center, Y_center, Z_center, normalV[0], normalV[1], normalV[2], color='black', length=50)
            self._ax.plot_wireframe(X, Y, Z, color='lightcyan', linewidth=0.3)
            theta = np.linspace(0, 2*np.pi, 100)
            x = R*np.cos(theta) + X_center
            y = R*np.sin(theta) + Y_center
            z = np.zeros_like(theta) + Z_center
            self._ax.plot(x, y, z, color='k', linewidth=0.3)

    def plot_plane(self, params):
        """
        ミラー(plane)を描画する。

        Parameters
        ----------
        params : list or ndarray
            光学素子(plane)のパラメータを格納したリストまたはndarray。
            [[pos_x, pos_y, pos_z], [normalV_x, normalV_y, normalV_z], R, np.inf]

        Returns
        -------
        None
        """
        X_center = params[0][0]
        Y_center = params[0][1]
        Z_center = params[0][2]
        normalV = params[1]
        R = params[2]

        argmax_index = np.argmax(np.abs(params[1]))

        if argmax_index == 0:
            # 円盤生成
            geneNum = 200
            y = np.linspace(Y_center-R, Y_center+R, geneNum)
            z = np.linspace(Z_center-R, Z_center+R, geneNum)
            Y, Z = np.meshgrid(y, z)
            if normalV[0] == 0:
                X = X_center - (normalV[1]*(Y-Y_center) +
                                normalV[2]*(Z-Z_center)) / 0.01
            else:
                X = X_center - (normalV[1]*(Y-Y_center) +
                                normalV[2]*(Z-Z_center)) / normalV[0]
            for i in range(geneNum):
                for j in range(geneNum):
                    if (X[i][j]-X_center)**2 + (Y[i][j]-Y_center)**2 + (Z[i][j]-Z_center)**2 > R**2:
                        Z[i][j] = np.nan
            #self._ax.quiver(X_center, Y_center, Z_center, normalV[0], normalV[1], normalV[2], color='black', length=50)
            self._ax.plot_wireframe(X, Y, Z, linewidth=0.3)
        elif argmax_index == 1:
            # 円盤生成
            geneNum = 200
            x = np.linspace(X_center-R, X_center+R, geneNum)
            z = np.linspace(Z_center-R, Z_center+R, geneNum)
            X, Z = np.meshgrid(x, z)
            if normalV[1] == 0:
                Y = Y_center - (normalV[0]*(X-X_center) +
                                normalV[2]*(Z-Z_center)) / 0.01
            else:
                Y = Y_center - (normalV[0]*(X-X_center) +
                                normalV[2]*(Z-Z_center)) / normalV[1]
            for i in range(geneNum):
                for j in range(geneNum):
                    if (X[i][j]-X_center)**2 + (Y[i][j]-Y_center)**2 + (Z[i][j]-Z_center)**2 > R**2:
                        Z[i][j] = np.nan
            #self._ax.quiver(X_center, Y_center, Z_center, normalV[0], normalV[1], normalV[2], color='black', length=50)
            self._ax.plot_wireframe(X, Y, Z, linewidth=0.3)
        elif argmax_index == 2:
            # 円盤生成
            geneNum = 200
            x = np.linspace(X_center-R, X_center+R, geneNum)
            y = np.linspace(Y_center-R, Y_center+R, geneNum)
            X, Y = np.meshgrid(x, y)
            if normalV[2] == 0:
                Z = Z_center - (normalV[0]*(X-X_center) +
                                normalV[1]*(Y-Y_center)) / 0.01
            else:
                Z = Z_center - (normalV[0]*(X-X_center) +
                                normalV[1]*(Y-Y_center)) / normalV[2]
            for i in range(geneNum):
                for j in range(geneNum):
                    if (X[i][j]-X_center)**2 + (Y[i][j]-Y_center)**2 + (Z[i][j]-Z_center)**2 > R**2:
                        Z[i][j] = np.nan
            #self._ax.quiver(X_center, Y_center, Z_center, normalV[0], normalV[1], normalV[2], color='black', length=50)
            self._ax.plot_wireframe(X, Y, Z, linewidth=0.3)

    # レンズ描画
    def plot_lens(self, params):
        """
        球面レンズを描画する。

        Parameters
        ----------
        params : list or ndarray
            光学素子(plane)のパラメータを格納したリストまたはndarray。
            [[pos_x, pos_y, pos_z], [normalV_x, normalV_y, normalV_z], R, Lens_R(axis+-)]

        Returns
        -------
        None
        """
        geneNum = 300
        limitTheta = 2*np.pi  # theta生成数
        limitPhi = np.pi  # phi生成数
        theta = np.linspace(0, limitTheta, geneNum)
        phi = np.linspace(0, limitPhi, geneNum)

        argmax_index = np.argmax(np.abs(params[1]))

        if argmax_index == 0:
            Ys = np.outer(np.sin(theta), np.sin(phi))
            Zs = np.outer(np.ones(np.size(theta)), np.cos(phi))
            Ys1 = params[2] * Ys
            Zs1 = params[2] * Zs
            if params[3] < 0:
                Xs1 = -(params[3]**2-Ys1**2-Zs1**2)**0.5 - params[3]
                self._ax.plot_wireframe(
                    Xs1+params[0][0], Ys1+params[0][1], Zs1+params[0][2], linewidth=0.1)
            elif params[3] > 0:
                Xs1 = (params[3]**2-Ys1**2-Zs1**2)**0.5 - params[3]
                self._ax.plot_wireframe(
                    Xs1+params[0][0], Ys1+params[0][1], Zs1+params[0][2], linewidth=0.1)
        elif argmax_index == 1:
            Xs = np.outer(np.sin(theta), np.sin(phi))
            Zs = np.outer(np.ones(np.size(theta)), np.cos(phi))
            Xs1 = params[2] * Xs
            Zs1 = params[2] * Zs
            if params[3] < 0:
                Ys1 = -(params[3]**2-Xs1**2-Zs1**2)**0.5 - params[3]
                self._ax.plot_wireframe(
                    Xs1+params[0][0], Ys1+params[0][1], Zs1+params[0][2], linewidth=0.1)
            elif params[3] > 0:
                Ys1 = (params[3]**2-Xs1**2-Zs1**2)**0.5 - params[3]
                self._ax.plot_wireframe(
                    Xs1+params[0][0], Ys1+params[0][1], Zs1+params[0][2], linewidth=0.1)
        elif argmax_index == 2:
            Xs = np.outer(np.sin(theta), np.sin(phi))
            Ys = np.outer(np.ones(np.size(theta)), np.cos(phi))
            Xs1 = params[2] * Xs
            Ys1 = params[2] * Ys
            if params[3] < 0:
                Zs1 = -(params[3]**2-Xs1**2-Ys1**2)**0.5 - params[3]
                self._ax.plot_wireframe(
                    Xs1+params[0][0], Ys1+params[0][1], Zs1+params[0][2], linewidth=0.1)
            elif params[3] > 0:
                Zs1 = (params[3]**2-Xs1**2-Ys1**2)**0.5 - params[3]
                self._ax.plot_wireframe(
                    Xs1+params[0][0], Ys1+params[0][1], Zs1+params[0][2], linewidth=0.1)

    # 放物線描画
    def plot_parabola(self, params):
        """
        回転放物面を描画する。

        Parameters
        ----------
        params : list or ndarray
            光学素子(plane)のパラメータを格納したリストまたはndarray。
            [[pos_x, pos_y, pos_z], [normalV_x, normalV_y, normalV_z], R, parabola_R(axis+-)]

        Returns
        -------
        None
        """
        theta = np.linspace(0, 2*np.pi, 100)
        R = params[2]
        a = abs(1/(2*params[3]))
        """if params[3] < 0:
            a = a
        else:
            a = -a"""
        max_index = self._max_index(params[1])
        if max_index == 0:  # x軸向き配置
            if params[3] < 0:
                y1 = R*np.cos(theta)
                z1 = R*np.sin(theta)
                Y1, Z1 = np.meshgrid(y1, z1)
                X1 = a*Y1**2 + a*Z1**2
                for i in range(100):
                    for j in range(100):
                        if (Y1[i][j])**2 + Z1[i][j]**2 > R**2:
                            X1[i][j] = np.nan
                        else:
                            X1[i][j] = a*Y1[i][j]**2 + a*Z1[i][j]**2
                self._ax.plot_wireframe(X1+params[0][0], Y1+params[0]
                                        [1], Z1+params[0][2], color='b', linewidth=0.1)
            elif params[3] > 0:
                y1 = R*np.cos(theta)
                z1 = R*np.sin(theta)
                Y1, Z1 = np.meshgrid(y1, z1)
                X1 = -(a*Y1**2 + a*Z1**2)
                for i in range(100):
                    for j in range(100):
                        if (Y1[i][j])**2 + Z1[i][j]**2 > R**2:
                            X1[i][j] = np.nan
                        else:
                            X1[i][j] = -(a*Y1[i][j]**2 + a*Z1[i][j]**2)
                self._ax.plot_wireframe(X1+params[0][0], Y1+params[0]
                                        [1], Z1+params[0][2], color='b', linewidth=0.1)
        elif max_index == 1:  # y軸向き配置
            if params[3] < 0:
                a = a
            else:
                a = -a
            if params[3] < 0:
                x1 = R*np.cos(theta)
                z1 = R*np.sin(theta)
                X1, Z1 = np.meshgrid(x1, z1)
                Y1 = a*X1**2 + a*Z1**2
                for i in range(100):
                    for j in range(100):
                        if (X1[i][j])**2 + Z1[i][j]**2 > R**2:
                            Y1[i][j] = np.nan
                        else:
                            Y1[i][j] = a*X1[i][j]**2 + a*Z1[i][j]**2
                self._ax.plot_wireframe(X1+params[0][0], Y1+params[0]
                                        [1], Z1+params[0][2], color='b', linewidth=0.1)
            elif params[3] > 0:
                x1 = R*np.cos(theta)
                z1 = R*np.sin(theta)
                X1, Z1 = np.meshgrid(x1, z1)
                Y1 = -(a*X1**2 + a*Z1**2)
                for i in range(100):
                    for j in range(100):
                        if (X1[i][j])**2 + Z1[i][j]**2 > R**2:
                            Y1[i][j] = np.nan
                        else:
                            Y1[i][j] = -(a*X1[i][j]**2 + a*Z1[i][j]**2)
                self._ax.plot_wireframe(X1+params[0][0], Y1+params[0]
                                        [1], Z1+params[0][2], color='b', linewidth=0.1)
        elif max_index == 2:  # z軸向き配置
            if params[3] < 0:
                x1 = R*np.cos(theta)
                y1 = R*np.sin(theta)
                X1, Y1 = np.meshgrid(x1, y1)
                Z1 = a*X1**2 + a*Y1**2
                for i in range(100):
                    for j in range(100):
                        if (X1[i][j])**2 + Y1[i][j]**2 > R**2:
                            Z1[i][j] = np.nan
                        else:
                            Z1[i][j] = a*X1[i][j]**2 + a*Y1[i][j]**2
                self._ax.plot_wireframe(X1+params[0][0], Y1+params[0]
                                        [1], Z1+params[0][2], color='b', linewidth=0.1)
            elif params[3] > 0:
                x1 = R*np.cos(theta)
                y1 = R*np.sin(theta)
                X1, Y1 = np.meshgrid(x1, y1)
                Z1 = -(a*X1**2 + a*Y1**2)
                for i in range(100):
                    for j in range(100):
                        if (X1[i][j])**2 + Y1[i][j]**2 > R**2:
                            Z1[i][j] = np.nan
                        else:
                            Z1[i][j] = -(a*X1[i][j]**2 + a*Y1[i][j]**2)
                self._ax.plot_wireframe(X1+params[0][0], Y1+params[0]
                                        [1], Z1+params[0][2], color='b', linewidth=0.1)

    # コーニック面描画
    def plot_conic(self, params):
        """
        コーニック面を描画する。

        Parameters
        ----------
        params : list or ndarray
            光学素子(plane)のパラメータを格納したリストまたはndarray。
            [[pos_x, pos_y, pos_z], [normalV_x, normalV_y, normalV_z], limit_R, conic_R(axis+-), conic_K]

        Returns
        -------
        None
        """
        theta = np.linspace(0, 2*np.pi, 100)
        R = params[2]
        conic_R = params[3]
        c = abs(1/conic_R)
        conic_K = params[4]
        tmp_index = self._max_index(params[1])
        if tmp_index == 0:  # x軸向き配置
            y1 = R*np.cos(theta)
            z1 = R*np.sin(theta)
            Y1, Z1 = np.meshgrid(y1, z1)
            if conic_R < 0:
                X1 = (c*(Y1**2+Z1**2)) / \
                    (1+np.sqrt(1-(1+conic_K)*(c**2)*(Y1**2+Z1**2)))
            elif conic_R > 0:
                X1 = -(c*(Y1**2+Z1**2)) / \
                    (1+np.sqrt(1-(1+conic_K)*(c**2)*(Y1**2+Z1**2)))
            for i in range(100):
                for j in range(100):
                    if (Y1[i][j])**2 + Z1[i][j]**2 > R**2:
                        X1[i][j] = np.nan
            self._ax.plot_wireframe(
                X1+params[0][0], Y1+params[0][1], Z1+params[0][2], color='purple', linewidth=0.1)
        elif tmp_index == 1:  # y軸向き配置
            x1 = R*np.cos(theta)
            z1 = R*np.sin(theta)
            X1, Z1 = np.meshgrid(x1, z1)
            if conic_R < 0:
                Y1 = (c*(X1**2+Z1**2)) / \
                    (1+np.sqrt(1-(1+conic_K)*(c**2)*(X1**2+Z1**2)))
            elif conic_R > 0:
                Y1 = -(c*(X1**2+Z1**2)) / \
                    (1+np.sqrt(1-(1+conic_K)*(c**2)*(X1**2+Z1**2)))
            for i in range(100):
                for j in range(100):
                    if (X1[i][j])**2 + Z1[i][j]**2 > R**2:
                        Y1[i][j] = np.nan
            self._ax.plot_wireframe(
                X1+params[0][0], Y1+params[0][1], Z1+params[0][2], color='purple', linewidth=0.1)
        elif tmp_index == 2:  # z軸向き配置
            x1 = R*np.cos(theta)
            y1 = R*np.sin(theta)
            X1, Y1 = np.meshgrid(x1, y1)
            if conic_R < 0:
                Z1 = (c*(X1**2+Y1**2)) / \
                    (1+np.sqrt(1-(1+conic_K)*(c**2)*(X1**2+Y1**2)))
            elif conic_R > 0:
                Z1 = -(c*(X1**2+Y1**2)) / \
                    (1+np.sqrt(1-(1+conic_K)*(c**2)*(X1**2+Y1**2)))
            for i in range(100):
                for j in range(100):
                    if (X1[i][j])**2 + Y1[i][j]**2 > R**2:
                        Z1[i][j] = np.nan
            self._ax.plot_wireframe(
                X1+params[0][0], Y1+params[0][1], Z1+params[0][2], color='purple', linewidth=0.1)

    # 光線計算
    # 受け取ったx,y,z座標から(x,y,z)の組を作る関数
    def make_points(self, point0, point1, point2, shape0, shape1=3):
        """
        受け取ったx,y,z座標から(x,y,z)の組を作る。

        Parameters
        ----------
        point0 : list or ndarray
            x座標を格納したリストまたはndarray。
        point1 : list or ndarray
            y座標を格納したリストまたはndarray。
        point2 : list or ndarray
            z座標を格納したリストまたはndarray。
        shape0 : int
            reshapeするときの第一引数。基本的にpoint0の長さを指定。
        shape1 : int
            reshapeするときの第二引数。基本的に3を指定(3次元)。

        Returns
        -------
        result : ndarray
            (x,y,z)の組を格納したndarray。shapeは(shape0, shape1)。
        """
        result = [None]*(len(point0)+len(point1)+len(point2))
        result[::3] = point0
        result[1::3] = point1
        result[2::3] = point2
        result = np.array(result)
        result = result.reshape(shape0, shape1)
        return result

    # ベクトルの最大成分のインデックスを返す関数
    def _max_index(self, vector):
        """
        べクトルの成分の絶対値を計算し、最大成分のインデックスを返す。

        Parameters
        ----------
        vector : list or ndarray
            ベクトルを格納したリストまたはndarray。

        Returns
        -------
        argmax_index : int
            最大成分のインデックス。
        """
        abs_vector = np.abs(vector)
        argmax_index = np.argmax(abs_vector)
        return argmax_index

    # ベクトルの最小成分のインデックスを返す関数
    def _min_index(self, vector):
        """
        べクトルの成分の絶対値を計算し、最小成分のインデックスを返す。

        Parameters
        ----------
        vector : list or ndarray
            ベクトルを格納したリストまたはndarray。

        Returns
        -------
        argmin_index : int
            最小成分のインデックス。
        """
        abs_vector = np.abs(vector)
        argmin_index = np.argmin(abs_vector)
        return argmin_index

    # 光学素子の面情報を登録する関数
    def set_surface(self, surface, refractive_index_before=1.0, refractive_index_after=1.0, surface_name='surface'):
        """
        登録関数。光学素子の面情報を登録する。

        Parameters
        ----------
        surface : list or ndarray
            光学素子のパラメータを格納したリストまたはndarray。
            [[pos_x, pos_y, pos_z], [normalV_x, normalV_y, normalV_z], limit_R, _lens_or_parabola_R]
        refractive_index_before : float
            光学素子の前の屈折率(入射側)。デフォルトは1.0。
        refractive_index_after : float
            光学素子の後の屈折率(出射側)。デフォルトは1.0。
        surface_name : str
            光学素子の名前。デフォルトは'surface'。

        Returns
        -------
        None
        """
        self._surface_pos = np.array(surface[0])  # [x, y, z]
        self._normalV_optical_element = np.array(
            surface[1])/np.linalg.norm(surface[1])  # [nV_x, nV_y, nV_z]
        self._normalV_refract_or_reflect = np.array(
            surface[1])/np.linalg.norm(surface[1])  # [nV_x, nV_y, nV_z]
        self._limit_R = surface[2]  # 光学素子の境界を制限する半径
        self._lens_or_parabola_R = surface[3]  # 曲率半径または焦点距離
        if len(surface) == 5:
            self._conic_K = surface[4]  # 光学素子のコーニック定数
        else:
            self._conic_K = 0.0  # 球面の場合は0.0
        if type(self._surface_name) == str:
            self._surface_name = surface_name
        else:
            print('set_surface : surface_name must be str.')
        # 光学素子の前の屈折率(入射側)
        self._refractive_index_calc_optical_path_length = self._refractive_index_after
        self._refractive_index_before = refractive_index_before
        self._refractive_index_after = refractive_index_after  # 光学素子の後の屈折率

    # 光学素子の面情報を読み出す関数
    def get_surface(self):
        """
        読み出し関数。光学素子の面情報を読み出す。

        Parameters
        ----------
        None

        Returns
        -------
        surface : list
            光学素子のパラメータを格納したリスト。
            [[pos_x, pos_y, pos_z], [normalV_x, normalV_y, normalV_z], _limit_R, _lens_or_parabola_R, _conic_K]
        surface_name : str
            光学素子の名前。
        refractive_index_before : float
            光学素子の前の屈折率(入射側)。
        refractive_index_after : float
            光学素子の後の屈折率(出射側)。
        """
        surface = [self._surface_pos, self._normalV_optical_element,
                   self._limit_R, self._lens_or_parabola_R, self._conic_K]
        print("\nsurface_name :", self._surface_name)
        print(
            "[[pos_x, pos_y, pos_z], [normalV_x, normalV_y, normalV_z], _limit_R, _lens_R(中心曲率半径), _conic_K]")
        print(surface)
        print("refraction_index :")
        print("before_refractive_index (incident)")
        print(self._refractive_index_before)
        print("after_refractive_index")
        print(self._refractive_index_after, "\n")
        return surface, self._surface_name, self._refractive_index_before, self._refractive_index_after

    # 平板のレイトレーシング
    def raytrace_plane(self):
        """
        set_surface()で登録した平面へレイトレーシングを行う。
        終点座標を計算し、self.ray_end_posに格納する。

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        centerV = self._surface_pos
        normalV = self._normalV_optical_element
        length_ray_start_dir = len(self.ray_start_dir)
        #print("length_ray_start_dir", length_ray_start_dir)
        if length_ray_start_dir == 3:
            nV = np.array(normalV)/np.linalg.norm(normalV)
            T = (np.dot(centerV, nV)-np.dot(self.ray_start_pos, nV)) / \
                (np.dot(self.ray_start_dir, nV))
            self.ray_end_pos = self.ray_start_pos + T*self.ray_start_dir
            #print("VF:shape(ray_end_pos)!!!!!!!!!!!!!!!!!!!!!2", np.shape(self.ray_end_pos))
            self.optical_path_length += T*self._refractive_index_calc_optical_path_length
            self._normalV_refract_or_reflect = nV
        else:  # 光線群の場合
            nV = np.array(normalV)/np.linalg.norm(normalV)
            T = []
            for i in range(length_ray_start_dir):
                if np.dot(self.ray_start_dir[i], np.array([1, 1, 1])) == 0:
                    T_tmp = 0
                    print("VF, raytrace_plane, T_tmp", T_tmp)
                    T.append(T_tmp)
                else:
                    T_tmp = (np.dot(centerV, nV)-np.dot(np.array(self.ray_start_pos[i]), nV)) / (
                        np.dot(np.array(self.ray_start_dir[i]), nV))
                    T.append(T_tmp)
            T = np.array(T)
            #T = [(np.dot(centerV, nV)-np.dot(self.ray_start_pos[i], nV)) / (np.dot(self.ray_start_dir[i], nV)) for i in range(length_ray_start_dir)]
            #print("VF:shape(ray_end_pos)!!!!!!!!!!!!!!!!!!!!!1", np.shape(self.ray_end_pos), np.shape(T), np.shape(self.ray_end_dir))
            self.ray_end_pos = self.ray_start_pos + \
                np.array([V*T for V, T in zip(self.ray_start_dir, T)])
            #print("VF:shape(ray_end_pos)!!!!!!!!!!!!!!!!!!!!!2", np.shape(self.ray_end_pos), np.shape(T), np.shape(self.ray_end_dir), np.shape([V*T for V, T in zip(self.ray_start_dir, T)]))
            self.optical_path_length += np.array(T) * \
                self._refractive_index_calc_optical_path_length
            self._normalV_refract_or_reflect = [nV]*length_ray_start_dir

    # 球面のレイトレーシング
    def raytrace_sphere(self):
        """
        set_surface()で登録した球面へレイトレーシングを行う。
        終点座標を計算し、self.ray_end_posに格納する。

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        lens_pos = self._surface_pos
        lens_R = self._lens_or_parabola_R
        length_ray_start_dir = len(self.ray_start_pos)
        if length_ray_start_dir == 3:
            tmp_V = np.zeros_like(self.ray_start_pos)
            tmp_index = self._max_index(self.ray_start_dir)
            tmp_V[tmp_index] = lens_R
            test_dot = np.dot(tmp_V, self.ray_start_dir)
            tmp_V = np.zeros_like(self.ray_start_pos)
            tmp_index = self._max_index(self.ray_start_dir)
            tmp_V[tmp_index] = lens_R
            shiftV = lens_pos - tmp_V
            #print("VFtest!!!!, shiftV =", shiftV)
            if test_dot > 0:  # 凹レンズ
                ray_pos = self.ray_start_pos - shiftV
                A = np.dot(self.ray_start_dir, self.ray_start_dir)
                B = np.dot(self.ray_start_dir, ray_pos)
                C = np.dot(ray_pos, ray_pos) - abs(lens_R)**2
                T = (-B + np.sqrt(B**2 - A*C)) / A
            elif test_dot < 0:  # 凸レンズ
                ray_pos = self.ray_start_pos - shiftV
                A = np.dot(self.ray_start_dir, self.ray_start_dir)
                B = np.dot(self.ray_start_dir, ray_pos)
                C = np.dot(ray_pos, ray_pos) - abs(lens_R)**2
                T = (-B - np.sqrt(B**2 - A*C)) / A
            else:
                T = np.nan
            self.ray_end_pos = self.ray_start_pos + T*self.ray_start_dir
            self.optical_path_length += np.array(T) * \
                self._refractive_index_calc_optical_path_length
            self._normalV_refract_or_reflect = self._calc_normalV_sphere()
        else:  # 光線群の場合
            tmp_V = np.zeros_like(self.ray_start_pos)
            tmp_index = self._max_index(self._normalV_optical_element)
            tmp_V[:, tmp_index] = lens_R
            test_dot = np.dot(tmp_V[0], self.ray_start_dir[0])
            tmp_V = np.zeros(3)
            tmp_index = self._max_index(self.ray_start_dir[0])
            tmp_V[tmp_index] = lens_R
            shiftV = lens_pos - tmp_V
            #print("VFtest!!!!, shiftV =", shiftV)
            if test_dot > 0:  # 凹レンズ
                ray_pos = self.ray_start_pos - \
                    np.array([shiftV]*length_ray_start_dir)
                A = np.diag(np.dot(self.ray_start_dir,
                                   np.array(self.ray_start_dir).T))
                B = np.diag(np.dot(self.ray_start_dir, ray_pos.T))
                C = np.diag(np.dot(ray_pos, ray_pos.T)) - abs(lens_R)**2
                T = []
                for i in range(length_ray_start_dir):
                    if np.dot(self.ray_start_dir[i], np.array([1, 1, 1])) == 0:
                        T_tmp = 0
                        T.append(T_tmp)
                    else:
                        T_tmp = (-B[i] + np.sqrt(B[i]**2 - A[i]*C[i])) / A[i]
                        T.append(T_tmp)
                T = np.array(T)
            elif test_dot < 0:  # 凸レンズ
                ray_pos = self.ray_start_pos - \
                    np.array([shiftV]*length_ray_start_dir)
                A = np.diag(np.dot(self.ray_start_dir,
                                   np.array(self.ray_start_dir).T))
                B = np.diag(np.dot(self.ray_start_dir, ray_pos.T))
                C = np.diag(np.dot(ray_pos, ray_pos.T)) - abs(lens_R)**2
                T = (-B - np.sqrt(B**2 - A*C)) / A
            else:
                T = np.zeros(length_ray_start_dir)
            self.ray_end_pos = self.ray_start_pos + \
                [V*T for V, T in zip(self.ray_start_dir, T)]
            self.optical_path_length += np.array(T) * \
                self._refractive_index_calc_optical_path_length
            self._normalV_refract_or_reflect = self._calc_normalV_sphere()

    # 球面の法線ベクトルを計算する関数
    def _calc_normalV_sphere(self):
        """
        球面レイトレース関数で用いる球面の法線ベクトルを計算する。

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        surface_pos = self._surface_pos
        lens_R = self._lens_or_parabola_R
        length_ray_start_dir = len(self.ray_start_pos)
        if length_ray_start_dir == 3:
            tmp_V = np.zeros(3)
            tmp_index = self._max_index(self.ray_start_dir)
            tmp_V[tmp_index] = lens_R
            normalV = self.ray_end_pos - surface_pos + tmp_V
            normalV = normalV / np.linalg.norm(normalV)
            print("normalV =", normalV)
            return np.array(normalV)
        else:  # 光線群の場合
            tmp_V = np.zeros(3)
            tmp_index = self._max_index(self.ray_start_dir[0])
            tmp_V[tmp_index] = lens_R
            normalV = []
            for i in range(length_ray_start_dir):
                tmp_normalV = self.ray_end_pos[i] - surface_pos + tmp_V
                normalV.append(tmp_normalV/np.linalg.norm(tmp_normalV))
            return np.array(normalV)

    # 放物線のレイトレーシング
    def raytrace_parabola(self):
        """
        set_surface()で登録した回転放物面へレイトレーシングを行う。
        終点座標を計算し、self.ray_end_posに格納する。

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        parabola_pos = self._surface_pos
        parabola_R = self._lens_or_parabola_R
        length_ray_start_dir = len(self.ray_start_pos)
        """tmp_V = np.zeros_like(self.ray_start_pos)
        tmp_index = self._max_index(self.ray_start_dir)
        tmp_V[tmp_index] = parabola_R
        test_dot = np.dot(tmp_V, self.ray_start_dir)
        print("test_dot =", test_dot)"""
        if length_ray_start_dir == 3:  # 光線1本の場合
            max_index = self._max_index(self._normalV_optical_element)
            ray_pos = self.ray_start_pos - parabola_pos
            ray_dir = self.ray_start_dir
            ray_dir = ray_dir / np.linalg.norm(ray_dir)
            tmp_V = np.zeros(3)
            tmp_V[max_index] = parabola_R
            test_dot = np.dot(self.ray_start_dir, tmp_V)
            a = 1/abs(2*parabola_R)
            if parabola_R < 0:
                a = a
            else:
                a = -a
            if max_index == 0:  # x軸向き配置
                if ray_dir[1] == 0 and ray_dir[2] == 0:  # x軸に平行な光線, たぶんOK
                    T = a*(ray_pos[1]**2 + ray_pos[2]**2) - ray_pos[0]
                    self.ray_end_pos = self.ray_start_pos + T*self.ray_start_dir
                    self.optical_path_length += np.array(T) * \
                        self._refractive_index_calc_optical_path_length
                    self._normalV_refract_or_reflect = self._calc_normalV_parabola()
                else:  # x軸に平行でない光線, たぶんOK
                    if test_dot < 0:  # 凸, たぶんOK
                        A = ray_dir[1]**2 + ray_dir[2]**2
                        B = ray_pos[1]*ray_dir[1] + ray_pos[2] * \
                            ray_dir[2] - ray_dir[0]/(2*a)
                        C = ray_pos[1]**2 + ray_pos[2]**2 - ray_pos[0]/a
                        T = (-B - np.sqrt(B**2 - A*C)) / A
                        self.ray_end_pos = self.ray_start_pos + T*self.ray_start_dir
                        self.optical_path_length += np.array(T) * \
                            self._refractive_index_calc_optical_path_length
                        self._normalV_refract_or_reflect = self._calc_normalV_parabola()
                    else:  # 凹, 実装中, たぶんOK
                        A = ray_dir[1]**2 + ray_dir[2]**2
                        B = ray_pos[1]*ray_dir[1] + ray_pos[2] * \
                            ray_dir[2] - ray_dir[0]/(2*a)
                        C = ray_pos[1]**2 + ray_pos[2]**2 - ray_pos[0]/a
                        T = (-B + np.sqrt(B**2 - A*C)) / A
                        self.ray_end_pos = self.ray_start_pos + T*self.ray_start_dir
                        self.optical_path_length += np.array(T) * \
                            self._refractive_index_calc_optical_path_length
                        self._normalV_refract_or_reflect = self._calc_normalV_parabola()
            elif max_index == 1:  # y軸向き配置
                if ray_dir[0] == 0 and ray_dir[2] == 0:  # y軸に平行な光線
                    T = a*(ray_pos[0]**2 - ray_pos[1] /
                           a + ray_pos[0]**2) / ray_dir[1]
                    self.ray_end_pos = self.ray_start_pos + T*self.ray_start_dir
                    self.optical_path_length += np.array(T) * \
                        self._refractive_index_calc_optical_path_length
                    self._normalV_refract_or_reflect = self._calc_normalV_parabola()
                else:  # y軸に平行でない光線
                    if test_dot < 0:  # 凸, たぶんOK
                        A = ray_dir[0]**2 + ray_dir[2]**2
                        B = ray_pos[0]*ray_dir[0] + ray_pos[2] * \
                            ray_dir[2] - ray_dir[1]/(2*a)
                        C = ray_pos[0]**2 + ray_pos[2]**2 - ray_pos[1]/a
                        T = (-B - np.sqrt(B**2 - A*C)) / A
                        self.ray_end_pos = self.ray_start_pos + T*self.ray_start_dir
                        self.optical_path_length += np.array(T) * \
                            self._refractive_index_calc_optical_path_length
                        self._normalV_refract_or_reflect = self._calc_normalV_parabola()
                    else:  # 凹, たぶんOK
                        A = ray_dir[0]**2 + ray_dir[2]**2
                        B = ray_pos[0]*ray_dir[0] + ray_pos[2] * \
                            ray_dir[2] - ray_dir[1]/(2*a)
                        C = ray_pos[0]**2 + ray_pos[2]**2 - ray_pos[1]/a
                        T = (-B + np.sqrt(B**2 - A*C)) / A
                        self.ray_end_pos = self.ray_start_pos + T*self.ray_start_dir
                        self.optical_path_length += np.array(T) * \
                            self._refractive_index_calc_optical_path_length
                        self._normalV_refract_or_reflect = self._calc_normalV_parabola()
            elif max_index == 2:  # z軸向き配置
                if ray_dir[0] == 0 and ray_dir[1] == 0:  # z軸に平行な光線
                    a = 1/abs(2*parabola_R)
                    if parabola_R < 0:
                        a = a
                    else:
                        a = -a
                    T = a*(ray_pos[0]**2 - ray_pos[2] /
                           a + ray_pos[0]**2) / ray_dir[2]
                    self.ray_end_pos = self.ray_start_pos + T*self.ray_start_dir
                    self.optical_path_length += np.array(T) * \
                        self._refractive_index_calc_optical_path_length
                    self._normalV_refract_or_reflect = self._calc_normalV_parabola()
                else:  # z軸に平行でない光線
                    if test_dot < 0:  # 凸, たぶんOK
                        A = ray_dir[0]**2 + ray_dir[1]**2
                        B = ray_pos[0]*ray_dir[0] + ray_pos[1] * \
                            ray_dir[1] - ray_dir[2]/(2*a)
                        C = ray_pos[0]**2 + ray_pos[1]**2 - ray_pos[2]/a
                        T = (-B - np.sqrt(B**2 - A*C)) / A
                        self.ray_end_pos = self.ray_start_pos + T*self.ray_start_dir
                        self.optical_path_length += np.array(T) * \
                            self._refractive_index_calc_optical_path_length
                        self._normalV_refract_or_reflect = self._calc_normalV_parabola()
                    else:  # 凹, たぶんOK
                        A = ray_dir[0]**2 + ray_dir[1]**2
                        B = ray_pos[0]*ray_dir[0] + ray_pos[1] * \
                            ray_dir[1] - ray_dir[2]/(2*a)
                        C = ray_pos[0]**2 + ray_pos[1]**2 - ray_pos[2]/a
                        T = (-B + np.sqrt(B**2 - A*C)) / A
                        self.ray_end_pos = self.ray_start_pos + T*self.ray_start_dir
                        self.optical_path_length += np.array(T) * \
                            self._refractive_index_calc_optical_path_length
                        self._normalV_refract_or_reflect = self._calc_normalV_parabola()
        else:  # 光線群の場合
            print("raytrace_parabola: インデックス自動化、x,z方向, 動作未確認 注意")
            max_index = self._max_index(self._normalV_optical_element)
            ray_pos = self.ray_start_pos - parabola_pos
            ray_dir = self.ray_start_dir
            length_ray_start_dir = len(self.ray_start_dir)
            tmp_V = np.zeros(3)
            tmp_V[max_index] = parabola_R
            test_dot = np.dot(
                self.ray_start_dir[int(length_ray_start_dir/2)], tmp_V)
            a = 1/abs(2*parabola_R)
            if parabola_R < 0:
                a = a
            else:
                a = -a
            if max_index == 0:  # x軸向き配置, 未確認注意
                print("ray_dir, 自動正規化 x,z向き, 未確認注意")
                if ray_dir[0][0] == 0 and ray_dir[0][2] == 0:  # x軸に平行な光線
                    T = [a*(ray_pos[i][1]**2 - ray_pos[i][0]/a + ray_pos[i][1]
                            ** 2) / ray_dir[i][0] for i in range(length_ray_start_dir)]
                    self.ray_end_pos = self.ray_start_pos + T*self.ray_start_dir
                    self.optical_path_length += np.array(T) * \
                        self._refractive_index_calc_optical_path_length
                    self._normalV_refract_or_reflect = self._calc_normalV_parabola()
                else:  # x軸に平行でない光線
                    if test_dot < 0:  # 凸
                        A = [ray_dir[i][1]**2 + ray_dir[i][2] ** 2
                             for i in range(length_ray_start_dir)]
                        B = [ray_pos[i][1]*ray_dir[i][1] + ray_pos[i][2] * ray_dir[i][2] - ray_dir[i][0]/(2*a)
                             for i in range(length_ray_start_dir)]
                        C = [ray_pos[i][1]**2 + ray_pos[i][2]**2 - ray_pos[i][0]/a
                             for i in range(length_ray_start_dir)]
                        T = [(-B[i] - np.sqrt(B[i]**2 - A[i]*C[i])) / A[i]
                             for i in range(length_ray_start_dir)]
                        self.ray_end_pos = self.ray_start_pos + T*self.ray_start_dir
                        self.optical_path_length += np.array(T) * \
                            self._refractive_index_calc_optical_path_length
                        self._normalV_refract_or_reflect = self._calc_normalV_parabola()
                    else:  # 凹
                        A = [ray_dir[i][1]**2 + ray_dir[i][2] ** 2
                             for i in range(length_ray_start_dir)]
                        B = [ray_pos[i][1]*ray_dir[i][1] + ray_pos[i][2] * ray_dir[i][2] - ray_dir[i][0]/(2*a)
                             for i in range(length_ray_start_dir)]
                        C = [ray_pos[i][1]**2 + ray_pos[i][2]**2 - ray_pos[i][0]/a
                             for i in range(length_ray_start_dir)]
            elif max_index == 1:  # y軸向き配置, たぶんOK
                if ray_dir[0][0] == 0 and ray_dir[0][2] == 0:  # y軸に平行な光線
                    T = [a*(ray_pos[i][0]**2 - ray_pos[i][1]/a + ray_pos[i][0]
                            ** 2) / ray_dir[i][1] for i in range(length_ray_start_dir)]
                    self.ray_end_pos = self.ray_start_pos + T*self.ray_start_dir
                    self.optical_path_length += np.array(T) * \
                        self._refractive_index_calc_optical_path_length
                    self._normalV_refract_or_reflect = self._calc_normalV_parabola()
                else:  # y軸に平行でない光線
                    if test_dot < 0:  # 凸
                        A = [ray_dir[i][0]**2 + ray_dir[i][2] **
                             2 for i in range(length_ray_start_dir)]
                        B = [ray_pos[i][0]*ray_dir[i][0] + ray_pos[i][2]*ray_dir[i]
                             [2] - ray_dir[i][1]/(2*a) for i in range(length_ray_start_dir)]
                        C = [ray_pos[i][0]**2 + ray_pos[i][2]**2 - ray_pos[i]
                             [1]/a for i in range(length_ray_start_dir)]
                        T = [(-B[i] - np.sqrt(B[i]**2 - A[i]*C[i])) / A[i]
                             for i in range(length_ray_start_dir)]
                        self.ray_end_pos = [
                            self.ray_start_pos[i] + T[i]*self.ray_start_dir[i] for i in range(length_ray_start_dir)]
                        self.optical_path_length += [np.array(
                            T[i])*self._refractive_index_calc_optical_path_length for i in range(length_ray_start_dir)]
                        self._normalV_refract_or_reflect = self._calc_normalV_parabola()
                    else:  # 凹
                        A = [ray_dir[i][0]**2 + ray_dir[i][2] **
                             2 for i in range(length_ray_start_dir)]
                        B = [ray_pos[i][0]*ray_dir[i][0] + ray_pos[i][2]*ray_dir[i]
                             [2] - ray_dir[i][1]/(2*a) for i in range(length_ray_start_dir)]
                        C = [ray_pos[i][0]**2 + ray_pos[i][2]**2 - ray_pos[i]
                             [1]/a for i in range(length_ray_start_dir)]
                        T = [(-B[i] + np.sqrt(B[i]**2 - A[i]*C[i])) / A[i]
                             for i in range(length_ray_start_dir)]
                        self.ray_end_pos = [
                            self.ray_start_pos[i] + T[i]*self.ray_start_dir[i] for i in range(length_ray_start_dir)]
                        self.optical_path_length += [np.array(
                            T[i])*self._refractive_index_calc_optical_path_length for i in range(length_ray_start_dir)]
                        self._normalV_refract_or_reflect = self._calc_normalV_parabola()
            elif max_index == 2:  # z軸向き配置, 未確認注意
                print("ray_dir, 自動正規化 x,z向き, 未確認注意")
                if ray_dir[0][0] == 0 and ray_dir[0][1] == 0:  # z軸に平行な光線
                    T = [a*(ray_pos[i][0]**2 - ray_pos[i][2]/a + ray_pos[i][0]
                            ** 2) / ray_dir[i][2] for i in range(length_ray_start_dir)]
                    self.ray_end_pos = self.ray_start_pos + T*self.ray_start_dir
                    self.optical_path_length += np.array(T) * \
                        self._refractive_index_calc_optical_path_length
                    self._normalV_refract_or_reflect = self._calc_normalV_parabola()
                else:  # z軸に平行でない光線, 未確認
                    if test_dot < 0:  # 凸
                        A = [ray_dir[i][0]**2 + ray_dir[i][1] **
                             2 for i in range(length_ray_start_dir)]
                        B = [ray_pos[i][0]*ray_dir[i][0] + ray_pos[i][1]*ray_dir[i]
                             [1] - ray_dir[i][2]/(2*a) for i in range(length_ray_start_dir)]
                        C = [ray_pos[i][0]**2 + ray_pos[i][1]**2 - ray_pos[i]
                             [2]/a for i in range(length_ray_start_dir)]
                        T = [(-B[i] - np.sqrt(B[i]**2 - A[i]*C[i])) / A[i]
                             for i in range(length_ray_start_dir)]
                        self.ray_end_pos = [
                            self.ray_start_pos[i] + T[i]*self.ray_start_dir[i] for i in range(length_ray_start_dir)]
                        self.optical_path_length += [np.array(
                            T[i])*self._refractive_index_calc_optical_path_length for i in range(length_ray_start_dir)]
                        self._normalV_refract_or_reflect = self._calc_normalV_parabola()
                    else:  # 凹
                        A = [ray_dir[i][0]**2 + ray_dir[i][1] **
                             2 for i in range(length_ray_start_dir)]
                        B = [ray_pos[i][0]*ray_dir[i][0] + ray_pos[i][1]*ray_dir[i]
                             [1] - ray_dir[i][2]/(2*a) for i in range(length_ray_start_dir)]
                        C = [ray_pos[i][0]**2 + ray_pos[i][1]**2 - ray_pos[i]
                             [2]/a for i in range(length_ray_start_dir)]
                        T = [(-B[i] + np.sqrt(B[i]**2 - A[i]*C[i])) / A[i]
                             for i in range(length_ray_start_dir)]
                        self.ray_end_pos = [
                            self.ray_start_pos[i] + T[i]*self.ray_start_dir[i] for i in range(length_ray_start_dir)]
                        self.optical_path_length += [np.array(
                            T[i])*self._refractive_index_calc_optical_path_length for i in range(length_ray_start_dir)]
                        self._normalV_refract_or_reflect = self._calc_normalV_parabola()

    # 放物線の法線ベクトルを計算する関数
    def _calc_normalV_parabola(self):
        """
        回転放物面レイトレース関数で用いる回転放物面の法線ベクトルを計算する。

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        parabola_pos = self._surface_pos
        parabola_R = self._lens_or_parabola_R
        length_ray_start_dir = len(self.ray_start_pos)
        a = abs(1/(2*parabola_R))
        if parabola_R < 0:
            a = a
        else:
            a = -a
        if length_ray_start_dir == 3:
            tmp_index = self._max_index(
                self._normalV_optical_element)  # 方向の計算に使う
            ray_pos = self.ray_end_pos
            normalV = 2*a*(ray_pos - parabola_pos)
            normalV[tmp_index] = -1
            normalV = normalV/np.linalg.norm(normalV)
            return normalV
        else:  # 光線群の場合
            tmp_index = self._max_index(
                self._normalV_optical_element)  # 方向の計算に使う
            ray_pos = self.ray_end_pos
            normalV = 2*a*(ray_pos - parabola_pos)
            normalV[:, tmp_index] = -1
            normalV = np.array([V/np.linalg.norm(V) for V in normalV])
            return normalV

    # 非球面のレイトレーシング
    def raytrace_aspherical(self):
        """
        set_surface()で設定した非球面面でのレイトレーシングを行う。
        終点の座標を計算し、self.ray_end_posに格納する。

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        print("非球面のレイトレーシングは実装中です。")
        if self._conic_K == -1.000:  # 放物線の場合
            self.raytrace_parabola()
        else:  # 放物線以外の非球面の場合, 光線１本, たぶんOK
            aspherical_pos = self._surface_pos
            aspherical_R = self._lens_or_parabola_R
            length_ray_start_dir = len(self.ray_start_pos)
            """tmp_V = np.zeros(3)
            tmp_index = self._max_index(self._normalV_optical_element)
            tmp_V[tmp_index] = aspherical_R
            print("tmp_V = ", tmp_V)"""
            if length_ray_start_dir == 3:
                print("self.ray_start_pos = ", self.ray_start_pos)
                print("aspherical_pos = ", aspherical_pos)
                ray_pos = np.array(self.ray_start_pos) - \
                    np.array(aspherical_pos)
                print("ray_pos = ", ray_pos)
                ray_dir = self.ray_start_dir
                if ray_dir[1] == 0 and ray_dir[2] == 0:  # x軸に平行な光線の場合, たぶんOK
                    print("x軸に平行な光線の場合 : インデックス自動化未実装")
                    if aspherical_R < 0:  # 凸面の場合, たぶんOK
                        print("凸面の場合, たぶんOK")
                        c = abs(1/aspherical_R)
                        conic_K = self._conic_K
                        T = -1*ray_pos[0] + (1-np.sqrt(1-(1+conic_K)*(c**2) *
                                                       ((ray_pos[1]**2)+(ray_pos[2]**2))))/(c*(1+self._conic_K))
                    elif aspherical_R > 0:  # 凹面の場合, たぶんOK
                        print("凹面の場合, たぶんOK")
                        c = abs(1/aspherical_R)
                        conic_K = self._conic_K
                        T = -1*ray_pos[0] + (1-np.sqrt(1-(1+conic_K)*(c**2) *
                                                       ((ray_pos[1]**2)+(ray_pos[2]**2))))/(-c*(1+self._conic_K))
                else:  # x軸に平行でない光線の場合, たぶんOK
                    print("x軸に平行でない光線の場合 : インデックス自動化未実装")
                    if aspherical_R < 0:  # 凸面の場合, たぶんOK
                        #print("凸面の場合, (1+K)が正の場合はOK")
                        c = abs(1/aspherical_R)  # abs()を取る必要があるかも
                        print("c = ", c)
                        conic_K = self._conic_K
                        print("conic_K = ", conic_K)
                        print("(1+conic_K)", (1+conic_K))
                        if 1+conic_K > 0:  # 1+Kが正の場合, たぶんOK
                            A_1 = (c**2)*((1+conic_K)**2)*(ray_dir[0]**2)
                            B_1 = 2*(c**2)*((1+conic_K)**2) * \
                                (ray_pos[0]*ray_dir[0]) - \
                                2*c*(1+conic_K)*ray_dir[0]
                            C_1 = (c**2)*((1+conic_K)**2) * \
                                (ray_pos[0]**2) - 2*c * \
                                (1+conic_K)*ray_pos[0] + 1
                            A_2 = -1*(c**2)*(1+conic_K) * \
                                ((ray_dir[1]**2)+(ray_dir[2]**2))
                            B_2 = -2*(c**2)*(1+conic_K) * \
                                (ray_pos[1]*ray_dir[1] + ray_pos[2]*ray_dir[2])
                            C_2 = 1 - (1+conic_K)*(c**2) * \
                                ((ray_pos[1]**2)+(ray_pos[2]**2))
                            A = A_1-A_2
                            B = B_1-B_2
                            C = C_1-C_2
                            T = (-B - np.sqrt(B**2 - 4*A*C)) / (2*A)
                        elif 1+conic_K < 0:  # コーニック定数が負の場合, たぶんOK
                            A_1 = (c**2)*((1+conic_K)**2)*(ray_dir[0]**2)
                            B_1 = 2*(c**2)*((1+conic_K)**2) * \
                                (ray_pos[0]*ray_dir[0]) - \
                                2*c*abs(1+conic_K)*ray_dir[0]
                            C_1 = (c**2)*((1+conic_K)**2) * \
                                (ray_pos[0]**2) - 2*c * \
                                abs(1+conic_K)*ray_pos[0] + 1
                            A_2 = -1*(c**2)*abs(1+conic_K) * \
                                ((ray_dir[1]**2)+(ray_dir[2]**2))
                            B_2 = -2*(c**2)*abs(1+conic_K) * \
                                (ray_pos[1]*ray_dir[1] + ray_pos[2]*ray_dir[2])
                            C_2 = 1 - abs(1+conic_K)*(c**2) * \
                                ((ray_pos[1]**2)+(ray_pos[2]**2))
                            A = A_1-A_2
                            B = B_1-B_2
                            C = C_1-C_2
                            T = (-B - np.sqrt(B**2 - 4*A*C)) / (2*A)
                    elif aspherical_R > 0:  # 凹面の場合, たぶんOK
                        c = -abs(1/aspherical_R)  # abs()を取る必要があるかも
                        conic_K = self._conic_K
                        if 1+conic_K > 0:  # コーニック定数が正の場合, たぶんOK
                            print("たぶんOK")
                            A_1 = (c**2)*((1+conic_K)**2)*(ray_dir[0]**2)
                            B_1 = 2*(c**2)*((1+conic_K)**2) * \
                                (ray_pos[0]*ray_dir[0]) - \
                                2*c*(1+conic_K)*ray_dir[0]
                            C_1 = (c**2)*((1+conic_K)**2) * \
                                (ray_pos[0]**2) - 2*c * \
                                (1+conic_K)*ray_pos[0] + 1
                            A_2 = -1*(c**2)*(1+conic_K) * \
                                ((ray_dir[1]**2)+(ray_dir[2]**2))
                            B_2 = -2*(c**2)*(1+conic_K) * \
                                (ray_pos[1]*ray_dir[1] + ray_pos[2]*ray_dir[2])
                            C_2 = 1 - (1+conic_K)*(c**2) * \
                                ((ray_pos[1]**2)+(ray_pos[2]**2))
                            A = A_1-A_2
                            B = B_1-B_2
                            C = C_1-C_2
                            T = (-B + np.sqrt(B**2 - 4*A*C)) / (2*A)
                        elif 1+conic_K < 0:  # コーニック定数が負の場合, たぶんOK
                            A_1 = (c**2)*((1+conic_K)**2)*(ray_dir[0]**2)
                            B_1 = 2*(c**2)*((1+conic_K)**2) * \
                                (ray_pos[0]*ray_dir[0]) - \
                                2*c*abs(1+conic_K)*ray_dir[0]
                            C_1 = (c**2)*((1+conic_K)**2) * \
                                (ray_pos[0]**2) - 2*c * \
                                abs(1+conic_K)*ray_pos[0] + 1
                            A_2 = -1*(c**2)*abs(1+conic_K) * \
                                ((ray_dir[1]**2)+(ray_dir[2]**2))
                            B_2 = -2*(c**2)*abs(1+conic_K) * \
                                (ray_pos[1]*ray_dir[1] + ray_pos[2]*ray_dir[2])
                            C_2 = 1 - abs(1+conic_K)*(c**2) * \
                                ((ray_pos[1]**2)+(ray_pos[2]**2))
                            A = A_1-A_2
                            B = B_1-B_2
                            C = C_1-C_2
                            T = (-B + np.sqrt(B**2 - 4*A*C)) / (2*A)
                self.ray_end_pos = self.ray_start_pos + T*self.ray_start_dir
                self.optical_path_length += np.array(
                    T)*self._refractive_index_calc_optical_path_length
                self._normalV_refract_or_reflect = self._calc_normalV_aspherical()
            else:  # 光線群の場合
                print("raytrace_aspherical : 光線群の場合は未実装です。")
                print("self.ray_start_pos = ", self.ray_start_pos)
                print("aspherical_pos = ", aspherical_pos)
                ray_pos = np.array(self.ray_start_pos) - \
                    np.array(aspherical_pos)
                print("ray_pos = ", ray_pos)
                ray_dir = self.ray_start_dir
                if ray_dir[1] == 0 and ray_dir[2] == 0:  # x軸に平行な光線の場合, たぶんOK
                    print("x軸に平行な光線の場合 : インデックス自動化未実装")
                    if aspherical_R < 0:  # 凸面の場合, たぶんOK
                        print("凸面の場合, たぶんOK")
                        c = abs(1/aspherical_R)
                        conic_K = self._conic_K
                        T = -1*ray_pos[0] + (1-np.sqrt(1-(1+conic_K)*(c**2) *
                                                       ((ray_pos[1]**2)+(ray_pos[2]**2))))/(c*(1+self._conic_K))
                    elif aspherical_R > 0:  # 凹面の場合, たぶんOK
                        print("凹面の場合, たぶんOK")
                        c = abs(1/aspherical_R)
                        conic_K = self._conic_K
                        T = -1*ray_pos[0] + (1-np.sqrt(1-(1+conic_K)*(c**2) *
                                                       ((ray_pos[1]**2)+(ray_pos[2]**2))))/(-c*(1+self._conic_K))
                else:  # x軸に平行でない光線の場合, たぶんOK
                    print("x軸に平行でない光線の場合 : インデックス自動化未実装")
                    if aspherical_R < 0:  # 凸面の場合, たぶんOK
                        #print("凸面の場合, (1+K)が正の場合はOK")
                        c = abs(1/aspherical_R)  # abs()を取る必要があるかも
                        print("c = ", c)
                        conic_K = self._conic_K
                        print("conic_K = ", conic_K)
                        print("(1+conic_K)", (1+conic_K))
                        if 1+conic_K > 0:  # 1+Kが正の場合, たぶんOK
                            A_1 = (c**2)*((1+conic_K)**2)*(ray_dir[0]**2)
                            B_1 = 2*(c**2)*((1+conic_K)**2) * \
                                (ray_pos[0]*ray_dir[0]) - \
                                2*c*(1+conic_K)*ray_dir[0]
                            C_1 = (c**2)*((1+conic_K)**2) * \
                                (ray_pos[0]**2) - 2*c * \
                                (1+conic_K)*ray_pos[0] + 1
                            A_2 = -1*(c**2)*(1+conic_K) * \
                                ((ray_dir[1]**2)+(ray_dir[2]**2))
                            B_2 = -2*(c**2)*(1+conic_K) * \
                                (ray_pos[1]*ray_dir[1] + ray_pos[2]*ray_dir[2])
                            C_2 = 1 - (1+conic_K)*(c**2) * \
                                ((ray_pos[1]**2)+(ray_pos[2]**2))
                            A = A_1-A_2
                            B = B_1-B_2
                            C = C_1-C_2
                            T = (-B - np.sqrt(B**2 - 4*A*C)) / (2*A)
                        elif 1+conic_K < 0:  # コーニック定数が負の場合, たぶんOK
                            A_1 = (c**2)*((1+conic_K)**2)*(ray_dir[0]**2)
                            B_1 = 2*(c**2)*((1+conic_K)**2) * \
                                (ray_pos[0]*ray_dir[0]) - \
                                2*c*abs(1+conic_K)*ray_dir[0]
                            C_1 = (c**2)*((1+conic_K)**2) * \
                                (ray_pos[0]**2) - 2*c * \
                                abs(1+conic_K)*ray_pos[0] + 1
                            A_2 = -1*(c**2)*abs(1+conic_K) * \
                                ((ray_dir[1]**2)+(ray_dir[2]**2))
                            B_2 = -2*(c**2)*abs(1+conic_K) * \
                                (ray_pos[1]*ray_dir[1] + ray_pos[2]*ray_dir[2])
                            C_2 = 1 - abs(1+conic_K)*(c**2) * \
                                ((ray_pos[1]**2)+(ray_pos[2]**2))
                            A = A_1-A_2
                            B = B_1-B_2
                            C = C_1-C_2
                            T = (-B - np.sqrt(B**2 - 4*A*C)) / (2*A)
                    elif aspherical_R > 0:  # 凹面の場合, たぶんOK
                        c = -abs(1/aspherical_R)  # abs()を取る必要があるかも
                        conic_K = self._conic_K
                        if 1+conic_K > 0:  # コーニック定数が正の場合, たぶんOK
                            print("たぶんOK")
                            A_1 = (c**2)*((1+conic_K)**2)*(ray_dir[0]**2)
                            B_1 = 2*(c**2)*((1+conic_K)**2) * \
                                (ray_pos[0]*ray_dir[0]) - \
                                2*c*(1+conic_K)*ray_dir[0]
                            C_1 = (c**2)*((1+conic_K)**2) * \
                                (ray_pos[0]**2) - 2*c * \
                                (1+conic_K)*ray_pos[0] + 1
                            A_2 = -1*(c**2)*(1+conic_K) * \
                                ((ray_dir[1]**2)+(ray_dir[2]**2))
                            B_2 = -2*(c**2)*(1+conic_K) * \
                                (ray_pos[1]*ray_dir[1] + ray_pos[2]*ray_dir[2])
                            C_2 = 1 - (1+conic_K)*(c**2) * \
                                ((ray_pos[1]**2)+(ray_pos[2]**2))
                            A = A_1-A_2
                            B = B_1-B_2
                            C = C_1-C_2
                            T = (-B + np.sqrt(B**2 - 4*A*C)) / (2*A)
                        elif 1+conic_K < 0:  # コーニック定数が負の場合, たぶんOK
                            A_1 = (c**2)*((1+conic_K)**2)*(ray_dir[0]**2)
                            B_1 = 2*(c**2)*((1+conic_K)**2) * \
                                (ray_pos[0]*ray_dir[0]) - \
                                2*c*abs(1+conic_K)*ray_dir[0]
                            C_1 = (c**2)*((1+conic_K)**2) * \
                                (ray_pos[0]**2) - 2*c * \
                                abs(1+conic_K)*ray_pos[0] + 1
                            A_2 = -1*(c**2)*abs(1+conic_K) * \
                                ((ray_dir[1]**2)+(ray_dir[2]**2))
                            B_2 = -2*(c**2)*abs(1+conic_K) * \
                                (ray_pos[1]*ray_dir[1] + ray_pos[2]*ray_dir[2])
                            C_2 = 1 - abs(1+conic_K)*(c**2) * \
                                ((ray_pos[1]**2)+(ray_pos[2]**2))
                            A = A_1-A_2
                            B = B_1-B_2
                            C = C_1-C_2
                            T = (-B + np.sqrt(B**2 - 4*A*C)) / (2*A)
                self.ray_end_pos = self.ray_start_pos + T*self.ray_start_dir
                self.optical_path_length += np.array(
                    T)*self._refractive_index_calc_optical_path_length
                self._normalV_refract_or_reflect = self._calc_normalV_aspherical()

    # 非球面の法線ベクトルを計算する関数
    def _calc_normalV_aspherical(self):
        """
        非球面レンズレイトレース関数で用いる非球面レンズの法線ベクトルを計算する。

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        aspherical_pos = self._surface_pos
        aspherical_R = self._lens_or_parabola_R
        c = 1/aspherical_R
        conic_K = self._conic_K
        length_ray_start_dir = len(self.ray_start_pos)
        if length_ray_start_dir == 3:
            print("光線1本")
            tmp_index = self._max_index(
                self._normalV_optical_element)  # 方向の計算に使う
            ray_pos = self.ray_end_pos
            if tmp_index == 0:  # x軸方向配置の場合
                tmp_root = np.sqrt(
                    1-(1+conic_K)*(ray_pos[1]**2+ray_pos[2]**2)*(c**2))
                normalV_0 = 1
                normalV_1 = ((2*c*ray_pos[1]*(1+tmp_root))-((ray_pos[1]**2+ray_pos[2]**2)*c*(-1*(
                    c**2)*ray_pos[1]*(1+conic_K))/tmp_root))/((1+tmp_root)**2)
                normalV_2 = ((2*c*ray_pos[2]*(1+tmp_root))-((ray_pos[1]**2+ray_pos[2]**2)*c*(-1*(
                    c**2)*ray_pos[2]*(1+conic_K))/tmp_root))/((1+tmp_root)**2)
                normalV = np.array([-normalV_0, -normalV_1, -normalV_2])
                normalV = normalV/np.linalg.norm(normalV)
                return normalV
        else:
            print("光線群")

    # 反射計算
    def reflect(self):
        """
        set_surface()で設定した反射面での反射計算を行う。
        終点の方向ベクトルを計算し、self.ray_end_dirに格納する。

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        normalV = self._normalV_refract_or_reflect
        length_ray_start_dir = len(self.ray_start_pos)
        if length_ray_start_dir == 3:
            ray_dir = np.array(self.ray_start_dir) / \
                np.linalg.norm(self.ray_start_dir)
            normalV = np.array(normalV)/np.linalg.norm(normalV)
            outRayV = ray_dir - 2*(np.dot(ray_dir, normalV))*normalV
            # 正規化
            outRayV = outRayV/np.linalg.norm(outRayV)
            self.ray_end_dir = outRayV
        else:  # 光線群の場合
            ray_dir = [V/np.linalg.norm(V) for V in self.ray_start_dir]
            normalV = [V/np.linalg.norm(V) for V in normalV]
            #dot_tmp = np.dot(ray_dir, normalV)
            dot_tmp = [np.dot(ray_dir[i], normalV[i])
                       for i in range(length_ray_start_dir)]
            dot_tmp2 = [np.dot(dot_tmp[i], normalV[i])
                        for i in range(length_ray_start_dir)]
            outRayV = np.array(ray_dir) - 2*np.array(dot_tmp2)
            # 正規化
            outRayV = [V/np.linalg.norm(V) for V in outRayV]
            self.ray_end_dir = outRayV

    # スネルの法則
    def refract(self):
        """
        set_surface()で設定した屈折面での屈折計算を行う。
        終点の方向ベクトルを計算し、self.ray_end_dirに格納する。

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        normalV = self._normalV_refract_or_reflect
        Nin = self._refractive_index_before
        Nout = self._refractive_index_after
        length_ray_start_dir = len(self.ray_start_pos)
        if length_ray_start_dir == 3:
            # 正規化
            ray_dir = self.ray_start_dir/np.linalg.norm(self.ray_start_dir)
            normalV = normalV/np.linalg.norm(normalV)
            if np.dot(ray_dir, normalV) <= 0:
                # print("内積が負です")
                # 係数A
                A = Nin/Nout
                # 入射角
                cos_t_in = abs(np.dot(ray_dir, normalV))
                # 量子化誤差対策
                if cos_t_in < -1.:
                    cos_t_in = -1.
                elif cos_t_in > 1.:
                    cos_t_in = 1.
                # スネルの法則
                sin_t_in = np.sqrt(1.0 - cos_t_in**2)
                sin_t_out = sin_t_in*A
                if sin_t_out > 1.0:
                    # 全反射する場合
                    return np.zeros(3)
                cos_t_out = np.sqrt(1 - sin_t_out**2)
                # 係数B
                B = A*cos_t_in - cos_t_out
                # 出射光線の方向ベクトル
                outRayV = A*ray_dir + B*normalV
                # 正規化
                outRayV = outRayV/np.linalg.norm(outRayV)
            else:
                # print("内積が正です")
                # 係数A
                A = Nin/Nout
                # 入射角
                cos_t_in = abs(np.dot(ray_dir, normalV))
                # 量子化誤差対策
                if cos_t_in < -1.:
                    cos_t_in = -1.
                elif cos_t_in > 1.:
                    cos_t_in = 1.
                # スネルの法則
                sin_t_in = np.sqrt(1.0 - cos_t_in**2)
                sin_t_out = sin_t_in*A
                if sin_t_out > 1.0:
                    # 全反射する場合
                    return np.zeros(3)
                cos_t_out = np.sqrt(1 - sin_t_out**2)
                # 係数B
                B = -A*cos_t_in + cos_t_out
                # 出射光線の方向ベクトル
                outRayV = A*ray_dir + B*normalV
                # 正規化
                outRayV = outRayV/np.linalg.norm(outRayV)
            self.ray_end_dir = outRayV
        else:
            # 正規化
            ray_dir = [V/np.linalg.norm(V) for V in self.ray_start_dir]
            normalV = [V/np.linalg.norm(V) for V in normalV]
            tmp_V = np.zeros(3)
            if len(normalV) == 3:
                tmp_index = self._max_index(normalV)
            else:
                tmp_index = self._max_index(normalV[0])
            tmp_V[tmp_index] = 1.
            test_dot = np.dot(ray_dir[0], tmp_V)
            # if test_dot <= 0:
            if np.dot(ray_dir[0], normalV[0]) <= 0:
                # print("内積が負です")
                # 係数A
                A = Nin/Nout
                # 入射角
                cos_t_in = np.abs(
                    np.diag(np.dot(ray_dir, np.array(normalV).T)))
                outRayV = []
                for i in range(length_ray_start_dir):
                    i_cos_t_in = cos_t_in[i]
                    i_ray_dir = ray_dir[i]
                    i_normalV = normalV[i]
                    # 量子化誤差対策
                    if i_cos_t_in < -1.:
                        i_cos_t_in = -1.
                    elif i_cos_t_in > 1.:
                        i_cos_t_in = 1.
                    # スネルの法則
                    sin_t_in = np.sqrt(1.0 - i_cos_t_in**2)
                    sin_t_out = sin_t_in*A
                    if sin_t_out > 1.0:
                        # 全反射する場合
                        return np.zeros(3)
                    cos_t_out = np.sqrt(1 - sin_t_out**2)
                    # 係数B
                    B = A*i_cos_t_in - cos_t_out
                    # 出射光線の方向ベクトル
                    tmp_outRayV = A*i_ray_dir + B*i_normalV
                    # 正規化
                    tmp_outRayV = tmp_outRayV/np.linalg.norm(tmp_outRayV)
                    outRayV.append(tmp_outRayV)
            else:
                # print("内積が正です")
                # 係数A
                A = Nin/Nout
                # 入射角
                cos_t_in = np.abs(
                    np.diag(np.dot(ray_dir, np.array(normalV).T)))
                outRayV = []
                for i in range(length_ray_start_dir):
                    i_cos_t_in = cos_t_in[i]
                    i_ray_dir = ray_dir[i]
                    i_normalV = normalV[i]
                    # 量子化誤差対策
                    if i_cos_t_in < -1.:
                        i_cos_t_in = -1.
                    elif i_cos_t_in > 1.:
                        i_cos_t_in = 1.
                    # スネルの法則
                    sin_t_in = np.sqrt(1.0 - i_cos_t_in**2)
                    sin_t_out = sin_t_in*A
                    if sin_t_out > 1.0:
                        # 全反射する場合
                        return np.zeros(3)
                    cos_t_out = np.sqrt(1 - sin_t_out**2)
                    # 係数B
                    B = -A*i_cos_t_in + cos_t_out
                    # 出射光線の方向ベクトル
                    tmp_outRayV = A*i_ray_dir + B*i_normalV
                    # 正規化
                    tmp_outRayV = tmp_outRayV/np.linalg.norm(tmp_outRayV)
                    outRayV.append(tmp_outRayV)
            self.ray_end_dir = outRayV

    # 焦点距離を計算する関数
    def calc_focal_length(self, ray_start_pos_init):
        """
        焦点距離を計算する。

        Parameters
        ----------
        ray_start_pos_init : list or numpy.ndarray
            光線の座標の初期値。

        Returns
        -------
        focal_length : float
            焦点距離。
        """
        length_ray_dir = len(self.ray_end_pos)
        if length_ray_dir == 3:
            argmin_index = self._min_index(self.ray_end_dir)
            tmp_V = -1.*self.ray_end_dir * \
                ray_start_pos_init[argmin_index]/self.ray_end_dir[argmin_index]
            argmax_index = self._max_index(tmp_V)
            focal_length = tmp_V[argmax_index]
            return focal_length
        else:  # 光線が複数の場合
            argmin_index = self._min_index(self.ray_end_dir[0])
            focal_length = []
            for i in range(length_ray_dir):
                tmp_V = -1. * \
                    self.ray_end_dir[i]*ray_start_pos_init[i][argmin_index] / \
                    self.ray_end_dir[i][argmin_index]
                argmax_index = self._max_index(tmp_V)
                focal_length.append(tmp_V[argmax_index])
            # 並び替え
            focal_length.sort()
            return focal_length[0]

    # 焦点位置を計算する関数
    def calc_focal_pos(self, ray_start_pos_init):
        """
        焦点位置を計算する。

        Parameters
        ----------
        ray_start_pos_init : list or numpy.ndarray
            光線の座標の初期値。

        Returns
        -------
        focal_length : float
            焦点位置。
        """
        length_ray_dir = len(self.ray_end_pos)
        if length_ray_dir == 3:
            argmin_index = self._min_index(self.ray_end_dir)
            tmp_V = -1.*self.ray_end_dir * \
                self.ray_end_pos[argmin_index]/self.ray_end_dir[argmin_index]
            focal_point = tmp_V + self.ray_end_pos
            argmax_index = self._max_index(tmp_V)
            print("!!!!正確な焦点位置を得るには近軸光線を計算する必要があります!!!!")
            return focal_point[argmax_index]
        else:
            argmin_index = self._min_index(self.ray_end_dir[0])
            focal_point_list = []
            for i in range(length_ray_dir):
                tmp_V = -1. * \
                    self.ray_end_dir[i]*self.ray_end_pos[i][argmin_index] / \
                    self.ray_end_dir[i][argmin_index]
                focal_point_list.append(tmp_V + self.ray_end_pos[i])
            r_list = []
            focus_pos = []
            for i in range(length_ray_dir):
                r_list.append(
                    np.sqrt(ray_start_pos_init[i][1]**2 + ray_start_pos_init[i][2]**2))
                focus_pos.append(focal_point_list[i][0])
            # 並び替え
            r_list, focus_pos = zip(*sorted(zip(r_list, focus_pos)))
            #print("focus_pos = ", focus_pos)
            focal_point = focus_pos[0]
            print("!!!!正確な焦点位置を得るには近軸光線を計算する必要があります!!!!")
            return focal_point

    # ２点の位置ベクトルから直線を引く関数
    def plot_line_blue(self):
        """
        2点の位置ベクトルから直線を引く。

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        length_ray_start_dir = len(self.ray_start_pos)
        if length_ray_start_dir == 3:
            startPointV = self.ray_start_pos
            endPointV = self.ray_end_pos
            startX = startPointV[0]
            startY = startPointV[1]
            startZ = startPointV[2]
            endX = endPointV[0]
            endY = endPointV[1]
            endZ = endPointV[2]
            self._ax.plot([startX, endX], [startY, endY], [startZ, endZ],
                          'o-', ms='2', linewidth=0.5, color='blue')
        else:
            for i in range(length_ray_start_dir):
                startPointV = self.ray_start_pos[i]
                endPointV = self.ray_end_pos[i]
                startX = startPointV[0]
                startY = startPointV[1]
                startZ = startPointV[2]
                endX = endPointV[0]
                endY = endPointV[1]
                endZ = endPointV[2]
                self._ax.plot([startX, endX], [startY, endY], [startZ, endZ],
                              'o-', ms='2', linewidth=0.5, color='blue')

    def plot_line_green(self):
        """
        2点の位置ベクトルから直線を引く。

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        length_ray_start_dir = len(self.ray_start_pos)
        if length_ray_start_dir == 3:
            startPointV = self.ray_start_pos
            endPointV = self.ray_end_pos
            startX = startPointV[0]
            startY = startPointV[1]
            startZ = startPointV[2]
            endX = endPointV[0]
            endY = endPointV[1]
            endZ = endPointV[2]
            self._ax.plot([startX, endX], [startY, endY], [startZ, endZ],
                          'o-', ms='2', linewidth=0.5, color='green')
        else:
            for i in range(length_ray_start_dir):
                startPointV = self.ray_start_pos[i]
                endPointV = self.ray_end_pos[i]
                startX = startPointV[0]
                startY = startPointV[1]
                startZ = startPointV[2]
                endX = endPointV[0]
                endY = endPointV[1]
                endZ = endPointV[2]
                self._ax.plot([startX, endX], [startY, endY], [startZ, endZ],
                              'o-', ms='2', linewidth=0.5, color='green')

    def plot_line_red(self):
        """
        2点の位置ベクトルから直線を引く。

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        length_ray_start_dir = len(self.ray_start_pos)
        if length_ray_start_dir == 3:
            startPointV = self.ray_start_pos
            endPointV = self.ray_end_pos
            startX = startPointV[0]
            startY = startPointV[1]
            startZ = startPointV[2]
            endX = endPointV[0]
            endY = endPointV[1]
            endZ = endPointV[2]
            self._ax.plot([startX, endX], [startY, endY], [startZ, endZ],
                          'o-', ms='2', linewidth=0.5, color='r')
        else:
            for i in range(length_ray_start_dir):
                startPointV = self.ray_start_pos[i]
                endPointV = self.ray_end_pos[i]
                startX = startPointV[0]
                startY = startPointV[1]
                startZ = startPointV[2]
                endX = endPointV[0]
                endY = endPointV[1]
                endZ = endPointV[2]
                self._ax.plot([startX, endX], [startY, endY], [startZ, endZ],
                              'o-', ms='2', linewidth=0.5, color='r')

    def plot_line_orange(self):
        """
        2点の位置ベクトルから直線を引く。

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        length_ray_start_dir = len(self.ray_start_pos)
        if length_ray_start_dir == 3:
            startPointV = self.ray_start_pos
            endPointV = self.ray_end_pos
            startX = startPointV[0]
            startY = startPointV[1]
            startZ = startPointV[2]
            endX = endPointV[0]
            endY = endPointV[1]
            endZ = endPointV[2]
            self._ax.plot([startX, endX], [startY, endY], [startZ, endZ],
                          'o-', ms='2', linewidth=0.5, color='orange')
        else:
            for i in range(length_ray_start_dir):
                startPointV = self.ray_start_pos[i]
                endPointV = self.ray_end_pos[i]
                startX = startPointV[0]
                startY = startPointV[1]
                startZ = startPointV[2]
                endX = endPointV[0]
                endY = endPointV[1]
                endZ = endPointV[2]
                self._ax.plot([startX, endX], [startY, endY], [startZ, endZ],
                              'o-', ms='2', linewidth=0.5, color='orange')

    def plot_four_beam_line(self, i):
        """
        4つの光について、2点の位置ベクトルから直線を引く。

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        if i == 0:
            self.plot_line_blue()
        elif i == 1:
            self.plot_line_green()
        elif i == 2:
            self.plot_line_red()
        elif i == 3:
            self.plot_line_orange()
        else:
            print("fourBeamPlotLine, iの値が不正です")

    def plot_line_purple(self, startPointV, endPointV):
        """
        2点の位置ベクトルから直線を引く。

        Parameters
        ----------
        startPointV : list or numpy.ndarray
            直線の始点の位置ベクトル
        endPointV : list or numpy.ndarray
            直線の終点の位置ベクトル

        Returns
        -------
        None
        """
        length_ray_start_dir = len(self.ray_start_pos)
        if length_ray_start_dir == 3:
            startX = startPointV[0]
            startY = startPointV[1]
            startZ = startPointV[2]
            endX = endPointV[0]
            endY = endPointV[1]
            endZ = endPointV[2]
            self._ax.plot([startX, endX], [startY, endY], [startZ, endZ],
                          'o-', ms='2', linewidth=0.5, color='purple')
        else:
            for i in range(length_ray_start_dir):
                startPointV = self.ray_start_pos[i]
                endPointV = self.ray_end_pos[i]
                startX = startPointV[0]
                startY = startPointV[1]
                startZ = startPointV[2]
                endX = endPointV[0]
                endY = endPointV[1]
                endZ = endPointV[2]
                self._ax.plot([startX, endX], [startY, endY], [startZ, endZ],
                              'o-', ms='2', linewidth=0.5, color='purple')

    def plot_line_black(self, startPointV, endPointV):
        """
        2点の位置ベクトルから直線を引く。

        Parameters
        ----------
        startPointV : list or numpy.ndarray
            直線の始点の位置ベクトル
        endPointV : list or numpy.ndarray
            直線の終点の位置ベクトル

        Returns
        -------
        None
        """
        length_ray_start_dir = len(self.ray_start_pos)
        if length_ray_start_dir == 3:
            startX = startPointV[0]
            startY = startPointV[1]
            startZ = startPointV[2]
            endX = endPointV[0]
            endY = endPointV[1]
            endZ = endPointV[2]
            self._ax.plot([startX, endX], [startY, endY], [startZ, endZ],
                          'o-', ms='2', linewidth=0.5, color='black')
        else:
            for i in range(length_ray_start_dir):
                startPointV = self.ray_start_pos[i]
                endPointV = self.ray_end_pos[i]
                startX = startPointV[0]
                startY = startPointV[1]
                startZ = startPointV[2]
                endX = endPointV[0]
                endY = endPointV[1]
                endZ = endPointV[2]
                self._ax.plot([startX, endX], [startY, endY], [startZ, endZ],
                              'o-', ms='2', linewidth=0.5, color='black')
