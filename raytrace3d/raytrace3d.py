import numpy as np
import plotly.graph_objs as go


"""光学計算や光学系の描画を行うクラス。"""


class VectorFunctions:
    """
    光学計算や光学系の描画を行うクラス。

    Attributes
    ----------
    _ax : Axes3D
        3次元グラフを描画するためのAxes3Dオブジェクト。
    ray_init_pos : list or ndarray
        光源の位置ベクトル。
    ray_init_dir : list or ndarray
        光源の方向ベクトル。
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
        self._fig_plotly = None  # plotlyで描画
        self.ray_init_pos = np.array([0, 0, 0])  # 光源の位置ベクトル
        self.ray_init_dir = np.array([0, 0, 0])  # 光源の方向ベクトル
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

    def set_fig_plotly(self, initial_camera=None):
        """
        描画関数のためのplotlyのFigureオブジェクトをセットする。

        tmp
        """
        if initial_camera is None:
            # 初期カメラ設定（視点を調整）
            initial_camera = dict(
                eye=dict(x=-1.2, y=-1.8, z=0.1),
                center=dict(x=-0.1, y=0, z=-0.1),
                up=dict(x=0, y=0, z=1)
            )
        else:
            initial_camera = initial_camera
        # グラフのレイアウト設定
        layout = go.Layout(
            scene=dict(
                xaxis=dict(title='X [mm]'),
                yaxis=dict(title='Y [mm]'),
                zaxis=dict(title='Z [mm]'),
                camera=initial_camera,
                aspectmode='data',
            ),
        )
        self._fig_plotly = go.Figure(layout=layout)

    def get_fig_plotly(self):
        """
        描画関数のためのplotlyのFigureオブジェクトを取得する。

        tmp
        """
        return self._fig_plotly

    def show_fig_plotly(self):
        """
        plotlyで描画した図を表示する。

        tmp
        """
        self._fig_plotly.show()

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

        # 円盤生成
        geneNum = 200
        if argmax_index == 0:  # 光軸:x軸
            ax_opt_center = X_center
            ax_no_opt_1_center = Y_center
            ax_no_opt_2_center = Z_center
            nV_arg_opt = normalV[0]
            nV_arg_no_opt_1 = normalV[1]
            nV_arg_no_opt_2 = normalV[2]
        elif argmax_index == 1:  # 光軸:y軸
            ax_opt_center = Y_center
            ax_no_opt_1_center = Z_center
            ax_no_opt_2_center = X_center
            nV_arg_opt = normalV[1]
            nV_arg_no_opt_1 = normalV[2]
            nV_arg_no_opt_2 = normalV[0]
        elif argmax_index == 2:  # 光軸:z軸
            ax_opt_center = Z_center
            ax_no_opt_1_center = X_center
            ax_no_opt_2_center = Y_center
            nV_arg_opt = normalV[2]
            nV_arg_no_opt_1 = normalV[0]
            nV_arg_no_opt_2 = normalV[1]
        ax_no_opt_1 = np.linspace(ax_no_opt_1_center-R, ax_no_opt_1_center+R, geneNum)
        ax_no_opt_2 = np.linspace(ax_no_opt_2_center-R, ax_no_opt_2_center+R, geneNum)
        ax_no_opt_1, ax_no_opt_2 = np.meshgrid(ax_no_opt_1, ax_no_opt_2)
        if nV_arg_opt == 0:
            ax_opt = ax_opt_center - (nV_arg_no_opt_1*(ax_no_opt_1-ax_no_opt_1_center) +
                                    nV_arg_no_opt_2*(ax_no_opt_2-ax_no_opt_2_center)) / 0.01
        else:
            ax_opt = ax_opt_center - (nV_arg_no_opt_1*(ax_no_opt_1-ax_no_opt_1_center) +
                                    nV_arg_no_opt_2*(ax_no_opt_2-ax_no_opt_2_center)) / nV_arg_opt

        for i in range(geneNum):
            for j in range(geneNum):
                if (ax_opt[i][j]-ax_opt_center)**2 + (ax_no_opt_1[i][j]-ax_no_opt_1_center)**2 + (ax_no_opt_2[i][j]-ax_no_opt_2_center)**2 > R**2:
                    ax_no_opt_2[i][j] = np.nan

        if argmax_index == 0:
            self._ax.plot_wireframe(ax_opt, ax_no_opt_1, ax_no_opt_2, color=color, linewidth=0.3)
        elif argmax_index == 1:
            self._ax.plot_wireframe(ax_no_opt_2, ax_opt, ax_no_opt_1, color=color, linewidth=0.3)
        elif argmax_index == 2:
            self._ax.plot_wireframe(ax_no_opt_1, ax_no_opt_2, ax_opt, color=color, linewidth=0.3)

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

        # 円盤生成
        geneNum = 200
        if argmax_index == 0:  # 光軸:x軸
            ax_opt_center = X_center
            ax_no_opt_1_center = Y_center
            ax_no_opt_2_center = Z_center
            nV_arg_opt = normalV[0]
            nV_arg_no_opt_1 = normalV[1]
            nV_arg_no_opt_2 = normalV[2]
        elif argmax_index == 1:  # 光軸:y軸
            ax_opt_center = Y_center
            ax_no_opt_1_center = Z_center
            ax_no_opt_2_center = X_center
            nV_arg_opt = normalV[1]
            nV_arg_no_opt_1 = normalV[2]
            nV_arg_no_opt_2 = normalV[0]
        elif argmax_index == 2:  # 光軸:z軸
            ax_opt_center = Z_center
            ax_no_opt_1_center = X_center
            ax_no_opt_2_center = Y_center
            nV_arg_opt = normalV[2]
            nV_arg_no_opt_1 = normalV[0]
            nV_arg_no_opt_2 = normalV[1]
        ax_no_opt_1 = np.linspace(ax_no_opt_1_center-R, ax_no_opt_1_center+R, geneNum)
        ax_no_opt_2 = np.linspace(ax_no_opt_2_center-R, ax_no_opt_2_center+R, geneNum)
        ax_no_opt_1, ax_no_opt_2 = np.meshgrid(ax_no_opt_1, ax_no_opt_2)
        if nV_arg_opt == 0:
            ax_opt = ax_opt_center - (nV_arg_no_opt_1*(ax_no_opt_1-ax_no_opt_1_center) +
                                    nV_arg_no_opt_2*(ax_no_opt_2-ax_no_opt_2_center)) / 0.01
        else:
            ax_opt = ax_opt_center - (nV_arg_no_opt_1*(ax_no_opt_1-ax_no_opt_1_center) +
                                    nV_arg_no_opt_2*(ax_no_opt_2-ax_no_opt_2_center)) / nV_arg_opt

        for i in range(geneNum):
            for j in range(geneNum):
                if (ax_opt[i][j]-ax_opt_center)**2 + (ax_no_opt_1[i][j]-ax_no_opt_1_center)**2 + (ax_no_opt_2[i][j]-ax_no_opt_2_center)**2 > R**2:
                    ax_no_opt_2[i][j] = np.nan

        if argmax_index == 0:
            self._ax.plot_wireframe(ax_opt, ax_no_opt_1, ax_no_opt_2, color='lightcyan', linewidth=0.3)
        elif argmax_index == 1:
            self._ax.plot_wireframe(ax_no_opt_2, ax_opt, ax_no_opt_1, color='lightcyan', linewidth=0.3)
        elif argmax_index == 2:
            self._ax.plot_wireframe(ax_no_opt_1, ax_no_opt_2, ax_opt, color='lightcyan', linewidth=0.3)

        theta = np.linspace(0, 2*np.pi, 100)
        ax_opt = np.zeros_like(theta) + ax_opt_center
        ax_no_opt_1 = R*np.cos(theta) + ax_no_opt_1_center
        ax_no_opt_2 = R*np.sin(theta) + ax_no_opt_2_center
        if argmax_index == 0:
            self._ax.plot(ax_opt, ax_no_opt_1, ax_no_opt_2, color='black', linewidth=0.3)
        elif argmax_index == 1:
            self._ax.plot(ax_no_opt_2, ax_opt, ax_no_opt_1, color='black', linewidth=0.3)
        elif argmax_index == 2:
            self._ax.plot(ax_no_opt_1, ax_no_opt_2, ax_opt, color='black', linewidth=0.3)

    def plot_plane(self, params, color="gray"):
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

        # 円盤生成
        geneNum = 200
        if argmax_index == 0:  # 光軸:x軸
            ax_opt_center = X_center
            ax_no_opt_1_center = Y_center
            ax_no_opt_2_center = Z_center
            nV_arg_opt = normalV[0]
            nV_arg_no_opt_1 = normalV[1]
            nV_arg_no_opt_2 = normalV[2]
        elif argmax_index == 1:  # 光軸:y軸
            ax_opt_center = Y_center
            ax_no_opt_1_center = Z_center
            ax_no_opt_2_center = X_center
            nV_arg_opt = normalV[1]
            nV_arg_no_opt_1 = normalV[2]
            nV_arg_no_opt_2 = normalV[0]
        elif argmax_index == 2:  # 光軸:z軸
            ax_opt_center = Z_center
            ax_no_opt_1_center = X_center
            ax_no_opt_2_center = Y_center
            nV_arg_opt = normalV[2]
            nV_arg_no_opt_1 = normalV[0]
            nV_arg_no_opt_2 = normalV[1]
        ax_no_opt_1 = np.linspace(ax_no_opt_1_center-R, ax_no_opt_1_center+R, geneNum)
        ax_no_opt_2 = np.linspace(ax_no_opt_2_center-R, ax_no_opt_2_center+R, geneNum)
        ax_no_opt_1, ax_no_opt_2 = np.meshgrid(ax_no_opt_1, ax_no_opt_2)
        if nV_arg_opt == 0:
            ax_opt = ax_opt_center - (nV_arg_no_opt_1*(ax_no_opt_1-ax_no_opt_1_center) +
                                    nV_arg_no_opt_2*(ax_no_opt_2-ax_no_opt_2_center)) / 0.01
        else:
            ax_opt = ax_opt_center - (nV_arg_no_opt_1*(ax_no_opt_1-ax_no_opt_1_center) +
                                    nV_arg_no_opt_2*(ax_no_opt_2-ax_no_opt_2_center)) / nV_arg_opt

        for i in range(geneNum):
            for j in range(geneNum):
                if (ax_opt[i][j]-ax_opt_center)**2 + (ax_no_opt_1[i][j]-ax_no_opt_1_center)**2 + (ax_no_opt_2[i][j]-ax_no_opt_2_center)**2 > R**2:
                    ax_no_opt_2[i][j] = np.nan

        if argmax_index == 0:
            self._ax.plot_wireframe(ax_opt, ax_no_opt_1, ax_no_opt_2, color=color, linewidth=0.3)
        elif argmax_index == 1:
            self._ax.plot_wireframe(ax_no_opt_2, ax_opt, ax_no_opt_1, color=color, linewidth=0.3)
        elif argmax_index == 2:
            self._ax.plot_wireframe(ax_no_opt_1, ax_no_opt_2, ax_opt, color=color, linewidth=0.3)

    def plot_square(self, params, color='gray'):
        """
        ミラー(square)を描画する。関数"plot_mirror"の派生型。

        Parameters
        ----------
        params : list or ndarray
            光学素子(square)のパラメータを格納したリストまたはndarray。
            [[pos_x, pos_y, pos_z], [normalV_x, normalV_y, normalV_z], a, np.inf]

        Returns
        -------
        None
        """
        X_center = params[0][0]
        Y_center = params[0][1]
        Z_center = params[0][2]
        normalV = params[1]
        a = params[2]  # aperture_R, 正方形の一辺の長さの半分

        argmax_index = np.argmax(np.abs(params[1]))

        # 正方形生成
        geneNum = 10

        if argmax_index == 0:
            ax_opt_center = X_center
            ax_no_opt_1_center = Y_center
            ax_no_opt_2_center = Z_center
            nV_arg_opt = normalV[0]
            nV_arg_no_opt_1 = normalV[1]
            nV_arg_no_opt_2 = normalV[2]
        elif argmax_index == 1:
            ax_opt_center = Y_center
            ax_no_opt_1_center = Z_center
            ax_no_opt_2_center = X_center
            nV_arg_opt = normalV[1]
            nV_arg_no_opt_1 = normalV[2]
            nV_arg_no_opt_2 = normalV[0]
        elif argmax_index == 2:
            ax_opt_center = Z_center
            ax_no_opt_1_center = X_center
            ax_no_opt_2_center = Y_center
            nV_arg_opt = normalV[2]
            nV_arg_no_opt_1 = normalV[0]
            nV_arg_no_opt_2 = normalV[1]

        ax_no_opt_1 = np.linspace(ax_no_opt_1_center-a, ax_no_opt_1_center+a, geneNum)
        ax_no_opt_2 = np.linspace(ax_no_opt_2_center-a, ax_no_opt_2_center+a, geneNum)
        ax_no_opt_1, ax_no_opt_2 = np.meshgrid(ax_no_opt_1, ax_no_opt_2)
        if nV_arg_opt == 0:
            ax_opt = ax_opt_center - (nV_arg_no_opt_1*(ax_no_opt_1-ax_no_opt_1_center) +
                                    nV_arg_no_opt_2*(ax_no_opt_2-ax_no_opt_2_center)) / 0.01
        else:
            ax_opt = ax_opt_center - (nV_arg_no_opt_1*(ax_no_opt_1-ax_no_opt_1_center) +
                                    nV_arg_no_opt_2*(ax_no_opt_2-ax_no_opt_2_center)) / nV_arg_opt

        for i in range(geneNum):
            for j in range(geneNum):
                if (ax_opt[i][j]-ax_opt_center) > a or (ax_no_opt_1[i][j]-ax_no_opt_1_center) > a or (ax_no_opt_2[i][j]-ax_no_opt_2_center) > a:
                    ax_no_opt_2[i][j] = np.nan

        if argmax_index == 0:
            self._ax.plot_wireframe(ax_opt, ax_no_opt_1, ax_no_opt_2, color=color, linewidth=0.3)
        elif argmax_index == 1:
            self._ax.plot_wireframe(ax_no_opt_2, ax_opt, ax_no_opt_1, color=color, linewidth=0.3)
        elif argmax_index == 2:
            self._ax.plot_wireframe(ax_no_opt_1, ax_no_opt_2, ax_opt, color=color, linewidth=0.3)

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
        # argmax_index==0->x軸向き配置
        # argmax_index==1->y軸向き配置
        # argmax_index==2->z軸向き配置
        tmp_outer_1 = np.outer(np.sin(theta), np.sin(phi))
        tmp_outer_2 = np.outer(np.ones(np.size(theta)), np.cos(phi))
        ax_no_opt_1 = params[2] * tmp_outer_1
        ax_no_opt_2 = params[2] * tmp_outer_2
        if params[3] <= 0:
            ax_opt = -(params[3]**2-ax_no_opt_1**2-ax_no_opt_2**2)**0.5 - params[3]
        elif params[3] > 0:
            ax_opt = (params[3]**2-ax_no_opt_1**2-ax_no_opt_2**2)**0.5 - params[3]
        if argmax_index == 0:  # 光軸: x軸
            Xs = ax_opt
            Ys = ax_no_opt_1
            Zs = ax_no_opt_2
        elif argmax_index == 1:  # 光軸: y軸
            Xs = ax_no_opt_2
            Ys = ax_opt
            Zs = ax_no_opt_1
        elif argmax_index == 2:  # 光軸: z軸
            Xs = ax_no_opt_1
            Ys = ax_no_opt_2
            Zs = ax_opt
        # レンズ描画
        self._ax.plot_wireframe(
            Xs+params[0][0], Ys+params[0][1], Zs+params[0][2], linewidth=0.1)

    # レンズ描画(plotly)
    def plot_lens_plotly(self, params):
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
        geneNum = 30
        limitTheta = 2*np.pi  # theta生成数
        limitPhi = np.pi  # phi生成数
        theta = np.linspace(0, limitTheta, geneNum)
        phi = np.linspace(0, limitPhi, geneNum)

        argmax_index = np.argmax(np.abs(params[1]))

        # argmax_index==0->x軸向き配置
        # argmax_index==1->y軸向き配置
        # argmax_index==2->z軸向き配置
        tmp_outer_1 = np.outer(np.sin(theta), np.sin(phi))
        tmp_outer_2 = np.outer(np.ones(np.size(theta)), np.cos(phi))
        ax_no_opt_1 = params[2] * tmp_outer_1
        ax_no_opt_2 = params[2] * tmp_outer_2
        if params[3] <= 0:
            ax_opt = -(params[3]**2-ax_no_opt_1**2-ax_no_opt_2**2)**0.5 - params[3]
        elif params[3] > 0:
            ax_opt = (params[3]**2-ax_no_opt_1**2-ax_no_opt_2**2)**0.5 - params[3]
        if argmax_index == 0:  # 光軸: x軸
            Xs = ax_opt
            Ys = ax_no_opt_1
            Zs = ax_no_opt_2
        elif argmax_index == 1:  # 光軸: y軸
            Xs = ax_no_opt_2
            Ys = ax_opt
            Zs = ax_no_opt_1
        elif argmax_index == 2:  # 光軸: z軸
            Xs = ax_no_opt_1
            Ys = ax_no_opt_2
            Zs = ax_opt
        # レンズ描画
        self._fig_plotly.add_trace(go.Surface(
            x=Xs+params[0][0],
            y=Ys+params[0][1],
            z=Zs+params[0][2],
            showscale=False,
            colorscale='jet',
            opacity=0.1,
            cmin=0,
            cmax=1,
            surfacecolor=np.ones_like(Xs)*0.3,  # aqua blue
        ))


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
        if params[3] < 0:
            a = a
        else:
            a = -a

        max_index = self._max_index(params[1])

        if max_index == 0:  # x軸向き配置
            ax_opt_center = params[0][0]
            ax_no_opt_1_center = params[0][1]
            ax_no_opt_2_center = params[0][2]
        elif max_index == 1:  # y軸向き配置
            ax_opt_center = params[0][1]
            ax_no_opt_1_center = params[0][2]
            ax_no_opt_2_center = params[0][0]
        elif max_index == 2:  # z軸向き配置
            ax_opt_center = params[0][2]
            ax_no_opt_1_center = params[0][0]
            ax_no_opt_2_center = params[0][1]

        ax_no_opt_1 = R*np.cos(theta)
        ax_no_opt_2 = R*np.sin(theta)
        ax_no_opt_1, ax_no_opt_2 = np.meshgrid(ax_no_opt_1, ax_no_opt_2)
        if params[3] < 0:
            ax_opt = a*ax_no_opt_1**2 + a*ax_no_opt_2**2
        elif params[3] > 0:
            ax_opt = -(a*ax_no_opt_1**2 + a*ax_no_opt_2**2)

        for i in range(100):
            for j in range(100):
                if (ax_no_opt_1[i][j])**2 + ax_no_opt_2[i][j]**2 > R**2:
                    ax_opt[i][j] = np.nan
                else:
                    if params[3] < 0:
                        ax_opt[i][j] = a*ax_no_opt_1[i][j]**2 + a*ax_no_opt_2[i][j]**2
                    elif params[3] > 0:
                        ax_opt[i][j] = -(a*ax_no_opt_1[i][j]**2 + a*ax_no_opt_2[i][j]**2)

        if max_index == 0:  # x軸向き配置
            self._ax.plot_wireframe(ax_opt+ax_opt_center, ax_no_opt_1+ax_no_opt_1_center, ax_no_opt_2+ax_no_opt_2_center, color='b', linewidth=0.1)
        elif max_index == 1:  # y軸向き配置
            self._ax.plot_wireframe(ax_no_opt_2+ax_no_opt_2_center, ax_opt+ax_opt_center, ax_no_opt_1+ax_no_opt_1_center, color='b', linewidth=0.1)
        elif max_index == 2:  # z軸向き配置
            self._ax.plot_wireframe(ax_no_opt_1+ax_no_opt_1_center, ax_no_opt_2+ax_no_opt_2_center, ax_opt+ax_opt_center, color='b', linewidth=0.1)

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
        result = np.column_stack((point0, point1, point2))
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

    # 光線の終点に注目し、光学素子の中心からaperture_Rの範囲内にあるかどうかを判定する関数
    def _is_in_aperture(self, surface, ray_end_point):
        """
        光線の終点に注目し、光学素子の中心からaperture_Rの範囲内にあるかどうかを判定する。

        Parameters
        ----------
        ray_end_point : ndarray
            光線の終点の座標を格納したndarray。

        Returns
        -------
        bool
            光学素子の中心からaperture_Rの範囲内にあるかどうか。
            光線の数だけbool値を格納したリストを返す。
        """
        max_index = self._max_index(surface[1])
        aperture_R = surface[2]

        if max_index == 0:  # x軸向き配置
            # 光線の終点のy座標とz座標を一度に比較
            diff = np.abs(ray_end_point[:, [1, 2]] - np.array(surface[0])[[1, 2]])
            in_aperture = np.sqrt(np.sum(diff**2, axis=1)) <= aperture_R
            return in_aperture

        if max_index == 1:  # y軸向き配置
            # 光線の終点のx座標とz座標を一度に比較
            diff = np.abs(ray_end_point[:, [0, 2]] - np.array(surface[0])[[0, 2]])
            in_aperture = np.sqrt(np.sum(diff**2, axis=1)) <= aperture_R
            return in_aperture

        if max_index == 2:  # z軸向き配置
            # 光線の終点のx座標とy座標を一度に比較
            diff = np.abs(ray_end_point[:, [0, 1]] - np.array(surface[0])[[0, 1]])
            in_aperture = np.sqrt(np.sum(diff**2, axis=1)) <= aperture_R
            return in_aperture

    def _is_in_aperture_square(self, surface, ray_end_point):
        """
        光線の終点に注目し、光学素子の中心からaperture_Rの範囲内にあるかどうかを判定する。

        Parameters
        ----------
        ray_end_point : ndarray
            光線の終点の座標を格納したndarray。

        Returns
        -------
        bool
            光学素子の中心からaperture_Rの範囲内にあるかどうか。
            光線の数だけbool値を格納したリストを返す。
        """
        max_index = self._max_index(surface[1])
        aperture_R = surface[2]

        if max_index == 0:
            # 光線の終点のy座標とz座標を一度に比較
            diff = np.abs(ray_end_point[:, [1, 2]] - surface[0][[1, 2]])
            in_aperture = np.all(diff <= aperture_R, axis=1)
            return in_aperture
        elif max_index == 1:
            # 光線の終点のx座標とz座標を一度に比較
            diff = np.abs(ray_end_point[:, [0, 2]] - surface[0][[0, 2]])
            in_aperture = np.all(diff <= aperture_R, axis=1)
            return in_aperture
        elif max_index == 2:
            # 光線の終点のx座標とy座標を一度に比較
            diff = np.abs(ray_end_point[:, [0, 1]] - surface[0][[0, 1]])
            in_aperture = np.all(diff <= aperture_R, axis=1)
            return in_aperture

        # max_index != 0 の場合、すべての光線は中心のy座標とz座標内にあります
        return np.ones(len(ray_end_point), dtype=bool)

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
        print("refraction_index (in, out):", self._refractive_index_before, ",", self._refractive_index_after, "\n")
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

        nV = np.array(normalV) / np.linalg.norm(normalV)

        dot_product = np.dot(centerV, nV)
        dot_product_start = np.dot(self.ray_start_pos, nV)
        dot_product_dir = np.dot(self.ray_start_dir, nV)

        if length_ray_start_dir == 3:
            T = (dot_product - dot_product_start) / dot_product_dir
            self.ray_end_pos = self.ray_start_pos + T * self.ray_start_dir
            self.optical_path_length += T * self._refractive_index_calc_optical_path_length
            self._normalV_refract_or_reflect = nV
        else:  # 光線群の場合
            T = np.where(dot_product_dir == 0, 0, (dot_product -
                         dot_product_start) / dot_product_dir)
            self.ray_end_pos = self.ray_start_pos + \
                (np.array(self.ray_start_dir).T * T).T
            self.optical_path_length += T * self._refractive_index_calc_optical_path_length
            self._normalV_refract_or_reflect = np.tile(
                nV, (length_ray_start_dir, 1))

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

        tmp_V = np.zeros(3)
        tmp_index = self._max_index(self.ray_start_dir[0])
        tmp_V[tmp_index] = lens_R
        tmp_V = np.array([tmp_V] * length_ray_start_dir)
        test_dot = np.sum(tmp_V * self.ray_start_dir, axis=1)  # 内積を計算

        shiftV = lens_pos - tmp_V

        T = np.zeros(length_ray_start_dir)
        convex = test_dot < 0  # 凸

        ray_pos = self.ray_start_pos - shiftV
        A = np.diag(np.dot(self.ray_start_dir, np.array(self.ray_start_dir).T))
        B = np.diag(np.dot(self.ray_start_dir, ray_pos.T))
        C = np.diag(np.dot(ray_pos, ray_pos.T)) - abs(lens_R)**2

        non_zero_indices = np.where(
            np.dot(self.ray_start_dir, np.array([1, 1, 1])) != 0)[0]
        T[non_zero_indices] = np.where(convex, (-B[non_zero_indices] - np.sqrt(B[non_zero_indices]**2 - A[non_zero_indices]*C[non_zero_indices])) /
                                       A[non_zero_indices], (-B[non_zero_indices] + np.sqrt(B[non_zero_indices]**2 - A[non_zero_indices]*C[non_zero_indices])) / A[non_zero_indices])

        self.ray_end_pos = self.ray_start_pos + \
            np.array([V * T for V, T in zip(self.ray_start_dir, T)])
        self.optical_path_length += T * self._refractive_index_calc_optical_path_length
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
            # # ===========================================================
            # normalV = []
            # for i in range(length_ray_start_dir):
            #     tmp_normalV = self.ray_end_pos[i] - surface_pos + tmp_V
            #     normalV.append(tmp_normalV/np.linalg.norm(tmp_normalV))
            # # ===========================================================
            # ===========================================================
            # 各光線の終点からのベクトルを一括計算
            ray_end_pos_shifted = self.ray_end_pos - surface_pos + tmp_V

            # ベクトルの正規化を一括計算
            norms = np.linalg.norm(ray_end_pos_shifted, axis=1)
            normalV = ray_end_pos_shifted / norms[:, np.newaxis]
            # ===========================================================
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
        # tmp_V = np.zeros_like(self.ray_start_pos)
        # tmp_index = self._max_index(self.ray_start_dir)
        # tmp_V[tmp_index] = parabola_R
        # test_dot = np.dot(tmp_V, self.ray_start_dir)
        # print("test_dot =", test_dot)
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
                # if ray_dir[0][1] == 0 and ray_dir[0][2] == 0:  # x軸に平行な光線
                #     T = [a*(ray_pos[i][1]**2 - ray_pos[i][0]/a + ray_pos[i][1]
                #             ** 2) / ray_dir[i][0] for i in range(length_ray_start_dir)]
                #     self.ray_end_pos = self.ray_start_pos + T*self.ray_start_dir
                #     self.optical_path_length += np.array(T) * \
                #         self._refractive_index_calc_optical_path_length
                #     self._normalV_refract_or_reflect = self._calc_normalV_parabola()
                # else:  # x軸に平行でない光線
                #     if test_dot < 0:  # 凸
                #         A = [ray_dir[i][1]**2 + ray_dir[i][2] ** 2
                #              for i in range(length_ray_start_dir)]
                #         B = [ray_pos[i][1]*ray_dir[i][1] + ray_pos[i][2] * ray_dir[i][2] - ray_dir[i][0]/(2*a)
                #              for i in range(length_ray_start_dir)]
                #         C = [ray_pos[i][1]**2 + ray_pos[i][2]**2 - ray_pos[i][0]/a
                #              for i in range(length_ray_start_dir)]
                #         T = [(-B[i] - np.sqrt(B[i]**2 - A[i]*C[i])) / A[i]
                #              for i in range(length_ray_start_dir)]
                #         self.ray_end_pos = self.ray_start_pos + T*self.ray_start_dir
                #         self.optical_path_length += np.array(T) * \
                #             self._refractive_index_calc_optical_path_length
                #         self._normalV_refract_or_reflect = self._calc_normalV_parabola()
                #     else:  # 凹
                #         A = [ray_dir[i][1]**2 + ray_dir[i][2] ** 2
                #              for i in range(length_ray_start_dir)]
                #         B = [ray_pos[i][1]*ray_dir[i][1] + ray_pos[i][2] * ray_dir[i][2] - ray_dir[i][0]/(2*a)
                #              for i in range(length_ray_start_dir)]
                #         C = [ray_pos[i][1]**2 + ray_pos[i][2]**2 - ray_pos[i][0]/a
                #              for i in range(length_ray_start_dir)]
                #         T = [(-B[i] + np.sqrt(B[i]**2 - A[i]*C[i])) / A[i]
                #              for i in range(length_ray_start_dir)]
                #         self.ray_end_pos = self.ray_start_pos + T*self.ray_start_dir
                #         self.optical_path_length += np.array(T) * \
                #             self._refractive_index_calc_optical_path_length
                #         self._normalV_refract_or_reflect = self._calc_normalV_parabola()
                if np.all(ray_dir[:, 1] == 0) and np.all(ray_dir[:, 2] == 0):  # x軸に平行な光線
                    T = (a * (ray_pos[:, 1]**2 - ray_pos[:, 0] / a +
                         ray_pos[:, 1]**2) / ray_dir[:, 0]).reshape(-1, 1)
                    self.ray_end_pos = self.ray_start_pos + T * self.ray_start_dir
                    self.optical_path_length += (
                        T * self._refractive_index_calc_optical_path_length).flatten()
                    self._normalV_refract_or_reflect = self._calc_normalV_parabola()
                else:  # x軸に平行でない光線
                    #test_dot = np.dot(ray_dir, ray_dir)
                    convex = test_dot < 0  # 凸
                    A = ray_dir[:, 1]**2 + ray_dir[:, 2]**2
                    B = ray_pos[:, 1] * ray_dir[:, 1] + ray_pos[:,
                                                                2] * ray_dir[:, 2] - ray_dir[:, 0] / (2 * a)
                    C = ray_pos[:, 1]**2 + ray_pos[:, 2]**2 - ray_pos[:, 0] / a
                    discriminant = B**2 - A * C
                    sqrt_discriminant = np.sqrt(discriminant)

                    T1 = (-B - sqrt_discriminant) / A
                    T2 = (-B + sqrt_discriminant) / A

                    T = np.where(convex, T1, T2).reshape(-1, 1)
                    self.ray_end_pos = self.ray_start_pos + T * self.ray_start_dir
                    # .flatten() を使用して形状を (173,) に変更
                    self.optical_path_length += (
                        T * self._refractive_index_calc_optical_path_length).flatten()
                    self._normalV_refract_or_reflect = self._calc_normalV_parabola()
            elif max_index == 1:  # y軸向き配置, たぶんOK
                # if ray_dir[0][0] == 0 and ray_dir[0][2] == 0:  # y軸に平行な光線
                #     T = [a*(ray_pos[i][0]**2 - ray_pos[i][1]/a + ray_pos[i][0]
                #             ** 2) / ray_dir[i][1] for i in range(length_ray_start_dir)]
                #     self.ray_end_pos = self.ray_start_pos + T*self.ray_start_dir
                #     self.optical_path_length += np.array(T) * \
                #         self._refractive_index_calc_optical_path_length
                #     self._normalV_refract_or_reflect = self._calc_normalV_parabola()
                # else:  # y軸に平行でない光線
                #     if test_dot < 0:  # 凸
                #         A = [ray_dir[i][0]**2 + ray_dir[i][2] **
                #              2 for i in range(length_ray_start_dir)]
                #         B = [ray_pos[i][0]*ray_dir[i][0] + ray_pos[i][2]*ray_dir[i]
                #              [2] - ray_dir[i][1]/(2*a) for i in range(length_ray_start_dir)]
                #         C = [ray_pos[i][0]**2 + ray_pos[i][2]**2 - ray_pos[i]
                #              [1]/a for i in range(length_ray_start_dir)]
                #         T = [(-B[i] - np.sqrt(B[i]**2 - A[i]*C[i])) / A[i]
                #              for i in range(length_ray_start_dir)]
                #         self.ray_end_pos = np.array([
                #             self.ray_start_pos[i] + T[i]*self.ray_start_dir[i] for i in range(length_ray_start_dir)])
                #         self.optical_path_length += np.array([np.array(
                #             T[i])*self._refractive_index_calc_optical_path_length for i in range(length_ray_start_dir)])
                #         self._normalV_refract_or_reflect = self._calc_normalV_parabola()
                #     else:  # 凹
                #         A = [ray_dir[i][0]**2 + ray_dir[i][2] **
                #              2 for i in range(length_ray_start_dir)]
                #         B = [ray_pos[i][0]*ray_dir[i][0] + ray_pos[i][2]*ray_dir[i]
                #              [2] - ray_dir[i][1]/(2*a) for i in range(length_ray_start_dir)]
                #         C = [ray_pos[i][0]**2 + ray_pos[i][2]**2 - ray_pos[i]
                #              [1]/a for i in range(length_ray_start_dir)]
                #         T = [(-B[i] + np.sqrt(B[i]**2 - A[i]*C[i])) / A[i]
                #              for i in range(length_ray_start_dir)]
                #         self.ray_end_pos = np.array([
                #             self.ray_start_pos[i] + T[i]*self.ray_start_dir[i] for i in range(length_ray_start_dir)])
                #         self.optical_path_length += np.array([np.array(
                #             T[i])*self._refractive_index_calc_optical_path_length for i in range(length_ray_start_dir)])
                #         self._normalV_refract_or_reflect = self._calc_normalV_parabola()
                if np.all(ray_dir[:, 0] == 0) and np.all(ray_dir[:, 2] == 0):  # y軸に平行な光線
                    T = (a * (ray_pos[:, 0]**2 - ray_pos[:, 1] / a +
                         ray_pos[:, 0]**2) / ray_dir[:, 1]).reshape(-1, 1)
                    self.ray_end_pos = self.ray_start_pos + T * self.ray_start_dir
                    self.optical_path_length += (
                        T * self._refractive_index_calc_optical_path_length).flatten()
                    self._normalV_refract_or_reflect = self._calc_normalV_parabola()
                else:  # y軸に平行でない光線
                    #test_dot = np.dot(ray_dir, ray_dir)
                    convex = test_dot < 0  # 凸
                    A = ray_dir[:, 0]**2 + ray_dir[:, 2]**2
                    B = ray_pos[:, 0] * ray_dir[:, 0] + ray_pos[:,
                                                                2] * ray_dir[:, 2] - ray_dir[:, 1] / (2 * a)
                    C = ray_pos[:, 0]**2 + ray_pos[:, 2]**2 - ray_pos[:, 1] / a
                    discriminant = B**2 - A * C
                    sqrt_discriminant = np.sqrt(discriminant)

                    T1 = (-B - sqrt_discriminant) / A
                    T2 = (-B + sqrt_discriminant) / A

                    T = np.where(convex, T1, T2).reshape(-1, 1)
                    self.ray_end_pos = self.ray_start_pos + T * self.ray_start_dir
                    # .flatten() を使用して形状を (173,) に変更
                    self.optical_path_length += (
                        T * self._refractive_index_calc_optical_path_length).flatten()
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
                        self.ray_end_pos = np.array([
                            self.ray_start_pos[i] + T[i]*self.ray_start_dir[i] for i in range(length_ray_start_dir)])
                        self.optical_path_length += np.array([np.array(
                            T[i])*self._refractive_index_calc_optical_path_length for i in range(length_ray_start_dir)])
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
                        self.ray_end_pos = np.array([
                            self.ray_start_pos[i] + T[i]*self.ray_start_dir[i] for i in range(length_ray_start_dir)])
                        self.optical_path_length += np.array([np.array(
                            T[i])*self._refractive_index_calc_optical_path_length for i in range(length_ray_start_dir)])
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
        # print("非球面のレイトレーシングは実装中です。")
        if self._conic_K == -1.000:  # 放物線の場合
            self.raytrace_parabola()
        else:  # 放物線以外の非球面の場合
            aspherical_pos = self._surface_pos
            aspherical_R = self._lens_or_parabola_R
            length_ray_start_dir = len(self.ray_start_pos)
            tmp_V = np.zeros(3)
            tmp_index = self._max_index(self._normalV_optical_element)
            tmp_V[tmp_index] = aspherical_R
            #print("tmp_V = ", tmp_V)
            test_dot = np.dot(self.ray_start_dir, tmp_V)
            c = abs(1/aspherical_R)
            if aspherical_R < 0:
                c = c
            else:
                c = -c
            if length_ray_start_dir == 3:  # 光線１本の場合
                ray_pos = np.array(self.ray_start_pos) - \
                    np.array(aspherical_pos)
                ray_dir = self.ray_start_dir
                if ray_dir[1] == 0 and ray_dir[2] == 0:  # x軸に平行な光線の場合
                    #print("x軸に平行な光線の場合 : インデックス自動化未実装")
                    conic_K = self._conic_K
                    T = -1*ray_pos[0] + (1-np.sqrt(1-(1+conic_K)*(c**2) *
                                                   ((ray_pos[1]**2)+(ray_pos[2]**2))))/(c*(1+self._conic_K))
                    # if test_dot < 0:  # 凸面の場合
                    #     print("平行光 : 非球面、凸面")
                    # elif test_dot > 0:  # 凹面の場合
                    #     print("平行光 : 非球面、凹面")
                else:  # x軸に平行でない光線の場合
                    #print("x軸に平行でない光線の場合 : インデックス自動化未実装")
                    if test_dot < 0:  # 凸面の場合
                        # print("非平行光 : 非球面、凸面")
                        conic_K = self._conic_K
                        if 1+conic_K > 0:  # 1+Kが正の場合
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
                        elif 1+conic_K < 0:  # コーニック定数が負の場合
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
                    elif test_dot > 0:  # 凹面の場合
                        # print("非平行光 : 凹面")
                        # c = -abs(1/aspherical_R)  # abs()を取る必要があるかも
                        conic_K = self._conic_K
                        if 1+conic_K > 0:  # コーニック定数が正の場合
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
                        elif 1+conic_K < 0:  # コーニック定数が負の場合
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
                max_index = self._max_index(self._normalV_optical_element)
                length_ray_start_dir = len(self.ray_start_dir)
                tmp_V = np.zeros(3)
                tmp_V[max_index] = aspherical_R
                test_dot = np.dot(
                    self.ray_start_dir[int(length_ray_start_dir/2)], tmp_V)

                ray_pos = np.array(self.ray_start_pos) - \
                    np.array(aspherical_pos)
                ray_dir = self.ray_start_dir
                if ray_dir[0][1] == 0 and ray_dir[0][2] == 0:  # x軸に平行な光線の場合
                    #print("x軸に平行な光線の場合 : インデックス自動化未実装")
                    conic_K = self._conic_K
                    # 係数Tの計算、ここでは凹凸関係なし
                    T = [-1*ray_pos[i][0] + (1-np.sqrt(1-(1+conic_K)*(c**2) * ((ray_pos[i][1]**2)+(ray_pos[i][2]**2))))/(c*(1+self._conic_K))
                         for i in range(length_ray_start_dir)]
                    # if test_dot < 0:  # 凸面の場合
                    #     print("平行光 : 非球面、凸面")
                    # elif test_dot > 0:  # 凹面の場合
                    #     print("平行光 : 非球面、凹面")
                else:  # x軸に平行でない光線の場合
                    #print("x軸に平行でない光線の場合 : インデックス自動化未実装")
                    if test_dot < 0:  # 凸面の場合
                        # print("非平行光 : 非球面、凸面")
                        conic_K = self._conic_K
                        if 1+conic_K > 0:  # 1+Kが正の場合
                            A_1 = [(c**2)*((1+conic_K)**2)*(ray_dir[i][0]**2)
                                   for i in range(length_ray_start_dir)]
                            B_1 = [2*(c**2)*((1+conic_K)**2) *
                                   (ray_pos[i][0]*ray_dir[i][0]) -
                                   2*c*(1+conic_K)*ray_dir[i][0]
                                   for i in range(length_ray_start_dir)]
                            C_1 = [(c**2)*((1+conic_K)**2) *
                                   (ray_pos[i][0]**2) - 2*c *
                                   (1+conic_K)*ray_pos[i][0] + 1
                                   for i in range(length_ray_start_dir)]
                            A_2 = [-1*(c**2)*(1+conic_K) *
                                   ((ray_dir[i][1]**2)+(ray_dir[i][2]**2))
                                   for i in range(length_ray_start_dir)]
                            B_2 = [-2*(c**2)*(1+conic_K) *
                                   (ray_pos[i][1]*ray_dir[i][1] +
                                    ray_pos[i][2]*ray_dir[i][2])
                                   for i in range(length_ray_start_dir)]
                            C_2 = [1 - (1+conic_K)*(c**2) *
                                   ((ray_pos[i][1]**2)+(ray_pos[i][2]**2))
                                   for i in range(length_ray_start_dir)]
                            A = np.array(A_1)-np.array(A_2)
                            B = np.array(B_1)-np.array(B_2)
                            C = np.array(C_1)-np.array(C_2)
                            T = [(-B[i] - np.sqrt(B[i]**2 - 4*A[i]*C[i])) / (2*A[i])
                                 for i in range(length_ray_start_dir)]
                        elif 1+conic_K < 0:  # コーニック定数が負の場合
                            A_1 = [(c**2)*((1+conic_K)**2)*(ray_dir[i][0]**2)
                                   for i in range(length_ray_start_dir)]
                            B_1 = [2*(c**2)*((1+conic_K)**2) *
                                   (ray_pos[i][0]*ray_dir[i][0]) -
                                   2*c*abs(1+conic_K)*ray_dir[i][0]
                                   for i in range(length_ray_start_dir)]
                            C_1 = [(c**2)*((1+conic_K)**2) *
                                   (ray_pos[i][0]**2) - 2*c *
                                   abs(1+conic_K)*ray_pos[i][0] + 1
                                   for i in range(length_ray_start_dir)]
                            A_2 = [-1*(c**2)*abs(1+conic_K) *
                                   ((ray_dir[i][1]**2)+(ray_dir[i][2]**2))
                                   for i in range(length_ray_start_dir)]
                            B_2 = [-2*(c**2)*abs(1+conic_K) *
                                   (ray_pos[i][1]*ray_dir[i][1] +
                                    ray_pos[i][2]*ray_dir[i][2])
                                   for i in range(length_ray_start_dir)]
                            C_2 = [1 - abs(1+conic_K)*(c**2) *
                                   ((ray_pos[i][1]**2)+(ray_pos[i][2]**2))
                                   for i in range(length_ray_start_dir)]
                            A = np.array(A_1)-np.array(A_2)
                            B = np.array(B_1)-np.array(B_2)
                            C = np.array(C_1)-np.array(C_2)
                            T = [(-B[i] - np.sqrt(B[i]**2 - 4*A[i]*C[i])) / (2*A[i])
                                 for i in range(length_ray_start_dir)]
                    elif test_dot > 0:  # 凹面の場合
                        # print("非平行光 : 凹面")
                        # c = -abs(1/aspherical_R)  # abs()を取る必要があるかも
                        conic_K = self._conic_K
                        if 1+conic_K > 0:  # コーニック定数が正の場合
                            A_1 = [(c**2)*((1+conic_K)**2)*(ray_dir[i][0]**2)
                                   for i in range(length_ray_start_dir)]
                            B_1 = [2*(c**2)*((1+conic_K)**2) *
                                   (ray_pos[i][0]*ray_dir[i][0]) -
                                   2*c*(1+conic_K)*ray_dir[i][0]
                                   for i in range(length_ray_start_dir)]
                            C_1 = [(c**2)*((1+conic_K)**2) *
                                   (ray_pos[i][0]**2) - 2*c *
                                   (1+conic_K)*ray_pos[i][0] + 1
                                   for i in range(length_ray_start_dir)]
                            A_2 = [-1*(c**2)*(1+conic_K) *
                                   ((ray_dir[i][1]**2)+(ray_dir[i][2]**2))
                                   for i in range(length_ray_start_dir)]
                            B_2 = [-2*(c**2)*(1+conic_K) *
                                   (ray_pos[i][1]*ray_dir[i][1] +
                                    ray_pos[i][2]*ray_dir[i][2])
                                   for i in range(length_ray_start_dir)]
                            C_2 = [1 - (1+conic_K)*(c**2) *
                                   ((ray_pos[i][1]**2)+(ray_pos[i][2]**2))
                                   for i in range(length_ray_start_dir)]
                            A = np.array(A_1)-np.array(A_2)
                            B = np.array(B_1)-np.array(B_2)
                            C = np.array(C_1)-np.array(C_2)
                            T = [(-B[i] + np.sqrt(B[i]**2 - 4*A[i]*C[i])) / (2*A[i])
                                 for i in range(length_ray_start_dir)]
                        elif 1+conic_K < 0:  # コーニック定数が負の場合
                            A_1 = [(c**2)*((1+conic_K)**2)*(ray_dir[i][0]**2)
                                   for i in range(length_ray_start_dir)]
                            B_1 = [2*(c**2)*((1+conic_K)**2) *
                                   (ray_pos[i][0]*ray_dir[i][0]) -
                                   2*c*abs(1+conic_K)*ray_dir[i][0]
                                   for i in range(length_ray_start_dir)]
                            C_1 = [(c**2)*((1+conic_K)**2) *
                                   (ray_pos[i][0]**2) - 2*c *
                                   abs(1+conic_K)*ray_pos[i][0] + 1
                                   for i in range(length_ray_start_dir)]
                            A_2 = [-1*(c**2)*abs(1+conic_K) *
                                   ((ray_dir[i][1]**2)+(ray_dir[i][2]**2))
                                   for i in range(length_ray_start_dir)]
                            B_2 = [-2*(c**2)*abs(1+conic_K) *
                                   (ray_pos[i][1]*ray_dir[i][1] +
                                    ray_pos[i][2]*ray_dir[i][2])
                                   for i in range(length_ray_start_dir)]
                            C_2 = [1 - abs(1+conic_K)*(c**2) *
                                   ((ray_pos[i][1]**2)+(ray_pos[i][2]**2))
                                   for i in range(length_ray_start_dir)]
                            A = np.array(A_1)-np.array(A_2)
                            B = np.array(B_1)-np.array(B_2)
                            C = np.array(C_1)-np.array(C_2)
                            T = [(-B[i] + np.sqrt(B[i]**2 - 4*A[i]*C[i])) / (2*A[i])
                                 for i in range(length_ray_start_dir)]
                self.ray_end_pos = self.ray_start_pos + \
                    [V*T for V, T in zip(self.ray_start_dir, T)]
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
        if length_ray_start_dir == 3:  # 光線1本の場合
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
                print("Error: _calc_normalV_aspherical")
        else:  # 光線群の場合
            tmp_index = self._max_index(
                self._normalV_optical_element)  # 方向の計算に使う
            ray_pos = self.ray_end_pos
            if tmp_index == 0:  # x軸方向配置の場合
                normalV_list = []
                for i in range(length_ray_start_dir):
                    tmp_root = np.sqrt(
                        1-(1+conic_K)*(ray_pos[i][1]**2+ray_pos[i][2]**2)*(c**2))
                    normalV_0 = 1
                    normalV_1 = ((2*c*ray_pos[i][1]*(1+tmp_root))-((ray_pos[i][1]**2+ray_pos[i][2]**2)*c*(-1*(
                        c**2)*ray_pos[i][1]*(1+conic_K))/tmp_root))/((1+tmp_root)**2)
                    normalV_2 = ((2*c*ray_pos[i][2]*(1+tmp_root))-((ray_pos[i][1]**2+ray_pos[i][2]**2)*c*(-1*(
                        c**2)*ray_pos[i][2]*(1+conic_K))/tmp_root))/((1+tmp_root)**2)
                    normalV = np.array([-normalV_0, -normalV_1, -normalV_2])
                    normalV = normalV/np.linalg.norm(normalV)
                    normalV_list.append(normalV)
                return normalV_list
            else:
                print("Error: _calc_normalV_aspherical")

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
            self.ray_end_dir = np.array(outRayV)

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
        length_ray_start_dir = len(self.ray_start_pos)
        # print("normalV", normalV[int(length_ray_start_dir/2)])
        Nin = self._refractive_index_before
        Nout = self._refractive_index_after
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
            #print(np.dot(ray_dir[0], normalV[0]))
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
            # print("outRayV", outRayV[int(length_ray_start_dir/2)])
            self.ray_end_dir = np.array(outRayV)

    # スネルの法則(逆向き)
    def refract_reverse(self):
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
        length_ray_start_dir = len(self.ray_start_pos)
        print("reverse, normalV", normalV[int(length_ray_start_dir/2)])
        Nin = self._refractive_index_before
        Nout = self._refractive_index_after
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
            #print(np.dot(ray_dir[0], normalV[0]))
            ray_dir_tmp = ray_dir[0]
            normalV_tmp = [1, 0, 0]
            for V in ray_dir:
                # print(V)
                if V[0] == np.nan:
                    continue
                else:
                    ray_dir_tmp = V
            # for V in normalV:
            #     if V[0] == np.nan:
            #         continue
            #     else:
            #         normalV_tmp = V
            print("ray_dir_tmp", ray_dir_tmp)
            print("normalV_tmp", normalV_tmp)
            if np.dot(ray_dir_tmp, normalV_tmp) > 0:
                print("reverse, 内積が負です")
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
                print("reverse, 内積が正です")
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
            print("reverse, outRayV", outRayV[int(length_ray_start_dir/2)])
            self.ray_end_dir = outRayV

    # 焦点距離を計算する関数
    def calc_focal_length(self, ray_start_pos_init=None):
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
        if ray_start_pos_init is None:
            ray_start_pos_init = self.ray_init_pos
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
            # print("argmin_index = ", argmin_index)
            focal_length = []
            for i in range(length_ray_dir):
                tmp_V = -1. * \
                    self.ray_end_dir[i]*ray_start_pos_init[i][argmin_index] / \
                    self.ray_end_dir[i][argmin_index]
                argmax_index = self._max_index(tmp_V)
                focal_length.append(tmp_V[argmax_index])
            # 並び替え
            focal_length.sort()
            focal_length_mean = np.round(np.nanmean(focal_length), 5)
            focal_length_std = np.round(np.nanstd(focal_length), 5)
            focal_length_max = np.round(np.nanmax(focal_length), 5)
            focal_length_min = np.round(np.nanmin(focal_length), 5)
            print("focal_length_mean: ", focal_length_mean,
                  "std: ", focal_length_std,
                  "max: ", focal_length_max, "min: ", focal_length_min)
            return focal_length

    # 焦点位置を計算する関数
    def calc_focal_pos(self, ray_start_pos_init=None):
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
        if ray_start_pos_init is None:
            ray_start_pos_init = self.ray_init_pos
        length_ray_dir = len(self.ray_end_pos)
        if length_ray_dir == 3:  # 光線が1本の場合
            argmin_index = self._min_index(self.ray_end_dir)
            tmp_V = -1.*self.ray_end_dir * \
                self.ray_end_pos[argmin_index]/self.ray_end_dir[argmin_index]
            focal_point = tmp_V + self.ray_end_pos
            argmax_index = self._max_index(tmp_V)
            print("!!!!正確な焦点位置を得るには近軸光線を計算する必要があります!!!!")
            return focal_point[argmax_index]
        else:  # 光線が複数の場合
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
    def plot_line_plotly(self, color='red', alpha=0.7, ms=1.0):
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
            # self._ax.plot([startX, endX], [startY, endY], [startZ, endZ],
            #               fmt, ms=ms, linewidth=0.5, color='r', alpha=alpha)
            self._fig_plotly.add_trace(go.Scatter3d(
                x=[startX, endX],
                y=[startY, endY],
                z=[startZ, endZ],
                mode='lines+markers',
                opacity=alpha,
                line=dict(color=color, width=1),
                marker=dict(size=ms, color=color),
                showlegend=False,
                hoverinfo='none',
                ))
        else:
            startPointV = self.ray_start_pos
            endPointV = self.ray_end_pos
            startX = startPointV[:, 0]
            startY = startPointV[:, 1]
            startZ = startPointV[:, 2]
            endX = endPointV[:, 0]
            endY = endPointV[:, 1]
            endZ = endPointV[:, 2]
            for i in range(length_ray_start_dir):
                # self._ax.plot([startX[i], endX[i]], [startY[i], endY[i],], [startZ[i], endZ[i]],
                #               fmt, ms=ms, linewidth=0.5, color='r', alpha=alpha)
                self._fig_plotly.add_trace(go.Scatter3d(
                    x=[startX[i], endX[i]],
                    y=[startY[i], endY[i]],
                    z=[startZ[i], endZ[i]],
                    mode='lines+markers',
                    opacity=alpha,
                    line=dict(color=color, width=1),
                    marker=dict(size=ms, color=color),
                    showlegend=False,
                    hoverinfo='none',
                    ))

    def plot_line_blue(self, alpha=1.0, fmt='o-', ms=2):
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
                          fmt, ms=ms, linewidth=0.5, color='blue', alpha=alpha)
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
                              fmt, ms=ms, linewidth=0.5, color='blue', alpha=alpha)

    def plot_line_green(self, alpha=1.0, fmt='o-', ms=2):
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
                          fmt, ms=ms, linewidth=0.5, color='green', alpha=alpha)
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
                              fmt, ms=ms, linewidth=0.5, color='green', alpha=alpha)

    def plot_line_red(self, alpha=1.0, fmt='o-', ms=2):
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
                          fmt, ms=ms, linewidth=0.5, color='r', alpha=alpha)
        else:
            # for i in range(length_ray_start_dir):
            #     startPointV = self.ray_start_pos[i]
            #     endPointV = self.ray_end_pos[i]
            #     startX = startPointV[0]
            #     startY = startPointV[1]
            #     startZ = startPointV[2]
            #     endX = endPointV[0]
            #     endY = endPointV[1]
            #     endZ = endPointV[2]
            #     self._ax.plot([startX, endX], [startY, endY], [startZ, endZ],
            #                   fmt, ms=ms, linewidth=0.5, color='r', alpha=alpha)
            startPointV = self.ray_start_pos
            endPointV = self.ray_end_pos
            startX = startPointV[:, 0]
            startY = startPointV[:, 1]
            startZ = startPointV[:, 2]
            endX = endPointV[:, 0]
            endY = endPointV[:, 1]
            endZ = endPointV[:, 2]
            for i in range(length_ray_start_dir):
                self._ax.plot([startX[i], endX[i]], [startY[i], endY[i],], [startZ[i], endZ[i]],
                              fmt, ms=ms, linewidth=0.5, color='r', alpha=alpha)

    def plot_line_orange(self, alpha=1.0, fmt='o-', ms=2):
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
                          fmt, ms=ms, linewidth=0.5, color='orange', alpha=alpha)
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
                              fmt, ms=ms, linewidth=0.5, color='orange', alpha=alpha)

    def plot_four_beam_line(self, i, alpha=1.0, fmt='o-', ms=2):
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
            self.plot_line_blue(alpha=alpha, fmt=fmt, ms=ms)
        elif i == 1:
            self.plot_line_green(alpha=alpha, fmt=fmt, ms=ms)
        elif i == 2:
            self.plot_line_red(alpha=alpha, fmt=fmt, ms=ms)
        elif i == 3:
            self.plot_line_orange(alpha=alpha, fmt=fmt, ms=ms)
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

    def __str__(self) -> str:
        return f'ray_end_pos.shape: {self.ray_end_pos.shape}, ray_end_dir.shape: {self.ray_end_dir.shape}, optical_path_length.shape: {self.optical_path_length.shape}'

    def __repr__(self) -> str:
        return f'VectorFunctions("{self._ax}",\
                                "{self.ray_init_pos}",\
                                "{self.ray_init_dir}",\
                                "{self.ray_start_pos}",\
                                "{self.ray_start_dir}",\
                                "{self.ray_end_pos}",\
                                "{self.ray_end_dir}",\
                                "{self._surface_pos}",\
                                "{self._limit_R}",\
                                "{self._lens_or_parabola_R}",\
                                "{self._conic_K}",\
                                "{self._surface_name}",\
                                "{self._refractive_index_before}",\
                                "{self._refractive_index_after}",\
                                "{self._refractive_index_calc_optical_path_length}",\
                                "{self._normalV_optical_element}",\
                                "{self._normalV_refract_or_reflect}",\
                                "{self.optical_path_length}")'
