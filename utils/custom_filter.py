from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor

import numpy as np

# オリジナルのプログラムに合わせた形でインポート
from numpy import eye, dot, isscalar

from filterpy.kalman import UnscentedKalmanFilter, unscented_transform


class CustomUKF(UnscentedKalmanFilter):
    """
    filterpyのUnscentedKalmanFilterを拡張したクラス

    以下を拡張
    * 状態変数、中間変数、観測変数が扱えるように拡張
    * 観測変数が1次元でも処理が行えるように拡張
    """

    def __init__(
        self,
        dim_x,
        dim_z,
        dt,
        hx,
        fx,
        gx,
        points,
        sqrt_fn=None,
        x_mean_fn=None,
        z_mean_fn=None,
        residual_x=None,
        residual_z=None,
        state_add=None,
        max_workers=0,
    ):
        """
        コンストラクタ
        """

        super().__init__(
            dim_x=dim_x,
            dim_z=dim_z,
            dt=dt,
            hx=hx,
            fx=fx,
            points=points,
            sqrt_fn=sqrt_fn,
            x_mean_fn=x_mean_fn,
            z_mean_fn=z_mean_fn,
            residual_x=residual_x,
            residual_z=residual_z,
        )

        # Hidden states between x and z
        self.s = None
        self.sigmas_s = None
        self.s_shape = None

        self.gx = gx

        # Program of newer version of filterpy
        if state_add is None:
            self.state_add = np.add
        else:
            self.state_add = state_add

        self.max_workers = max_workers

        self.count_singular = 0

    def initialize_states(self, states):
        self.s_shape = states.shape
        self.s = states.flatten()

        self.sigmas_s = np.zeros((self._num_sigmas, self.s.shape[0]))
        for i in range(self._num_sigmas):
            self.sigmas_s[i] = self.s

    # NOTE Extended.
    # Add method to call gx with a single argument.
    def call_gx(self, sigmas):
        x = sigmas[0]
        s = sigmas[1]
        return self.gx(x, s, multi_thread="", **self.gx_args)

    def update(self, z, R=None, UT=None, hx=None, gx=None, hx_args={}, gx_args={}):

        # NOTE: Move to the top of this method, to use UT even if z is None.
        if UT is None:
            UT = unscented_transform

        # NOTE: Extended
        # Make a transition of hidden states
        if gx is None:
            gx = self.gx
        if self.s is not None:
            # NOTE: Extended
            # Use Multi Thred Processing
            if self.max_workers > 0:
                # Static argument for gx.
                self.gx_args = gx_args
                # Sigmas.
                sigmas = [[x, s.reshape(self.s_shape)] for x, s in zip(self.sigmas_f, self.sigmas_s)]

                with ThreadPoolExecutor(self.max_workers) as executor:
                    ret = executor.map(self.call_gx, sigmas)
                sigmas_s = [r.flatten() for r in ret]

            else:
                # sigmas_sの更新
                sigmas_s = []
                for x, s in zip(self.sigmas_f, self.sigmas_s):
                    sigmas_s.append(gx(x, s.reshape(self.s_shape), **gx_args).flatten())

            self.sigmas_s = np.atleast_2d(sigmas_s)
            self.s, _ = UT(self.sigmas_s, self.Wm, self.Wc)
            # if len(self.s[self.s < 0]) > 0:
            #     for i, s_ in enumerate(self.sigmas_s):
            #         if len(s_[s_ < 0]) > 0:
            #             print(i, s_)

        if z is None:
            self.z = np.array([[None] * self._dim_z]).T
            self.x_post = self.x.copy()
            self.P_post = self.P.copy()
            return

        if hx is None:
            hx = self.hx

        if R is None:
            R = self.R
        elif isscalar(R):
            R = eye(self._dim_z) * R

        # pass prior sigmas through h(x) to get measurement sigmas
        # the shape of sigmas_h will vary if the shape of z varies, so
        # recreate each time
        sigmas_h = []
        for s in self.sigmas_s:
            # NOTE: Extended
            # Function hx requires hidden states as an argument.
            sigmas_h.append(hx(s.reshape(self.s_shape), **hx_args))

        self.sigmas_h = np.atleast_2d(sigmas_h)

        # NOTE: Extended
        # Make this method available for dim_z = 1.
        if self._dim_z == 1:
            self.sigmas_h = self.sigmas_h.reshape((self.sigmas_h.shape[0], self.sigmas_h.shape[1]))

        # mean and covariance of prediction passed through unscented transform
        zp, self.S = UT(self.sigmas_h, self.Wm, self.Wc, R, self.z_mean, self.residual_z)
        try:
            self.SI = self.inv(self.S)
        except:
            if self.count_singular == 0:
                print("Warning: Singular matrix.")
                self.count_singular = 1
            self.SI = np.linalg.pinv(self.S)

        # compute cross variance of the state and the measurements
        Pxz = self.cross_variance(self.x, zp, self.sigmas_f, self.sigmas_h)
        self.K = dot(Pxz, self.SI)  # Kalman gain
        self.y = self.residual_z(z, zp)  # residual

        # update Gaussian state estimate (x, P)
        self.x = self.state_add(self.x, dot(self.K, self.y))
        self.P = self.P - dot(self.K, dot(self.S, self.K.T))

        # save measurement and posterior state
        self.z = deepcopy(z)
        self.x_post = self.x.copy()
        self.P_post = self.P.copy()

        # set to None to force recompute
        self._log_likelihood = None
        self._likelihood = None
        self._mahalanobis = None

        # NOTE: Extended
        # NOTE: 謎の処理。中間変数のシグマ点列をUT変換で得られた平均値に統一している。
        #       しかしこれでは次時点のsigmas_sの計算で前時点のsimgas_sとしてすべて同じ値を使用することになる。
        # If DA is done (if reach hear), integrate sigma points for hidden states
        # for i in range(self.sigmas_s.shape[0]):
        #     self.sigmas_s[i] = self.s
