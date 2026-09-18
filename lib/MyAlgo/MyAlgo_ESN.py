from ._compat import BaseMethod, TSData, pick_torch_device
import numpy as np
import torch

# pyrcn is incompatible with scikit-learn >= 1.4 (removed private helper).
# Provide a minimal shim so ESNRegressor can still import.
import sklearn.utils.validation as _sk_validation

if not hasattr(_sk_validation, "_check_fit_params"):

    def _check_fit_params(X, params, indices=None):  # noqa: N802
        return params

    _sk_validation._check_fit_params = _check_fit_params

from pyrcn.echo_state_network import ESNRegressor



class MyAlgo_ESN(BaseMethod):
    def __init__(self, params:dict) -> None:
        super().__init__()
        self.__anomaly_score = None
        self.score_train = None
        self.y_hat_train = None

        self.device = pick_torch_device()
        self.cuda = self.device.type == "cuda"
        print(f"=== Using {self.device.type.upper()} ===")

        params = params or {}
        self.p = int(params.get("window_size", 100))

        
    def train_valid_phase(self, tsTrain: TSData):
        n_future = 1

        train_x = tsTrain.train[:-n_future] 
        train_y = tsTrain.train[n_future:]


        self.model = ESNRegressor().fit(X=train_x.reshape(-1, 1), y=train_y)
        output = self.model.predict(train_x.reshape(-1, 1))
        self.y_hat_train = output
        self.score_train = np.square(output - train_y)

    
    def test_phase(self, tsData: TSData):

        # windows = np.lib.stride_tricks.sliding_window_view(tsData.test, window_shape=100) 
        # target_in_windows = np.array([w[0] for w in windows[1:]])
        # windows = windows[:-1]

        n_future = 1

        test_x = tsData.test[:-n_future] 
        test_y = tsData.test[n_future:]


        output = self.model.predict(test_x.reshape(-1, 1))
        
        loss = np.square(output - test_y)
        self.output = output
        self.__anomaly_score = loss

    def anomaly_score(self) -> np.ndarray:
        return self.__anomaly_score
    
    def get_output(self) -> np.ndarray:
        return self.output
