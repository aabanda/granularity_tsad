from ._compat import BaseMethod, TSData
import numpy as np
import torch
from pyrcn.echo_state_network import ESNRegressor



class MyAlgo_ESN(BaseMethod):
    def __init__(self, params:dict) -> None:
        super().__init__()
        self.__anomaly_score = None
        self.score_train = None
        self.y_hat_train = None

        self.cuda = True
        if self.cuda == True and torch.cuda.is_available():
            self.device = torch.device("cuda")
            print("=== Using CUDA ===")
        else:
            if self.cuda == True and not torch.cuda.is_available():
                print("=== CUDA is unavailable ===")
            self.device = torch.device("cpu")
            print("=== Using CPU ===")
            
        self.p = params["window_size"]

        
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
