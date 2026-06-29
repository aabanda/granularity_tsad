from MyAlgo._compat import BaseMethod, TSData
import numpy as np
from pyod.models.cblof import CBLOF




class MyAlgo_CLOF(BaseMethod):
    def __init__(self, hparams) -> None:
        super().__init__()
        self.__anomaly_score = None
        self.model_name = "CBLOF"
        self.device = "cuda"
        self.outliers_fraction = 0.05
        self.window_size = 100
        self.score_train = None
        self.model =  CBLOF(
            n_clusters=10,
            contamination=self.outliers_fraction,
            check_estimator=False,
        )


    def train_valid_phase(self, tsTrain: TSData):

        self.model.fit(tsTrain.train.reshape(-1, 1) )
        self.score_train = self.model.decision_function(tsTrain.train.reshape(-1, 1))


        
    def test_phase(self, tsData: TSData):

        self.__anomaly_score = self.model.decision_function(tsData.test.reshape(-1, 1))


        
    def train_valid_phase_all_in_one(self, tsTrains: dict[str, TSData]):
        # used for all-in-one and zero-shot mode
        pass

    def anomaly_score(self) -> np.ndarray:
        return self.__anomaly_score

    def param_statistic(self, save_file):
        pass

