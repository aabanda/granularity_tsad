import copy
import os
import json
import numpy as np
import sys
import pandas as pd
from .BaseSchema import BaseSchema
from ..Methods import BaseMethodMeta
from ..Controller.PathManager import PathManager

class Naive(BaseSchema):
    def __init__(self, dc, method, cfg_path:str=None, diff_order:int=None, preprocess:str=None) -> None:
        """
        Initializes an instance of the OneByOne class.

        Args:
            - `dc` (dict): Data configuration parameters.
            - `ec` (dict): Evaluation configuration parameters.
            - `method` (str): The method being used.
            - `cfg_path` (str, optional): Path to a custom configuration file. Defaults to None.
            - `diff_order` (int, optional): The differential order. Defaults to None.
            - `preprocess` (str, optional): The preprocessing method. Options: "raw", "min-max", "z-score". Defaults to None (equals to "raw"). 
        """
        super().__init__(dc, method, "naive", cfg_path, diff_order, preprocess)
        self.pm = PathManager.get_instance()
        
    def do_exp(self, tsDatas,replace, hparams=None):
        for dataset_name, value in tsDatas.items():
            if "Model_Params" in self.cfg:
                model_params = self.cfg["Model_Params"]["Default"]
                if dataset_name in self.cfg["Model_Params"]:
                    specific_params = self.cfg["Model_Params"][dataset_name]
                    for k, v in specific_params.items():
                        model_params[k] = v
                        
            if hparams is not None:
                model_params = hparams
            
            self.train_valid_timer.reset_total()
            self.test_timer.reset_total()
            
            for curve_name, curve in value.items():
                score_path = self.pm.get_score_path(self.method, self.schema, dataset_name, curve_name)
                if (os.path.isfile(score_path) & replace) or (not os.path.isfile(score_path)):
                    self.logger.info("    [{}] handling dataset {} | curve {} ".format(self.method, dataset_name, curve_name))
                    
                    ## training & test step
                    if self.method in BaseMethodMeta.registry:
                        method = BaseMethodMeta.registry[self.method](model_params)
                    else:
                        raise ValueError("Unknown method class \"{}\". Ensure that the method name matches the class name exactly (including case sensitivity).".format(self.method))
                    
                    statistic_path = self.pm.get_rt_statistic_path(self.method, self.schema, dataset_name)
                    method.param_statistic(statistic_path)
                    
                    self.train_valid_timer.tic()
                    method.train_valid_phase(curve)
                    self.train_valid_timer.toc()
                    
                    # pd.DataFrame(method.reconstructed_sequence).to_csv(self.pm.get_yhat_path(self.method, self.schema, dataset_name, curve_name), index=False)
                    # pd.DataFrame(method.y_hats).to_csv(self.pm.get_yhat_path(self.method, self.schema, dataset_name, curve_name), index=False)
                    try:
                        pd.DataFrame(method.y_hat_train).to_csv(self.pm.get_train_yhat_path(self.method, self.schema, dataset_name, curve_name), index=False)
                    except:
                        pass
                    pd.DataFrame(method.score_train).to_csv(self.pm.get_train_score_path(self.method, self.schema, dataset_name, curve_name), index=False)

                    self.test_timer.tic()
                    method.test_phase(curve)
                    self.test_timer.toc()
                    
                    # pd.DataFrame(method.reconstructed_sequence).to_csv(self.pm.get_yhat_path(self.method, self.schema, dataset_name, curve_name), index=False)
                    # pd.DataFrame(method.y_hats).to_csv(self.pm.get_yhat_path(self.method, self.schema, dataset_name, curve_name), index=False)
                    # pd.DataFrame(method.get_output()).to_csv(self.pm.get_yhat_path(self.method, self.schema, dataset_name, curve_name), index=False)
                    # pd.DataFrame(method.get_y_hat()).to_csv(self.pm.get_yhat_path(self.method, self.schema, dataset_name, curve_name), index=False)
                    # score = method.anomaly_score()

                    # np.save(score_path, score)
                    # a =np.load('/home/110646@TRI.LAN/PycharmProjects/tsad/Results/Scores/AE/naive/UCR/1.npy')
                    # b =np.load('/home/110646@TRI.LAN/PycharmProjects/tsad/Results/Test_Scores/AE/naive/UCR/1.npy')
                    # import matplotlib.pyplot as plt
                    # plt.plot(a, label='a')
                    # plt.plot(b, label='b')
                    # plt.legend()
                    
                    
            # save running time info
            time_path = self.pm.get_rt_time_path(self.method, self.schema, dataset_name)
            time_dict = {
                "train_and_valid": self.train_valid_timer.get_total_time(),
                "test": self.test_timer.get_total_time()
            }
            with open(time_path, 'w') as f:
                json.dump(time_dict, f, indent=4)
                
                