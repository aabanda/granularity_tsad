import pandas as pd
import numpy as np
from lib.tsadalia.tsdalia.functions import convert_to_windows, load_model
from lib.tsadalia.tsdalia.models import (
    transformer_autoencoder,
    diffusion,
    diffusion_autoencoder,
    basic_autoencoder,
)
from io import StringIO
from lib.tsadalia.tsdalia.models.diffusion import ConditionalDiffusionTrainingNetwork
from dataclasses import dataclass, field
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, precision_recall_curve, auc
import matplotlib.pyplot as plt
import keras
import os
import torch
from torch.utils.data import DataLoader
from pyod.models.cblof import CBLOF
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest
import sys


model_backprop_function = {
    "transformer_autoencoder": transformer_autoencoder.backprop,
    "diffusion": diffusion.backprop,
    "diffusion_autoencoder": diffusion_autoencoder.backprop,
}


@dataclass
class AnomalyDetectorBaseClass:
    window_size: int = 100
    ts_dims: int = 3
    batch_size: int = 128
    lr: float = 0.00001
    device: str = "cuda"
    model_name: str = None
    optimizer: any = None
    scheduler: any = None
    model: any = None
    train_error: any = None
    loss_test: any = None
    verbose: bool = False
 
    def create_windows(
        self,
        train_data: np.ndarray,
        test_data: np.ndarray,
        test_labels: np.array,
        normalize: bool = True,
        dataloader: bool = True,
    ):
        train_data_copy = train_data.copy()
        test_data_copy = test_data.copy()
        test_labels_copy = test_labels.copy()

        # remove windows in train with nans:
        windows = list(torch.split(torch.tensor(train_data_copy), self.window_size))
        for i in range(self.window_size - windows[-1].shape[0]):
            windows[-1] = torch.cat((windows[-1], windows[-1][-1].unsqueeze(0)))
        train_in_windows = torch.stack(windows)  
        train_in_windows = train_in_windows[:-1]
        if len(train_in_windows.shape)==2:
            train_in_windows = train_in_windows.unsqueeze(-1)
        count_nans = []
        for i in range(train_in_windows.shape[0]):
            if torch.sum(torch.isnan(train_in_windows[i, :, :])) > 0:
                count_nans.append(i)
        train_in_windows = train_in_windows.numpy()
        
        if len(count_nans)>1:
            train_in_windows = np.delete(
                train_in_windows, np.array(count_nans), axis=0
            )

        train_x = train_in_windows.reshape(
            (
                train_in_windows.shape[0] * train_in_windows.shape[1],
                train_in_windows.shape[2],
            )
        )

        if len(test_data_copy.shape)==1:
            test_x = test_data_copy[: int(test_data_copy.shape[0] / self.window_size) * self.window_size]
        else:
            test_x = test_data_copy[: int(test_data_copy.shape[0] / self.window_size) * self.window_size, :]
        test_y = test_labels_copy[: int(test_data_copy.shape[0] / self.window_size) * self.window_size]

        if normalize:

            if train_x.shape[1]>1:
                x_min = np.min(train_x, axis=0)
                x_max = np.max(train_x, axis=0)

                for d in range(train_x.shape[1]):
                    test_x[:, d] = (test_x[:, d] - x_min[d]) / (x_max[d] - x_min[d])
                    train_x[:, d] = (train_x[:, d] - x_min[d]) / (x_max[d] - x_min[d])
            else:
                test_x = (test_x - train_x.min()) / (train_x.max() - train_x.min())
                train_x = (train_x -train_x.min()) / (train_x.max() -train_x.min())


        if dataloader:
            train_x = DataLoader(train_x, batch_size=train_x.shape[0])
            test_x = DataLoader(test_x, batch_size=test_x.shape[0])

        return train_x, test_x, test_y

    def fit(self, train_x: pd.DataFrame, num_epochs: int = 100, plot_loss: bool = False):
        pass

    def predict(
        self, test_x: pd.DataFrame, binary: bool = False, train_error: float = None
    ):
        pass

    def evaluate(self, test_y: pd.Series, metric: str = "AUC"):

        if metric == "AUC":

            self.loss_test = self.loss_test.reshape(
                (
                    self.loss_test.shape[0] * self.loss_test.shape[1],
                    self.loss_test.shape[2],
                )
            )

            self.loss_test = np.mean(self.loss_test, axis=1)

            print("----------TEST------------------------")
            print(f"ROC: {roc_auc_score(test_y, self.loss_test)}")
            precision, recall, thresholds = precision_recall_curve(test_y, self.loss_test)
            print(f"PRC: {auc(recall, precision)}")


            return roc_auc_score(test_y, self.loss_test)

        elif metric == "binary":

            print(confusion_matrix(test_y, self.loss_test))


@dataclass
class TransformerAutoEncoderDetector(AnomalyDetectorBaseClass):

    def __post_init__(self):
        self.model_name = "transformer_autoencoder"

    def fit(self, train_x: pd.DataFrame, num_epochs: int = 100, plot_loss: bool = False):

        if not self.verbose:
           output = StringIO()
           sys.stdout = output
       

        train = next(iter(train_x))
        trainD = convert_to_windows(train, self.window_size).cpu()

        (
            self.model,
            self.optimizer,
            epoch,
        ) = load_model(
            model_name=self.model_name,
            lr=self.lr,
            window_size=self.window_size,
            dims=self.ts_dims,
            batch_size=self.batch_size,
            device=self.device,
        )

        len_dataloader = len(trainD) // self.model.batch
        if len(trainD) % self.model.batch:
            len_dataloader += 1
        num_training_steps = len_dataloader * num_epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, 0.1 * num_training_steps, num_training_steps
        )

        print("training")
        e = epoch + 1
        loss_train_hist = [100000000]

        for e in tqdm(list(range(epoch + 1, epoch + num_epochs + 1))):
            loss_train, recons_train = model_backprop_function[self.model_name](
                epoch=e,
                model=self.model,
                data=trainD,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                training=True,
                dims=self.ts_dims,
                batch_size=self.batch_size,
                window_size=self.window_size,
            )
            loss_train_hist.append(np.mean(loss_train))

        loss_train, recons_train = model_backprop_function[self.model_name](
            epoch=e,
            model=self.model,
            data=trainD,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            training=False,
            dims=self.ts_dims,
            batch_size=self.batch_size,
            window_size=self.window_size,
        )
        self.train_error = np.abs(trainD - recons_train).numpy()

        if plot_loss:
            plt.plot(loss_train_hist)

        self.train_error = self.train_error.reshape(
            self.train_error.shape[0] * self.train_error.shape[1], self.ts_dims
        )

    def predict(self, test_x: pd.DataFrame, binary: bool = False):

        test = next(iter(test_x))
        testD = convert_to_windows(test, self.window_size)

        self.loss_test, reconstruction = model_backprop_function[self.model_name](
            epoch=0,
            model=self.model,
            data=testD,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            training=False,
            dims=self.ts_dims,
            batch_size=self.batch_size,
            window_size=self.window_size,
        )

        if binary:
            threshold = np.mean(self.train_error) + 2 * np.std(self.train_error)
            test_error = np.abs(testD - reconstruction).numpy()
            test_error = test_error.reshape(
                test_error.shape[0] * test_error.shape[1], 3
            ).mean(axis=1)

            predictions = np.zeros_like(test_error)
            predictions[test_error > threshold] = 1

            self.loss_test = predictions

        if not self.verbose:
            sys.stdout = sys.__stdout__


@dataclass
class DiffusionDetector(AnomalyDetectorBaseClass):
    noise_steps: int = 100
    denoise_steps: int = 50
    diff_lambda: float = 0.1
    diffusion_training_net: any = None
    params_specific: dict = None

    def __post_init__(self) -> None:

        self.params_specific = {
            "noise_steps": self.noise_steps,
            "denoise_steps": self.denoise_steps,
            "diff_lambda": self.diff_lambda,
        }

        self.diffusion_training_net = ConditionalDiffusionTrainingNetwork(
            nr_feats=self.ts_dims,
            window_size=int(self.window_size),
            batch_size=self.batch_size,
            **self.params_specific,
        ).float()

        self.model_name = "diffusion"

    def fit(self, train_x: pd.DataFrame, num_epochs: int = 100, plot_loss: bool = False):
        if not self.verbose:
           output = StringIO()
           sys.stdout = output

        train = next(iter(train_x))
        trainD = convert_to_windows(train, self.window_size).cpu()

        (
            self.model,
            self.optimizer,
            epoch,
        ) = load_model(
            model_name=self.model_name,
            lr=self.lr,
            window_size=self.window_size,
            dims=self.ts_dims,
            batch_size=self.batch_size,
            device=self.device,
            params_specific=self.params_specific,
            diffusion_training_net=self.diffusion_training_net,
        )

        len_dataloader = len(trainD) // self.model.batch
        if len(trainD) % self.model.batch:
            len_dataloader += 1
        num_training_steps = len_dataloader * num_epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, 0.1 * num_training_steps, num_training_steps
        )

        print("training")
        e = epoch + 1
        loss_train_hist = [100000000]

        for e in tqdm(list(range(epoch + 1, epoch + num_epochs + 1))):
            loss_train, recons_train = model_backprop_function[self.model_name](
                epoch=e,
                model=self.model,
                data=trainD,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                training=True,
                dims=self.ts_dims,
                diffusion_training_net=self.diffusion_training_net,
                batch_size=self.batch_size,
                window_size=self.window_size,
                **self.params_specific,
            )
            loss_train_hist.append(np.mean(loss_train))

        loss_train, recons_train = model_backprop_function[self.model_name](
            epoch=e,
            model=self.model,
            data=trainD,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            training=False,
            dims=self.ts_dims,
            diffusion_training_net=self.diffusion_training_net,
            batch_size=self.batch_size,
            window_size=self.window_size,
            **self.params_specific,
        )
        self.train_error = np.abs(trainD - recons_train).numpy()

        if plot_loss:
            plt.plot(loss_train_hist)

        self.train_error = self.train_error.reshape(
            self.train_error.shape[0] * self.train_error.shape[1], self.train_error.shape[2]
        )

    def predict(self, test_x: pd.DataFrame, binary: bool = False):

        test = next(iter(test_x))
        testD = convert_to_windows(test, self.window_size)

        self.loss_test, reconstruction = model_backprop_function[self.model_name](
            epoch=0,
            model=self.model,
            data=testD,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            diffusion_training_net=self.diffusion_training_net,
            training=False,
            dims=self.ts_dims,
            batch_size=self.batch_size,
            window_size=self.window_size,
            **self.params_specific,
        )

        if binary:
            threshold = np.mean(self.train_error) + 2 * np.std(self.train_error)
            test_error = np.abs(testD - reconstruction).numpy()
            test_error = test_error.reshape(
                test_error.shape[0] * test_error.shape[1], 3
            ).mean(axis=1)

            predictions = np.zeros_like(test_error)
            predictions[test_error > threshold] = 1

            self.loss_test = predictions

        if not self.verbose:
            sys.stdout = sys.__stdout__

@dataclass
class BasicAutoEncoderDetector(AnomalyDetectorBaseClass):

    def __post_init__(self) -> None:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        self.model_name = "basic_autoencoder"

    def fit(self, train_x: pd.DataFrame, num_epochs: int = 100, plot_loss: bool = False):
        if not self.verbose:
           output = StringIO()
           sys.stdout = output

        train = next(iter(train_x))
        trainD = convert_to_windows(train, self.window_size).cpu().detach().numpy()

        self.model = basic_autoencoder.load_model(
            ts_dims=self.ts_dims, window_size=self.window_size
        )
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.00001), loss="mse"
        )

        history = self.model.fit(
            trainD,
            trainD,
            epochs=num_epochs,
            batch_size=self.batch_size,
            validation_split=0.2,
            callbacks=[
                keras.callbacks.EarlyStopping(
                    monitor="val_loss", patience=50, mode="min"
                )
            ],
        )

        loss_train_hist = history.history["loss"]
        recons_train = self.model.predict(trainD)
        self.train_error = np.abs(trainD - recons_train)

        if plot_loss:
            plt.plot(loss_train_hist)

        self.train_error = self.train_error.reshape(
            self.train_error.shape[0] * self.train_error.shape[1], self.train_error.shape[2]
        )

    def predict(self, test_x: pd.DataFrame, binary: bool = False):

        test = next(iter(test_x))
        testD = convert_to_windows(test, self.window_size).cpu().detach().numpy()
        if len(testD.shape)==2:
            testD = np.expand_dims(testD, 2)
        reconstruction = self.model.predict(testD)
        self.loss_test = np.abs(testD - reconstruction)

        if not self.verbose:
            sys.stdout = sys.__stdout__

@dataclass
class BenchmarkDetectors(AnomalyDetectorBaseClass):
    outliers_fraction: float = 0.05
    CBLOF: any = None
    HBOS: any = None
    IFORST: any = None
    scores_CBLOF: any = None
    scores_HBOS: any = None
    scores_IFORST: any = None

    def __post_init__(self) -> None:

        self.CBLOF = CBLOF(
            n_clusters=10,
            contamination=self.outliers_fraction,
            check_estimator=False,
        )
        self.HBOS = HBOS(contamination=self.outliers_fraction)
        self.IFORST = IForest(contamination=self.outliers_fraction)

    def fit(
        self,
        train_x: pd.DataFrame,
    ):

        self.CBLOF.fit(train_x)
        self.HBOS.fit(train_x)
        self.IFORST.fit(train_x)

    def predict(self, test_x: pd.DataFrame):

        self.scores_CBLOF = self.CBLOF.decision_function(test_x)
        self.scores_HBOS = self.HBOS.decision_function(test_x)
        self.scores_IFORST = self.IFORST.decision_function(test_x)

    def evaluate(self, test_y: pd.Series, metric: str = "AUC"):
        r1 = roc_auc_score(test_y[np.where(~np.isnan(self.scores_CBLOF))], 
                                    self.scores_CBLOF[np.where(~np.isnan(self.scores_CBLOF))])
        r2 = roc_auc_score(test_y[np.where(~np.isnan(self.scores_HBOS))], 
                                   self.scores_HBOS[np.where(~np.isnan(self.scores_HBOS))]) 
        r3 = roc_auc_score(test_y[np.where(~np.isnan(self.scores_IFORST))], 
                                     self.scores_IFORST[np.where(~np.isnan(self.scores_IFORST))])
        
        
        print(f"CBLOF: {r1}")
        print(f"HBOS: {r2}")
        print(f"IFORST: {r3}")


        return [r1, r2, r3]