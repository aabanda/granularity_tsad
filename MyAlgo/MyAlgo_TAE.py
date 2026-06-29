from MyAlgo._compat import BaseMethod, TSData, tsdalia
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm

_functions = tsdalia("functions")
convert_to_windows = _functions.convert_to_windows
load_model = _functions.load_model

transformer_autoencoder = tsdalia("models.transformer_autoencoder")
diffusion = tsdalia("models.diffusion")
diffusion_autoencoder = tsdalia("models.diffusion_autoencoder")
basic_autoencoder = tsdalia("models.basic_autoencoder")


model_backprop_function = {
    "transformer_autoencoder": transformer_autoencoder.backprop,
    "diffusion": diffusion.backprop,
    "diffusion_autoencoder": diffusion_autoencoder.backprop,
}



class MyAlgo_TAE(BaseMethod):
    def __init__(self, hparams) -> None:
        super().__init__()
        self.__anomaly_score = None
        self.window_size = 100
        self.ts_dims = 1
        self.batch_size = 256
        self.lr = 0.00001
        self.num_epochs = 1000
        self.model_name = "transformer_autoencoder"
        self.device = "cuda"
        self.score_train = None
        self.y_hat_train = None
        

    def train_valid_phase(self, tsTrain: TSData):


        train =  DataLoader(tsTrain.train, batch_size=tsTrain.train.shape[0])
        train = next(iter(train))
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
        num_training_steps = len_dataloader * self.num_epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, 0.1 * num_training_steps, num_training_steps
        )

        print("training")
        e = epoch + 1
        loss_train_hist = [100000000]

        for e in tqdm(list(range(epoch + 1, epoch + self.num_epochs + 1))):
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
        self.train_error = np.abs(trainD - recons_train[:,:,0]).numpy()

        self.train_error = self.train_error.reshape(
            self.train_error.shape[0] * self.train_error.shape[1], self.ts_dims
        )

        
        reconstruction = recons_train.reshape(
            (
                recons_train.shape[0] * self.window_size,
                recons_train.shape[2],
            )
        )


        loss_train = loss_train.reshape(
            (
                loss_train.shape[0] * loss_train.shape[1],
                loss_train.shape[2],
            )
        )

        self.score_train = np.mean(loss_train, axis=1)[:len(train)]
        self.y_hat_train = reconstruction

    def test_phase(self, tsData: TSData):

        test =  DataLoader(tsData.test, batch_size=tsData.test.shape[0])
        test = next(iter(test))
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

        reconstruction = reconstruction.reshape(
            (
                reconstruction.shape[0] * self.window_size,
                reconstruction.shape[2],
            )
        )


        # import matplotlib.pyplot as plt
        # plt.figure(figsize=(10, 5))
        # plt.plot(test, label="test")
        # plt.plot(reconstruction, label="reconstruction")
        # plt.legend()

        self.loss_test = self.loss_test.reshape(
            (
                self.loss_test.shape[0] * self.loss_test.shape[1],
                self.loss_test.shape[2],
            )
        )

        self.loss_test = np.mean(self.loss_test, axis=1)[:len(test)]

        self.__anomaly_score =  self.loss_test

    def train_valid_phase_all_in_one(self, tsTrains: dict[str, TSData]):
        # used for all-in-one and zero-shot mode
        pass

    def anomaly_score(self) -> np.ndarray:
        return self.__anomaly_score

    def param_statistic(self, save_file):
        pass

