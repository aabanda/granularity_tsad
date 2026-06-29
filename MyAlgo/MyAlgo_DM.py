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
ConditionalDiffusionTrainingNetwork = diffusion.ConditionalDiffusionTrainingNetwork
Unet = tsdalia("models.unet").Unet

model_backprop_function = {
    "transformer_autoencoder": transformer_autoencoder.backprop,
    "diffusion": diffusion.backprop,
    "diffusion_autoencoder": diffusion_autoencoder.backprop,
}



class MyAlgo_DM(BaseMethod):
    def __init__(self, hparams) -> None:
        super().__init__()
        self.__anomaly_score = None
        self.window_size = 100
        self.ts_dims = 1
        self.batch_size = 128
        self.lr = 0.00001
        self.num_epochs = 1000
        self.noise_steps: int = 100
        self.denoise_steps: int = 50
        self.diff_lambda: float = 0.1
        self.diffusion_training_net: any = None
        self.params_specific: dict = None
        self.score_train = None
        self.model_name = "diffusion"
        self.device = "cuda"
        self.y_hat_train = None

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


    def train_valid_phase(self, tsTrain: TSData):
        
        self.denoise_fn = Unet(
            dim=self.ts_dims,
            channels=1,
            resnet_block_groups=1,
            init_size=torch.Size([self.ts_dims, self.window_size, self.ts_dims]),
        )
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
            diffusion_training_net=self.diffusion_training_net,
            params_specific=self.params_specific,
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
                diffusion_training_net=self.diffusion_training_net,
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
            batch_size=self.batch_size,
            window_size=self.window_size,
            diffusion_training_net=self.diffusion_training_net,
            **self.params_specific,
        )
        self.train_error = np.abs(trainD - recons_train[:,:,0]).numpy()

        self.train_error = self.train_error.reshape(
            self.train_error.shape[0] * self.train_error.shape[1], self.ts_dims
        )


        self.loss_test = loss_train.reshape(
            (
                loss_train.shape[0] * loss_train.shape[1],
                loss_train.shape[2],
            )
        )

        loss_train = np.mean(loss_train, axis=1)[:len(train)]

        self.score_train =  loss_train
        self.y_hat_train =  recons_train.reshape(
            (
                recons_train.shape[0] * self.window_size,
                recons_train.shape[2],
            )
        )
        
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
            diffusion_training_net=self.diffusion_training_net,
            **self.params_specific,
        )


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

