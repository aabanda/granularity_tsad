import torch.nn as nn
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from .diffusion import ConditionalDiffusionTrainingNetwork


def backprop(epoch: int, model: any, data: torch.Tensor,
             optimizer, scheduler, k: int = 1, batch_size: int = 128, window_size: int = 100,
             training: bool = True, device='cuda',
             dims: int = 1,
             noise_steps: int = 100,
             denoise_steps: int = 50,
             diff_lambda: float = 0.1,
             diffusion_training_net = None):
    diff_lambda = 0.5
    mse_loss = nn.MSELoss(reduction='none')
    data_x = torch.tensor(data, dtype=torch.float32)
    dataset = TensorDataset(data_x, data_x)

    diffusion_prediction_net = ConditionalDiffusionTrainingNetwork(nr_feats=dims,
                                                                   window_size=int(window_size),
                                                                   batch_size=batch_size,
                                                                   train=False,
                                                                   noise_steps=noise_steps,
                                                                   denoise_steps=denoise_steps).float()
    diffusion_training_net = diffusion_training_net.to(device)
    diffusion_prediction_net = diffusion_prediction_net.to(device)

    bs = diffusion_training_net.batch if not model else model.batch
    dataloader = DataLoader(dataset, batch_size=bs)
    w_size = diffusion_training_net.window_size
    l1s, ae_losses, diff_losses, samples = [], [], [], []
    if training:
        model.train()
        diffusion_training_net.train()
        for d, _ in dataloader:
            window = d
            window = window.to(device)
            ae_reconstruction = model(window, window)
            ae_loss = mse_loss(ae_reconstruction, window)
            ae_reconstruction = ae_reconstruction.reshape(-1, w_size, dims)

            diffusion_loss, _ = diffusion_training_net(ae_reconstruction)

            ae_losses.append(torch.mean(ae_loss).item())
            diff_losses.append(torch.mean(diffusion_loss).item())

            loss = diff_lambda * diffusion_loss + torch.mean(ae_loss)
            l1s.append(loss.item())

            optimizer.zero_grad()
            loss.backward()                                                                                                     
            optimizer.step()
            scheduler.step()

        tqdm.write(f'Epoch {epoch},\tLoss total: dif_lambda * loss_diff + loss_ae = {np.mean(l1s)}')
        # tqdm.write(f'Epoch {epoch},\tAE = {np.mean(ae_losses)}')
        # tqdm.write(f'Epoch {epoch},\tDiff = {np.mean(diff_losses)}')
        return np.mean(l1s), [np.mean(ae_losses), np.mean(diff_losses)]
    else:
        with torch.no_grad():
            model.eval()
            diffusion_prediction_net.load_state_dict(diffusion_training_net.state_dict())
            diffusion_prediction_net.eval()
            # diffusion_training_net.eval()

            l1s = []  # scores
            sum_losses = []
            ae_losses = []
            diff_losses = []
            recons = []
            for d, _ in dataloader:
                window = d
                window = window.to(device)
                window_reshaped = window.reshape(-1, w_size, dims)

                ae_reconstruction = model(window, window)
                ae_reconstruction_reshaped = ae_reconstruction.reshape(-1, w_size, dims)
                recons.append(ae_reconstruction_reshaped)
                ae_loss = mse_loss(ae_reconstruction, window)
                ae_losses.append(torch.mean(ae_loss).item())
                


                _, diff_sample = diffusion_prediction_net(ae_reconstruction_reshaped)
                diff_sample = torch.squeeze(diff_sample, 1)
                diffusion_loss = mse_loss(diff_sample, window_reshaped)
                diffusion_loss = torch.mean(diffusion_loss).item()

                sum_losses.append(torch.mean(ae_loss).item() + diffusion_loss)
                diff_losses.append(diffusion_loss)

                samples.append(diff_sample)
                loss = mse_loss(diff_sample, window_reshaped)
                l1s.append(loss)

        return torch.cat(l1s).detach().cpu().numpy(), [torch.cat(samples).detach().cpu().numpy(), torch.cat(recons).detach().cpu().numpy()]
