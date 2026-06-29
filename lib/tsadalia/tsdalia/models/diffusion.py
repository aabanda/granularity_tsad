import torch.nn as nn
from .unet import Unet
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
import numpy as np


class Diffusion_model(nn.Module):
    def __init__(
        self,
        feats: int,
        lr: float,
        window_size: int,
        batch_size: int,
        diff_lambda: float,
        noise_steps: int = 100,
        denoise_steps: int = 50,
        train: bool = True,
    ):
        super().__init__()
        self.dim = min(feats, 16)
        self.n_feats = feats
        self.lr = lr
        self.n_window = window_size
        self.batch = batch_size

        self.training = train
        self.timesteps = noise_steps
        self.denoise_steps = denoise_steps

        self.denoise_fn = Unet(
            dim=self.dim,
            channels=1,
            resnet_block_groups=1,
            init_size=torch.Size([self.dim, self.n_window, self.n_feats]),
        )

    def forward(self, x):
        diffusion_loss = None
        x_recon = None

        x = x.reshape(-1, 1, self.window_size, self.n_feats)
        if self.training:
            t = torch.randint(0, self.timesteps, (x.shape[0],), device="cuda").long()
            diffusion_loss = p_losses(self.denoise_fn, x, t)
        else:
            x_recon = sample(
                self.denoise_fn,
                shape=(x.shape[0], 1, self.window_size, self.n_feats),
                x_start=x,
                denoise_steps=self.denoise_steps,
            )

        return diffusion_loss, x_recon


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)


def quadratic_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start**0.5, beta_end**0.5, timesteps) ** 2


def sigmoid_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    betas = torch.linspace(-6, 6, timesteps)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start


timesteps = 300

# define beta schedule
betas = linear_beta_schedule(timesteps=timesteps)

# define alphas
alphas = 1.0 - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

# calculations for diffusion q(x_t | x_{t-1}) and others
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

# calculations for posterior q(x_{t-1} | x_t, x_0)
posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)


def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


# forward diffusion (using the nice property)
def q_sample(x_start, t, noise=None):
    if noise is None:
        noise = torch.randn_like(x_start)

    sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, x_start.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, x_start.shape
    )

    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise


def p_losses(denoise_model, x_start, t, noise=None, loss_type="l2"):
    if noise is None:
        noise = torch.randn_like(x_start)

    x_noisy = q_sample(x_start=x_start, t=t, noise=noise)
    predicted_noise = denoise_model(x_noisy, t)
    # if train:
    if loss_type == "l1":
        loss = F.l1_loss(noise, predicted_noise)
    elif loss_type == "l2":
        loss = F.mse_loss(noise, predicted_noise)
    elif loss_type == "huber":
        loss = F.smooth_l1_loss(noise, predicted_noise)
    else:
        raise NotImplementedError()
    # else:
    #     x_recon = (x_noisy - extract(sqrt_one_minus_alphas_cumprod, t, x_noisy.shape) * predicted_noise) / extract(
    #     sqrt_alphas_cumprod, t, x_noisy.shape)
    #     loss = F.mse_loss(predicted_noise, noise, reduction='none')
    return loss


##### SAMPLING #######


@torch.no_grad()
def p_sample(model, x, t, t_index):
    betas_t = extract(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(sqrt_one_minus_alphas_cumprod, t, x.shape)
    sqrt_recip_alphas_t = extract(sqrt_recip_alphas, t, x.shape)

    # Equation 11 in the paper
    # Use our model (noise predictor) to predict the mean
    model_mean = sqrt_recip_alphas_t * (
        x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
    )

    if t_index == 0:
        return model_mean
    else:
        posterior_variance_t = extract(posterior_variance, t, x.shape)
        noise = torch.randn_like(x)
        # Algorithm 2 line 4:
        return model_mean + torch.sqrt(posterior_variance_t) * noise

    # Algorithm 2 (including returning all images)


@torch.no_grad()
def p_sample_loop(model, shape, x_start, denoise_steps):
    # device = next(model.parameters()).device
    # timesteps = 200
    timesteps = denoise_steps
    device = "cuda"

    b = shape[0]
    # start from pure noise (for each example in the batch)
    # img = torch.randn(shape, device=device)
    noise = torch.randn_like(x_start)
    img = q_sample(
        x_start=x_start,
        t=torch.full((b,), timesteps, device=device, dtype=torch.long),
        noise=noise,
    )
    # imgs = []

    for i in tqdm(
        reversed(range(0, timesteps)), desc="sampling loop time step", total=timesteps
    ):
        img = p_sample(
            model, img, torch.full((b,), i, device=device, dtype=torch.long), i
        )
        # imgs.append(img.cpu().numpy())
    return img


@torch.no_grad()
def sample(model, shape, x_start, denoise_steps):
    return p_sample_loop(
        model, shape=shape, x_start=x_start, denoise_steps=denoise_steps
    )


# ESte es  el diffusion
class ConditionalDiffusionTrainingNetwork(nn.Module):
    def __init__(
        self,
        nr_feats,
        window_size,
        batch_size,
        noise_steps,
        denoise_steps,
        train=True,
        diff_lambda=None,
    ):
        super().__init__()
        self.dim = min(nr_feats, 16)
        self.nr_feats = nr_feats
        self.window_size = window_size
        self.batch_size = batch_size

        self.training = train
        self.timesteps = noise_steps
        self.denoise_steps = denoise_steps

        self.denoise_fn = Unet(
            dim=self.dim,
            channels=1,
            resnet_block_groups=1,
            init_size=torch.Size([self.dim, self.window_size, self.nr_feats]),
        )

    def forward(self, x):
        diffusion_loss = None
        x_recon = None

        x = x.reshape(-1, 1, self.window_size, self.nr_feats)
        if self.training:
            t = torch.randint(0, self.timesteps, (x.shape[0],), device="cuda").long()
            diffusion_loss = p_losses(self.denoise_fn, x, t)
        else:
            x_recon = sample(
                self.denoise_fn,
                shape=(x.shape[0], 1, self.window_size, self.nr_feats),
                x_start=x,
                denoise_steps=self.denoise_steps,
            )

        return diffusion_loss, x_recon


# epoch = e, model = model, data = trainD, optimizer = optimizer, scheduler = scheduler,
# training = True, dims = labels.shape[1], batch_size = batch_size, window_size = window_size)


def backprop(
    epoch: int,
    model: any,
    data: torch.Tensor,
    optimizer,
    scheduler,
    k: int = 1,
    batch_size: int = 128,
    window_size: int = 100,
    training: bool = True,
    device="cuda",
    dims: int = 1,
    noise_steps: int = 100,
    denoise_steps: int = 50,
    diff_lambda: float = 0.1,
    diffusion_training_net=None,
):

    mse_loss = nn.MSELoss(reduction="none")
    data_x = torch.tensor(data, dtype=torch.float32)
    dataset = TensorDataset(data_x, data_x)

    diffusion_prediction_net = ConditionalDiffusionTrainingNetwork(
        nr_feats=dims,
        window_size=int(window_size),
        batch_size=batch_size,
        train=False,
        noise_steps=noise_steps,
        denoise_steps=denoise_steps,
    ).float()
    diffusion_training_net = diffusion_training_net.to(device)
    diffusion_prediction_net = diffusion_prediction_net.to(device)

    bs = diffusion_training_net.batch if not model else model.batch
    dataloader = DataLoader(dataset, batch_size=bs)
    w_size = diffusion_training_net.window_size
    l1s, samples = [], []
    if training:
        # model.train()
        diffusion_training_net.train()
        for d, _ in dataloader:
            window = d
            window = window.to(device)
            window = window.reshape(-1, w_size, dims)
            loss, _ = diffusion_training_net(window)
            l1s.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
        tqdm.write(f"Epoch {epoch},\tLoss = {np.mean(l1s)}")
        return np.mean(l1s), window
    else:
        with torch.no_grad():
            # model.eval()
            diffusion_prediction_net.load_state_dict(
                diffusion_training_net.state_dict()
            )
            diffusion_prediction_net.eval()
            diffusion_training_net.eval()
            l1s = []
            for d, _ in dataloader:
                window = d
                window = window.to(device)
                window_reshaped = window.reshape(-1, w_size, dims)
                _, x_recon = diffusion_prediction_net(window_reshaped)
                x_recon = torch.squeeze(x_recon, 1)
                samples.append(x_recon)
                loss = mse_loss(x_recon, window_reshaped)
                l1s.append(loss)

        return (
            torch.cat(l1s).detach().cpu().numpy(),
            torch.cat(samples).detach().cpu().numpy(),
        )
