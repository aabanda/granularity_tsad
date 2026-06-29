import torch
from lib.tsadalia.tsdalia.models.transformer_autoencoder import Transformer_Autoencoder
from lib.tsadalia.tsdalia.models.diffusion import Diffusion_model
import pathlib
import pandas as pd
import random
from collections import Counter


model_class_dict = {
    "transformer_autoencoder": Transformer_Autoencoder,
    "diffusion": Diffusion_model,
    "diffusion_autoencoder": Transformer_Autoencoder,
}

file_path = pathlib.Path(__file__).parent.resolve()
data_folder = file_path.parents[3] / "data" / "processed" 


def load_dataset(dataset: str = "ia4tes"):

    train_x = pd.read_csv(f"{data_folder}/{dataset}/train_x.txt", header=None, delimiter='\s+')
    test_x = pd.read_csv(f"{data_folder}/{dataset}/test_x.txt", header=None, delimiter='\s+')
    test_y = pd.read_csv(f"{data_folder}/{dataset}/test_y.txt", header=None, delimiter='\s+')

    return train_x.values, test_x.values, test_y.values


def load_model(
    model_name: str = "basic_autoencoder",
    lr: float = 1e-3,
    window_size: int = 100,
    dims: int = 1,
    batch_size: int = 128,
    device: str = "cuda",
    params_specific: dict = None,
    diffusion_training_net=None,
):
    model_class = model_class_dict[model_name]
    if model_name == "diffusion":
        model = model_class(
            feats=dims,
            lr=float(lr),
            window_size=int(window_size),
            batch_size=batch_size,
            **params_specific,
        ).float()
        optimizer = torch.optim.Adam(diffusion_training_net.parameters(), lr=float(lr))

    elif model_name == "diffusion_autoencoder":
        model = model_class(
            feats=dims,
            lr=float(lr),
            window_size=int(window_size),
            batch_size=batch_size,
        ).float()
        optimizer = torch.optim.Adam(
            list(model.parameters()) + list(diffusion_training_net.parameters()),
            lr=model.lr,
        )

    else:
        model = model_class(
            feats=dims,
            lr=float(lr),
            window_size=int(window_size),
            batch_size=batch_size,
        ).float()
        if model_name == "basic_autoencoder":
            optimizer = torch.optim.Adam(model.parameters(), lr=model.lr)
        else:
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=model.lr, weight_decay=1e-5
            )

    epoch = -1
    accuracy_list = []
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    size_all_mb = (param_size + buffer_size) / 1024**2
    print("model size: {:.3f}MB".format(size_all_mb))
    model = model.to(device)
    return model, optimizer, epoch


def convert_to_windows(data, n_window: int, window_strategy: int = 1):

    if window_strategy == 1:
        data = torch.Tensor(data)
        windows = list(torch.split(data, n_window))
        for i in range(n_window - windows[-1].shape[0]):
            windows[-1] = torch.cat((windows[-1], windows[-1][-1].unsqueeze(0)))

        a = torch.stack(windows)

    return torch.tensor(a)


def majority_vote_for_instances(labels_list):
    majority_votes = []

    for l_ind in range(labels_list.shape[1]):
        counter = Counter(labels_list[:, l_ind])
        most_common = counter.most_common()
        if len(most_common) > 1 and most_common[0][1] == most_common[1][1]:
            most_common_label = random.choice(most_common)[0]
            # most_common_label = most_common[0][0]
            # most_common_label = most_common[1][0]
        else:
            most_common_label = most_common[0][0]
        majority_votes.append(most_common_label)
    return majority_votes
