import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch import Tensor

import numpy as np
from edflow import TemplateIterator, get_logger
from typing import *
from edflow.data.util import adjust_support

import torch
from torch import nn
from torch.nn import functional as F

from edflow.util import retrieve
from fid import fid_callback


def rec_fid_callback(*args, **kwargs):
    return fid_callback.fid(
        *args,
        **kwargs,
        im_in_key="img",
        im_in_support="-1->1",
        im_out_key="reconstructions",
        im_out_support="0->255",
        name="fid_recons"
    )


def sample_fid_callback(*args, **kwargs):
    return fid_callback.fid(
        *args,
        **kwargs,
        im_in_key="img",
        im_in_support="-1->1",
        im_out_key="samples",
        im_out_support="0->255",
        name="fid_samples"
    )


def reconstruction_callback(root, data_in, data_out, config):
    log = {"scalars": dict()}
    log["scalars"]["rec_loss"] = np.mean(data_out.labels["rec_loss"])
    log["scalars"]["kl_loss"] = np.mean(data_out.labels["kl_loss"])
    return log


class KLDLoss(nn.Module):
    def __init__(self, reduction="sum"):
        super(KLDLoss, self).__init__()
        self.reduction = reduction

    def forward(self, mean, logvar):
        # KLD loss
        kld_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp(), 1)
        # Size average
        if self.reduction == "mean":
            kld_loss = torch.mean(kld_loss)
        elif self.reduction == "sum":
            kld_loss = torch.sum(kld_loss)
        return kld_loss


class Model(torch.nn.Module):
    def __init__(self, config, **kwargs) -> None:
        super(Model, self).__init__()

        in_channels = config["in_channels"]
        latent_dim = config["latent_dim"]
        hidden_dims = config["hidden_dims"]
        beta = config.get("beta", 1)

        self.latent_dim = latent_dim
        self.beta = beta

        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        out_channels=h_dim,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                    ),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU(),
                )
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1] * 4, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1] * 4, latent_dim)

        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 4)

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        hidden_dims[i],
                        hidden_dims[i + 1],
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        output_padding=1,
                    ),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU(),
                )
            )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(
                hidden_dims[-1],
                hidden_dims[-1],
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
            nn.BatchNorm2d(hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dims[-1], out_channels=3, kernel_size=3, padding=1),
            nn.Tanh(),
        )

    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        result = self.decoder_input(z)
        result = result.view(-1, 512, 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return [self.decode(z), input, mu, log_var]

    def loss_function(self, recons, input, mu, log_var) -> dict:
        # make batch of losses
        recons_loss = F.mse_loss(recons, input, reduction="none")
        recons_loss = recons_loss.mean(dim=[1, 2, 3])

        # batch of losses
        kld_loss = -0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1)

        loss = recons_loss + self.beta * kld_loss

        return {"loss": loss, "Reconstruction_Loss": recons_loss, "KLD": kld_loss}

    def sample(self, num_samples: int, current_device: int, **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples, self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]


class Iterator(TemplateIterator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # loss and optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001)

    @property
    def callbacks(self):
        return {}

    def save(self, checkpoint_path):
        state = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        torch.save(state, checkpoint_path)

    def restore(self, checkpoint_path):
        state = torch.load(checkpoint_path)
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])

    def step_op(self, model, **kwargs):
        inputs = kwargs["img"]
        # inputs = adjust_support(
        #     inputs, "-1->1", "0->1"
        # )  # make sure adjust support preservers datatype
        inputs = inputs.astype(np.float32)
        inputs = torch.tensor(inputs)
        inputs = inputs.permute(0, 3, 1, 2)

        def train_op():
            # compute loss
            recons, _, mu, log_var = model(inputs)
            loss_dict = model.loss_function(recons, inputs, mu, log_var)
            loss = loss_dict["loss"].mean()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        def log_op():
            with torch.no_grad():
                recons, _, mu, log_var = model(inputs)
                loss_dict = model.loss_function(recons, inputs, mu, log_var)
                loss = loss_dict["loss"].mean()
                loss_rec = loss_dict["Reconstruction_Loss"].mean()
                loss_kld = loss_dict["KLD"].mean()

                image_logs = {
                    "inputs": inputs.detach().permute(0, 2, 3, 1).numpy(),
                    "recons": recons.detach().permute(0, 2, 3, 1).numpy(),
                }
                scalar_logs = {"loss": loss, "loss_rec": loss_rec, "loss_kld": loss_kld}

            return {"images": image_logs, "scalars": scalar_logs}

        def eval_op():
            with torch.no_grad():
                recons, _, mu, log_var = model(inputs)
                samples = model.sample(inputs.shape[0], recons.device)
                loss_dict = model.loss_function(recons, inputs, mu, log_var)
                loss_rec = loss_dict["Reconstruction_Loss"]
                loss_kld = loss_dict["KLD"]
            return {
                "reconstructions": recons.detach().permute(0, 2, 3, 1).numpy(),
                "samples": samples.detach()
                .permute(0, 2, 3, 1)
                .numpy(),  # TODO: replace with samples
                "labels": {
                    "rec_loss": loss_rec.detach().numpy(),
                    "kl_loss": loss_kld.detach().numpy(),
                },
            }

        return {"train_op": train_op, "log_op": log_op, "eval_op": eval_op}

    @property
    def callbacks(self):
        cbs = {"eval_op": {"reconstruction": reconstruction_callback}}
        cbs["eval_op"]["fid_reconstruction"] = rec_fid_callback
        cbs["eval_op"]["fid_samples"] = sample_fid_callback
        return cbs

