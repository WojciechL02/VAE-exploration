import torch
from torch import nn
from torch.nn import functional as F
from .base import BaseVAE
from .distributions import log_normal_diag, log_standard_normal


class Encoder(nn.Module):
    def __init__(self, encoder_net) -> None:
        super(Encoder, self).__init__()

        self.encoder = encoder_net

    @staticmethod
    def reparametrization(mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + std * eps

    def encode(self, x):
        h_e = self.encoder(x)

        mu_e, log_var_e = torch.chunk(h_e, 2, dim=1)

        return mu_e, log_var_e

    def sample(self, x=None, mu_e=None, log_var_e=None):
        if (mu_e is None) and (log_var_e is None):
            mu_e, log_var_e = self.encode(x)
        else:
            if (mu_e is None) or (log_var_e is None):
                raise ValueError("Missing mean or variance!")

        z = self.reparametrization(mu_e, log_var_e)
        return z

    # log-probability for ELBO
    def log_prob(self, x=None, mu_e=None, log_var_e=None, z=None):
        if x is not None:
            mu_e, log_var_e = self.encode(x)
            z = self.sample(mu_e=mu_e, log_var_e=log_var_e)
        else:
            if (mu_e is None) or (log_var_e is None) or (z is None):
                raise ValueError("Mean, variance and Z cannot be None!")

        return log_normal_diag(z, mu_e, log_var_e)

    def forward(self, x, type='log_prob'):
        assert type in ['encode', 'log_prob'], 'Type could be either encode or log_prob'
        if type == 'log_prob':
            return self.log_prob(x)
        else:
            return self.sample(x)


class Decoder(nn.Module):
    def __init__(self, decoder_net) -> None:
        super(Decoder, self).__init__()

        self.decoder = decoder_net

    def decode(self, z):
        x_new = self.decoder(z)
        return x_new

    def reconstruction_loss(self, x, z):
        x_new = self.decode(z)
        recons_loss = F.mse_loss(x_new, x)
        return recons_loss

    def forward(self, z, x=None, type='loss'):
        assert type in ['decode', 'loss'], 'Type could be either decode or loss'
        if type == 'loss':
            return self.reconstruction_loss(x, z)
        else:
            return self.decode(z)


class Prior(nn.Module):
    def __init__(self, L) -> None:
        super(Prior, self).__init__()
        self.L = L

    def sample(self, batch_size):
        z = torch.randn((batch_size, self.L))
        return z

    def log_prob(self, z):
        return log_standard_normal(z)


class VAE(BaseVAE):
    def __init__(self, encoder_net, decoder_net, L, regularize=True) -> None:
        super(VAE, self).__init__()

        self.encoder = Encoder(encoder_net)
        self.decoder = Decoder(decoder_net)
        self.prior = Prior(L)
        self.regularize = regularize

    def forward(self, x):
        mu_e, log_var_e = self.encoder.encode(x)
        z = self.encoder.sample(mu_e=mu_e, log_var_e=log_var_e)

        # LOSS
        RE = self.decoder.reconstruction_loss(x, z)
        KL = 0.
        if self.regularize:
            # KL = torch.mean(-0.5 * torch.sum(1 + log_var_e - mu_e ** 2 - log_var_e.exp(), dim=1), dim=0)
            KL = (self.encoder.log_prob(mu_e=mu_e, log_var_e=log_var_e, z=z) - self.prior.log_prob(z)).sum(-1).mean()

        return (RE + 0.0005 * KL)

    def sample(self, num_samples: int):
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :return: (Tensor)
        """
        z = self.prior.sample(num_samples)

        samples = self.decoder.decode(z)
        return samples
