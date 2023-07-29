from torch import nn
from abc import abstractmethod


class BaseVAE(nn.Module):
    def __init__(self) -> None:
        super(BaseVAE, self).__init__()

    def sample(self, num_samples: int):
        raise NotImplementedError

    @abstractmethod
    def forward(self, *inputs):
        pass
