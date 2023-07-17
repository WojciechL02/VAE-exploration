from torch.utils.data import Dataset, random_split
from torchvision.datasets import MNIST


class MnistDataset(Dataset):
    def __init__(self, mode='train', transforms=None) -> None:
        super().__init__()
        self.transforms = transforms

        if mode in ['train', 'val']:
            data = MNIST('./data', train=True, download=True, transform=transforms)
            train_set, val_set = random_split(data, [50000, 10000])
            if mode == 'train':
                self.data = train_set
            elif mode == 'val':
                self.data = val_set
        else:
            self.data = MNIST('./data', train=False, transform=transforms)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index][0].flatten()
