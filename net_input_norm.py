import torch

class NetInputNorm:
    def __init__(self, device=None):
        self.norm_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.norm_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

        if device is not None:
            self.norm_mean = self.norm_mean.to(device)
            self.norm_std = self.norm_std.to(device)

    def to(self, device):
        return NetInputNorm(device)

    def __call__(self, input):
        return (input - self.norm_mean) / self.norm_std

