import torch

def total_variation(input):
    diff_h = ((input[:, :, 1:, :] - input[:, :, :-1, :]) ** 2).mean()
    diff_v = ((input[:, :, :, 1:] - input[:, :, :, :-1]) ** 2).mean()
    return torch.sqrt(diff_h + diff_v)

