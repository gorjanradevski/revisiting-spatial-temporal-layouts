import torch


def generate_square_subsequent_mask(sz: int) -> torch.Tensor:
    # https://pytorch.org/docs/stable/_modules/torch/nn/modules/transformer.html#Transformer.generate_square_subsequent_mask
    mask = ~(torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    return mask
