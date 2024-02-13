import numpy as np
import torch


def quantize_8bit(input):
    """Quantize a tensor to a given precision.

    Args:
        input (torch.Tensor): The tensor to quantize.
        precision (int): The number of bits to quantize to.

    Returns:
        torch.Tensor: The quantized tensor.
    """
    print('starting quantize_8bit')
    offset = input.min(axis=0).values
    print('did offset')
    scale = (input.max(axis=0).values - offset) / 255
    print('did scale, closest to 0 is', abs(scale).min())
    quant = ((input - offset) / scale).float().round().clamp(0,
                                                             255).to(torch.uint8)
    print('did quant')
    return quant, offset, scale


def unquantize_8bit(input, offset, scale):
    """Unquantize a tensor to a given precision.

    Args:
        input (torch.Tensor): The tensor to quantize.
        precision (int): The number of bits to quantize to.

    Returns:
        torch.Tensor: The quantized tensor.
    """
    return input.to(torch.float16) * scale + offset
