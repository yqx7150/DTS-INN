import math
import torch


def compute_same_pad(kernel_size, stride):
    if isinstance(kernel_size, int):
        kernel_size = [kernel_size]

    if isinstance(stride, int):
        stride = [stride]

    assert len(stride) == len(
        kernel_size
    ), "Pass kernel size and stride both as int, or both as equal length iterable"

    return [((k - 1) * s + 1) // 2 for k, s in zip(kernel_size, stride)]


def uniform_binning_correction(x, n_bits=8):
    """Replaces x^i with q^i(x) = U(x, x + 1.0 / 256.0).

    Args:
        x: 4-D Tensor of shape (NCHW)
        n_bits: optional.
    Returns:
        x: x ~ U(x, x + 1.0 / 256)
        objective: Equivalent to -q(x)*log(q(x)).
    """
    b, c, h, w = x.size()
    n_bins = 2 ** n_bits
    chw = c * h * w
    x += torch.zeros_like(x).uniform_(0,
                                      1.0 / n_bins)  # torch.zeros_like(x)产生一个与x相同shape的tensor，里面的数值全为0。uniform_(0, 1.0 / n_bins)使张量里面的数值在0~1.0/n_bins之间，并且在torch.zeros_like(x).uniform_(0, 1.0 / n_bins)产生的数值并不为0.

    objective = -math.log(n_bins) * chw * torch.ones(b, device=x.device)
    return x, objective


def split_feature(tensor, type="split"):
    """
    type = ["split", "cross"]
    """
    C = tensor.size(1)
    if type == "split":
        return tensor[:, : C // 2, ...], tensor[:, C // 2:, ...]
        # return tensor[:, :10, ...], tensor[:,10:, ...]
    elif type == "cross":
        # return tensor[:, 0::2, ...], tensor[:, 1::2, ...]
        return tensor[:, 0::2, ...], tensor[:, 1::2, ...]
