from typing import List

import torch

from e3nn.math import normalize2mom
from e3nn.util.jit import compile_mode


@compile_mode('script')
class _Layer(torch.nn.Module):
    h_in: float
    h_out: float
    var_in: float
    var_out: float
    _profiling_str: str

    def __init__(self, h_in, h_out, act, var_in, var_out):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(h_in, h_out))
        # LoRA weights initialization
        self.LoRA_weight = []
        self.alpha = 16
        self.r = 16
        self.LoRA_weight.append(torch.nn.Parameter(torch.randn(h_in, self.r)))
        self.LoRA_weight.append(torch.nn.Parameter(torch.zeros(self.r, h_out)))
        self.LoRA_weight = torch.nn.ParameterList(self.LoRA_weight)
        self.act = act

        self.h_in = h_in
        self.h_out = h_out
        self.var_in = var_in
        self.var_out = var_out

        self._profiling_str = repr(self)

    def __repr__(self):
        act = self.act
        if hasattr(act, '__name__'):
            act = act.__name__
        elif isinstance(act, torch.nn.Module):
            act = act.__class__.__name__

        return f"Layer({self.h_in}->{self.h_out}, act={act})"

    def forward(self, x: torch.Tensor):
        # - PROFILER - with torch.autograd.profiler.record_function(self._profiling_str):
        weight = self.weight + self.alpha / self.r * self.LoRA_weight[0] @ self.LoRA_weight[1]
        if self.act is not None:
            w = weight / (self.h_in * self.var_in)**0.5
            x = x @ w
            x = self.act(x)
            x = x * self.var_out**0.5
        else:
            w = weight / (self.h_in * self.var_in / self.var_out)**0.5
            x = x @ w
        return x
    
    def merge_LoRA(self):
        self.weight.data = self.weight + self.alpha / self.r * self.LoRA_weight[0] @ self.LoRA_weight[1]
        del self.LoRA_weight
        del self.alpha
        del self.r

@compile_mode('script')
class FullyConnectedNet(torch.nn.Sequential):
    r"""Fully-connected Neural Network

    Parameters
    ----------
    hs : list of int
        input, internal and output dimensions

    act : function
        activation function :math:`\phi`, it will be automatically normalized by a scaling factor such that

        .. math::

            \int_{-\infty}^{\infty} \phi(z)^2 \frac{e^{-z^2/2}}{\sqrt{2\pi}} dz = 1
    """
    hs: List[int]

    def __init__(self, hs, act=None, variance_in=1, variance_out=1, out_act=False):
        super().__init__()
        self.hs = list(hs)
        if act is not None:
            act = normalize2mom(act)
        var_in = variance_in

        for i, (h1, h2) in enumerate(zip(self.hs, self.hs[1:])):
            if i == len(self.hs) - 2:
                var_out = variance_out
                a = act if out_act else None
            else:
                var_out = 1
                a = act

            layer = _Layer(h1, h2, a, var_in, var_out)
            setattr(self, f"layer{i}", layer)

            var_in = var_out

    def __repr__(self):
        return f"{self.__class__.__name__}{self.hs}"
