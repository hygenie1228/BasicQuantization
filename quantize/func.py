import torch
import torch.nn as nn 

class LinearQuant(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, n_bit):
        if n_bit == 32:
            out = input
        elif n_bit == 1:
            out = torch.sign(input)
        else:
            n = float(2 ** n_bit - 1)
            out = torch.round(input * n) / n
        return out

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None

class Quantizer(nn.Module):
    def __init__(self):
        super(Quantizer, self).__init__()
        self.w_bit = 8
        self.a_bit = 8
        self.ratio = 1.0
    
    def set_quantizer(self, wbit, abit, ratio):
        self.w_bit = wbit
        self.a_bit = abit
        self.ratio = ratio

    def weight_quantize_fn(self, weight):
        if self.w_bit == 32:
            return 1.0, weight
        elif self.w_bit == 1:
            scale = torch.mean(torch.abs(weight)).detach()
            weight_q = LinearQuant.apply(weight / scale, self.w_bit)
        else:
            weight = torch.tanh(weight)
            scale = torch.max(torch.abs(weight)).detach()
            weight = weight / 2 / scale + 0.5
            weight_q = 2 * LinearQuant.apply(weight, self.w_bit) - 1

        return scale, weight_q


    def activation_quantize_fn(self, act):
        if self.a_bit == 32:
            return act
        act = LinearQuant.apply(torch.clamp(act, 0, 1), self.a_bit)
        return act

quantizer = Quantizer()