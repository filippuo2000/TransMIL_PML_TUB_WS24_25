import torch
import torch.nn as nn
from captum.attr import IntegratedGradients


class IGWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        x = self.model(x)
        return x['logits']


def integrated_gradients(model, features, label):
    ig_wrapper = IGWrapper(model)
    ig = IntegratedGradients(forward_func=ig_wrapper)

    baseline = torch.zeros_like(features, requires_grad=True)

    target = torch.tensor([label])

    attribution = ig.attribute(
        inputs=features, baselines=baseline, target=target, n_steps=10
    )
    attribution = attribution.sum(dim=-1)
    attribution = torch.squeeze(attribution)
    attribution = attribution.abs()
    attribution = (attribution - attribution.min()) / (
        attribution.max() - attribution.min()
    )
    attribution = attribution.detach().tolist()

    return attribution
