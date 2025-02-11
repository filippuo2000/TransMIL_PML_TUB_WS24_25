import torch


def saliency_gradients(model, features):
    model.eval()
    y = model(features)
    logits = y['logits']
    class_idx = logits.argmax().item()
    logits[:, class_idx].backward()
    grads = features.grad
    grads = grads.abs()
    grads_sum = torch.sum(grads, dim=2)
    grads_sum = (grads_sum - torch.min(grads_sum)) / (
        torch.max(grads_sum) - torch.min(grads_sum)
    )
    grads_list = torch.squeeze(grads_sum).tolist()
    return grads_list
