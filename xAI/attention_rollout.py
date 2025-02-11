import torch


def attention_rollout(model, features):
    model.model.at1.att._forward_hooks.clear()
    model.model.at2.att._forward_hooks.clear()

    attention = []

    def att_hook(module, input, output):
        attention.append(output[1].detach())

    model.model.at1.att.register_forward_hook(att_hook)
    model.model.at2.att.register_forward_hook(att_hook)

    print(f"len of attention before forward pass: {len(attention)}")
    model.eval()
    # attention [] is appended during runtime
    _ = model(features)
    print(f"len of attention after the forward pass: {len(attention)}")

    # create an Identity matrix of attention shape
    results = torch.eye(attention[0].shape[-1])
    for att in attention:
        # create an Identity matrix of attention shape
        I_matrix = torch.eye((att.shape[-1]))
        att = torch.squeeze(torch.mean(att, dim=1))
        att_i = (att + I_matrix) / 2

        # normalize the rows to form the prob distribution
        att_i = att_i / att_i.sum(dim=-1, keepdim=True)
        results = torch.matmul(att_i, results)

    # extract without the zero padding
    final_results = results[: len(features[0]) + 1, : len(features[0]) + 1]

    # extract the class token
    patch_att = final_results[0, 1:]

    # re-normalize
    patch_att = (patch_att - patch_att.min()) / (
        patch_att.max() - patch_att.min()
    )
    patch_att = patch_att.tolist()
    return patch_att
