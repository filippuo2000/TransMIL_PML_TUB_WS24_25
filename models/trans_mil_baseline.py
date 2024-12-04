import torch
import torch.nn as nn


class TransMILBaseline(nn.Module):
    def __init__(self, new_num_features, n_heads, num_classes, device):
        super().__init__()
        self.device = device
        self.fc1 = nn.Linear(
            768, new_num_features
        )  # [B, n, new_num_features], n - no of instances in a processed bag
        self.cls_token = nn.Parameter(
            torch.rand((1, 1, new_num_features))
        )  # [1, new_num_features] - will have to expand this
        self.fc2 = nn.Linear(new_num_features, num_classes)
        self.ln = nn.LayerNorm(new_num_features)
        self.at1 = nn.MultiheadAttention(
            embed_dim=new_num_features,
            num_heads=n_heads,
            dropout=0.3,
            batch_first=True,
        )
        self.at2 = nn.MultiheadAttention(
            embed_dim=new_num_features,
            num_heads=n_heads,
            dropout=0.3,
            batch_first=True,
        )

    def forward(self, x):
        batch_size, _, _ = x.shape  # x shape is [B, n, new_num_features]

        x = self.fc1(x)  # [B, n, new_num_features]
        class_token = self.cls_token.expand(batch_size, -1, -1).to(
            self.device
        )  # [B, 1, new_num_features]
        x = torch.cat((class_token, x), dim=1)  # [B, n+1, new_num_features]
        x, _ = self.at1(x, x, x)
        x, _ = self.at2(x, x, x)

        class_token = x[:, 0]  # [B, 1, new_num_features]
        class_token = self.ln(class_token)  # [B, 1, new_num_features]
        class_token = self.fc2(class_token)  # [B, num_class]
        return class_token
