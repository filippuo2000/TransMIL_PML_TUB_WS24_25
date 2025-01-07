import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from nystrom_attention import NystromAttention


class PPEG(nn.Module):
    def __init__(self, in_out_dims):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_out_dims,
            out_channels=in_out_dims,
            kernel_size=3,
            stride=1,
            padding=3 // 2,
            groups=in_out_dims,
        )
        self.conv2 = nn.Conv2d(
            in_channels=in_out_dims,
            out_channels=in_out_dims,
            kernel_size=5,
            stride=1,
            padding=5 // 2,
            groups=in_out_dims,
        )
        self.conv3 = nn.Conv2d(
            in_channels=in_out_dims,
            out_channels=in_out_dims,
            kernel_size=7,
            stride=1,
            padding=7 // 2,
            groups=in_out_dims,
        )

    def forward(self, x, H, W):
        B, _, C = x.shape
        class_token, x = (
            x[:, 0, :],
            x[:, 1:, :],
        )  # [B, num_features], [B, N, num_features]
        x = x.transpose(1, 2)  # [B, num_features, N]
        x = x.view(B, C, H, W)  # [B, num_features, sqrt(N), sqrt(N)]
        x = x + self.conv1(x) + self.conv2(x) + self.conv3(x)
        x = x.flatten(2).transpose(1, 2)  # [B, N, num_features]
        x = torch.cat(
            (class_token.unsqueeze(dim=1), x), dim=1
        )  # [B, N+1, num_features]

        return x


class SelfNystromAttention(nn.Module):
    def __init__(self, in_out_dims):
        super().__init__()
        self.att = NystromAttention(
            dim=in_out_dims,
            dim_head=in_out_dims // 8,
            heads=8,
            num_landmarks=in_out_dims // 2,
            pinv_iterations=6,  # moore-penrose iterations for approx pinverse.
            # 6 recommended by the Nystrom paper
            residual=True,  # if to do an extra residual with the value
            # supposedly faster convergence if turned on
            dropout=0.1,
        )
        self.norm = nn.LayerNorm(in_out_dims)

    def forward(self, x):
        x_norm = self.norm(x)
        x = x + self.att(x_norm)
        return x


class TransMILSquaring(nn.Module):
    def __init__(self, new_num_features, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(
            768, new_num_features
        )  # [B, n, new_num_features], n - no of instances in a processed bag
        self.cls_token = nn.Parameter(
            torch.rand((1, 1, new_num_features))
        )  # [1, new_num_features] - will have to expand this
        self.fc2 = nn.Linear(new_num_features, num_classes)
        self.relu = nn.ReLU()
        self.ln = nn.LayerNorm(new_num_features)
        self.at1 = SelfNystromAttention(new_num_features)
        self.at2 = SelfNystromAttention(new_num_features)
        self.ppeg = PPEG(new_num_features)

    def forward(self, x):
        batch_size, _, _ = x.shape  # x shape is [B, n, new_num_features]

        x = self.fc1(x)  # [B, n, new_num_features]
        x = self.relu(x)

        # squaring of the sequence, find N and M
        n = x.shape[1]
        N = np.ceil(np.sqrt(n))
        H_ppeg, W_ppeg = int(N), int(N)
        M = int(N**2 - n)

        class_token = self.cls_token.expand(
            batch_size, -1, -1
        )  # [B, 1, new_num_features]

        # for no ppeg
        # x = torch.cat((class_token, x), dim=1)  # [B, N+1, new_num_features]

        x = torch.cat(
            (class_token, x, x[:, :M, :]), dim=1
        )  # [B, N+1, new_num_features]

        # first self-attention layer
        x = self.at1(x)

        # PPEG module
        x = self.ppeg(x, H_ppeg, W_ppeg)

        # second self-attention layer
        x = self.at2(x)

        class_token = x[:, 0]  # [B, 1, new_num_features]
        class_token = self.ln(class_token)  # [B, 1, new_num_features]
        logits = self.fc2(class_token)  # [B, num_class]
        logits_prob = F.softmax(logits, dim=1)
        prediction = torch.argmax(logits, dim=1)
        return {
            'logits': logits,
            'y_prob': logits_prob,
            'prediction': prediction,
        }


class TransMILBaseline(nn.Module):
    def __init__(self, new_num_features, n_heads, num_classes):
        super().__init__()
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

        class_token = self.cls_token.expand(
            batch_size, -1, -1
        )  # [B, 1, new_num_features]
        x = torch.cat((class_token, x), dim=1)  # [B, n+1, new_num_features]
        x, _ = self.at1(x, x, x)
        x, _ = self.at2(x, x, x)

        class_token = x[:, 0]  # [B, 1, new_num_features]
        class_token = self.ln(class_token)  # [B, 1, new_num_features]
        logits = self.fc2(class_token)  # [B, num_class]
        logits_prob = F.softmax(logits, dim=1)
        # prediction = torch.argmax(logits, dim=1)
        return {'logits': logits, 'y_prob': logits_prob}
