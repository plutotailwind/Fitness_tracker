# simple_gcn.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleGCN(nn.Module):
    """
    Input:  x [B, C, T, V]
    Output: per-frame embeddings [B, hidden, T]
    - Graph message passing over joints (V)
    - Pool over joints only; keep time
    """
    def __init__(self, in_channels=2, num_joints=33, hidden=64):
        super().__init__()
        self.A = self.build_adjacency(num_joints)  # [V, V]
        self.conv1 = nn.Conv2d(in_channels, hidden, kernel_size=1, bias=False)
        self.bn1   = nn.BatchNorm2d(hidden)
        self.conv2 = nn.Conv2d(hidden, hidden, kernel_size=1, bias=False)
        self.bn2   = nn.BatchNorm2d(hidden)

    def build_adjacency(self, V: int) -> torch.Tensor:
        # Simple chain + self connections. Replace with a real skeleton graph if you want.
        A = torch.eye(V)
        for i in range(V - 1):
            A[i, i + 1] = 1.0
            A[i + 1, i] = 1.0
        # Optional degree normalization
        D_inv_sqrt = torch.diag(1.0 / torch.sqrt(A.sum(dim=1) + 1e-6))
        A_norm = D_inv_sqrt @ A @ D_inv_sqrt
        return A_norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T, V]
        A = self.A.to(x.device)
        # 1x1 conv over channels (feature lift)
        x = self.bn1(self.conv1(x))               # [B, hidden, T, V]
        # message passing over joints: multiply along V
        x = torch.einsum('nctv,vw->nctw', x, A)   # [B, hidden, T, V]
        x = F.relu(self.bn2(self.conv2(x)))       # [B, hidden, T, V]
        # pool ONLY over joints, keep time
        x = x.mean(dim=-1)                        # [B, hidden, T]
        return x

