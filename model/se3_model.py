import torch
from torch import nn


class Se3Model(nn.Module):
    """Depth recurrent policy that outputs CTBR raw commands."""

    def __init__(self, dim_obs=10, dim_action=4, hidden_dim=192):
        super().__init__()
        self.dim_obs = dim_obs
        self.dim_action = dim_action
        self.hidden_dim = hidden_dim

        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, 3, 2, 1, bias=False), 
            nn.LeakyReLU(0.05),
            nn.Conv2d(32, 64, 3, 2, 1, bias=False),
            nn.LeakyReLU(0.05),
            nn.Conv2d(64, 128, 3, 2, 1, bias=False),
            nn.LeakyReLU(0.05),
            nn.Flatten(),
            nn.Linear(128 * 3 * 4, hidden_dim, bias=False),
        )
        self.observation_fc = nn.Linear(dim_obs, hidden_dim)
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.action_fc = nn.Linear(hidden_dim, dim_action, bias=False)
        self.activation = nn.LeakyReLU(0.05)

    def reset(self):
        pass

    def forward(self, depth, state, hx=None):
        feat = self.stem(depth)
        feat = self.activation(feat + self.observation_fc(state))
        hx = self.gru(feat, hx)
        raw_action = self.action_fc(self.activation(hx))

        thrust_raw = raw_action[:, :1]
        body_rate_raw = raw_action[:, 1:4]
        ctbr_raw = torch.cat([thrust_raw, body_rate_raw], -1)
        return ctbr_raw, hx
