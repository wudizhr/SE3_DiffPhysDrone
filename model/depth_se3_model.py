import torch
from torch import nn


class Model(nn.Module):
    """Depth recurrent policy that outputs CTBR raw commands."""

    def __init__(self, dim_obs=10, dim_action=4, hidden_dim=192) -> None:
        super().__init__()
        self.dim_obs = int(dim_obs)
        self.dim_action = int(dim_action)
        self.hidden_dim = int(hidden_dim)
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, 2, 2, bias=False),
            nn.LeakyReLU(0.05),
            nn.Conv2d(32, 64, 3, bias=False),
            nn.LeakyReLU(0.05),
            nn.Conv2d(64, 128, 3, bias=False),
            nn.LeakyReLU(0.05),
            nn.Flatten(),
            nn.Linear(128 * 2 * 4, self.hidden_dim, bias=False),
        )
        self.observation_fc = nn.Linear(self.dim_obs, self.hidden_dim)
        self.gru = nn.GRUCell(self.hidden_dim, self.hidden_dim)
        self.action_fc = nn.Linear(self.hidden_dim, self.dim_action, bias=False)
        self.activation = nn.LeakyReLU(0.05)

    def reset(self):
        pass

    def forward(self, x: torch.Tensor, state, hx=None):
        if state is None:
            raise ValueError("depth_se3_model requires a state tensor")
        if x.ndim != 4:
            raise ValueError(f"depth input must be 4D, got shape {tuple(x.shape)}")
        img_feat = self.stem(x)
        fused = self.activation(img_feat + self.observation_fc(state))
        hx = self.gru(fused, hx)
        ctbr_raw = self.action_fc(self.activation(hx))
        return ctbr_raw, None, hx


if __name__ == "__main__":
    model = Model()
    print(model(torch.ones(2, 1, 12, 16), torch.zeros(2, 10))[0].shape)
