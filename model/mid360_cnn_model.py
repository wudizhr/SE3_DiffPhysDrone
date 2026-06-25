import torch
from torch import nn


class Model(nn.Module):
    def __init__(
        self,
        dim_obs=10,
        dim_action=6,
        hidden_dim=192,
        input_height=12,
        input_width=60,
    ) -> None:
        super().__init__()
        self.dim_obs = int(dim_obs)
        self.dim_action = int(dim_action)
        self.hidden_dim = int(hidden_dim)
        self.input_height = int(input_height)
        self.input_width = int(input_width)

        self.lidar_stem = nn.Sequential(
            nn.Conv2d(1, 32, 5, 2, 2, bias=False),
            nn.LeakyReLU(0.05),
            nn.Conv2d(32, 64, 3, 2, 1, bias=False),
            nn.LeakyReLU(0.05),
            nn.Conv2d(64, 128, 3, 2, 1, bias=False),
            nn.LeakyReLU(0.05),
            nn.AdaptiveAvgPool2d((2, 4)),
            nn.Flatten(),
            nn.Linear(128 * 2 * 4, self.hidden_dim, bias=False),
        )
        self.observation_fc = nn.Linear(self.dim_obs, self.hidden_dim)
        self.gru = nn.GRUCell(self.hidden_dim, self.hidden_dim)
        self.action_fc = nn.Linear(self.hidden_dim, self.dim_action, bias=False)
        self.activation = nn.LeakyReLU(0.05)

    def reset(self):
        pass

    def forward(self, obs, state=None, hx=None):
        if isinstance(obs, dict):
            pseudo_image = obs["mid360_pseudo_image"]
            state = obs["state"]
        else:
            pseudo_image = obs
        if state is None:
            raise ValueError("mid360_cnn_model requires a state tensor")
        if pseudo_image.ndim != 4:
            raise ValueError(f"mid360_pseudo_image must be 4D, got shape {tuple(pseudo_image.shape)}")
        if pseudo_image.shape[1] != 1 and pseudo_image.shape[-1] == 1:
            pseudo_image = pseudo_image.permute(0, 3, 1, 2).contiguous()

        lidar_feat = self.lidar_stem(pseudo_image)
        fused = self.activation(lidar_feat + self.observation_fc(state))
        hx = self.gru(fused, hx)
        action = self.action_fc(self.activation(hx))
        return action, None, hx


if __name__ == "__main__":
    model = Model()
    obs = {
        "mid360_pseudo_image": torch.ones(2, 1, 12, 60),
        "state": torch.zeros(2, 10),
    }
    print(model(obs)[0].shape)
