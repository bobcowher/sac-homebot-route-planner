import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from models.base import BaseModel

LOG_SIG_MIN = -20
LOG_SIG_MAX = 2


class GaussianActor(BaseModel):
    """Gaussian actor with tanh squashing for continuous SAC.

    Architecture mirrors the Q-DQN champion: conv backbone → projection (256) →
    goal encoder → motion encoder → 4×512 head → (mu, log_sigma).

    goal_dim=6: [robot_x, robot_y, goal_x, goal_y, sin(angle), cos(angle)].
    The sin/cos heading is required because the continuous action (linear, angular)
    is in the robot frame — the actor needs to know current heading to navigate.
    """

    def __init__(self, action_dim=2, input_shape=(3, 96, 96),
                 goal_dim=6,
                 goal_scale=(864., 576., 864., 576., 1., 1.),
                 goal_hidden=128, goal_layers=2,
                 fc_hidden=512, head_layers=4,
                 use_motion=True, motion_in_dim=None, motion_hidden=64):
        super().__init__()
        self.action_dim = action_dim
        self.use_motion = use_motion

        assert len(goal_scale) == goal_dim
        self.register_buffer("goal_scale",
                             torch.tensor(goal_scale, dtype=torch.float32))

        # Convolutional backbone (same as Q-DQN)
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape)
            flat_size = self._conv_forward(dummy).shape[1]

        self.conv_projection = nn.Linear(flat_size, 256)

        # Goal encoder: goal_layers linear layers, ReLU between (not after last)
        g_layers, in_dim = [], goal_dim
        for _ in range(goal_layers):
            g_layers.append(nn.Linear(in_dim, goal_hidden))
            in_dim = goal_hidden
        self.goal_encoder = nn.ModuleList(g_layers)

        # Motion encoder (optional): 2-layer MLP
        motion_feat = 0
        if use_motion:
            if motion_in_dim is None:
                from motion import motion_dim
                motion_in_dim = motion_dim(action_dim)  # window folded in externally
            self.motion_encoder = nn.Sequential(
                nn.Linear(motion_in_dim, motion_hidden),
                nn.ReLU(),
                nn.Linear(motion_hidden, motion_hidden),
            )
            motion_feat = motion_hidden
        else:
            self.motion_encoder = None

        # 4×512 head
        h_layers, in_dim = [], 256 + goal_hidden + motion_feat
        for _ in range(head_layers):
            h_layers.append(nn.Linear(in_dim, fc_hidden))
            in_dim = fc_hidden
        self.head = nn.ModuleList(h_layers)

        # Output: mu and log_sigma for each action dim
        self.output = nn.Linear(fc_hidden, action_dim * 2)

        self.apply(self._weights_init)
        print(f"GaussianActor: action_dim={action_dim}, goal_dim={goal_dim}, "
              f"goal_layers={goal_layers}, head_layers={head_layers}, "
              f"use_motion={use_motion}")

    def _conv_forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return x.flatten(1)

    def _encode_goal(self, goal):
        g = goal / self.goal_scale
        last = len(self.goal_encoder) - 1
        for i, layer in enumerate(self.goal_encoder):
            g = layer(g)
            if i < last:
                g = F.relu(g)
        return g

    def _features(self, obs, goal, motion=None):
        x = F.relu(self.conv_projection(self._conv_forward(obs)))
        g = self._encode_goal(goal)
        parts = [x, g]
        if self.motion_encoder is not None:
            assert motion is not None
            parts.append(F.relu(self.motion_encoder(motion)))
        h = torch.cat(parts, dim=1)
        for layer in self.head:
            h = F.relu(layer(h))
        return h

    def forward(self, obs, goal, motion=None):
        """Returns (mu, log_sigma) before squashing."""
        h = self._features(obs, goal, motion)
        out = self.output(h)
        mu, log_sigma = out.chunk(2, dim=-1)
        log_sigma = torch.clamp(log_sigma, LOG_SIG_MIN, LOG_SIG_MAX)
        return mu, log_sigma

    def sample(self, obs, goal, motion=None):
        """Sample action with reparameterization trick. Returns (action, log_prob, mean).

        action ∈ [-1, 1]^action_dim (tanh squashed).
        log_prob accounts for the tanh squashing Jacobian.
        """
        mu, log_sigma = self.forward(obs, goal, motion)
        sigma = log_sigma.exp()
        dist = Normal(mu, sigma)
        u = dist.rsample()                           # reparameterized sample
        action = torch.tanh(u)

        # log_prob with tanh correction: sum_i [log N(u_i) - log(1 - tanh^2(u_i))]
        log_prob = dist.log_prob(u) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)  # (B, 1)

        mean_action = torch.tanh(mu)
        return action, log_prob, mean_action

    def mean_action(self, obs, goal, motion=None):
        """Deterministic mean action (tanh(mu)) for greedy eval."""
        mu, _ = self.forward(obs, goal, motion)
        return torch.tanh(mu)

    def _weights_init(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
