import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base import BaseModel


class _SingleQNet(nn.Module):
    """One Q-network for Q(s, a, g). Used inside TwinQCritic."""

    def __init__(self, action_dim, input_shape, goal_dim, goal_scale,
                 goal_hidden, goal_layers, fc_hidden, head_layers,
                 use_motion, motion_in_dim, motion_hidden):
        super().__init__()
        self.use_motion = use_motion

        self.register_buffer("goal_scale",
                             torch.tensor(goal_scale, dtype=torch.float32))

        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape)
            flat_size = self._conv_forward(dummy).shape[1]

        self.conv_projection = nn.Linear(flat_size, 256)

        g_layers, in_dim = [], goal_dim
        for _ in range(goal_layers):
            g_layers.append(nn.Linear(in_dim, goal_hidden))
            in_dim = goal_hidden
        self.goal_encoder = nn.ModuleList(g_layers)

        # Action encoder (continuous action ∈ [-1,1]^action_dim)
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, 64),
            nn.ReLU(),
        )
        action_feat = 64

        motion_feat = 0
        if use_motion:
            self.motion_encoder = nn.Sequential(
                nn.Linear(motion_in_dim, motion_hidden),
                nn.ReLU(),
                nn.Linear(motion_hidden, motion_hidden),
            )
            motion_feat = motion_hidden
        else:
            self.motion_encoder = None

        h_layers, in_dim = [], 256 + goal_hidden + action_feat + motion_feat
        for _ in range(head_layers):
            h_layers.append(nn.Linear(in_dim, fc_hidden))
            in_dim = fc_hidden
        self.head = nn.ModuleList(h_layers)
        self.output = nn.Linear(fc_hidden, 1)

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

    def forward(self, obs, action, goal, motion=None):
        x = F.relu(self.conv_projection(self._conv_forward(obs)))
        g = self._encode_goal(goal)
        a = self.action_encoder(action)
        parts = [x, g, a]
        if self.motion_encoder is not None:
            assert motion is not None
            parts.append(F.relu(self.motion_encoder(motion)))
        h = torch.cat(parts, dim=1)
        for layer in self.head:
            h = F.relu(layer(h))
        return self.output(h)

    def _weights_init(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


class TwinQCritic(BaseModel):
    """Twin Q-critics for continuous SAC.

    Two independent Q-networks with separate weights; the target uses min(Q1, Q2).
    Architecture: conv backbone → projection (256) → goal encoder → action encoder
    → motion encoder → 4×512 head → scalar.
    """

    def __init__(self, action_dim=2, input_shape=(3, 96, 96),
                 goal_dim=6,
                 goal_scale=(864., 576., 864., 576., 1., 1.),
                 goal_hidden=128, goal_layers=2,
                 fc_hidden=512, head_layers=4,
                 use_motion=True, motion_in_dim=None, motion_hidden=64):
        super().__init__()

        if use_motion and motion_in_dim is None:
            from motion import motion_dim
            motion_in_dim = motion_dim(action_dim)

        shared = dict(
            action_dim=action_dim, input_shape=input_shape,
            goal_dim=goal_dim, goal_scale=goal_scale,
            goal_hidden=goal_hidden, goal_layers=goal_layers,
            fc_hidden=fc_hidden, head_layers=head_layers,
            use_motion=use_motion,
            motion_in_dim=motion_in_dim if use_motion else 0,
            motion_hidden=motion_hidden,
        )
        self.q1 = _SingleQNet(**shared)
        self.q2 = _SingleQNet(**shared)

        self.q1.apply(self.q1._weights_init)
        self.q2.apply(self.q2._weights_init)

        print(f"TwinQCritic: action_dim={action_dim}, goal_dim={goal_dim}, "
              f"goal_layers={goal_layers}, head_layers={head_layers}, "
              f"use_motion={use_motion}")

    def forward(self, obs, action, goal, motion=None):
        """Returns (Q1, Q2) as (B, 1) tensors."""
        q1 = self.q1(obs, action, goal, motion)
        q2 = self.q2(obs, action, goal, motion)
        return q1, q2

    def Q1(self, obs, action, goal, motion=None):
        """Q1 only — used for actor gradient (cheaper than both)."""
        return self.q1(obs, action, goal, motion)
