import torch
import os


class ReplayBuffer:
    def __init__(self, max_size, input_shape, action_dim, input_device,
                 output_device='cpu', goal_dim=6, motion_dim=0):
        self.mem_size = max_size
        self.mem_ctr  = 0

        override = os.getenv("REPLAY_BUFFER_MEMORY")
        if override in ["cpu", "cuda:0", "cuda:1"]:
            print("Received replay buffer memory override.")
            self.input_device = override
        else:
            self.input_device = input_device

        print(f"Replay buffer memory on: {self.input_device}")
        self.output_device = output_device

        self.state_memory      = torch.zeros((max_size, *input_shape), dtype=torch.uint8,   device=self.input_device)
        self.next_state_memory = torch.zeros((max_size, *input_shape), dtype=torch.uint8,   device=self.input_device)
        # Continuous actions stored as float vectors.
        self.action_memory     = torch.zeros((max_size, action_dim),   dtype=torch.float32, device=self.input_device)
        self.reward_memory     = torch.zeros(max_size,                 dtype=torch.float32, device=self.input_device)
        self.terminal_memory   = torch.zeros(max_size,                 dtype=torch.bool,    device=self.input_device)
        # Per-transition bootstrap discount (gamma**n_eff); n_eff varies near episode
        # ends / hindsight-goal hits, so this can't be a single global gamma**n constant.
        self.discount_memory   = torch.zeros(max_size,                 dtype=torch.float32, device=self.input_device)
        self.goal_memory       = torch.zeros((max_size, goal_dim),     dtype=torch.float32, device=self.input_device)
        self.next_goal_memory  = torch.zeros((max_size, goal_dim),     dtype=torch.float32, device=self.input_device)

        self.use_motion = motion_dim > 0
        if self.use_motion:
            self.motion_memory      = torch.zeros((max_size, motion_dim), dtype=torch.float32, device=self.input_device)
            self.next_motion_memory = torch.zeros((max_size, motion_dim), dtype=torch.float32, device=self.input_device)

    def can_sample(self, batch_size: int) -> bool:
        return self.mem_ctr >= batch_size * 10

    def store_transition(self, state, action, reward, next_state, done, goal, next_goal,
                         motion=None, next_motion=None, discount=None):
        idx = self.mem_ctr % self.mem_size
        self.state_memory[idx]      = torch.as_tensor(state,      dtype=torch.uint8,   device=self.input_device)
        self.next_state_memory[idx] = torch.as_tensor(next_state, dtype=torch.uint8,   device=self.input_device)
        self.action_memory[idx]     = torch.as_tensor(action,     dtype=torch.float32, device=self.input_device)
        self.reward_memory[idx]     = float(reward)
        self.terminal_memory[idx]   = bool(done)
        self.goal_memory[idx]       = torch.as_tensor(goal,      dtype=torch.float32, device=self.input_device)
        self.next_goal_memory[idx]  = torch.as_tensor(next_goal, dtype=torch.float32, device=self.input_device)
        self.discount_memory[idx]   = float(discount)
        if self.use_motion:
            self.motion_memory[idx]      = torch.as_tensor(motion,      dtype=torch.float32, device=self.input_device)
            self.next_motion_memory[idx] = torch.as_tensor(next_motion, dtype=torch.float32, device=self.input_device)
        self.mem_ctr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_ctr, self.mem_size)
        batch   = torch.randint(0, max_mem, (batch_size,), device=self.input_device, dtype=torch.int64)

        states      = self.state_memory[batch].to(self.output_device,      dtype=torch.float32)
        next_states = self.next_state_memory[batch].to(self.output_device, dtype=torch.float32)
        actions     = self.action_memory[batch].to(self.output_device)
        rewards     = self.reward_memory[batch].to(self.output_device)
        dones       = self.terminal_memory[batch].to(self.output_device)
        goals       = self.goal_memory[batch].to(self.output_device)
        next_goals  = self.next_goal_memory[batch].to(self.output_device)
        discounts   = self.discount_memory[batch].to(self.output_device)

        motions = next_motions = None
        if self.use_motion:
            motions      = self.motion_memory[batch].to(self.output_device)
            next_motions = self.next_motion_memory[batch].to(self.output_device)

        return states, actions, rewards, next_states, dones, goals, next_goals, motions, next_motions, discounts
