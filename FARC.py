import torch
import torch.nn as nn


class FARCtor(nn.Module):
    def __init__(self, input_size=150, hidden_size=64, masked_dim=50):
        super(FARCtor, self).__init__()
        self.name = "farctor"
        self.num_channels = masked_dim
        self.hidden_size = hidden_size
        self.batch_norm = nn.BatchNorm1d(1)
        self.action_scaler = 100000.
        # FC Layers for both short-term and long-term memory pathways
        self.st_path = nn.Sequential(
            nn.Linear(masked_dim // 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 2),
            nn.ReLU()
        )
        self.lt_path = nn.Sequential(
            nn.Linear(masked_dim // 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 2),
            nn.ReLU()
        )
        # combine outputs of the pathway models
        self.combined_path = nn.Sequential(
            nn.Linear(4, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, 2),
            nn.ReLU()
        )
        # prepare a mask tensor for filtering
        self.register_buffer('mask', self.prepare_filter_mask())

    def prepare_filter_mask(self):
        weights = {
            1: 1.0,  # Receiving rate
            2: 1.0,  # Number of received packets
            3: 0.0,  # Received bytes
            4: 1.0,  # Queuing delay
            5: 0.0,  # Delay
            6: 0.0,  # Minimum seen delay
            7: 0.0,  # Delay ratio
            8: 0.0,  # Delay average minimum difference
            9: 0.0,  # Interarrival time
            10: 0.0,  # Jitter
            11: 1.0,  # Packet loss ratio
            12: 1.0,  # Average number of lost packets
            13: 0.0,  # Video packets probability
            14: 0.0,  # Audio packets probability
            15: 0.0   # Probing packets probability
        }

        mask = [weight > 0 for ind, weight in weights.items() for _ in range(10)]
        mask_tensor = torch.tensor(mask, dtype=torch.bool)
        return mask_tensor

    def forward(self, x, h, c):
        x = x[:, :, self.mask]
        # convert receiving rate from bps to mbps
        x[:, :, 0:10] = x[:, :, 0:10] / self.action_scaler
        x = self.batch_norm(x)
        # split input into short-term and long-term features
        st_features = x[:, :, :self.num_channels // 2]
        lt_features = x[:, :, self.num_channels // 2:]

        # process short-term features
        st_output = self.st_path(st_features)

        # process long-term features
        lt_output = self.lt_path(lt_features)

        # concatenate the last outputs of both pathways
        combined_input = torch.cat((st_output, lt_output), dim=2)

        # combine outputs using additional layers
        action = self.combined_path(combined_input)
        # scale back to bps space
        action = action * self.action_scaler

        return action, h, c


class FARCritic(nn.Module):
    # predicts the video quality for the given bw prediction and state
    def __init__(self, input_dim=150, masked_dim=50):
        super(FARCritic, self).__init__()
        self.name="farcritic"
        self.action_scaler = 100000.
        self.batch_norm = nn.BatchNorm1d(masked_dim)
        self.state_path = nn.Sequential(
            nn.Linear(masked_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 2), # video quality and audio quality
            nn.ReLU()
        )
        self.final_path = nn.Sequential(
            nn.Linear(3, 8),
            nn.ReLU(),
            nn.Linear(8, 2), # video quality and audio quality
            nn.ReLU()
        )
        # prepare a mask tensor for filtering
        self.register_buffer('mask', self.prepare_filter_mask())

    def prepare_filter_mask(self):
        weights = {
            1: 1.0,  # Receiving rate
            2: 1.0,  # Number of received packets
            3: 0.0,  # Received bytes
            4: 1.0,  # Queuing delay
            5: 0.0,  # Delay
            6: 0.0,  # Minimum seen delay
            7: 0.0,  # Delay ratio
            8: 0.0,  # Delay average minimum difference
            9: 0.0,  # Interarrival time
            10: 0.0,  # Jitter
            11: 1.0,  # Packet loss ratio
            12: 1.0,  # Average number of lost packets
            13: 0.0,  # Video packets probability
            14: 0.0,  # Audio packets probability
            15: 0.0   # Probing packets probability
        }

        mask = [weight > 0 for ind, weight in weights.items() for _ in range(10)]
        mask_tensor = torch.tensor(mask, dtype=torch.bool)
        return mask_tensor

    def forward(self, state, action):
        state = state[:, :, self.mask].squeeze(1)
        # convert receiving rate from bps to mbps
        state[:, 0:10] = state[:, 0:10] / self.action_scaler
        action = action / self.action_scaler

        state = self.batch_norm(state)
        state = self.state_path(state).squeeze()
        state_action = torch.cat([state, action], dim=1)

        return self.final_path(state_action)
