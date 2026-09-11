from lstm_unit import LSTMUnit
from torch import nn
import torch

class SocialPooling(nn.Module):
    def __init__(self, spatial_size=32, grid_size=8, hidden_dim=128):
        """
        Social Pooling Layer from the Social LSTM paper.
        
        Args:
            spatial_size (float): The side length N_o of the square neighborhood region.
            grid_size (int): Grid division factor (e.g., 8 -> 8x8 spatial grid cells).
            hidden_dim (int): Hidden state vector dimension D of each pedestrian's LSTM.  """
       
        super(SocialPooling, self).__init__()
        self.spatial_size = spatial_size
        self.grid_size = grid_size
        self.hidden_dim = hidden_dim
        self.cell_size = spatial_size / grid_size

    def forward(self, current_positions, hidden_states):
        
        num_peds = current_positions.size(0)
        
        # Initialize output tensor H_t with zeros
        social_tensor = torch.zeros(
            (num_peds, self.grid_size, self.grid_size, self.hidden_dim),
            device=current_positions.device,
            dtype=hidden_states.dtype
        )

        for i in range(num_peds):
            pos_i = current_positions[i]
            
            # Compute relative coordinates for all neighbors relative to pedestrian i
            rel_pos = current_positions - pos_i  # Shape: (Num_pedestrians, 2)
            
            # Calculate top-left boundary of the N_o x N_o bounding box centered at pos_i
            half_size = self.spatial_size / 2.0
            
            # Convert relative coordinates to grid indices (m, n)
            # Center of the grid is at (0, 0) relative to pos_i
            grid_x = torch.floor((rel_pos[:, 0] + half_size) / self.cell_size).long()
            grid_y = torch.floor((rel_pos[:, 1] + half_size) / self.cell_size).long()
            
            # Mask out neighbors outside the N_o x N_o spatial neighborhood or self
            valid_mask = (
                (grid_x >= 0) & (grid_x < self.grid_size) &
                (grid_y >= 0) & (grid_y < self.grid_size) &
                (torch.arange(num_peds, device=current_positions.device) != i)
            )
            
            # Pool hidden states of neighbors by summing them into their assigned grid cells
            for j in range(num_peds):
                if valid_mask[j]:
                    m = grid_x[j].item()
                    n = grid_y[j].item()
                    social_tensor[i, m, n, :] += hidden_states[j]

        # Flatten grid and hidden state dimensions into a vector for each pedestrian
        return social_tensor.view(num_peds, -1)