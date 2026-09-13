from lstm_unit import LSTMUnit
from torch import nn
import torch
import lightning as L 

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
    
class SocialLSTM(L.LightningModule):
    def __init__(
        self,
        coord_dim=2,
        embedding_dim=64,
        hidden_dim=128,
        spatial_size=32,
        grid_size=8,
        lr=0.003
    ):
        super(SocialLSTM, self).__init__()
        self.save_hyperparameters()

        self.coord_dim = coord_dim
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.lr = lr

        # 1. Spatial Coordinate Embedding Layer (e_t)
        self.coord_embedding = nn.Sequential(
            nn.Linear(coord_dim, embedding_dim),
            nn.ReLU()
        )

        # 2. Social Pooling Layer
        self.social_pooling = SocialPooling(
            spatial_size=spatial_size,
            grid_size=grid_size,
            hidden_dim=hidden_dim
        )

        # 3. Social Tensor Embedding Layer (a_t)
        pooled_input_dim = grid_size * grid_size * hidden_dim
        self.social_embedding = nn.Sequential(
            nn.Linear(pooled_input_dim, embedding_dim),
            nn.ReLU()
        )

        # 4. Standard PyTorch LSTM Cell (Input size: spatial_emb + social_emb)
        self.lstm_cell = nn.LSTMCell(embedding_dim + embedding_dim, hidden_dim)

        # 5. Output Trajectory Distribution Parameter Generator (W_p)
        # Predicts 5 parameters: mu_x, mu_y, log_sigma_x, log_sigma_y, atanh_rho
        self.output_layer = nn.Linear(hidden_dim, 5)

    def forward_step(self, pos, h_prev, c_prev):
        """
        Single step pass for all pedestrians in the scene at time t.
        
        Args:
            pos: (num_peds, 2) spatial coordinates at time t
            h_prev: (num_peds, hidden_dim) previous hidden state
            c_prev: (num_peds, hidden_dim) previous cell state
        Returns:
            dist_params: (num_peds, 5) predicted bivariate gaussian parameters
            h_next: (num_peds, hidden_dim) updated hidden state
            c_next: (num_peds, hidden_dim) updated cell state
        """
        # Embed spatial coordinates e_t
        e_t = self.coord_embedding(pos)

        # Compute Social Tensor and embed it a_t
        social_tensor = self.social_pooling(pos, h_prev)
        a_t = self.social_embedding(social_tensor)

        # Concatenate embeddings
        lstm_input = torch.cat([e_t, a_t], dim=-1)

        # LSTM Cell update
        h_next, c_next = self.lstm_cell(lstm_input, (h_prev, c_prev))

        # Predict Bivariate Gaussian Parameters
        dist_params = self.output_layer(h_next)

        return dist_params, h_next, c_next

    @staticmethod
    def bivariate_gaussian_loss(dist_params, target_pos):
        """
        Calculates Negative Log-Likelihood (NLL) loss for Bivariate Gaussian.
        
        Args:
            dist_params: (num_peds, 5) output from linear prediction layer
            target_pos: (num_peds, 2) actual target positions (x, y)
        """
        mu_x = dist_params[:, 0]
        mu_y = dist_params[:, 1]
        sigma_x = torch.exp(dist_params[:, 2])  # Ensure positive std dev
        sigma_y = torch.exp(dist_params[:, 3])
        rho = torch.tanh(dist_params[:, 4])      # Ensure correlation in (-1, 1)

        x = target_pos[:, 0]
        y = target_pos[:, 1]

        norm_x = (x - mu_x) / sigma_x
        norm_y = (y - mu_y) / sigma_y

        z = norm_x**2 + norm_y**2 - 2 * rho * norm_x * norm_y
        one_minus_rho2 = 1.0 - rho**2 + 1e-6  # Epsilon for numerical stability

        # Bivariate Gaussian Negative Log-Likelihood Formula
        log_prob = -0.5 * (z / one_minus_rho2) - torch.log(
            2 * math.pi * sigma_x * sigma_y * torch.sqrt(one_minus_rho2) + 1e-6
        )

        return -torch.mean(log_prob)

    def forward(self, observed_seq, pred_len):
        """
        Full Sequence Inference/Forward Pass.
        
        Args:
            observed_seq: (T_obs, num_peds, 2) ground-truth positions during observation
            pred_len: int, number of future steps to predict (T_pred - T_obs)
        """
        t_obs, num_peds, _ = observed_seq.shape
        device = observed_seq.device

        h = torch.zeros(num_peds, self.hidden_dim, device=device)
        c = torch.zeros(num_peds, self.hidden_dim, device=device)

        # 1. Process observed history
        for t in range(t_obs):
            current_pos = observed_seq[t]
            params, h, c = self.forward_step(current_pos, h, c)

        # 2. Autoregressive trajectory generation for future timesteps
        predictions = []
        current_pos = observed_seq[-1]

        for _ in range(pred_len):
            params, h, c = self.forward_step(current_pos, h, c)
            
            # Extract mean predicted position for deterministic sampling
            next_pos = params[:, 0:2]
            predictions.append(next_pos)
            current_pos = next_pos

        return torch.stack(predictions, dim=0)  # Shape: (pred_len, num_peds, 2)

    def training_step(self, batch, batch_idx):
        """
        batch shape: (seq_len, num_peds, 2) where seq_len = T_obs + T_pred
        """
        seq_len, num_peds, _ = batch.shape
        t_obs = self.hparams.get("t_obs", 8)  # Default: 8 observed frames (3.2s)

        h = torch.zeros(num_peds, self.hidden_dim, device=batch.device)
        c = torch.zeros(num_peds, self.hidden_dim, device=batch.device)

        loss = 0.0

        for t in range(seq_len - 1):
            current_pos = batch[t]
            target_pos = batch[t + 1]

            params, h, c = self.forward_step(current_pos, h, c)

            # Compute loss on steps after observation window (T_obs)
            if t >= t_obs - 1:
                step_loss = self.bivariate_gaussian_loss(params, target_pos)
                loss += step_loss

        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        # RMSprop or Adam as described in Section 3.2 of the paper
        return torch.optim.RMSprop(self.parameters(), lr=self.lr)