"""Datasets used to train the Social LSTM implementation. Data downloading options are included so see README"""

import torch
from torch.utils.data import Dataset


class SyntheticTrajectoryDataset(Dataset):

    """
        Small, deterministic collection of interacting pedestrian trajectories.

        Each item has shape ``(sequence_length, pedestrians, 2)``.  This makes the
        repository runnable without first downloading ETH or UCY while preserving
        the input layout expected by :class:`SocialLSTM`.
    """

    def __init__(
        self,
        num_scenes: int = 64,
        sequence_length: int = 20,
        num_pedestrians: int = 5,
        seed: int = 7,
    ) -> None:
        if num_scenes < 1 or sequence_length < 2 or num_pedestrians < 1:
            raise ValueError("num_scenes and num_pedestrians must be positive; sequence_length must be at least 2")

        generator = torch.Generator().manual_seed(seed)
        time = torch.arange(sequence_length, dtype=torch.float32).view(-1, 1, 1)
        scenes = []

        for _ in range(num_scenes):
            start = (torch.rand(num_pedestrians, 2, generator=generator) - 0.5) * 12.0
            velocity = (torch.rand(num_pedestrians, 2, generator=generator) - 0.5) * 0.8
            acceleration = (torch.rand(num_pedestrians, 2, generator=generator) - 0.5) * 0.015
            noise = torch.randn(sequence_length, num_pedestrians, 2, generator=generator) * 0.015
            trajectory = start.unsqueeze(0) + time * velocity.unsqueeze(0)
            trajectory += 0.5 * time.square() * acceleration.unsqueeze(0)
            scenes.append(trajectory + noise)

        self.trajectories = torch.stack(scenes)

    def __len__(self) -> int:
        return self.trajectories.size(0)

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.trajectories[index]
