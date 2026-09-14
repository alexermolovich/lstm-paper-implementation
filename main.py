"""Train Social LSTM on the bundled synthetic trajectory data."""

import argparse

import lightning as L
from torch.utils.data import DataLoader

from src.data_processing import SyntheticTrajectoryDataset
from src.social_lstm import SocialLSTM


def train(args: argparse.Namespace) -> SocialLSTM:
    L.seed_everything(args.seed, workers=True)
    dataset = SyntheticTrajectoryDataset(
        num_scenes=args.scenes,
        sequence_length=args.sequence_length,
        num_pedestrians=args.pedestrians,
        seed=args.seed,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    model = SocialLSTM(lr=args.learning_rate, t_obs=args.observed_length)
    trainer = L.Trainer(
        max_epochs=args.epochs,
        accelerator="auto",
        devices=1,
        logger=False,
        enable_checkpointing=False,
    )
    trainer.fit(model, train_dataloaders=loader)
    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--scenes", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--sequence-length", type=int, default=20)
    parser.add_argument("--observed-length", type=int, default=8)
    parser.add_argument("--pedestrians", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    if args.sequence_length <= args.observed_length:
        parser.error("--sequence-length must be greater than --observed-length")
    return args


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
