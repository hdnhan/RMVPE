import os
from datetime import datetime

import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from argparse import ArgumentParser

from src import MIR1K, E2E0, bce
from evaluate import evaluate

config = ArgumentParser()
config.add_argument("--data_dir", type=str, default="data")
config.add_argument("--epochs", type=int, default=200)
config.add_argument("--lr", type=float, default=1e-4)


def train(args):
    time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    args.hsize = 320
    logdir = f"runs/{time}-{args.hsize}"
    os.makedirs(logdir, exist_ok=True)
    writer = SummaryWriter(logdir)

    batch_size = 16
    clip_grad_norm = 3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = MIR1K(args.data_dir, args.hsize, ["train"], whole_audio=False, use_aug=True)
    validation_dataset = MIR1K(args.data_dir, args.hsize, ["test"], whole_audio=True, use_aug=False)
    data_loader = DataLoader(
        train_dataset,
        batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        persistent_workers=True,
        num_workers=2,
    )

    model = E2E0(4, 1, (2, 2)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    # summary(model)

    prev_rpa = 0
    for epoch in range(args.epochs):
        print(f"Epoch {epoch + 1}/{args.epochs}", "lr:", scheduler.get_last_lr()[0])

        # Training
        model.train()
        losses = []
        for data in tqdm(data_loader):
            optimizer.zero_grad()
            mel, pitch_label = data["mel"].to(device), data["pitch"].to(device)

            pitch_pred = model(mel)
            loss = bce(pitch_pred, pitch_label)

            loss.backward()
            if clip_grad_norm:
                clip_grad_norm_(model.parameters(), clip_grad_norm)
            optimizer.step()
            losses.append(loss.item())
        writer.add_scalar("train/loss", np.mean(losses), global_step=epoch)
        writer.add_scalar("train/lr", scheduler.get_last_lr()[0], global_step=epoch)
        print(f"Loss: {np.mean(losses)}")
        scheduler.step()

        # Validation
        model.eval()
        with torch.no_grad():
            metrics = evaluate(validation_dataset, model, args.hsize, device)
            for key, value in metrics.items():
                writer.add_scalar("validation/" + key, np.mean(value), global_step=epoch)
            rpa = np.mean(metrics["RPA"])
            print(f"RPA: {rpa}")
            if rpa > prev_rpa:
                print(f"Saving model...")
                torch.save({"model": model.state_dict()}, os.path.join(logdir, "rmvpe.pth"))
                prev_rpa = rpa
        print()


if __name__ == "__main__":
    args = config.parse_args()
    train(args)
