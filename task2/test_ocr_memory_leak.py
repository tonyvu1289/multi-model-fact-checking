import argparse
import gc
import os

import psutil
import torch

from read_data import get_dataset
from train import ClaimVerificationDataset, make_batch, FocalLoss
from model import MultiModalClassification


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, required=True)
    parser.add_argument('--claim_pt', type=str, default='roberta-base')
    parser.add_argument('--vision_pt', type=str, default='ocr_easyocr')
    parser.add_argument('--long_pt', type=str, default='longformer')
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--sample_limit', type=int, default=6)
    parser.add_argument('--steps', type=int, default=8)
    parser.add_argument('--cpu_leak_mb_threshold', type=float, default=600.0)
    parser.add_argument('--gpu_leak_mb_threshold', type=float, default=600.0)
    return parser.parse_args()


def rss_mb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def gpu_allocated_mb(device):
    if device.type != 'cuda':
        return 0.0
    return torch.cuda.memory_allocated(device) / (1024 * 1024)


def main():
    args = parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)

    train, _, _ = get_dataset(args.path)
    train = train[: max(1, args.sample_limit)]
    train_ds = ClaimVerificationDataset(train)

    model = MultiModalClassification(
        device=device,
        claim_pt=args.claim_pt,
        vision_pt=args.vision_pt,
        long_pt=args.long_pt,
    ).to(device)
    model.train()

    loss_fn = FocalLoss(gamma=2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    X, y, _ = make_batch(train_ds, batch_size=args.batch_size, shuffle=False)
    if not X:
        raise RuntimeError('No batches created. Check dataset path/sample_limit.')

    cpu_track = []
    gpu_track = []

    for step in range(args.steps):
        batch_idx = step % len(X)
        optimizer.zero_grad()
        score, lb = model(X[batch_idx], y[batch_idx])
        loss = loss_fn(score.to(device), lb.to(device))
        loss.backward()
        optimizer.step()

        gc.collect()
        if device.type == 'cuda':
            torch.cuda.synchronize(device)

        cpu_m = rss_mb()
        gpu_m = gpu_allocated_mb(device)
        cpu_track.append(cpu_m)
        gpu_track.append(gpu_m)

        print(
            f'step={step + 1} loss={loss.item():.6f} '
            f'cpu_rss_mb={cpu_m:.2f} gpu_alloc_mb={gpu_m:.2f}'
        )

    cpu_drift = cpu_track[-1] - cpu_track[0]
    gpu_drift = gpu_track[-1] - gpu_track[0]

    print('\nMemory drift report')
    print(f'cpu_drift_mb={cpu_drift:.2f}')
    print(f'gpu_drift_mb={gpu_drift:.2f}')

    cpu_ok = cpu_drift <= args.cpu_leak_mb_threshold
    gpu_ok = gpu_drift <= args.gpu_leak_mb_threshold

    if not (cpu_ok and gpu_ok):
        raise RuntimeError(
            'Potential memory leak detected: '
            f'cpu_drift_mb={cpu_drift:.2f}, gpu_drift_mb={gpu_drift:.2f}'
        )

    print('Leak smoke-test: PASS')


if __name__ == '__main__':
    main()
