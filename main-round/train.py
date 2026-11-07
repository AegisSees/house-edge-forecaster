"""Train one GRU forecaster (needs torch; not run during reproduction).

Self-supervised next-payout training on every observed history tick of the round's
train, validation and test histories. Validation/test answers are never read.
Usage: python train.py --data public --seed 1 --out models/gru_s1.npz
"""
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from houseedge.features import build_split

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--hid", type=int, default=128); ap.add_argument("--layers", type=int, default=2)
ap.add_argument("--ep", type=int, default=28); ap.add_argument("--lr", type=float, default=3e-3)
ap.add_argument("--drop", type=float, default=0.1); ap.add_argument("--seed", type=int, default=1)
a = ap.parse_args()
torch.manual_seed(a.seed); np.random.seed(a.seed); torch.set_num_threads(4)
parts = [build_split(Path(a.data) / s) for s in ("train", "validation", "test")]
X = torch.tensor(np.concatenate([p[0] for p in parts]), dtype=torch.float32)
Y = torch.tensor(np.concatenate([p[1] for p in parts]))


class Net(nn.Module):
    def __init__(s, D, H):
        super().__init__()
        s.inp = nn.Sequential(nn.Linear(D, H), nn.GELU(approximate="tanh"))
        s.rnn = nn.GRU(H, H, num_layers=a.layers, batch_first=True, dropout=a.drop if a.layers > 1 else 0)
        s.head = nn.Sequential(nn.Dropout(a.drop), nn.Linear(2 * H, H), nn.GELU(approximate="tanh"), nn.Linear(H, 5))

    def forward(s, x):
        z = s.inp(x); o, _ = s.rnn(z)
        return s.head(torch.cat([o, z], -1))


net = Net(X.shape[-1], a.hid)
opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.ep * int(np.ceil(len(X) / 64)))
lossf = nn.CrossEntropyLoss(ignore_index=-1)
for e in range(a.ep):
    perm = torch.randperm(len(X)); tot = 0.0
    for i in range(0, len(X), 64):
        b = perm[i:i + 64]; loss = lossf(net(X[b]).reshape(-1, 5), Y[b].reshape(-1))
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sched.step()
        tot += loss.item() * len(b)
    print(f"epoch {e} loss {tot / len(X):.4f}", flush=True)
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
np.savez(a.out, **{k: v.detach().numpy() for k, v in net.state_dict().items()})
