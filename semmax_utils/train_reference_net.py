#!/usr/bin/env python
# coding=utf-8

import os, re, math, argparse, gc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datasets import load_from_disk
from sentence_transformers import SentenceTransformer
import os, random
import numpy as np
import torch
from repro import set_seed

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def split_sentences(text):
    """Same splitter as sem_max_nn.py — generator and detector must agree."""
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]

class ResidualBlock(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim, dim), nn.LayerNorm(dim))

    def forward(self, x):
        return x + self.net(x)


class RobustNetwork(nn.Module):
    def __init__(self, embed_dim=768, hidden_dim=2048, num_blocks=2, dropout=0.15):
        super().__init__()
        self.embed_dim = embed_dim
        self.blocks = nn.ModuleList([
            ResidualBlock(embed_dim, hidden_dim, dropout) for _ in range(num_blocks)])

    def forward(self, x):
        for b in self.blocks:
            x = b(x)
        return x / torch.norm(x, dim=-1, keepdim=True) * math.sqrt(self.embed_dim)


def ema_context(H, alpha=0.6):
    """H: (B, w, d) oldest->newest. Mirrors the detector's loop exactly."""
    c = H[:, 0]
    for j in range(1, H.shape[1]):
        c = alpha * H[:, j] + (1.0 - alpha) * c
    return c / c.norm(dim=-1, keepdim=True) * math.sqrt(H.shape[-1])


def load_data(dataset_path, window=5, embedder_name='./semmax_utils/finetuned_embedder'):
    """
    Returns
        S    (N,d) sentence embeddings (originals)
        Ppar (M,d) paraphrase embeddings, aligned with Sidx
        Sidx (M,)  index into S of each aligned original sentence
        ctx  (K,w) context windows (indices into S)
        nxt  (K,)  next-sentence index
    """
    window = window
    cache = f"cached_v6_w{window}_{embedder_name.replace('/','_')}.pt"
    CACHE=cache
    if os.path.exists(cache):
        print(f"Loading cache {CACHE}...")
        b = torch.load(cache, map_location=device)
        return (b["S"].to(device), b["Ppar"].to(device), b["Sidx"].to(device),
                b["ctx"].to(device), b["nxt"].to(device))

    train = load_from_disk(os.path.join(dataset_path, "train"))

    sents, paras, sidx, ctx, nxt = [], [], [], [], []
    n_aligned_docs = 0
    for t, pt in zip(train["text"], train["para_text"]):
        # accept either a list of sentences or a raw paragraph string
        o = t if isinstance(t, list) else split_sentences(t)
        p = pt if isinstance(pt, list) else split_sentences(pt)
        o = [x for x in o if x.strip()]
        p = [x for x in p if x.strip()]
        if len(o) < 2:
            continue

        base = len(sents)
        sents.extend(o)

        # (context, next) windows for L_null — every doc contributes
        for i in range(1, len(o)):
            lo = max(0, i - window)
            win = list(range(base + lo, base + i))
            if len(win) < window:                       # left-pad by repeating oldest
                win = [win[0]] * (window - len(win)) + win
            ctx.append(win); nxt.append(base + i)

        # aligned sentence pairs for L_nce — only if the split lines up
        if len(p) == len(o):
            n_aligned_docs += 1
            for j in range(len(o)):
                paras.append(p[j]); sidx.append(base + j)

    print(f"{len(sents)} sentences | {len(ctx)} (context,next) windows | "
          f"{len(paras)} aligned paraphrase pairs from {n_aligned_docs} docs")
    if not paras:
        raise ValueError("No aligned sentence pairs — original and paraphrase "
                         "split into different sentence counts everywhere.")

    print(f"Encoding with {embedder_name}...")
    emb = SentenceTransformer(embedder_name).to(device)
    S = emb.encode(sents, convert_to_tensor=True, normalize_embeddings=True,
                   show_progress_bar=True, batch_size=256)
    Ppar = emb.encode(paras, convert_to_tensor=True, normalize_embeddings=True,
                      show_progress_bar=True, batch_size=256)
    Sidx = torch.tensor(sidx, dtype=torch.long)
    ctx = torch.tensor(ctx, dtype=torch.long)
    nxt = torch.tensor(nxt, dtype=torch.long)

    torch.save({"S": S.cpu(), "Ppar": Ppar.cpu(), "Sidx": Sidx,
                "ctx": ctx, "nxt": nxt}, CACHE)
    print(f"Saved -> {CACHE}")
    return S.to(device), Ppar.to(device), Sidx.to(device), ctx.to(device), nxt.to(device)

def nce_loss(r_a, r_p, tau=0.07):
    """Single-sentence InfoNCE: invariance + anti-collapse + discriminability."""
    a = F.normalize(r_a, dim=-1)
    p = F.normalize(r_p, dim=-1)
    logits = (a @ p.T) / tau
    return F.cross_entropy(logits, torch.arange(len(r_a), device=r_a.device))


def null_moment_loss(s):
    """s: detection-matched scores on natural continuations. mean->0, var->1."""
    return s.mean().pow(2) + (s.var(unbiased=False) - 1.0).pow(2)


def train(model, S, Ppar, Sidx, ctx, nxt, window=5, alpha=0.6, epochs=30,
          bs_nce=256, bs_null=256, w_nce=1.0, w_null=1.0,
          tau=0.07, val_frac=0.1):
    d = model.embed_dim
    K = ctx.shape[0]
    perm = torch.randperm(K, device=device)
    n_val = int(K * val_frac)
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    M = Ppar.shape[0]
    print(f"\n{len(tr_idx)} null windows (train) / {len(val_idx)} (val) | {M} nce pairs")
    print(f"weights: nce={w_nce} null={w_null} \n")

    opt = optim.Adam(model.parameters(), lr=2e-4, weight_decay=1e-4)
    sch = lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)
    hist = {k: [] for k in ["total", "nce", "null", "val_mean", "val_std"]}

    steps = max(len(tr_idx) // bs_null, 1)
    for ep in range(epochs):
        model.train()
        order = tr_idx[torch.randperm(len(tr_idx), device=device)]
        agg = {"total": 0., "nce": 0., "null": 0.}
        for st in range(steps):
            opt.zero_grad()

            # ---- L_nce : single sentences, no EMA ----
            # ni = torch.randint(0, M, (bs_nce,), device=device)
            ni = torch.randperm(M, device=device)[:bs_nce]
            r_a = model(S[Sidx[ni]])
            r_p = model(Ppar[ni])
            L_nce = nce_loss(r_a, r_p, tau)

            # ---- L_null : EMA context vs true next sentence ----
            g = order[st * bs_null:(st + 1) * bs_null]
            B = len(g)
            H = model(S[ctx[g]].view(B * window, d)).view(B, window, d)
            r = ema_context(H, alpha)
            s = (S[nxt[g]] * r).sum(-1)                # ||r|| = sqrt(d)
            L_null = null_moment_loss(s)


            loss = w_nce * L_nce + w_null * L_null 
            loss.backward(); opt.step()
            agg["total"] += loss.item(); agg["nce"] += L_nce.item()
            agg["null"] += L_null.item(); 

        sch.step()
        for k in ["total", "nce", "null"]:
            hist[k].append(agg[k] / steps)

        model.eval()
        with torch.no_grad():
            vs = []
            for i in range(0, len(val_idx), 512):
                g = val_idx[i:i + 512]; B = len(g)
                H = model(S[ctx[g]].view(B * window, d)).view(B, window, d)
                vs.append((S[nxt[g]] * ema_context(H, alpha)).sum(-1))
            vs = torch.cat(vs)
            hist["val_mean"].append(float(vs.mean())); hist["val_std"].append(float(vs.std()))

        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"Ep {ep+1:02d}/{epochs} | nce {hist['nce'][-1]:.4f} "
                  f"| null {hist['null'][-1]:.4f} |  "
                  f"|| VAL null mean {hist['val_mean'][-1]:+.4f} std {hist['val_std'][-1]:.4f}")
    return hist, val_idx


@torch.no_grad()
def evaluate(model, S, Ppar, Sidx, ctx, nxt, val_idx, hist,args, window=5, alpha=0.6):
    model.eval(); d = model.embed_dim
    s_all = []
    for i in range(0, len(val_idx), 512):
        g = val_idx[i:i + 512]; B = len(g)
        H = model(S[ctx[g]].view(B * window, d)).view(B, window, d)
        s_all.append((S[nxt[g]] * ema_context(H, alpha)).sum(-1))
    s_all = torch.cat(s_all).cpu().numpy()

    ni = torch.randperm(Ppar.shape[0], device=device)[:4000]
    inv = F.cosine_similarity(model(S[Sidx[ni]]), model(Ppar[ni]), dim=-1).cpu().numpy()

    print("\n" + "=" * 62)
    print(f"per-sentence NULL (held-out, EMA context)")
    print(f"   mean {s_all.mean():+.4f}   (target 0)")
    print(f"   std  {s_all.std():.4f}   (target 1)")
    print(f"paraphrase invariance cos: mean {inv.mean():.4f}  min {inv.min():.4f}")
    print("-" * 62)
    print("DOCUMENT-level  Z = sum(s)/sqrt(n)   -- want mean ~0 at EVERY n")
    for n in [5, 10, 15]:
        zs = [s_all[k:k + n].sum() / np.sqrt(n)
              for k in np.random.randint(0, len(s_all) - n, 400)]
        print(f"   n={n:3d}   mean {np.mean(zs):+.3f}   std {np.std(zs):.3f}")
    print("=" * 62)
    print("mean staying ~0 as n grows => the mu*sqrt(n) bias is gone at the source.")

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    for k in ["total", "nce", "null"]:
        ax[0].plot(hist[k], label=k)
    ax[0].set_xlabel("epoch"); ax[0].set_title("losses"); ax[0].legend(); ax[0].grid(True, ls="--", alpha=.4)
    ax[1].plot(hist["val_mean"], label="val null mean"); ax[1].axhline(0, color="k", ls=":")
    ax[1].plot(hist["val_std"], label="val null std"); ax[1].axhline(1, color="r", ls="--")
    ax[1].set_xlabel("epoch"); ax[1].set_title("detection null during training")
    ax[1].legend(); ax[1].grid(True, ls="--", alpha=.4)
    ax[2].hist(s_all, bins=60, alpha=.7, color="tab:gray"); ax[2].axvline(0, color="k", ls=":")
    ax[2].set_title(f"per-sentence null (mean {s_all.mean():+.3f}, std {s_all.std():.3f})")
    ax[2].grid(True, ls="--", alpha=.4)
    plt.tight_layout(); os.makedirs("semmax_utils/plots", exist_ok=True)
    plt.savefig("semmax_utils/plots/train_v6.png", dpi=170)
    torch.save(model.state_dict(), "semmax_utils/robust_weights.pth")
    print("\nSaved -> semmax_utils/robust_weights.pth ; semmax_utils/plots/train_v6.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="./semmax_utils/data/original-c4-texts-8000-pegasus-bigram=False-threshold=0.0-split")
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--alpha", type=float, default=0.6)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--w-nce", type=float, default=1.0)
    ap.add_argument("--w-null", type=float, default=1.0)
    ap.add_argument("--tau", type=float, default=0.07)
    ap.add_argument("--seed", type=int, default=29)
    args = ap.parse_args()
    
    set_seed(args.seed)

    print(f"seed={args.seed} torch={torch.__version__} "
      f"cuda={torch.version.cuda} device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")

    S, Ppar, Sidx, ctx, nxt = load_data(args.data, args.window)
    model = RobustNetwork(embed_dim=S.shape[1]).to(device)
    hist, val_idx = train(model, S, Ppar, Sidx, ctx, nxt, window=args.window,
                          alpha=args.alpha, epochs=args.epochs, w_nce=args.w_nce,
                          w_null=args.w_null, tau=args.tau)
    evaluate(model, S, Ppar, Sidx, ctx, nxt, val_idx, hist,args, args.window, args.alpha)
    del S, Ppar, Sidx, ctx, nxt, model; gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()