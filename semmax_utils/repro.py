import os, random
import numpy as np
import torch
import hashlib


def set_seed(seed: int = 29, strict: bool = False):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # cuDNN: deterministic conv/matmul selection
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False          

    # matmul determinism (Ampere+ TF32 can vary)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    # cuBLAS workspace config required for deterministic matmuls
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    if strict:
        # will raise if any op lacks a deterministic kernel — use to AUDIT, then
        # relax if it errors on an op you cannot avoid.
        torch.use_deterministic_algorithms(True)

    # print(f"[repro] seed={seed} strict={strict} "
    #       f"cudnn.deterministic=True benchmark=False tf32=False")

def stable_seed(base: int, *parts) -> int:
    h = hashlib.sha256("::".join(map(str, parts)).encode("utf-8")).hexdigest()
    return (base + int(h[:8], 16)) % (2**31 - 1)



def seeded_generator(seed: int, device):
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    return g