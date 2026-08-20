
import os
import json
import math
import argparse


def load_lines(path):
    out = []
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        out.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    return out



def convert(obj):
    if isinstance(obj, dict):
        return {k: convert(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert(v) for v in obj]
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


def _cleanup(*objs):
    for o in objs:
        try:
            del o
        except Exception:
            pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()



def done_indices(path: str):
    """Indices already successfully written (parse-robust against a torn last line)."""
    done = set()
    if not os.path.exists(path):
        return done
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue  # skip a partially-written trailing line
            if rec.get("error") is None and "idx" in rec:
                done.add(rec["idx"])
    return done



def set_seed(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)



def load_prompts(path: str, n: int):
    items = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            items.append({"prompt": obj["prompt"], "natural_text": obj["natural_text"]})
            if len(items) >= n:
                break
    return items



def akey(a):
    """Canonical 2-decimal string for filenames/keys so 0.1 and 0.10 never diverge."""
    return f"{a:.2f}"



def write_rec(fout, rec):
    fout.write(json.dumps(convert(rec), ensure_ascii=False) + "\n")
    fout.flush(); os.fsync(fout.fileno())




def ntokens(tok, text):
    return len(tok.encode(text, add_special_tokens=False)) if text else 0


def to_jsonable(x):
    if isinstance(x, dict):
        return {k: to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_jsonable(v) for v in x]
    if isinstance(x, np.generic):
        x = x.item()
    elif isinstance(x, torch.Tensor):
        x = x.item() if x.numel() == 1 else x.tolist()
    if isinstance(x, float):
        return x if math.isfinite(x) else None   # NaN/inf -> null (valid JSON)
    if isinstance(x, list):
        return [to_jsonable(v) for v in x]
    return x



def append_line(path, obj):
    with open(path, "a") as f:
        f.write(json.dumps(to_jsonable(obj), ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def load_outcome(path):
    """Return (done_idx, finite_scores)"""
    last = {}
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("idx") is not None:
                    last[r["idx"]] = r

    done, finite = set(), {}
    for i, r in last.items():
        val = None
        if r.get("error") is None and r.get("score") is not None:
            try:
                v = float(r["score"])
                if math.isfinite(v):
                    val = v
            except (TypeError, ValueError):
                pass
        if val is not None:
            finite[i] = val
            done.add(i)
        else:
            err = str(r.get("error") or "")
            if not err.startswith("attack "):   # deterministic failure -> done (a miss)
                done.add(i)
    return done, finite


def strip_prompt(text, prompt):
    if prompt and text.startswith(prompt):
        return text[len(prompt):].lstrip()
    return text



def sent_split(text):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]




def write_summary_csv(summary_path, csv_path):
    seen = {}
    with open(summary_path) as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                seen[(r["method"], r["attack"])] = r
    cols = ["method", "attack", "n_pos", "n_neg", "AUROC"]
    for rule, _ in RULES:
        for lab in THRESHOLD_LABELS:
            cols.append(f"{rule}_{lab}")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for (m, a), r in sorted(seen.items()):
            row = [m, a, r.get("n_pos"), r.get("n_neg"), r.get("AUROC")]
            for rule, _ in RULES:
                met = r.get(rule, {})
                for lab in THRESHOLD_LABELS:
                    row.append(met.get(lab) if isinstance(met, dict) else None)
            w.writerow(row)
    print(f"\nwrote {csv_path}")




def load_generations(method, gen_dir ):
    path = os.path.join(gen_dir, f"{method}.jsonl")
    recs = []
    if not os.path.exists(path):
        print(f"!! missing {path} — run generate.py for {method} first")
        return recs
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("error") is None and r.get("watermarked_text"):
                recs.append(r)
    by_idx = {r["idx"]: r for r in recs}
    return [by_idx[i] for i in sorted(by_idx)]



# def last_by_idx(path):
#     last = {}
#     if not os.path.exists(path):
#         return last
#     with open(path) as f:
#         for line in f:
#             line = line.strip()
#             if not line:
#                 continue
#             try:
#                 r = json.loads(line)
#             except json.JSONDecodeError:
#                 continue
#             if r.get("idx") is not None:
#                 last[r["idx"]] = r
#     return last



def last_by_idx(path):
    return {r["idx"]: r for r in load_lines(path) if r.get("idx") is not None}



def load_cache(path):
    cache = {}
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                cache[f"{r['method']}:{r['role']}:{r['idx']}"] = r.get("scores", {})
    return cache






def load_kv(path, keyfn, valfn):
    d = {}
    for r in load_lines(path):
        try:
            d[keyfn(r)] = valfn(r)
        except Exception:
            pass
    return d


def append(path, obj):
    with open(path, "a") as f:
        f.write(json.dumps(obj) + "\n")
        f.flush(); os.fsync(f.fileno())

