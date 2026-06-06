"""Tiny content-addressed result cache for experiment scripts.

Keys are derived from (1) a JSON-serializable description of the cell
(label, config, lr, …) and (2) the byte contents of source files the
cell depends on. If anything in either changes, the hash changes and the
cell is recomputed.

Usage:

    from linattn.cache import cached

    key = {"label": "DeltaNet", "cfg": dataclasses.asdict(cfg), "lr": 3e-3}
    sources = [
        "linattn/models/deltanet.py",
        "linattn/models/backbone.py",
        "linattn/models/factory.py",
        "linattn/models/ffn.py",
        "linattn/train.py",
        "linattn/data.py",
        "linattn/utils.py",
    ]
    hit, save = cached(key, sources, rerun=args.rerun)
    if hit is not None:
        result = hit
        print(f"  [cached]")
    else:
        result = ...   # compute
        save(result)
        print(f"  [fresh]")

Cache lives in `.experiment_cache/` at the repo root. Delete the directory
to clear everything.
"""

import hashlib
import json
import pickle
from pathlib import Path

CACHE_DIR = Path(__file__).parent / ".experiment_cache"


def _hash_key(key_obj, source_paths) -> str:
    h = hashlib.sha256()
    h.update(json.dumps(key_obj, sort_keys=True, default=str).encode())
    for p in source_paths:
        h.update(b"\x00" + str(p).encode() + b"\x00")
        h.update(Path(p).read_bytes())
    return h.hexdigest()[:16]


def cached(key_obj, source_paths, rerun: bool = False):
    """Returns (cached_value_or_None, save_fn).

    If a cache entry exists and `rerun` is False, returns (value, None).
    Otherwise returns (None, save_fn) — call save_fn(result) to persist.
    """
    CACHE_DIR.mkdir(exist_ok=True)
    digest = _hash_key(key_obj, source_paths)
    path = CACHE_DIR / f"{digest}.pkl"

    if path.exists() and not rerun:
        return pickle.loads(path.read_bytes()), None

    def save(value):
        path.write_bytes(pickle.dumps(value))

    return None, save
