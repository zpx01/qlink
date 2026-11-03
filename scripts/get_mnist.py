"""Download MNIST (ylecun/mnist on the Hugging Face hub) to data/mnist28.npz (uint8 images).

  uv run --with pyarrow --with pillow python scripts/get_mnist.py
"""

import io
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import snapshot_download
from PIL import Image


def main():
    root = Path(snapshot_download("ylecun/mnist", repo_type="dataset"))
    out = {}
    for split, key in (("train", "tr"), ("test", "te")):
        t = pq.read_table(next(root.glob(f"mnist/{split}-*.parquet"))).to_pydict()
        out["X" + key] = np.stack([np.array(Image.open(io.BytesIO(im["bytes"]))) for im in t["image"]])
        out["y" + key] = np.array(t["label"])
    Path("data").mkdir(exist_ok=True)
    np.savez_compressed("data/mnist28.npz", **out)
    print({k: v.shape for k, v in out.items()})


if __name__ == "__main__":
    main()
