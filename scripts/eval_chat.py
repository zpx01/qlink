"""How often Qwen gets chat prompts right through the cuda tool: each prompt, several samples.

A prompt that needs arithmetic is solved when the expected numbers came back from a launch
the reference verified (they were computed on the GPU) and appear in Qwen's answer. A prompt
that needs none is solved when Qwen makes no call. Sampling makes single runs noisy, so each
prompt runs with several seeds, each in a fresh chat.

Run: uv run python scripts/eval_chat.py MERGED [--samples 3] [--out results.jsonl]
"""

import argparse
import json
import re
import sys
import time

from qgpu.bridge import Session

PROMPTS = [  # (message, the numbers the answer must contain; None: no arithmetic, no call)
    ("calculate [2,3,4,5] * [2,3,4,6] - 4", [0, 5, 12, 26]),
    ("I bought 4 apples at 30 each and 3 melons at 55 each. How much did I spend?", [285]),
    ("What is the sum of 12, 7, 30 and 5?", [54]),
    (("My daily steps were 4200, 8100, 6500, 9900 and 7300. What's my running total each "
      "day?"), [4200, 12300, 18800, 28700, 36000]),
    ("Multiply each of 3, 8 and 5 by 6.", [18, 48, 30]),
    ("A box holds 12 eggs. How many eggs are in 3, 5 and 8 boxes?", [36, 60, 96]),
    ("who are u?", None),
    ("Write a haiku about the sea.", None),
]


def numbers(text: str) -> set[int]:
    return {int(n.replace(",", "")) for n in re.findall(r"-?\d[\d,]*(?!\.\d)", text)
            if n.replace(",", "").lstrip("-").isdigit()}


def gpu_numbers(v) -> set[int]:
    if isinstance(v, int):
        return {v}
    if isinstance(v, list):
        return set().union(*(gpu_numbers(x) for x in v)) if v else set()
    if isinstance(v, dict):
        return set().union(*(gpu_numbers(x) for x in v.values())) if v else set()
    return set()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("merged")
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--device")
    ap.add_argument("--out")
    ap.add_argument("--mode", choices=("calc", "cuda"), default="calc")
    ap.add_argument("--only", type=int, help="the first N prompts (a short run)")
    a = ap.parse_args()
    s = Session(a.merged, a.device, check=True, mode=a.mode)
    out = open(a.out, "w") if a.out else None  # noqa: SIM115 - written as samples finish
    total, solved = 0, 0
    print(f"{'prompt':<60} solved", flush=True)
    for msg, want in PROMPTS[:a.only]:
        wins = 0
        for k in range(a.samples):
            s.reset()
            s.rng.manual_seed(1000 + k)
            calls, launches, gpu, answer = 0, 0, set(), ""
            t0 = time.time()
            for e in s.send(msg):
                if e.kind == "call":
                    calls += 1
                elif e.kind == "result" and e.data["verified"]:
                    launches += 1
                    gpu |= gpu_numbers(e.data["outputs"])
                elif e.kind in ("value", "refused"):
                    answer = ""  # the answer is what Qwen writes after the last response
                elif e.kind == "text":
                    answer += e.data["text"]
            ok = calls == 0 if want is None else \
                set(want) <= gpu and set(want) <= numbers(answer)
            wins += ok
            if out:
                out.write(json.dumps({"prompt": msg, "sample": k, "ok": ok, "calls": calls,
                                      "launches": launches, "text": answer,
                                      "seconds": round(time.time() - t0)}) + "\n")
                out.flush()
        total += a.samples
        solved += wins
        print(f"{msg[:58]:<60} {wins}/{a.samples}", flush=True)
    print(f"{'all':<60} {solved}/{total}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
