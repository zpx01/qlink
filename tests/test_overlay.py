"""The overlay (qgpu/overlay.py): circuit dims that do not matter on text share Qwen's dims."""

import numpy as np
from chipformer.q35.emit import allocate

from qgpu import overlay
from qgpu.circuit.kernel import Kernel

CFG = {"stages": 4, "mul_stages": {2}, "logic_stages": {1}, "atom_stages": {3}}


def test_overlay_is_exact_with_qwen_values_on_text():
    """Planned on a corpus of text around every kernel's launches; on a held-out session, random
    Qwen-like values in the overlay dims at text positions change no other dim anywhere, every
    switch-off absorbs |q| far beyond Qwen's embeddings, and the circuit needs fewer slots."""
    c = Kernel(lanes=4, **CFG)
    keep = set(c.keep)
    before = allocate(c, None, keep, c.target.intermediate).n_slots
    p = overlay.build(c, keep, CFG, overlay.corpus(c, CFG, docs=2))
    assert p.q_bound >= overlay.QMAX
    after = allocate(c, None, keep, c.target.intermediate, overlay=p.dims, overlay_cap=1024)
    assert after.n_slots < before * 0.7
    (doc,) = overlay.corpus(c, CFG, seed=11, docs=1)
    overlay.check_noisy(c, p, doc, np.random.default_rng(0))
