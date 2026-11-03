"""qccl on real NVIDIA GPUs: the same CUDA kernel (kernels/qccl/prim.cu, the same PTX), the
same host schedule (qccl.Comm), real NVLink P2P stores into peer memory.

CudaGPU has the interface qccl.Comm uses (malloc, map, memcpy_h2d, launch, read), so a
communicator over real GPUs is built exactly like one over QwenGPUs. Device memory is a
128-word int32 buffer per GPU, so word addresses mean the same thing on both machines. A kernel
argument Remote(a), an address in an aperture, becomes the peer's own pointer (peer access is
enabled between every pair of GPUs), so the kernel's stores cross NVLink, as NCCL's P2P
transport does; everything else is a local pointer. All launches are asynchronous on one stream
per GPU, and no host code orders one GPU after another: the kernels do, by spinning on the
flags their peers write.
"""

import ctypes
from pathlib import Path

import torch

from qgpu.dist.device import Remote
from qgpu.lower import DevPtr

ROOT = Path(__file__).resolve().parent.parent.parent
PTX = {"qccl_prim": ROOT / "kernels" / "qccl" / "prim.ptx",
       **{k: ROOT / "kernels" / "train" / "mlp.ptx" for k in ("mlp_forward", "mlp_backward", "sgd")}}


def _check(res):
    err = res[0] if isinstance(res, tuple) else res
    if int(err) != 0:
        raise RuntimeError(f"CUDA error {err}")
    return res[1:] if isinstance(res, tuple) and len(res) > 1 else None


class CudaGPU:
    registry: dict = {}  # rank -> CudaGPU (one process drives every GPU of the node)

    def __init__(self, rank: int, index: int, words: int = 128):
        from cuda.bindings import driver as cu

        self.cu, self.rank, self.index = cu, rank, index
        self.device = torch.device(f"cuda:{index}")
        self.mem = torch.zeros(words, dtype=torch.int32, device=self.device)
        self.base = self.mem.data_ptr()
        self.stream = torch.cuda.Stream(self.device)
        self.heap, self.apertures, self.funcs, self.launches = 0, [], {}, 0
        CudaGPU.registry[rank] = self

    @staticmethod
    def enable_peer_access(indices) -> None:
        from cuda.bindings import runtime as rt

        for i in indices:
            torch.cuda.set_device(i)
            for j in indices:
                if i != j:
                    err = rt.cudaDeviceEnablePeerAccess(j, 0)[0]
                    if int(err) not in (0, int(rt.cudaError_t.cudaErrorPeerAccessAlreadyEnabled)):
                        raise RuntimeError(f"peer access {i} -> {j}: {err}")

    def malloc(self, n: int, name: str = "") -> int:
        a = self.heap
        self.heap += n
        assert self.heap <= self.mem.numel()
        return a

    def map(self, lo: int, n: int, routes, local: bool = False) -> None:
        self.apertures.append((lo, n, tuple(routes)))

    def memcpy_h2d(self, addr: int, words) -> None:
        src = torch.tensor([w - (1 << 32) if w >> 31 else w for w in (x & 0xFFFFFFFF for x in words)],
                           dtype=torch.int32).pin_memory()
        with torch.cuda.stream(self.stream):
            self.mem[addr:addr + len(src)].copy_(src, non_blocking=True)

    def ptr(self, a: int) -> int:
        return self.base + 4 * a

    def remote(self, a: int) -> int:
        for lo, n, routes in self.apertures:
            if lo <= a < lo + n:
                (peer, base), = routes  # unicast apertures only
                return CudaGPU.registry[peer].ptr(base + a - lo)
        raise KeyError(f"rank {self.rank}: word {a} is in no aperture")

    def preload(self) -> None:
        """Load every kernel now. Loading a module allocates device memory, which with peer
        access enabled can wait for the peers, and a peer may be spinning on a flag this GPU has
        not written yet: load before anything runs."""
        for name in PTX:
            self._func(name)

    def _func(self, name: str):
        if name not in self.funcs:
            torch.cuda.set_device(self.index)
            torch.cuda.current_stream(self.device).synchronize()  # the primary context is current
            mod = _check(self.cu.cuModuleLoadData(PTX[name].read_bytes() + b"\0"))[0]
            self.funcs[name] = _check(self.cu.cuModuleGetFunction(mod, name.encode()))[0]
        return self.funcs[name]

    def launch(self, k, threads: int, args: list, block: int | None = None,
               name: str | None = None) -> None:
        block = block or threads
        vals, types = [], []
        for a in args:
            if isinstance(a, Remote):
                vals.append(self.remote(a.word))
                types.append(ctypes.c_uint64)
            elif isinstance(a, DevPtr):
                vals.append(self.ptr(a.word))
                types.append(ctypes.c_uint64)
            else:
                vals.append(a)
                types.append(ctypes.c_int32)
        f = self._func(k.name)
        stream = self.cu.CUstream(self.stream.cuda_stream)
        _check(self.cu.cuLaunchKernel(f, threads // block, 1, 1, block, 1, 1, 0, stream,
                                      (tuple(vals), tuple(types)), 0))
        self.launches += 1

    def synchronize(self) -> None:
        self.stream.synchronize()

    def read(self, addr: int, n: int, signed: bool = True) -> list[int]:
        self.synchronize()
        vals = self.mem[addr:addr + n].tolist()
        return vals if signed else [v & 0xFFFFFFFF for v in vals]
