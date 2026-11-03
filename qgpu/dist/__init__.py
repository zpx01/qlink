"""qgpu.dist: several QwenGPUs as one multi-GPU system.

A QwenGPU writes a store to global memory as a record of three tokens, MEM(a) VAL VAL (the
write-back of `st.global`, see circuit/kernel.py), and the same three tokens, fed to a model,
write word `a` of its memory: they are how the host's cudaMemcpy reaches the GPU. So the
interconnect needs no new tokens and no new weights. QLink, the link between QwenGPUs, carries
those records: a store whose address falls in a peer *aperture* leaves the GPU that computed it,
and the fabric feeds the same three tokens (the address remapped) into the peer's context at its
next record boundary, where the peer's own layers write them into its memory. Kernels on the
peer see them as stores from another GPU, like NVLink P2P stores into peer memory.

On top of that, qccl (qccl.py) implements NCCL's collectives with NCCL's protocol: FIFO slots
in the receiver's memory, a tail flag the sender posts after its data and a head flag the
receiver posts when a slot is free again, all written by CUDA kernels the QwenGPUs run.

Modules: device.py (a QwenGPU: device memory, streams, apertures, record boundaries),
backends.py (reference interpreter, exact circuit, merged Qwen checkpoint), transport.py (the
fabric between processes or threads), qccl.py (communicators and collectives), train.py
(data-parallel training on QwenGPUs).
"""
