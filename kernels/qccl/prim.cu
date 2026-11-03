#include "../qcuda.h"
// qccl_prim: one step of a collective over a ring channel, as NCCL's primitives run it
// (recv / reduce / copy / send). One thread block, one thread per element of a chunk.
//
// Flow control is NCCL's: a FIFO of slots at the receiver, a `tail` word the sender bumps
// after its data (the receiver waits for it), and a `head` word the receiver bumps when it
// has consumed a slot (the sender waits for a free slot). rtail/rhead are local words a peer
// writes; stail/shead are pointers into the peer's memory (a QLink aperture).
enum { RECV = 1, LOCAL = 2, COPY = 4, SEND = 8, SPLIT = 16 };
// SPLIT (a grouped send + recv, ncclGroupStart/End): y gets what arrived, the peer gets x

extern "C" __global__ void qccl_prim(const int *x, int *y, const volatile int *rslot,
                                     int *sslot, const volatile int *rtail, int *stail,
                                     const volatile int *rhead, int *shead, int rwait,
                                     int swait, int spost, int rpost, int mode) {
  int i = threadIdx.x;
  if (mode & RECV)
    while (*rtail < rwait) {  // the previous rank's chunk is in our slot
    }
  if ((mode & SEND) && swait > 0)
    while (*rhead < swait) {  // the next rank has a free slot for ours
    }
  int r = 0, v = 0;
  if (mode & RECV) r = rslot[i];
  if (mode & LOCAL) v = x[i];
  if (!(mode & SPLIT)) v += r;
  if (mode & COPY) y[i] = (mode & SPLIT) ? r : v;
  if (mode & SEND) sslot[i] = v;
  __threadfence_system();  // our data is visible to the peer before the flag
  __syncthreads();
  if (i != 0) return;
  if (mode & SEND) *stail = spost;  // -> next rank: chunk spost - 1 has arrived
  if (mode & RECV) *shead = rpost;  // -> previous rank: the slot is free again
}
