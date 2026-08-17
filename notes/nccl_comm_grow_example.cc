/*************************************************************************
 * Minimal ncclCommGrow example (NCCL 2.31+)
 *
 * Single process, multiple GPUs:
 *   1) GPU 0 creates a 1-rank communicator
 *   2) Grow to all GPUs via a NEW communicator
 *   3) Run AllReduce + Send/Recv on the new comm
 *
 * Need >= 2 GPUs.
 *
 *   nvcc -o grow_example notes/nccl_comm_grow_example.cc -lnccl
 *   ./grow_example
 *************************************************************************/

#include "cuda_runtime.h"
#include "nccl.h"

#include <stdio.h>
#include <stdlib.h>

#define NCCLCHECK(cmd)                                                         \
  do {                                                                         \
    ncclResult_t res = cmd;                                                    \
    if (res != ncclSuccess) {                                                  \
      fprintf(stderr, "NCCL error %s:%d: %s\n", __FILE__, __LINE__,            \
              ncclGetErrorString(res));                                        \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

#define CUDACHECK(cmd)                                                         \
  do {                                                                         \
    cudaError_t err = cmd;                                                     \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,            \
              cudaGetErrorString(err));                                        \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

int main() {
  int nGpus = 0;
  CUDACHECK(cudaGetDeviceCount(&nGpus));
  if (nGpus < 2) {
    fprintf(stderr, "Need at least 2 GPUs, found %d\n", nGpus);
    return EXIT_FAILURE;
  }
  printf("GPUs: %d. Grow communicator 1 -> %d\n", nGpus, nGpus);

  // -------------------------------------------------------------------------
  // 1) Old world: only rank 0 / GPU 0
  // -------------------------------------------------------------------------
  ncclUniqueId initId;
  ncclComm_t oldComm = NULL;
  NCCLCHECK(ncclGetUniqueId(&initId));
  CUDACHECK(cudaSetDevice(0));
  NCCLCHECK(ncclCommInitRank(&oldComm, /*nranks=*/1, initId, /*rank=*/0));
  printf("old comm: rank 0 / 1 on GPU 0\n");

  // -------------------------------------------------------------------------
  // 2) Coordinator (rank 0) makes a grow id, then everyone builds a NEW comm
  //
  // Existing ranks: ncclCommGrow(old, nGpus, NULL, -1, &newComm, NULL)
  // New ranks:      ncclCommGrow(NULL, nGpus, &growId, newRank, &newComm, NULL)
  // -------------------------------------------------------------------------
  ncclUniqueId growId;
  NCCLCHECK(ncclCommGetUniqueId(oldComm, &growId));

  ncclComm_t* newComms = (ncclComm_t*)malloc(nGpus * sizeof(ncclComm_t));
  cudaStream_t* streams = (cudaStream_t*)malloc(nGpus * sizeof(cudaStream_t));
  float** buf = (float**)malloc(nGpus * sizeof(float*));

  NCCLCHECK(ncclGroupStart());
  // existing rank 0 keeps rank 0 in the new comm
  CUDACHECK(cudaSetDevice(0));
  NCCLCHECK(ncclCommGrow(oldComm, nGpus, /*uniqueId=*/NULL, /*rank=*/-1, &newComms[0], NULL));
  // new ranks 1..nGpus-1 join
  for (int r = 1; r < nGpus; r++) {
    CUDACHECK(cudaSetDevice(r));
    NCCLCHECK(ncclCommGrow(/*comm=*/NULL, nGpus, &growId, r, &newComms[r], NULL));
  }
  NCCLCHECK(ncclGroupEnd());

  // parent comm is done; later send/recv/allreduce use newComms only
  NCCLCHECK(ncclCommDestroy(oldComm));
  printf("new comm ready: ranks 0..%d\n", nGpus - 1);

  for (int r = 0; r < nGpus; r++) {
    int rank = -1, nranks = -1;
    NCCLCHECK(ncclCommUserRank(newComms[r], &rank));
    NCCLCHECK(ncclCommCount(newComms[r], &nranks));
    printf("  GPU %d -> rank %d / %d\n", r, rank, nranks);

    CUDACHECK(cudaSetDevice(r));
    CUDACHECK(cudaStreamCreate(&streams[r]));
    CUDACHECK(cudaMalloc(&buf[r], sizeof(float)));
  }

  // -------------------------------------------------------------------------
  // 3) Collective on the grown comm: each rank contributes 1, sum == nGpus
  // -------------------------------------------------------------------------
  for (int r = 0; r < nGpus; r++) {
    float one = 1.0f;
    CUDACHECK(cudaSetDevice(r));
    CUDACHECK(cudaMemcpy(buf[r], &one, sizeof(float), cudaMemcpyHostToDevice));
  }
  NCCLCHECK(ncclGroupStart());
  for (int r = 0; r < nGpus; r++) {
    NCCLCHECK(ncclAllReduce(buf[r], buf[r], 1, ncclFloat, ncclSum, newComms[r], streams[r]));
  }
  NCCLCHECK(ncclGroupEnd());
  for (int r = 0; r < nGpus; r++) {
    float got = 0;
    CUDACHECK(cudaSetDevice(r));
    CUDACHECK(cudaStreamSynchronize(streams[r]));
    CUDACHECK(cudaMemcpy(&got, buf[r], sizeof(float), cudaMemcpyDeviceToHost));
    printf("AllReduce rank %d got %.1f (expect %d)\n", r, got, nGpus);
  }

  // -------------------------------------------------------------------------
  // 4) P2P on the same grown comm: rank 0 -> last rank
  // -------------------------------------------------------------------------
  float sendVal = 42.0f, recvVal = -1.0f;
  CUDACHECK(cudaSetDevice(0));
  CUDACHECK(cudaMemcpy(buf[0], &sendVal, sizeof(float), cudaMemcpyHostToDevice));
  CUDACHECK(cudaSetDevice(nGpus - 1));
  CUDACHECK(cudaMemset(buf[nGpus - 1], 0, sizeof(float)));

  NCCLCHECK(ncclGroupStart());
  NCCLCHECK(ncclSend(buf[0], 1, ncclFloat, nGpus - 1, newComms[0], streams[0]));
  NCCLCHECK(ncclRecv(buf[nGpus - 1], 1, ncclFloat, 0, newComms[nGpus - 1], streams[nGpus - 1]));
  NCCLCHECK(ncclGroupEnd());
  CUDACHECK(cudaSetDevice(0));
  CUDACHECK(cudaStreamSynchronize(streams[0]));
  CUDACHECK(cudaSetDevice(nGpus - 1));
  CUDACHECK(cudaStreamSynchronize(streams[nGpus - 1]));
  CUDACHECK(cudaMemcpy(&recvVal, buf[nGpus - 1], sizeof(float), cudaMemcpyDeviceToHost));
  printf("Send/Recv 0 -> %d got %.1f (expect 42)\n", nGpus - 1, recvVal);

  // -------------------------------------------------------------------------
  // 5) Cleanup
  // -------------------------------------------------------------------------
  for (int r = 0; r < nGpus; r++) {
    CUDACHECK(cudaSetDevice(r));
    CUDACHECK(cudaStreamSynchronize(streams[r]));
    NCCLCHECK(ncclCommDestroy(newComms[r]));
    CUDACHECK(cudaFree(buf[r]));
    CUDACHECK(cudaStreamDestroy(streams[r]));
  }
  free(newComms);
  free(streams);
  free(buf);
  printf("Success\n");
  return 0;
}
