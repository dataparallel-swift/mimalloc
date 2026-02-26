#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <pthread.h>

#include <cuda.h>
#include <mimalloc.h>

#define MI_CUDA_TEST_THREADS 8
#define MI_CUDA_TEST_ITERS   20000

#if MI_MALLOC_OVERRIDE
#define custom_malloc(s)      malloc(s)
#define custom_realloc(p,s)   realloc(p,s)
#define custom_free(p)        free(p)
#else
#define custom_malloc(s)      mi_malloc(s)
#define custom_realloc(p,s)   mi_realloc(p,s)
#define custom_free(p)        mi_free(p)
#endif

static int check_device_accessible(void* p) {
  CUdeviceptr dptr = 0;
  CUresult status = cuMemHostGetDevicePointer(&dptr, p, 0);
  if (status != CUDA_SUCCESS) {
    fprintf(stderr, "cuMemHostGetDevicePointer failed (status=%d, host=%p)\n", (int)status, p);
    return 10;
  }
  if ((uintptr_t)dptr != (uintptr_t)p) {
    fprintf(stderr, "device pointer mismatch (host=%p, device=%p)\n", p, (void*)(uintptr_t)dptr);
    return 11;
  }
  return 0;
}

static void* cuda_interpose_worker(void* arg) {
  uintptr_t tid = (uintptr_t)arg;
  CUdevice device = 0;
  CUcontext context = NULL;
  CUresult status = cuDeviceGet(&device, 0);
  if (status != CUDA_SUCCESS) return (void*)20;
  status = cuDevicePrimaryCtxRetain(&context, device);
  if (status != CUDA_SUCCESS || context == NULL) return (void*)21;
  status = cuCtxPushCurrent(context);
  if (status != CUDA_SUCCESS) return (void*)22;

  int rc = 0;
  for (size_t i = 0; i < MI_CUDA_TEST_ITERS; i++) {
    size_t sz = 16 + ((i + tid) % 4096);
    void* p = custom_malloc(sz);
    if (p == NULL) { rc = 1; break; }
    rc = check_device_accessible(p);
    if (rc != 0) { custom_free(p); break; }
    memset(p, (int)(i & 0x7F), (sz > 64 ? 64 : sz));

    if ((i % 5) == 0) {
      void* q = custom_realloc(p, sz + 33);
      if (q == NULL) {
        custom_free(p);
        rc = 2;
        break;
      }
      p = q;
      rc = check_device_accessible(p);
      if (rc != 0) { custom_free(p); break; }
    }
    custom_free(p);

    if ((i % 11) == 0) {
      void* m = mi_malloc(128 + (i % 256));
      if (m == NULL) { rc = 3; break; }
      rc = check_device_accessible(m);
      if (rc != 0) { mi_free(m); break; }
      mi_free(m);
    }

    if ((i % 128) == 0) {
      (void)cuInit(0);
    }
  }
  CUcontext popped = NULL;
  status = cuCtxPopCurrent(&popped);
  if (rc == 0 && status != CUDA_SUCCESS) rc = 23;
  status = cuDevicePrimaryCtxRelease(device);
  if (rc == 0 && status != CUDA_SUCCESS) rc = 24;
  return (rc == 0 ? NULL : (void*)(uintptr_t)rc);
}

int main(void) {
  mi_version();
  printf("Using %d threads with %d iteration%s\n", MI_CUDA_TEST_THREADS, MI_CUDA_TEST_ITERS, MI_CUDA_TEST_ITERS > 1 ? "s" : "");

  CUresult status = cuInit(0);
  if (status != CUDA_SUCCESS) {
    printf("skip: CUDA unavailable (cuInit=%d)\n", (int)status);
    return 0;
  }

  int device_count = 0;
  status = cuDeviceGetCount(&device_count);
  if (status != CUDA_SUCCESS || device_count <= 0) {
    printf("skip: no CUDA device available (status=%d, count=%d)\n", (int)status, device_count);
    return 0;
  }

  pthread_t threads[MI_CUDA_TEST_THREADS];
  for (uintptr_t i = 0; i < MI_CUDA_TEST_THREADS; i++) {
    if (pthread_create(&threads[i], NULL, &cuda_interpose_worker, (void*)i) != 0) {
      fprintf(stderr, "pthread_create failed\n");
      return 1;
    }
  }

  int result = 0;
  for (size_t i = 0; i < MI_CUDA_TEST_THREADS; i++) {
    void* ret = NULL;
    pthread_join(threads[i], &ret);
    if (ret != NULL) result = 2;
  }

  #ifndef NDEBUG
  // mi_collect(true);
  mi_debug_show_arenas();
  #endif
  mi_collect(true);
  mi_stats_print(NULL);

  return result;
}
