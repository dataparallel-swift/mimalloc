// This file is included in `src/prim/prim.c`

#ifndef _DEFAULT_SOURCE
#define _DEFAULT_SOURCE   // ensure mmap flags and syscall are defined
#endif

#include "mimalloc.h"
#include "mimalloc/internal.h"
#include "mimalloc/atomic.h"
#include "mimalloc/prim.h"

#include <cuda.h>
#include <errno.h>        // ENOMEM, ENODEV, EINVAL

#if !defined(_WIN32)
#include <sys/mman.h>     // mmap
#include <unistd.h>       // sysconf
#include <fcntl.h>        // open, close, read, access
#include <stdlib.h>       // getenv, arc4random_buf

#if defined(__linux__)
#include <sys/sysinfo.h>  // sysinfo
#endif
#endif

//------------------------------------------------------------------------------------
// Use syscalls for some primitives to allow for libraries that override open/read/close etc.
// and do allocation themselves; using syscalls prevents recursion when mimalloc is
// still initializing (issue #713)
// Declare inline to avoid unused function warnings.
//------------------------------------------------------------------------------------

#if defined(MI_HAS_SYSCALL_H) && defined(SYS_open) && defined(SYS_close) && defined(SYS_read) && defined(SYS_access)

static inline int mi_prim_open(const char* fpath, int open_flags) {
  return syscall(SYS_open,fpath,open_flags,0);
}
static inline ssize_t mi_prim_read(int fd, void* buf, size_t bufsize) {
  return syscall(SYS_read,fd,buf,bufsize);
}
static inline int mi_prim_close(int fd) {
  return syscall(SYS_close,fd);
}

#else

static inline int mi_prim_open(const char* fpath, int open_flags) {
  return open(fpath,open_flags);
}
static inline ssize_t mi_prim_read(int fd, void* buf, size_t bufsize) {
  return read(fd,buf,bufsize);
}
static inline int mi_prim_close(int fd) {
  return close(fd);
}

#endif


//---------------------------------------------
// Initialise
//---------------------------------------------

enum mi_cuda_init_e {
  MI_CUDA_INIT_UNINIT = 0,
  MI_CUDA_INIT_INITING = 1,
  MI_CUDA_INIT_READY = 2,
  MI_CUDA_INIT_FAILED = 3
};

enum mi_cuda_fallback_e {
  MI_CUDA_FALLBACK_UNINIT = 0,
  MI_CUDA_FALLBACK_INITING = 1,
  MI_CUDA_FALLBACK_READY = 2,
};

static _Atomic(uintptr_t) mi_cuda_init_state = MI_ATOMIC_VAR_INIT(MI_CUDA_INIT_UNINIT);
mi_decl_hidden void* mi_cuda_context = NULL;

#if defined(MI_MALLOC_OVERRIDE)
#define MI_CUDA_FALLBACK_RESERVE_SIZE (16*MI_MiB)

static _Atomic(uintptr_t) mi_cuda_fallback_state = MI_ATOMIC_VAR_INIT(MI_CUDA_FALLBACK_UNINIT);
static _Atomic(uintptr_t) mi_cuda_fallback_offset = MI_ATOMIC_VAR_INIT(0);

mi_decl_hidden uint8_t* mi_cuda_fallback_base = NULL;
mi_decl_hidden size_t mi_cuda_fallback_size = 0;

// Combined ready-check: true iff CUDA is initialized and we are not currently
// inside a CUDA API call. Written only on slow paths (init, enter/leave); read
// on every mi_malloc hot path via _mi_prim_cuda_ready().
mi_decl_hidden mi_decl_thread bool mi_cuda_thread_ready = false;
mi_decl_hidden mi_decl_thread bool mi_cuda_in_api = false;
#endif

static inline void mi_cuda_call_enter(void) {
  #if defined(MI_MALLOC_OVERRIDE)
  mi_assert_internal(!mi_cuda_in_api); // true means malloc() recursed inside a CUDA API call
  mi_cuda_in_api = true;
  mi_cuda_thread_ready = false;
  #endif
}

static inline void mi_cuda_call_leave(void) {
  #if defined(MI_MALLOC_OVERRIDE)
  mi_assert_internal(mi_cuda_in_api); // leave without matching enter
  mi_cuda_in_api = false;
  mi_cuda_thread_ready = (mi_cuda_context != NULL);
  #endif
}

static int mi_cuda_error(CUresult status, const char* api_name) {
  if (status == CUDA_SUCCESS) return 0;
  _mi_warning_message("CUDA call failed in %s (status: %d)\n", api_name, (int)status);
  switch(status) {
    case CUDA_ERROR_OUT_OF_MEMORY:      return ENOMEM;
    case CUDA_ERROR_NO_DEVICE:          return ENODEV;
    case CUDA_ERROR_NOT_INITIALIZED:    return ENODEV;
    default:                            return EINVAL;
  }
}

static int mi_cuda_cuInit(unsigned int flags) {
  mi_cuda_call_enter();
  CUresult status = cuInit(flags);
  mi_cuda_call_leave();
  return mi_cuda_error(status, "cuInit");
}

static int mi_cuda_cuDeviceGet(CUdevice* device, int ordinal) {
  mi_cuda_call_enter();
  CUresult status = cuDeviceGet(device, ordinal);
  mi_cuda_call_leave();
  return mi_cuda_error(status, "cuDeviceGet");
}

static int mi_cuda_cuDevicePrimaryCtxRetain(CUcontext* pctx, CUdevice dev) {
  mi_cuda_call_enter();
  CUresult status = cuDevicePrimaryCtxRetain(pctx, dev);
  mi_cuda_call_leave();
  return mi_cuda_error(status, "cuDevicePrimaryCtxRetain");
}

static int mi_cuda_cuCtxPushCurrent(CUcontext ctx) {
  mi_cuda_call_enter();
  CUresult status = cuCtxPushCurrent(ctx);
  mi_cuda_call_leave();
  return mi_cuda_error(status, "cuCtxPushCurrent");
}

static int mi_cuda_cuCtxPopCurrent(void) {
  mi_cuda_call_enter();
  CUresult status = cuCtxPopCurrent(NULL);
  mi_cuda_call_leave();
  return mi_cuda_error(status, "cuCtxPopCurrent");
}

static int mi_cuda_cuMemAllocHost(void** pp, size_t bytesize) {
  mi_cuda_call_enter();
  CUresult status = cuMemAllocHost(pp, bytesize);
  mi_cuda_call_leave();
  return mi_cuda_error(status, "cuMemAllocHost");
}

static int mi_cuda_cuMemFreeHost(void* p) {
  mi_cuda_call_enter();
  CUresult status = cuMemFreeHost(p);
  mi_cuda_call_leave();
  return mi_cuda_error(status, "cuMemFreeHost");
}

#if defined(MI_MALLOC_OVERRIDE)
static bool mi_cuda_fallback_ensure(void) {
  if (mi_atomic_load_acquire(&mi_cuda_fallback_state) == MI_CUDA_FALLBACK_READY) {
    return (mi_cuda_fallback_base != NULL);
  }

  uintptr_t expected = MI_CUDA_FALLBACK_UNINIT;
  if (mi_atomic_cas_strong_acq_rel(&mi_cuda_fallback_state, &expected, (uintptr_t)MI_CUDA_FALLBACK_INITING)) {
    const size_t reserve_size = MI_CUDA_FALLBACK_RESERVE_SIZE;
    void* p = mmap(NULL, reserve_size, (PROT_READ | PROT_WRITE), (MAP_PRIVATE | MAP_ANONYMOUS), -1, 0);
    if (p != MAP_FAILED) {
      mi_cuda_fallback_base = (uint8_t*)p;
      mi_cuda_fallback_size = reserve_size;
    }
    mi_atomic_store_release(&mi_cuda_fallback_state, (uintptr_t)MI_CUDA_FALLBACK_READY);
  }
  else {
    while (mi_atomic_load_acquire(&mi_cuda_fallback_state) == MI_CUDA_FALLBACK_INITING) {
      mi_atomic_yield();
    }
  }

  return (mi_cuda_fallback_base != NULL);
}

int _mi_cuda_fallback_alloc_aligned(size_t size, size_t alignment, void** addr) {
  mi_assert_internal(alignment >= 16 && (alignment & (alignment - 1)) == 0); // power of two, at least 16

  if (!mi_cuda_fallback_ensure()) return ENOMEM;

  // Round size up to a multiple of 16. This lets realloc safely copy rsize
  // bytes from a fallback allocation without reading out-of-bounds.
  const size_t rsize = _mi_align_up(size, 16);

  // Reserve alignment + rsize bytes.
  //
  // Invariant: the bump offset is always a multiple of 16. Since alignment >= 16
  // and rsize are both multiples of 16, every reservation keeps the offset
  // 16-aligned, so raw = base + offset is always 16-aligned.
  //
  // Returned ptr = align_up(raw + 16, alignment), which lies in [raw+16, raw+alignment].
  // The 16 bytes before ptr (i.e. [ptr-16, ptr)) are within our reserved region because
  // ptr >= raw + 16, so this is where we store rsize.  The data [ptr, ptr+rsize) ends at
  // most at raw + alignment + rsize = raw + reserve, which is exactly our reservation.
  const size_t reserve = alignment + rsize;
  uintptr_t offset = mi_atomic_add_acq_rel(&mi_cuda_fallback_offset, reserve);
  if (offset > (uintptr_t)mi_cuda_fallback_size || reserve > ((uintptr_t)mi_cuda_fallback_size - offset)) {
    _mi_error_message(ENOMEM, "CUDA fallback arena exhausted (requested: %zu bytes, reserved: %zu bytes)\n", size, mi_cuda_fallback_size);
    return ENOMEM;
  }

  const uintptr_t raw = (uintptr_t)(mi_cuda_fallback_base + offset);
  const uintptr_t ptr = _mi_align_up(raw + 16, alignment);

  // Store the rounded size in the header word immediately before the allocation.
  *(size_t*)(ptr - 16) = rsize;

  // mi_os_stat_counter_increase(cuda_fallback_alloc, 1);
  // mi_os_stat_counter_increase(cuda_fallback_bytes, reserve);
  *addr = (void*)ptr;
  return 0;
}

int _mi_cuda_fallback_alloc(size_t size, void** addr) {
  return _mi_cuda_fallback_alloc_aligned(size, 16, addr);
}

void _mi_cuda_fallback_free(void* addr) {
  MI_UNUSED(addr);
  // mi_os_stat_counter_increase(cuda_fallback_free, 1);
}
#endif

int _mi_prim_cuda_init(void) {
  // We only need to initialise CUDA once, and every thread will use the same
  // context. All host memory that we allocate (via cuMemAllocHost) will
  // automatically be immediately accessible to all contexts on all devices
  // which support unified addressing, and the device pointer that may be used
  // to access this host memory from those contexts is always equal to the
  // returned host pointer. Thus it is safe to initialise only the first device
  // and use the default context.
  uintptr_t state = mi_atomic_load_acquire(&mi_cuda_init_state);
  if (state == MI_CUDA_INIT_FAILED) {
    return ENODEV;
  }

  bool do_init = false;
  if (state == MI_CUDA_INIT_UNINIT) {
    uintptr_t expected = MI_CUDA_INIT_UNINIT;
    do_init = mi_atomic_cas_strong_acq_rel(&mi_cuda_init_state, &expected, (uintptr_t)MI_CUDA_INIT_INITING);
  }

  if (do_init) {
    CUdevice device = 0;
    CUcontext context = NULL;
    int err = mi_cuda_cuInit(0);
    if (err == 0) err = mi_cuda_cuDeviceGet(&device, 0);
    if (err == 0) err = mi_cuda_cuDevicePrimaryCtxRetain(&context, device);
    if (err != 0 || context == NULL) {
      mi_atomic_store_release(&mi_cuda_init_state, (uintptr_t)MI_CUDA_INIT_FAILED);
      return (err != 0 ? err : ENODEV);
    }
    mi_cuda_context = context;
    mi_atomic_store_release(&mi_cuda_init_state, (uintptr_t)MI_CUDA_INIT_READY);
  }
  else {
    do {
      state = mi_atomic_load_acquire(&mi_cuda_init_state);
      if (state == MI_CUDA_INIT_FAILED) return ENODEV;
      if (state == MI_CUDA_INIT_READY) break;
      mi_atomic_yield();
    } while(true);
  }

  mi_assert_internal(mi_cuda_context != NULL);
  #if defined(MI_MALLOC_OVERRIDE)
  mi_cuda_thread_ready = true;
  #endif
  return 0;
}

#if !defined(_WIN32)
// try to detect the physical memory dynamically (if possible)
static void unix_detect_physical_memory( size_t page_size, size_t* physical_memory_in_kib ) {
  #if defined(CTL_HW) && (defined(HW_PHYSMEM64) || defined(HW_MEMSIZE))  // freeBSD, macOS
    MI_UNUSED(page_size);
    int64_t physical_memory = 0;
    size_t length = sizeof(int64_t);
    #if defined(HW_PHYSMEM64)
    int mib[2] = { CTL_HW, HW_PHYSMEM64 };
    #else
    int mib[2] = { CTL_HW, HW_MEMSIZE };
    #endif
    const int err = sysctl(mib, 2, &physical_memory, &length, NULL, 0);
    if (err==0 && physical_memory > 0) {
      const int64_t phys_in_kib = physical_memory / MI_KiB;
      if (phys_in_kib > 0 && (uint64_t)phys_in_kib <= SIZE_MAX) {
        *physical_memory_in_kib = (size_t)phys_in_kib;
      }
    }
  #elif defined(__linux__)
    MI_UNUSED(page_size);
    struct sysinfo info; _mi_memzero_var(info);
    const int err = sysinfo(&info);
    if (err==0 && info.totalram > 0 && info.totalram <= SIZE_MAX) {
      *physical_memory_in_kib = (size_t)info.totalram / MI_KiB;
    }
  #elif defined(_SC_PHYS_PAGES)  // do not use by default as it might cause allocation (by using `fopen` to parse /proc/meminfo) (issue #1100)
    const long pphys = sysconf(_SC_PHYS_PAGES);
    const size_t psize_in_kib = page_size / MI_KiB;
    if (psize_in_kib > 0 && pphys > 0 && (unsigned long)pphys <= SIZE_MAX && (size_t)pphys <= (SIZE_MAX/psize_in_kib)) {
      *physical_memory_in_kib = (size_t)pphys * psize_in_kib;
    }
  #endif
}
#endif

void _mi_prim_mem_init( mi_os_mem_config_t* config ) {

#if defined(_WIN32)
  // get the page size
  SYSTEM_INFO si; _mi_memzero_var(si);
  GetSystemInfo(&si);
  if (si.dwPageSize > 0) { config->page_size = si.dwPageSize; }
  if (si.dwAllocationGranularity > 0) {
    config->alloc_granularity = si.dwAllocationGranularity;
    // win_allocation_granularity = si.dwAllocationGranularity;
  }
#else
  long psize = sysconf(_SC_PAGESIZE);
  if (psize > 0 && (unsigned long)psize < SIZE_MAX) {
    config->page_size = (size_t)psize;
    config->alloc_granularity = (size_t)psize;
    unix_detect_physical_memory(config->page_size, &config->physical_memory_in_kib);
  }
#endif
  config->has_overcommit = false;
  config->has_partial_free = false;
  config->has_virtual_reserve = false;
}


//---------------------------------------------
// Free
//---------------------------------------------

int _mi_prim_free(void* addr, size_t size) {
  if (size==0) return 0;
#if defined(MI_MALLOC_OVERRIDE)
  mi_assert_internal(!_mi_cuda_fallback_contains(addr));
#endif
  int err = mi_cuda_cuCtxPushCurrent(mi_cuda_context);
  if (err == 0) err = mi_cuda_cuMemFreeHost(addr);
  if (err == 0) err = mi_cuda_cuCtxPopCurrent();
  return err;
}


//---------------------------------------------
// Allocation
//---------------------------------------------

// Allocate OS memory. Return NULL on error.
// The `try_alignment` is just a hint and the returned pointer does not have to be aligned.
int _mi_prim_alloc(void* hint_addr, size_t size, size_t try_alignment, bool commit, bool allow_large, bool* is_large, bool* is_zero, void** addr) {
  MI_UNUSED(hint_addr);
  MI_UNUSED(allow_large);

  // cuMemAllocHost always returns committed memory
  MI_UNUSED(commit);

  // cuMemAllocHost only guarantees alignment suitable for any C variable (~16
  // bytes per spec). In practice on x86-64 and ARM64, CUDA returns page-aligned
  // (≥4 KiB) allocations that empirically meet MI_ARENA_SLICE_ALIGN (64 KiB).
  // If not, the caller (mi_os_prim_alloc_aligned in os.c) detects misalignment
  // and handles it by over-allocating size+alignment and aligning within;
  // has_partial_free=false ensures it keeps the original cuMemAllocHost pointer
  // for cuMemFreeHost.
  MI_UNUSED(try_alignment);

  mi_assert_internal(size > 0 && (size % _mi_os_page_size()) == 0);

  *is_large = false;
  *is_zero = false;
  *addr = NULL;

#if defined(MI_MALLOC_OVERRIDE)
  // In override mode, we should only reach this point if we are ready to call
  // the actual CUDA allocator. Any reentrant allocations should have been
  // diverted to the fallback allocator already, so that we are sure that all of
  // the internal mimalloc state is CUDA accessible.
  mi_assert_internal(mi_cuda_context != NULL);
  mi_assert_internal(!mi_cuda_in_api);
#else
  // Lazily initialise the CUDA context. Any reentrant allocations will be
  // handled by the default (non-overridden) malloc() implementation.
  if mi_unlikely(mi_cuda_context == NULL) {
    const int err = _mi_prim_cuda_init();
    if (err != 0) return err;
  }
#endif

  // Ensure that the calling thread has the correct context set before
  // attempting to allocate. This is a little expensive (manipulating the TLS)
  // but allocations should be very rare, so this is acceptable (one
  // cuMemAllocHost call amortized over potentially millions of user allocations
  // from each slab). This also supports the use case where the user's code is
  // manipulating the CUDA context stack, which we can't know about, so this
  // push/pop approach is safer in a (potentially) mixed environment.
  //
  // We could potentially reduce the cost of this by setting the context once
  // per thread in mi_thread_init(), and first checking if the current context
  // is the one we expect (cuCtxGetCurrent, presumably cheap with only a TLS
  // read), and skip push/pop if it is.
  int err = mi_cuda_cuCtxPushCurrent(mi_cuda_context);
  if (err == 0) err = mi_cuda_cuMemAllocHost(addr, size);
  if (err == 0) err = mi_cuda_cuCtxPopCurrent();
  return err;
}


//---------------------------------------------
// Commit/Reset/Protect
//---------------------------------------------

int _mi_prim_commit(void* addr, size_t size, bool* is_zero) {
  MI_UNUSED(addr);
  MI_UNUSED(size);
  *is_zero = false;
  return 0;
}

int _mi_prim_decommit(void* addr, size_t size, bool* needs_recommit) {
  MI_UNUSED(addr);
  MI_UNUSED(size);
  *needs_recommit = false;
  return 0;
}

int _mi_prim_reset(void* addr, size_t size) {
  MI_UNUSED(addr);
  MI_UNUSED(size);
  return 0;
}

int _mi_prim_reuse(void* addr, size_t size) {
  MI_UNUSED(addr);
  MI_UNUSED(size);
  return 0;
}

int _mi_prim_protect(void* addr, size_t size, bool protect) {
  MI_UNUSED(addr);
  MI_UNUSED(size);
  MI_UNUSED(protect);
  return 0;
}


//---------------------------------------------
// Huge pages and NUMA nodes
//---------------------------------------------

int _mi_prim_alloc_huge_os_pages(void* hint_addr, size_t size, int numa_node, bool* is_zero, void** addr) {
  MI_UNUSED(numa_node);

  bool commit = true;
  bool is_large = false;
  bool allow_large = false;
  return _mi_prim_alloc(hint_addr, size, MI_ARENA_SLICE_ALIGN, commit, allow_large, &is_large, is_zero, addr);
}

size_t _mi_prim_numa_node(void) {
  return 0;
}

size_t _mi_prim_numa_node_count(void) {
  return 1;
}


//----------------------------------------------------------------
// Clock
//----------------------------------------------------------------

#if defined(_WIN32)

static mi_msecs_t mi_to_msecs(LARGE_INTEGER t) {
  static LARGE_INTEGER mfreq; // = 0
  if (mfreq.QuadPart == 0LL) {
    LARGE_INTEGER f;
    QueryPerformanceFrequency(&f);
    mfreq.QuadPart = f.QuadPart/1000LL;
    if (mfreq.QuadPart == 0) mfreq.QuadPart = 1;
  }
  return (mi_msecs_t)(t.QuadPart / mfreq.QuadPart);
}

mi_msecs_t _mi_prim_clock_now(void) {
  LARGE_INTEGER t;
  QueryPerformanceCounter(&t);
  return mi_to_msecs(t);
}

#else
#include <time.h>

#if defined(CLOCK_REALTIME) || defined(CLOCK_MONOTONIC)

mi_msecs_t _mi_prim_clock_now(void) {
  struct timespec t;
  #ifdef CLOCK_MONOTONIC
  clock_gettime(CLOCK_MONOTONIC, &t);
  #else
  clock_gettime(CLOCK_REALTIME, &t);
  #endif
  return ((mi_msecs_t)t.tv_sec * 1000) + ((mi_msecs_t)t.tv_nsec / 1000000);
}

#else

// low resolution timer
mi_msecs_t _mi_prim_clock_now(void) {
  #if !defined(CLOCKS_PER_SEC) || (CLOCKS_PER_SEC == 1000) || (CLOCKS_PER_SEC == 0)
  return (mi_msecs_t)clock();
  #elif (CLOCKS_PER_SEC < 1000)
  return (mi_msecs_t)clock() * (1000 / (mi_msecs_t)CLOCKS_PER_SEC);
  #else
  return (mi_msecs_t)clock() / ((mi_msecs_t)CLOCKS_PER_SEC / 1000);
  #endif
}

#endif
#endif // _WIN32


//----------------------------------------------------------------
// Process info
//----------------------------------------------------------------

#if defined(_WIN32)
#include <psapi.h>

static mi_msecs_t filetime_msecs(const FILETIME* ftime) {
  ULARGE_INTEGER i;
  i.LowPart = ftime->dwLowDateTime;
  i.HighPart = ftime->dwHighDateTime;
  mi_msecs_t msecs = (i.QuadPart / 10000); // FILETIME is in 100 nano seconds
  return msecs;
}

typedef BOOL (WINAPI *PGetProcessMemoryInfo)(HANDLE, PPROCESS_MEMORY_COUNTERS, DWORD);
static PGetProcessMemoryInfo pGetProcessMemoryInfo = NULL;

void _mi_prim_process_info(mi_process_info_t* pinfo)
{
  FILETIME ct;
  FILETIME ut;
  FILETIME st;
  FILETIME et;
  GetProcessTimes(GetCurrentProcess(), &ct, &et, &st, &ut);
  pinfo->utime = filetime_msecs(&ut);
  pinfo->stime = filetime_msecs(&st);

  // load psapi on demand
  if (pGetProcessMemoryInfo == NULL) {
    HINSTANCE hDll = LoadLibrary(TEXT("psapi.dll"));
    if (hDll != NULL) {
      pGetProcessMemoryInfo = (PGetProcessMemoryInfo)(void (*)(void))GetProcAddress(hDll, "GetProcessMemoryInfo");
    }
  }

  // get process info
  PROCESS_MEMORY_COUNTERS info; _mi_memzero_var(info);
  if (pGetProcessMemoryInfo != NULL) {
    pGetProcessMemoryInfo(GetCurrentProcess(), &info, sizeof(info));
  }
  pinfo->current_rss    = (size_t)info.WorkingSetSize;
  pinfo->peak_rss       = (size_t)info.PeakWorkingSetSize;
  pinfo->current_commit = (size_t)info.PagefileUsage;
  pinfo->peak_commit    = (size_t)info.PeakPagefileUsage;
  pinfo->page_faults    = (size_t)info.PageFaultCount;
}


#elif defined(__unix__) || defined(__unix) || defined(unix) || defined(__APPLE__) || defined(__HAIKU__)
#include <stdio.h>
#include <unistd.h>
#include <sys/resource.h>

#if defined(__APPLE__)
#include <mach/mach.h>
#endif

#if defined(__HAIKU__)
#include <kernel/OS.h>
#endif

static mi_msecs_t timeval_secs(const struct timeval* tv) {
  return ((mi_msecs_t)tv->tv_sec * 1000L) + ((mi_msecs_t)tv->tv_usec / 1000L);
}

void _mi_prim_process_info(mi_process_info_t* pinfo)
{
  struct rusage rusage;
  getrusage(RUSAGE_SELF, &rusage);
  pinfo->utime = timeval_secs(&rusage.ru_utime);
  pinfo->stime = timeval_secs(&rusage.ru_stime);
#if !defined(__HAIKU__)
  pinfo->page_faults = rusage.ru_majflt;
#endif
#if defined(__HAIKU__)
  // Haiku does not have (yet?) a way to
  // get these stats per process
  thread_info tid;
  area_info mem;
  ssize_t c;
  get_thread_info(find_thread(0), &tid);
  while (get_next_area_info(tid.team, &c, &mem) == B_OK) {
    pinfo->peak_rss += mem.ram_size;
  }
  pinfo->page_faults = 0;
#elif defined(__APPLE__)
  pinfo->peak_rss = rusage.ru_maxrss;         // macos reports in bytes
  #ifdef MACH_TASK_BASIC_INFO
  struct mach_task_basic_info info;
  mach_msg_type_number_t infoCount = MACH_TASK_BASIC_INFO_COUNT;
  if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO, (task_info_t)&info, &infoCount) == KERN_SUCCESS) {
    pinfo->current_rss = (size_t)info.resident_size;
  }
  #else
  struct task_basic_info info;
  mach_msg_type_number_t infoCount = TASK_BASIC_INFO_COUNT;
  if (task_info(mach_task_self(), TASK_BASIC_INFO, (task_info_t)&info, &infoCount) == KERN_SUCCESS) {
    pinfo->current_rss = (size_t)info.resident_size;
  }
  #endif
#else
  pinfo->peak_rss = rusage.ru_maxrss * 1024;  // Linux/BSD report in KiB
#endif
  // use defaults for commit
}

#else

#ifndef __wasi__
// WebAssembly instances are not processes
#pragma message("define a way to get process info")
#endif

void _mi_prim_process_info(mi_process_info_t* pinfo)
{
  // use defaults
  MI_UNUSED(pinfo);
}

#endif


//----------------------------------------------------------------
// Output
//----------------------------------------------------------------

#if defined(_WIN32)

void _mi_prim_out_stderr( const char* msg )
{
  // on windows with redirection, the C runtime cannot handle locale dependent output
  // after the main thread closes so we use direct console output.
  if (!_mi_preloading()) {
    // _cputs(msg);  // _cputs cannot be used as it aborts when failing to lock the console
    static HANDLE hcon = INVALID_HANDLE_VALUE;
    static bool hconIsConsole = false;
    if (hcon == INVALID_HANDLE_VALUE) {
      hcon = GetStdHandle(STD_ERROR_HANDLE);
      #ifdef MI_HAS_CONSOLE_IO
      CONSOLE_SCREEN_BUFFER_INFO sbi;
      hconIsConsole = ((hcon != INVALID_HANDLE_VALUE) && GetConsoleScreenBufferInfo(hcon, &sbi));
      #endif
    }
    const size_t len = _mi_strlen(msg);
    if (len > 0 && len < UINT32_MAX) {
      DWORD written = 0;
      if (hconIsConsole) {
        #ifdef MI_HAS_CONSOLE_IO
        WriteConsoleA(hcon, msg, (DWORD)len, &written, NULL);
        #endif
      }
      else if (hcon != INVALID_HANDLE_VALUE) {
        // use direct write if stderr was redirected
        WriteFile(hcon, msg, (DWORD)len, &written, NULL);
      }
      else {
        // finally fall back to fputs after all
        fputs(msg, stderr);
      }
    }
  }
}

#else

void _mi_prim_out_stderr( const char* msg ) {
  fputs(msg, stderr);
}

#endif  // _WIN32


//----------------------------------------------------------------
// Environment
//----------------------------------------------------------------

#if defined(_WIN32)

// On Windows use GetEnvironmentVariable instead of getenv to work
// reliably even when this is invoked before the C runtime is initialized.
// i.e. when `_mi_preloading() == true`.
// Note: on windows, environment names are not case sensitive.
bool _mi_prim_getenv(const char* name, char* result, size_t result_size) {
  result[0] = 0;
  size_t len = GetEnvironmentVariableA(name, result, (DWORD)result_size);
  return (len > 0 && len < result_size);
}

#else

#if !defined(MI_USE_ENVIRON) || (MI_USE_ENVIRON!=0)
// On Posix systemsr use `environ` to access environment variables
// even before the C runtime is initialized.
#if defined(__APPLE__) && defined(__has_include) && __has_include(<crt_externs.h>)
#include <crt_externs.h>
static char** mi_get_environ(void) {
  return (*_NSGetEnviron());
}
#else
extern char** environ;
static char** mi_get_environ(void) {
  return environ;
}
#endif
bool _mi_prim_getenv(const char* name, char* result, size_t result_size) {
  if (name==NULL) return false;
  const size_t len = _mi_strlen(name);
  if (len == 0) return false;
  char** env = mi_get_environ();
  if (env == NULL) return false;
  // compare up to 10000 entries
  for (int i = 0; i < 10000 && env[i] != NULL; i++) {
    const char* s = env[i];
    if (_mi_strnicmp(name, s, len) == 0 && s[len] == '=') { // case insensitive
      // found it
      _mi_strlcpy(result, s + len + 1, result_size);
      return true;
    }
  }
  return false;
}
#else
// fallback: use standard C `getenv` but this cannot be used while initializing the C runtime
bool _mi_prim_getenv(const char* name, char* result, size_t result_size) {
  // cannot call getenv() when still initializing the C runtime.
  if (_mi_preloading()) return false;
  const char* s = getenv(name);
  if (s == NULL) {
    // we check the upper case name too.
    char buf[64+1];
    size_t len = _mi_strnlen(name,sizeof(buf)-1);
    for (size_t i = 0; i < len; i++) {
      buf[i] = _mi_toupper(name[i]);
    }
    buf[len] = 0;
    s = getenv(buf);
  }
  if (s == NULL || _mi_strnlen(s,result_size) >= result_size)  return false;
  _mi_strlcpy(result, s, result_size);
  return true;
}
#endif  // !MI_USE_ENVIRON

#endif  // _WIN32


//----------------------------------------------------------------
// Random
//----------------------------------------------------------------

#if defined(_WIN32)

#if defined(MI_USE_RTLGENRANDOM) // || defined(__cplusplus)
// We prefer to use BCryptGenRandom instead of (the unofficial) RtlGenRandom but when using
// dynamic overriding, we observed it can raise an exception when compiled with C++, and
// sometimes deadlocks when also running under the VS debugger.
// In contrast, issue #623 implies that on Windows Server 2019 we need to use BCryptGenRandom.
// To be continued..
#pragma comment (lib,"advapi32.lib")
#define RtlGenRandom  SystemFunction036
mi_decl_externc BOOLEAN NTAPI RtlGenRandom(PVOID RandomBuffer, ULONG RandomBufferLength);

bool _mi_prim_random_buf(void* buf, size_t buf_len) {
  return (RtlGenRandom(buf, (ULONG)buf_len) != 0);
}

#else

#ifndef BCRYPT_USE_SYSTEM_PREFERRED_RNG
#define BCRYPT_USE_SYSTEM_PREFERRED_RNG 0x00000002
#endif

typedef LONG (NTAPI *PBCryptGenRandom)(HANDLE, PUCHAR, ULONG, ULONG);
static  PBCryptGenRandom pBCryptGenRandom = NULL;

bool _mi_prim_random_buf(void* buf, size_t buf_len) {
  if (pBCryptGenRandom == NULL) {
    HINSTANCE hDll = LoadLibrary(TEXT("bcrypt.dll"));
    if (hDll != NULL) {
      pBCryptGenRandom = (PBCryptGenRandom)(void (*)(void))GetProcAddress(hDll, "BCryptGenRandom");
    }
    if (pBCryptGenRandom == NULL) return false;
  }
  return (pBCryptGenRandom(NULL, (PUCHAR)buf, (ULONG)buf_len, BCRYPT_USE_SYSTEM_PREFERRED_RNG) >= 0);
}

#endif  // MI_USE_RTLGENRANDOM

#else

#if defined(__APPLE__) && defined(MAC_OS_X_VERSION_10_15) && (MAC_OS_X_VERSION_MIN_REQUIRED >= MAC_OS_X_VERSION_10_15)
#include <CommonCrypto/CommonCryptoError.h>
#include <CommonCrypto/CommonRandom.h>

bool _mi_prim_random_buf(void* buf, size_t buf_len) {
  // We prefer CCRandomGenerateBytes as it returns an error code while arc4random_buf
  // may fail silently on macOS. See PR #390, and <https://opensource.apple.com/source/Libc/Libc-1439.40.11/gen/FreeBSD/arc4random.c.auto.html>
  return (CCRandomGenerateBytes(buf, buf_len) == kCCSuccess);
}

#elif defined(__ANDROID__) || defined(__DragonFly__) || \
      defined(__FreeBSD__) || defined(__NetBSD__) || defined(__OpenBSD__) || \
      defined(__sun) || \
      (defined(__APPLE__) && (MAC_OS_X_VERSION_MIN_REQUIRED >= MAC_OS_X_VERSION_10_7))

bool _mi_prim_random_buf(void* buf, size_t buf_len) {
  arc4random_buf(buf, buf_len);
  return true;
}

#elif defined(__APPLE__) || defined(__linux__) || defined(__HAIKU__)   // also for old apple versions < 10.7 (issue #829)

#include <sys/types.h>
#include <sys/stat.h>
#include <errno.h>

bool _mi_prim_random_buf(void* buf, size_t buf_len) {
  // Modern Linux provides `getrandom` but different distributions either use `sys/random.h` or `linux/random.h`
  // and for the latter the actual `getrandom` call is not always defined.
  // (see <https://stackoverflow.com/questions/45237324/why-doesnt-getrandom-compile>)
  // We therefore use a syscall directly and fall back dynamically to /dev/urandom when needed.
  #if defined(MI_HAS_SYSCALL_H) && defined(SYS_getrandom)
    #ifndef GRND_NONBLOCK
    #define GRND_NONBLOCK (1)
    #endif
    static _Atomic(uintptr_t) no_getrandom; // = 0
    if (mi_atomic_load_acquire(&no_getrandom)==0) {
      ssize_t ret = syscall(SYS_getrandom, buf, buf_len, GRND_NONBLOCK);
      if (ret >= 0) return (buf_len == (size_t)ret);
      if (errno != ENOSYS) return false;
      mi_atomic_store_release(&no_getrandom, (uintptr_t)1); // don't call again, and fall back to /dev/urandom
    }
  #endif
  int flags = O_RDONLY;
  #if defined(O_CLOEXEC)
  flags |= O_CLOEXEC;
  #endif
  int fd = mi_prim_open("/dev/urandom", flags);
  if (fd < 0) return false;
  size_t count = 0;
  while(count < buf_len) {
    ssize_t ret = mi_prim_read(fd, (char*)buf + count, buf_len - count);
    if (ret<=0) {
      if (errno!=EAGAIN && errno!=EINTR) break;
    }
    else {
      count += ret;
    }
  }
  mi_prim_close(fd);
  return (count==buf_len);
}

#else

bool _mi_prim_random_buf(void* buf, size_t buf_len) {
  return false;
}

#endif

#endif  // _WIN32


//----------------------------------------------------------------
// Thread init/done
//----------------------------------------------------------------

#if defined(_WIN32)
#if defined(MI_SHARED_LIB) && !defined(MI_WIN_NOREDIRECT)
  #define MI_PRIM_HAS_ALLOCATOR_INIT 1

  static bool mi_redirected = false;   // true if malloc redirects to mi_malloc

  bool _mi_is_redirected(void) {
    return mi_redirected;
  }

  #ifdef __cplusplus
  extern "C" {
  #endif
  mi_decl_export void _mi_redirect_entry(DWORD reason) {
    // called on redirection; careful as this may be called before DllMain
    mi_win_tls_init(reason);
    if (reason == DLL_PROCESS_ATTACH) {
      mi_redirected = true;
    }
    else if (reason == DLL_PROCESS_DETACH) {
      mi_redirected = false;
    }
    else if (reason == DLL_THREAD_DETACH) {
      _mi_thread_done(NULL);
    }
  }
  __declspec(dllimport) bool mi_cdecl mi_allocator_init(const char** message);
  __declspec(dllimport) void mi_cdecl mi_allocator_done(void);
  #ifdef __cplusplus
  }
  #endif
  bool _mi_allocator_init(const char** message) {
    return mi_allocator_init(message);
  }
  void _mi_allocator_done(void) {
    mi_allocator_done();
  }
#endif

bool _mi_prim_thread_is_in_threadpool(void) {
  #if (MI_ARCH_X64 || MI_ARCH_X86 || MI_ARCH_ARM64)
  if (win_major_version >= 6) {
    // check if this thread belongs to a windows threadpool
    // see: <https://www.geoffchappell.com/studies/windows/km/ntoskrnl/inc/api/pebteb/teb/index.htm>
    struct _TEB* const teb = NtCurrentTeb();
    void* const pool_data = *((void**)((uint8_t*)teb + (MI_SIZE_BITS == 32 ? 0x0F90 : 0x1778)));
    return (pool_data != NULL);
  }
  #endif
  return false;
}

#else
#if defined(MI_USE_PTHREADS)

// use pthread local storage keys to detect thread ending
// (and used with MI_TLS_PTHREADS for the default heap)
pthread_key_t _mi_heap_default_key = (pthread_key_t)(-1);

static void mi_pthread_done(void* value) {
  if (value!=NULL) {
    _mi_thread_done((mi_theap_t*)value);
  }
}

void _mi_prim_thread_init_auto_done(void) {
  mi_assert_internal(_mi_heap_default_key == (pthread_key_t)(-1));
  pthread_key_create(&_mi_heap_default_key, &mi_pthread_done);
}

void _mi_prim_thread_done_auto_done(void) {
  if (_mi_heap_default_key != (pthread_key_t)(-1)) {  // do not leak the key, see issue #809
    pthread_key_delete(_mi_heap_default_key);
  }
}

void _mi_prim_thread_associate_default_theap(mi_theap_t* theap) {
  if (_mi_heap_default_key != (pthread_key_t)(-1)) {  // can happen during recursive invocation on freeBSD
    pthread_setspecific(_mi_heap_default_key, theap);
  }
}

#else

void _mi_prim_thread_init_auto_done(void) {
  // nothing
}

void _mi_prim_thread_done_auto_done(void) {
  // nothing
}

void _mi_prim_thread_associate_default_theap(mi_theap_t* theap) {
  MI_UNUSED(theap);
}

#endif

bool _mi_prim_thread_is_in_threadpool(void) {
  return false;
}

#endif  // _WIN32
