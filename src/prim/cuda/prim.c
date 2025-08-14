
// This file is included in `src/prim/prim.c`

#if defined(MI_MALLOC_OVERRIDE)
// Because initialisation of CUDA requires allocation, interposition with malloc
// becomes tricky. The furthest I have gotten thus far is that the child thread
// will crash in pthread_mutex_lock during cuDevicePrimaryCtxRetain(), though I
// have not yet dug through the sources of pthreads to try and guess why (no
// debug symbols).
#error "CUDA backend is currently incompatible with malloc interposition"
#endif

#ifndef _DEFAULT_SOURCE
#define _DEFAULT_SOURCE   // ensure mmap flags and syscall are defined
#endif

#include "mimalloc.h"
#include "mimalloc/internal.h"
#include "mimalloc/atomic.h"
#include "mimalloc/prim.h"

#include <cuda.h>

#if !defined(_WIN32)
#include <sys/mman.h>     // mmap
#include <unistd.h>       // sysconf
#include <fcntl.h>        // open, close, read, access
#include <stdlib.h>       // getenv, arc4random_buf

#if defined(__linux__)
#include <sys/sysinfo.h>  // sysinfo
#endif
#endif

#define CUDA_SAFE_CALL(it)                                                                                                         \
  do {                                                                                                                             \
    CUresult status = it;                                                                                                          \
    if mi_unlikely(CUDA_SUCCESS != status) {                                                                                       \
      const char* name;                                                                                                            \
      const char* description;                                                                                                     \
      cuGetErrorName(status, &name);                                                                                               \
      cuGetErrorString(status, &description);                                                                                      \
      _mi_error_message(status, "%s:%d CUDA call failed with error %s (%d): %s\n", __FILE__, __LINE__, name, status, description); \
      abort();                                                                                                                     \
    }                                                                                                                              \
  } while (0);


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

static mi_decl_thread CUcontext tld_context = NULL;

static void _mi_prim_cuda_init(void) {
  static CUdevice device = 0;
  static CUcontext context = NULL;
  static mi_atomic_once_t initialised = 0;

  // We only need to initialise CUDA once, but the context must be bound to
  // every CPU thread. All host memory that we allocate (via cuMemAllocHost)
  // will automatically be immediately accessible to all contexts on all devices
  // which support unified addressing, and the device pointer that may be used
  // to access this host memory from those contexts is always equal to the
  // returned host pointer. Thus it is safe to initialise only the first device
  // and use the default context.
  if (mi_atomic_once(&initialised)) {
    CUDA_SAFE_CALL(cuInit(0));
    CUDA_SAFE_CALL(cuDeviceGet(&device, 0));
    CUDA_SAFE_CALL(cuDevicePrimaryCtxRetain(&context, device));
  }
  mi_assert(context != NULL);

#if MI_DEBUG > 2
  CUcontext current = NULL;
  CUDA_SAFE_CALL(cuCtxGetCurrent(&current));
  mi_assert(current == NULL);
#endif

  mi_assert(tld_context == NULL);
  CUDA_SAFE_CALL(cuCtxPushCurrent(context));
  tld_context = context;
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
  CUDA_SAFE_CALL(cuMemFreeHost(addr));
  return 0;
}


//---------------------------------------------
// Allocation
//---------------------------------------------

// Allocate OS memory. Return NULL on error.
// The `try_alignment` is just a hint and the returned pointer does not have to be aligned.
int _mi_prim_alloc(void* hint_addr, size_t size, size_t try_alignment, bool commit, bool allow_large, bool* is_large, bool* is_zero, void** addr) {
  MI_UNUSED(hint_addr);
  MI_UNUSED(try_alignment);
  MI_UNUSED(commit);
  MI_UNUSED(allow_large);

  mi_assert_internal(size > 0 && (size % _mi_os_page_size()) == 0);

  *is_large = false;
  *is_zero = false;

  if mi_unlikely(tld_context == NULL) {
    _mi_prim_cuda_init();
  }
  CUDA_SAFE_CALL(cuMemAllocHost(addr, size));
  return 0;
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
  bool is_large = true;
  bool allow_large = true;
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
    _mi_thread_done((mi_heap_t*)value);
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

void _mi_prim_thread_associate_default_heap(mi_heap_t* heap) {
  if (_mi_heap_default_key != (pthread_key_t)(-1)) {  // can happen during recursive invocation on freeBSD
    pthread_setspecific(_mi_heap_default_key, heap);
  }
}

#else

void _mi_prim_thread_init_auto_done(void) {
  // nothing
}

void _mi_prim_thread_done_auto_done(void) {
  // nothing
}

void _mi_prim_thread_associate_default_heap(mi_heap_t* heap) {
  MI_UNUSED(heap);
}

#endif

bool _mi_prim_thread_is_in_threadpool(void) {
  return false;
}

#endif  // _WIN32

