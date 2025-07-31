/*
这段代码是一个为 Windows 系统实现的 Unix 系统调用兼容层，
主要包含内存映射（mmap）、内存保护（mprotect）、同步（msync）等函数的 Windows 实现，以及时间获取函数clock_gettime。
这些函数是 POSIX 标准的一部分，在 Linux 和 macOS 上原生支持，但在 Windows 上需要手动实现。
*/

#include "win.h"
#include <errno.h>
#include <io.h>

#ifndef FILE_MAP_EXECUTE
#define FILE_MAP_EXECUTE    0x0020
#endif /* FILE_MAP_EXECUTE */

//错误处理：将 Windows 错误码（如GetLastError()）映射到 Unix 错误码（如EINVAL） 
//当前仅返回原始错误码，未完全实现映射
static int __map_mman_error(const uint32_t err, const int deferr)
{
    if (err == 0)
        return 0;
    //TODO: implement
    return err;
}

// __map_mmap_prot_page和__map_mmap_prot_file均为内存管理辅助函数
//功能：将 Unix 的内存保护标志（如 PROT_READ）映射到 Windows 的对应标志（如 PAGE_READONLY）
static uint32_t __map_mmap_prot_page(const int prot)
{
    uint32_t protect = 0;
    
    if (prot == PROT_NONE)
        return protect;
        
    if ((prot & PROT_EXEC) != 0)
    {
        protect = ((prot & PROT_WRITE) != 0) ? 
                    PAGE_EXECUTE_READWRITE : PAGE_EXECUTE_READ;
    }
    else
    {
        protect = ((prot & PROT_WRITE) != 0) ?
                    PAGE_READWRITE : PAGE_READONLY;
    }
    
    return protect;
}

static uint32_t __map_mmap_prot_file(const int prot)
{
    uint32_t desiredAccess = 0;
    
    if (prot == PROT_NONE)
        return desiredAccess;
        
    //映射关系如下
    //Unix PROT_READ → Windows FILE_MAP_READ
    //Unix PROT_WRITE → Windows FILE_MAP_WRITE
    //Unix PROT_EXEC → Windows FILE_MAP_EXECUTE
    if ((prot & PROT_READ) != 0)
        desiredAccess |= FILE_MAP_READ;
    if ((prot & PROT_WRITE) != 0)
        desiredAccess |= FILE_MAP_WRITE;
    if ((prot & PROT_EXEC) != 0)
        desiredAccess |= FILE_MAP_EXECUTE;
    
    return desiredAccess;
}

//内存映射（mmap）实现：将文件或设备映射到内存，实现零拷贝 I/O
//prot：内存保护标志（PROT_READ/PROT_WRITE/PROT_EXEC）
//flags：映射标志（如 MAP_ANONYMOUS）
//fildes：文件描述符（Windows 通过_get_osfhandle转换）
void* mmap(void *addr, size_t len, int prot, int flags, int fildes, ssize_t off)
{
    HANDLE fm, h;
    void * map = MAP_FAILED;
    
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable: 4293)
#endif

    //64 位偏移处理
    const uint32_t dwFileOffsetLow = (uint32_t)(off & 0xFFFFFFFFL);
    const uint32_t dwFileOffsetHigh = (uint32_t)((off >> 32) & 0xFFFFFFFFL);

    const uint32_t protect = __map_mmap_prot_page(prot);
    const uint32_t desiredAccess = __map_mmap_prot_file(prot);

    const ssize_t maxSize = off + (ssize_t)len;

    const uint32_t dwMaxSizeLow = (uint32_t)(maxSize & 0xFFFFFFFFL);
    const uint32_t dwMaxSizeHigh = (uint32_t)((maxSize >> 32) & 0xFFFFFFFFL);

#ifdef _MSC_VER
#pragma warning(pop)
#endif

    errno = 0;
    
    if (len == 0 
        /* Unsupported flag combinations */
        || (flags & MAP_FIXED) != 0
        /* Unsupported protection combinations */
        || prot == PROT_EXEC)
    {
        errno = EINVAL;
        return MAP_FAILED;
    }
    
    h = ((flags & MAP_ANONYMOUS) == 0) ? 
                    (HANDLE)_get_osfhandle(fildes) : INVALID_HANDLE_VALUE;

    if ((flags & MAP_ANONYMOUS) == 0 && h == INVALID_HANDLE_VALUE)
    {
        errno = EBADF;
        return MAP_FAILED;
    }

    //创建文件映射对象
    fm = CreateFileMapping(h, NULL, protect, dwMaxSizeHigh, dwMaxSizeLow, NULL);

    if (fm == NULL)
    {
        errno = __map_mman_error(GetLastError(), EPERM);
        return MAP_FAILED;
    }

    //映射视图到内存
    map = MapViewOfFile(fm, desiredAccess, dwFileOffsetHigh, dwFileOffsetLow, len);

    CloseHandle(fm);

    if (map == NULL)
    {
        errno = __map_mman_error(GetLastError(), EPERM);
        return MAP_FAILED;
    }

    return map;
}

 // 解除内存映射
int munmap(void *addr, size_t len)
{
    if (UnmapViewOfFile(addr))//UnmapViewOfFile为Windows API，munmap为Linux/Unix API
        return 0;
        
    errno =  __map_mman_error(GetLastError(), EPERM);
    
    return -1;
}

//修改内存保护属性
int mprotect(void *addr, size_t len, int prot)
{
    uint32_t newProtect = __map_mmap_prot_page(prot);
    uint32_t oldProtect = 0;
    
    if (VirtualProtect(addr, len, newProtect, &oldProtect))//Windows API
        return 0;
    
    errno =  __map_mman_error(GetLastError(), EPERM);
    
    return -1;
}

//同步内存与磁盘
int msync(void *addr, size_t len, int flags)
{
    if (FlushViewOfFile(addr, len))//Windows API
        return 0;
    
    errno =  __map_mman_error(GetLastError(), EPERM);
    
    return -1;
}

// 锁定内存页
int mlock(const void *addr, size_t len)
{
    if (VirtualLock((LPVOID)addr, len))//Windows API
        return 0;
        
    errno =  __map_mman_error(GetLastError(), EPERM);
    
    return -1;
}

// 解锁内存页
int munlock(const void *addr, size_t len)
{
    if (VirtualUnlock((LPVOID)addr, len))//Windows API
        return 0;
        
    errno =  __map_mman_error(GetLastError(), EPERM);
    
    return -1;
}

// Portable clock_gettime function for Windows时间获取函数
//功能：获取高精度时间，兼容 POSIX 标准
/*
int clock_gettime(int clk_id, struct timespec *tp) {
    uint32_t ticks = GetTickCount();//使用 Windows 的GetTickCount获取毫秒级时间
    tp->tv_sec = ticks / 1000;
    tp->tv_nsec = (ticks % 1000) * 1000000;//转换为timespec结构（秒 + 纳秒）
    return 0;
}
*/
// 优化：使用QueryPerformanceCounter获取纳秒级精度
int clock_gettime(int clk_id, struct timespec *tp) {
    LARGE_INTEGER freq, count;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&count);
    tp->tv_sec = count.QuadPart / freq.QuadPart;
    tp->tv_nsec = ((count.QuadPart % freq.QuadPart) * 1000000000) / freq.QuadPart;
    return 0;
}
