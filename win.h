/*
这是一个用于 Windows 平台 的头文件（win.h），主要目的是为项目提供 类 Unix 系统调用的兼容层，
让原本为 Linux 或 POSIX 环境编写的代码（尤其是涉及内存映射、时钟函数等）能在 Windows 下编译和运行。
*/

//头文件保护：避免头文件被重复包含，防止因重复定义导致编译错误。    
#ifndef _WIN_H_
#define _WIN_H_

#define WIN32_LEAN_AND_MEAN      // 头文件精简： Windows 头文件只包含最核心的功能（如基础 API、数据类型），不包含老旧或极少使用的组件（如过时的控件、调试库），加速编译并减少冗余。
#include <windows.h>   //Windows 平台的核心头文件，提供 HANDLE、DWORD 等类型和 CreateFile、MapViewOfFile 等系统调用。
#include <time.h>
#include <stdint.h>

//ssize_t：POSIX 标准中表示 带符号的字节数（如 read/write 的返回值），Windows 原生无此类型，用 int64_t 兼容。
#define ssize_t int64_t
//ftell：POSIX 中用于获取文件指针位置，Windows 下用 _ftelli64（支持 64 位文件偏移）替代。
#define ftell _ftelli64

// Below code is originally from mman-win32
//
/*
 * sys/mman.h
 * mman-win32
 */

#ifndef _WIN32_WINNT            // Allow use of features specific to Windows XP or later.
#define _WIN32_WINNT    0x0501  // Change this to the appropriate value to target other versions of Windows.
#endif

/* All the headers include this file. */
#ifndef _MSC_VER
#include <_mingw.h>
#endif

#include <sys/types.h>

//C++ 兼容：确保头文件内容在 C++ 编译时按 C 语言规则处理（避免函数名修饰问题），让 C++ 项目能调用这些 C 风格函数。
#ifdef __cplusplus
extern "C" {
#endif

//权限与标志宏（兼容 POSIX 语义）：这些宏让 Windows 代码能 用 POSIX 语义调用接口，无需修改核心逻辑。
//PROT_*：内存保护标志（读、写、执行权限）。
#define PROT_NONE       0
#define PROT_READ       1
#define PROT_WRITE      2
#define PROT_EXEC       4

//MAP_*：内存映射标志（共享、私有、匿名映射等）。
#define MAP_FILE        0
#define MAP_SHARED      1
#define MAP_PRIVATE     2
#define MAP_TYPE        0xf
#define MAP_FIXED       0x10
#define MAP_ANONYMOUS   0x20
#define MAP_ANON        MAP_ANONYMOUS

#define MAP_FAILED      ((void *)-1)

//MS_*：msync 的同步标志（异步、同步、失效缓存）。
#define MS_ASYNC        1
#define MS_SYNC         2
#define MS_INVALIDATE   4

//CLOCK_REALTIME：clock_gettime 的时钟类型（实时时钟）。
#define CLOCK_REALTIME  0

//POSIX 系统调用声明（内存映射、时钟等）
//内存映射与保护
void*   mmap(void *addr, size_t len, int prot, int flags, int fildes, ssize_t off); // 内存映射文件/匿名内存
int     munmap(void *addr, size_t len);// 解除内存映射
int     mprotect(void *addr, size_t len, int prot);// 修改内存保护权限
int     msync(void *addr, size_t len, int flags);// 同步内存映射到磁盘
int     mlock(const void *addr, size_t len);// 锁定内存（防止换出到磁盘）
int     munlock(const void *addr, size_t len);// 解锁内存
//时钟函数
int     clock_gettime(int clk_id, struct timespec *tp);// 高精度时钟

#ifdef __cplusplus
};
#endif

#endif /*  _WIN_H_ */
