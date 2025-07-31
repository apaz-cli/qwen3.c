cl.exe /fp:fast /Ox /openmp /I. runq.c win.c 

REM cl.exe：Microsoft Visual C++ 编译器，Windows 平台的标准 C/C++ 编译器。
REM /fp:fast	启用快速浮点模式，牺牲浮点计算精度以换取性能优化（如忽略 IEEE 标准）。
REM /Ox	开启最高级优化（等价于/O2+ 其他优化选项），生成最快的代码。（函数内联、循环展开等）
REM /openmp	启用 OpenMP 并行计算，利用多核 CPU，允许代码使用多线程加速（需配合#pragma omp）。
REM /I.	将当前目录（.）添加到头文件搜索路径，用于查找#include的文件。
REM runq.c 和 win.c：需要编译的 C 语言源文件，包含程序的实现代码。