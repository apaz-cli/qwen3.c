/* Inference for Qwen-3 Transformer model in pure C, int8 quantized forward pass. */
/* 这段代码实现了 Qwen-3 Transformer 模型的纯 C 语言推理，特别针对 int8 量化进行了优化 */
//改进策略：增加 AVX2/AVX512 指令集，以优化矩阵乘法

#include <stdio.h>//基本输入输出和文件处理
#include <stdlib.h>//通用的实用函数
#include <ctype.h>//字符处理
#include <stdint.h>//引入了标准的整数类型定义，像int8_t（8 位有符号整数）、uint16_t（16 位无符号整数）等固定宽度的整数类型——可移植性
#include <time.h>
#include <math.h>
#include <string.h>//字符串操作
#include <fcntl.h>//文件控制（文件打开、设置文件属性等操作）

#if defined _WIN32  //判断当前编译环境是否是 Windows 平台（_WIN32 是 Windows 平台下编译器默认定义的宏）
    #include "win.h"    //如果是 Windows 平台，就包含自定义的win.h头文件
#else
    #include <unistd.h>
    #include <sys/mman.h>   //这两行都是类 Unix 系统的头文件
#endif

// ----------------------------------------------------------------------------
// 定义一个全局变量GS
int GS = 0; // 量化权重时的分组大小

// ----------------------------------------------------------------------------
// Transformer model

//1.整体架构与核心组件
//1.1.模型配置
//核心作用：定义模型的超参数，如维度、层数、注意力头数等
//多查询注意力：通过n_kv_heads实现 KV 头共享，减少内存占用
typedef struct {
    int magic_number; // checkpoint magic number            检查点魔数：验证文件格式
    int version; // file format version            模型文件格式版本：防止新旧版本不兼容
    int dim; // transformer dimension       transformer模型维度（关联着注意力、FFN 等模块的参数规模）
    int hidden_dim; // for ffn layers         FFN隐藏层维度：决定网络“加宽”程度，影响对复杂模式的拟合
    int n_layers; // number of layers          层数：层数越多，模型理论上能捕捉的序列依赖越深远
    int n_heads; // number of query heads            查询头的数量（多头注意力的基础）
    int n_kv_heads; // number of key/value heads (can be < query heads because of multiquery)        键值头的数量（支持多查询注意力）通过共享减少计算量
    int vocab_size; // vocabulary size, usually 256 (byte-level)          词汇表大小：决定模型能识别和生成的基本符号范围
    int seq_len; // max sequence length         最大序列长度：限制模型一次能处理的文本长度
    int head_dim; // head dimension     注意力头的维度，和 n_heads 共同决定 dim（dim = n_heads * head_dim ）
    int shared_classifier; // 1 if wcls == p_tokens     是否共享分类器权重（影响参数复用和输出计算
    int group_size; // quantization group size (export.py uses 64)      量化组的大小（通常64）（平衡精度和存储）
} Config;

//1.2.量化张量与权重结构
//1.2.1.量化张量
//量化设计：使用 int8 存储权重，通过分组缩放因子s恢复精度（反量化操作）
//内存优化：量化将权重存储从 float32（4字节）压缩到 int8（1字节），减少 75% 内存占用
typedef struct {
    int8_t *q;    // quantized values   量化值      用 int8 存量化后的值，大幅压缩存储
    float *s; // scaling factors         缩放因子，恢复时用 q[i] * s[分组索引] 还原
} QuantizedTensor;

//1.2.2.模型权重
typedef struct {
    // token embedding table
    //词嵌入相关，模型对输入 token 的基础映射
    QuantizedTensor *q_tokens; // (vocab_size, dim)         词嵌入表（量化）
    float *token_embedding_table; // same, but dequantized        词嵌入表（反量化）

    // weights for rmsnorms     RMSNorm 权重，让训练和推理时的归一化一致，稳定数值计算
    float *rms_att_weight; // (layer, dim) rmsnorm weights      注意力层RMSNorm权重
    float *rms_ffn_weight; // (layer, dim)                      FFN层RMSNorm权重

    // weights for matmuls. note dim == n_heads * head_size     注意力权重矩阵（QKV和输出矩阵）多查询注意力下 q、k、v 头数可不同
    QuantizedTensor *wq; // (layer, dim, n_heads * head_size)
    QuantizedTensor *wk; // (layer, dim, n_kv_heads * head_size)
    QuantizedTensor *wv; // (layer, dim, n_kv_heads * head_size)
    QuantizedTensor *wo; // (layer, n_heads * head_size, dim)

    // QK-RMSNorm for Qwen3     Qwen3 特有的 QK-RMSNorm权重
    float *q_ln_weights;
    float *k_ln_weights;

    // weights for ffn       FFN权重矩阵，SwiGLU 结构的关键支撑
    QuantizedTensor *w1; // (layer, hidden_dim, dim)
    QuantizedTensor *w2; // (layer, dim, hidden_dim)
    QuantizedTensor *w3; // (layer, hidden_dim, dim)
    //FFN：前馈神经网络（Feedforward Neural Network）
    //单向数据流：信息从输入层向输出层单向流动，中间可能包含多个隐藏层，没有反馈连接（区别于循环神经网络 RNN）。
    //层间全连接：典型的 FFN 结构中，每层神经元与下一层的所有神经元相连，称为全连接层（Fully Connected Layer）。

    //最终RMSNorm权重
    float *rms_final_weight; // (dim,)

    // (optional可选) classifier weights for the logits, on the last layer     分类器权重（可选共享词嵌入）
    QuantizedTensor *wcls;
} TransformerWeights;

//1.3.运行状态与缓存
//激活值的流转：从词嵌入的 x 开始，经注意力、FFN 计算，激活值在 xb、hb 等缓存间传递、变换，最终输出 logits
typedef struct {
    //激活值缓冲区
    float *x; // activation at current time stamp (dim,)        当前激活值
    float *xb; // same, but inside a residual branch (dim,)
    float *xb2; // an additional buffer just for convenience (dim,)

    //隐藏层缓冲区
    float *hb; // buffer for hidden dimension in the ffn (hidden_dim,)
    float *hb2; // buffer for hidden dimension in the ffn (hidden_dim,)

    //量化缓冲区：量化后的激活值，部分计算可能基于量化态加速
    QuantizedTensor xq; // quantized x (dim,)
    QuantizedTensor hq; // quantized hb (hidden_dim,)

    //查询query，键key，值value向量（与自注意力机制有关）
    float *q; // query (dim,)
    float *k; // key (dim,)        
    float *v; // value (dim,)

    float *att; // buffer for scores/attention values (n_heads, seq_len)    注意力分数

    float *logits; //输出 logits ，模型推理结果的原始概率分布

    //KV 缓存（层×序列×维度），存历史键值对(key-value pair)，推理时避免重复计算，是加速关键
    float *key_cache;   // (layer, seq_len, dim)       键缓存
    float *value_cache; // (layer, seq_len, dim)       值缓存
} RunState;

/*   Qwen-3 模型推理的核心管理单元   */
typedef struct {
    //整合模型核心组件：把配置、权重、运行时状态打包，让模型推理时能便捷获取各部分资源，实现从模型加载（读 config、weights）到推理（用 state 存中间结果）的一站式管理。
    Config config; // the hyperparameters of the architecture (the blueprint)   超参数蓝图
    TransformerWeights weights; //模型权重
    RunState state; // buffers for the "wave" of activations in the forward pass    推理运行时状态

    //内存映射优势：大模型权重文件大，用 mmap 映射到内存，避免全盘加载占满内存，让推理在有限内存机器也能跑。
    float *data; // 内存映射的数据指针，高效加载大模型权重
    ssize_t file_size; // 模型文件大小，辅助内存映射管理
} Transformer;

//内存管理函数1
//分配策略：根据 Config 里的参数，精准计算各部分内存需求，为注意力、FFN、KV 缓存等分配空间，保证推理时数据有地方存。
void malloc_run_state(RunState* s, Config *p) {
    // we calloc instead of malloc to keep valgrind happy
    //calloc：动态内存分配并清零（函数）
    //malloc：c语言中的动态存储分配函数

    //计算注意力头总维度、KV 头维度等，为内存分配做准备
    int all_heads_dim = p->n_heads * p->head_dim;
    int kv_dim = p->n_kv_heads * p->head_dim;

    // 逐个为运行时状态里的缓存、中间变量分配内存
    s->x = calloc(p->dim, sizeof(float));
    s->xb = calloc(all_heads_dim, sizeof(float));
    s->xb2 = calloc(p->dim, sizeof(float));
    s->hb = calloc(p->hidden_dim, sizeof(float));
    s->hb2 = calloc(p->hidden_dim, sizeof(float));
    //量化相关内存，按 group_size （GS）分组分配缩放因子数组
    s->xq = (QuantizedTensor) { .q = calloc(all_heads_dim, sizeof(int8_t)), .s = calloc(all_heads_dim / GS, sizeof(float)) };
    s->hq = (QuantizedTensor) { .q = calloc(p->hidden_dim, sizeof(int8_t)), .s = calloc(p->hidden_dim / GS, sizeof(float)) };
    s->q = calloc(all_heads_dim, sizeof(float));
    s->att = calloc(p->n_heads * p->seq_len, sizeof(float));
    s->logits = calloc(p->vocab_size, sizeof(float));
    // KV 缓存分配，按层、序列长度、维度分配大数组，存历史键值
    s->key_cache = calloc(p->n_layers * (uint64_t)p->seq_len * kv_dim, sizeof(float));
    s->value_cache = calloc(p->n_layers * (uint64_t)p->seq_len * kv_dim, sizeof(float));

    // ensure all mallocs went fine     内存分配校验，防止内存不足导致推理崩溃
    if (!s->x || !s->xb || !s->xb2 || !s->hb || !s->hb2 || !s->q
     || !s->att || !s->logits || !s->key_cache
     || !s->value_cache) {
        fprintf(stderr, "malloc failed!\n");
        exit(EXIT_FAILURE);
    }
}

//内存管理函数2
//推理结束后，及时释放 RunState 里动态分配的内存，否则多次推理会让内存持续增长，最终撑爆程序。
//释放顺序：和分配顺序对应，确保指针有效时释放，防止野指针问题，避免内存泄漏。
void free_run_state(RunState* s) {
    free(s->x);
    free(s->xb);
    free(s->xb2);
    free(s->hb);
    free(s->hb2);
    free(s->xq.q);
    free(s->xq.s);
    free(s->hq.q);
    free(s->hq.s);
    free(s->q);
    free(s->att);
    free(s->logits);
    free(s->key_cache);
    free(s->value_cache);
}

// ----------------------------------------------------------------------------
// Quantization functions
//2.量化与内存映射

//2.1. 量化 / 反量化函数
//精度损失：int8 量化引入约 1-2% 精度损失，但通过分组优化可大幅降低

//反量化函数
//将量化后的 int8 数据还原为 float32 格式
void dequantize(QuantizedTensor *qx, float *x, int n) {
    for (int i = 0; i < n; i++) {
        x[i] = qx->q[i] * qx->s[i / GS];
        //qx->q[i]：存储的量化后 int8 值（-127~127）
        //qx->s[i / GS]：当前元素对应的缩放因子（每组 GS 个元素共享一个缩放因子s）
    }
}

//量化函数
//分组量化：将权重按GS（通常 64）个元素一组，每组独立计算缩放因子，平衡精度与压缩率
void quantize(QuantizedTensor *qx, float *x, int n) {
    int num_groups = n / GS;
    float Q_MAX = 127.0f;// 映射到int8范围

    //分组找组内最大绝对值，计算缩放因子
    for (int group = 0; group < num_groups; group++) {
        //计算组内最大值
        float wmax = 0;
        for (int i = 0; i < GS; i++) {
            float val = fabs(x[group * GS + i]);
            if (val > wmax) {
                wmax = val;
            }
        }

        //计算缩放因子
        float scale = wmax / Q_MAX;
        qx->s[group] = scale;

        //量化数值，并存储
        for (int i = 0; i < GS; i++) {
            float quant_value = x[group * GS + i] / scale; // scale
            int8_t quantized = (int8_t) round(quant_value); // round and clamp
            qx->q[group * GS + i] = quantized;
        }
    }
}

//2.2. 内存映射与加载权重
/*内存布局：
量化值（int8）连续存储
缩放因子（float）紧跟其后
例如，布局（GS=64 时）：[q0,q1,...,q63,s0,q64,...,q127,s1,...]
*/

//2.2.1.量化张量初始化
//零拷贝设计：直接将内存块映射到QuantizedTensor结构。避免数据复制，提升加载速度
/* initialize `n` x quantized tensor (with `size_each` elements), starting from memory pointed at *ptr */
QuantizedTensor *init_quantized_tensors(void **ptr, int n, int size_each) {
    void *p = *ptr;
    QuantizedTensor *res = malloc(n * sizeof(QuantizedTensor));

    for (int i = 0; i < n; i++) {
        //map quantized int8 values    映射量化值
        res[i].q = (int8_t*)p;
        p = (int8_t*)p + size_each;
        // 映射缩放因子
        res[i].s = (float*)p;
        p = (float*)p + size_each / GS;
    }
    *ptr = p; //更新指针位置
    return res;
}

//2.2.2.权重内存映射
void memory_map_weights(TransformerWeights *w, Config *p, void *ptr) {
    // first are the parameters that are kept in fp32 (the rmsnorm (1D) weights)     加载FP32的RMSNorm权重
    float *fptr = (float*) ptr; // cast our pointer to float*

    w->rms_att_weight = fptr;
    fptr += p->n_layers * p->dim;
    w->rms_ffn_weight = fptr;
    fptr += p->n_layers * p->dim;
    w->rms_final_weight = fptr;
    fptr += p->dim;
    w->q_ln_weights = fptr;
    fptr += p->n_layers * p->head_dim;
    w->k_ln_weights = fptr;
    fptr += p->n_layers * p->head_dim;

    // now read all the quantized weights      加载量化权重（通过init_quantized_tensors函数）
    ptr = (void *)fptr; // now cast the pointer back to void*
    w->q_tokens = init_quantized_tensors(&ptr, 1, p->vocab_size * p->dim);

    // dequantize token embedding table     特殊处理：词嵌入表需反量化
    w->token_embedding_table = malloc(p->vocab_size * p->dim * sizeof(float));
    dequantize(w->q_tokens, w->token_embedding_table, p->vocab_size * p->dim);

    //加载注意力层和FFN层量化权重
    w->wq = init_quantized_tensors(&ptr, p->n_layers, p->dim * (p->n_heads * p->head_dim));
    w->wk = init_quantized_tensors(&ptr, p->n_layers, p->dim * (p->n_kv_heads * p->head_dim));
    w->wv = init_quantized_tensors(&ptr, p->n_layers, p->dim * (p->n_kv_heads * p->head_dim));
    w->wo = init_quantized_tensors(&ptr, p->n_layers, (p->n_heads * p->head_dim) * p->dim);

    w->w1 = init_quantized_tensors(&ptr, p->n_layers, p->dim * p->hidden_dim);
    w->w2 = init_quantized_tensors(&ptr, p->n_layers, p->hidden_dim * p->dim);
    w->w3 = init_quantized_tensors(&ptr, p->n_layers, p->dim * p->hidden_dim);

    w->wcls = p->shared_classifier ? w->q_tokens : init_quantized_tensors(&ptr, 1, p->dim * p->vocab_size);
}

//2.2.3.检查点读取
void read_checkpoint(char *checkpoint, Config *config, TransformerWeights* weights, float** data, ssize_t* file_size, int ctx_length) {
    //打开文件并获取大小
    FILE *file = fopen(checkpoint, "rb");
    if (!file) { fprintf(stderr, "Couldn't open checkpoint %s\n", checkpoint); exit(EXIT_FAILURE); }

    fseek(file, 0, SEEK_END); // move file pointer to end of file
    *file_size = ftell(file); // get the file size, in bytes

    //使用mmap进行内存映射
    /*
    mmap 优势：
    零拷贝加载：直接将文件映射到进程地址空间
    懒加载机制：只有实际访问的内存页才会被加载到物理内存
    节省内存带宽：避免传统 fread+fwrite 的数据复制
    */
    *data = mmap(NULL, *file_size, PROT_READ, MAP_PRIVATE, fileno(file), 0);
    if (*data == MAP_FAILED) { fprintf(stderr, "mmap failed!\n"); exit(EXIT_FAILURE); }
    fclose(file);//映射后可关闭文件

    // checkpoint format is 256-byte header, and then the model weights

    //读取配置信息
    memcpy(config, *data, sizeof(Config));
    if (config->magic_number != 0x616a6331) { fprintf(stderr, "File %s is not a qwen3.c checkpoint\n", checkpoint); exit(EXIT_FAILURE); }//验证文件签名
    if (config->version != 1) { fprintf(stderr, "Checkpoint %s is version %d, need version 1\n", checkpoint, config->version); exit(EXIT_FAILURE); }

    if (ctx_length != 0 && ctx_length <= config->seq_len)
        config->seq_len = ctx_length;

    printf("hidden_size=%d, intermediate_size=%d, num_hidden_layers=%d, num_attention_heads=%d, num_kv_heads=%d, head_dim=%d, ctx_length=%d, vocab_size=%d, shared_classifier=%d, quantization_block_size=%d\n\n", config->dim, config->hidden_dim, config->n_layers, config->n_heads, config->n_kv_heads, config->head_dim, config->seq_len, config->vocab_size, config->shared_classifier, config->group_size);

    GS = config->group_size; //设置全局量化参数 as it will be used in many places

    //映射权重
    void *weights_ptr = ((char *)*data) + 256; //跳过头部(256 bytes)
    memory_map_weights(weights, config, weights_ptr);
}

//5.模型构建与释放
//5.1.构建 Transformer 模型
//初始化流程：先加载静态权重（通过内存映射），再分配动态运行时缓存（KV 缓存、激活值等）？？？？？？？？？？？
void build_transformer(Transformer *t, char *checkpoint_path, int ctx_length) {
    //  读取检查点（配置+权重）
    read_checkpoint(checkpoint_path, &t->config, &t->weights, &t->data, &t->file_size, ctx_length);
    //通过ctx_length参数，可动态调整最大序列长度，支持比训练时更小的上下文长度（节省内存）

    //  分配运行时状态
    malloc_run_state(&t->state, &t->config);
}

//5.2.释放模型资源
void free_transformer(Transformer *t) {
    // free QuantizedTensors     释放量化张量
    free(t->weights.q_tokens);
    free(t->weights.token_embedding_table);
    free(t->weights.wq);
    free(t->weights.wk);
    free(t->weights.wv);
    free(t->weights.wo);
    free(t->weights.w1);
    free(t->weights.w2);
    free(t->weights.w3);
    if(t->weights.wcls != t->weights.q_tokens) { free(t->weights.wcls); }

    // close the memory mapping     解除内存映射（避免悬空指针）
    if (t->data != MAP_FAILED) { munmap(t->data, t->file_size); }
    
    //释放运行时状态
    free_run_state(&t->state);
}

// ----------------------------------------------------------------------------
// neural net blocks; the dynamics of the Transformer

//6.核心计算函数
//6.1. RMSNorm 归一化
void rmsnorm(float *o, float *x, float *weight, int size) {
    //计算均方根
    float ss = 0;
    for (int j = 0; j < size; j++) {
        ss += x[j] * x[j];
    }
    ss = 1.0f / sqrtf((ss / size) + 1e-6f);//添加 1e-6 防止除零错误

    // 归一化并缩放
    for (int j = 0; j < size; j++) {
        o[j] = weight[j] * (ss * x[j]);
    }
}

//6.2. Softmax 激活函数
void softmax(float *x, int size) {
    //计算最大值（数值稳定性优化）
    float max_val = 0;
    for (int i = 0; i < size; i++) {
        if (x[i] > max_val) {
            max_val = x[i];
        }
    }

    //指数化并求和
    float sum = 0;
    for (int i = 0; i < size; i++) {
        x[i] = expf(x[i] - max_val);//减去最大值防止指数溢出
        sum += x[i];
    }

    // normalize    归一化
    for (int i = 0; i < size; i++) {
        x[i] /= sum;
    }
}

//6.3.量化矩阵乘法（模型的核心计算）
//采用了openMP并行化
void matmul(float *xout, QuantizedTensor *x, QuantizedTensor *w, int n, int d) {
    // W (d,n) @ x (n,) -> xout (d,)
    // by far the most amount of time is spent inside this little function
    // inputs to this function are both quantized

    #pragma omp parallel for
    for (int i = 0; i < d; i++) {
        float val = 0;
        int in = i * n;

        // 按GS分组量化计算矩阵乘法：提升缓存命中率
        for (int j = 0; j <= n - GS; j += GS) {
            int32_t ival = 0;
            for (int k = 0; k < GS; k++) {
                ival += x->q[j + k] * w->q[in + j + k];
            }
            val += ((float) ival) * w->s[(in + j) / GS] * x->s[j / GS];
            //int8 乘法累加后再进行浮点运算：减少浮点操作次数
        }

        xout[i] = val;
    }
}

//自己添加：KV 缓存优化（）
// 动态调整KV缓存大小，支持超过seq_len的生成
void resize_kv_cache(RunState* s, Config* p, int new_size) {
    s->key_cache = realloc(s->key_cache, p->n_layers * new_size * p->dim * sizeof(float));
    s->value_cache = realloc(s->value_cache, p->n_layers * new_size * p->dim * sizeof(float));
}

//6.4.前向传播函数
float *forward(Transformer *transformer, int token, int pos) {
    //初始化变量和缓存引用
    Config *p = &transformer->config;
    TransformerWeights* w = &transformer->weights;
    RunState* s = &transformer->state;
    float *x = s->x;
    int dim = p->dim;
    int kv_dim = p->n_kv_heads * p->head_dim;
    int kv_mul = p->n_heads / p->n_kv_heads; // 多查询注意力KV头复用倍数
    int hidden_dim =  p->hidden_dim;
    int all_heads_dim = p->n_heads * p->head_dim;

    // copy the token embedding into x      加载当前token的词嵌入向量
    memcpy(x, w->token_embedding_table + token*dim, dim * sizeof(float));

    //**********************************核心循环 *********************************/注意力机制实现：旋转位置编码（RoPE）+多头注意力
    // forward all the layers        逐层计算
    for(int l = 0; l < p->n_layers; l++) {
        //设置KV缓存指针
        //KV 缓存：每次推理只计算当前位置 KV，历史 KV 直接从缓存读取
        uint64_t loff = l * (uint64_t)p->seq_len * kv_dim; // kv cache layer offset for convenience
        s->k = s->key_cache + loff + pos * kv_dim;// 当前位置的 K向量 缓存位置
        s->v = s->value_cache + loff + pos * kv_dim;// 当前位置的 V向量 缓存位置

        //注意力层：RMSNorm归一化
        rmsnorm(s->xb, x, w->rms_att_weight + l*dim, dim);

        //计算QKV向量       量化矩阵乘法（核心性能瓶颈）
        //对于矩阵乘法的优化：matmul函数使用 OpenMP 并行计算，按组优化内存访问
        quantize(&s->xq, s->xb, dim);
        matmul(s->q, &s->xq, w->wq + l, dim, all_heads_dim);// Query矩阵
        matmul(s->k, &s->xq, w->wk + l, dim, kv_dim);       // Key矩阵
        matmul(s->v, &s->xq, w->wv + l, dim, kv_dim);       // Value矩阵

        float *gq = w->q_ln_weights + l * p->head_dim;   // 128 floats
        float *gk = w->k_ln_weights + l * p->head_dim;   // 128 floats

        /* ------------ Q-RMSNorm + rotate each query head ------------- */
        //Q-RMSNorm + RoPE旋转位置编码（查询头）
        for (int h = 0; h < p->n_heads; h++) {
            float *q = s->q + h * p->head_dim;
            rmsnorm(q, q, gq, p->head_dim);

            //RoPE旋转（复数乘法实现）
            for (int j = 0; j < p->head_dim/2; j++) {
                float freq = powf(1e6, -(float)j / (p->head_dim/2));
                float cos_freq = cosf(pos * freq), sin_freq = sinf(pos * freq);

                float x = q[j]; // real part
                float y = q[j + p->head_dim/2]; // imag part
                
                // 三角函数实现旋转
                q[j] = x * cos_freq - y * sin_freq; //实部旋转
                q[j + p->head_dim/2] = x * sin_freq + y * cos_freq; //虚部旋转
            }
        }

        /* ------------ K-RMSNorm + rotate each key head ------------ */
        //K-RMSNorm + RoPE旋转位置编码（键头）
        for (int h = 0; h < p->n_kv_heads; h++) {
            float *k = s->k + h * p->head_dim;
            rmsnorm(k, k, gk, p->head_dim);

            //同样的RoPE旋转（与查询头使用相同频率但不同位置）
            for (int j = 0; j < p->head_dim/2; j++) {
                float freq = powf(1e6, -(float)j / (p->head_dim/2));
                float cos_freq = cosf(pos * freq), sin_freq = sinf(pos * freq);

                float x = k[j];
                float y = k[j + p->head_dim/2];

                k[j] = x * cos_freq - y * sin_freq;
                k[j + p->head_dim/2] = x * sin_freq + y * cos_freq;
            }
        }

        // 多头注意力计算. iterate over all heads
        #pragma omp parallel for    // OpenMP并行加速
        for (int h = 0; h < p->n_heads; h++) {
            // get the query vector for this head
            float *q = s->q + h * p->head_dim;
            // attention scores for this head
            float *att = s->att + h * p->seq_len;

            //计算当前查询与所有历史键的注意力分数
            for (int t = 0; t <= pos; t++) {
                // get the key vector for this head and at this timestep
                float *k = s->key_cache + loff + t * kv_dim + (h / kv_mul) * p->head_dim;
                // calculate the attention score as the dot product of q and k
                float score = 0;
                for (int i = 0; i < p->head_dim; i++) {
                    score += q[i] * k[i];// 点积计算相似度
                }
                score /= sqrtf(p->head_dim);// 缩放防止梯度消失
                // save the score to the attention buffer
                att[t] = score;
            }

            // softmax the scores to get attention weights, from 0..pos inclusively
            //对注意力得分进行softmax归一化
            softmax(att, pos + 1);

            // weighted sum of the values, store back into xb   根据注意力权重聚合值向量
            float *xb = s->xb + h * p->head_dim;
            memset(xb, 0, p->head_dim * sizeof(float));
            for (int t = 0; t <= pos; t++) {
                // get the value vector for this head and at this timestep
                float *v = s->value_cache + loff + t * kv_dim + (h / kv_mul) * p->head_dim;
                // get the attention weight for this timestep
                float a = att[t];
                // accumulate the weighted value into xb
                for (int i = 0; i < p->head_dim; i++) {
                    xb[i] += a * v[i];// 加权求和
                }
            }
        }

        // final matmul to get the output of the attention      注意力输出线性变换
        quantize(&s->xq, s->xb, all_heads_dim);
        matmul(s->xb2, &s->xq, w->wo + l, all_heads_dim, dim);

        // residual connection back into x      残差连接
        for (int i = 0; i < dim; i++) {
            x[i] += s->xb2[i];
        }

        //FFN层：RMSNorm
        rmsnorm(s->xb, x, w->rms_ffn_weight + l*dim, dim);

        // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
        // first calculate self.w1(x) and self.w3(x)
        quantize(&s->xq, s->xb, dim);
        //计算FFN的两个线性变换w1(x)和w3(x)
        matmul(s->hb, &s->xq, w->w1 + l, dim, hidden_dim);
        matmul(s->hb2, &s->xq, w->w3 + l, dim, hidden_dim);

        // SwiGLU non-linearity    FFN层计算（SwiGLU激活）
        // SwiGLU激活函数：f(x) = silu(w1(x)) * w3(x)，相比 ReLU，提升模型表达能力同时保持计算效率
        // ReLU线性整流函数：f(x)=max(0,x),只保留正值，负值赋为零
        for (int i = 0; i < hidden_dim; i++) {
            float val = s->hb[i];
            // silu(x)=x*σ(x), where σ(x) is the logistic sigmoid
            val *= (1.0f / (1.0f + expf(-val))); // silu激活函数
            // elementwise multiply with w3(x)
            val *= s->hb2[i]; // 逐元素相乘（门控机制）
            s->hb[i] = val;
        }

        // final matmul to get the output of the ffn    FFN输出线性变换 
        quantize(&s->hq, s->hb, hidden_dim);
        matmul(s->xb, &s->hq, w->w2 + l, hidden_dim, dim);//最终投影

        // residual connection  残差连接
        for (int i = 0; i < dim; i++) {
            x[i] += s->xb[i];
        }
    }

    // final rmsnorm    最终归一化
    rmsnorm(x, x, w->rms_final_weight, dim);

    // classifier into logits      最终分类器
    quantize(&s->xq, x, dim);
    matmul(s->logits, &s->xq, w->wcls, dim, p->vocab_size);

    return s->logits; // 返回词表上的概率分布
}

// ----------------------------------------------------------------------------
// The Byte Pair Encoding (BPE) Tokenizer that translates strings <-> tokens
//4.分词器与对话系统

//4.1. 分词器数据结构
typedef struct {
    char **vocab;   // 词表（字符串数组）
    float *merge_scores;    // 合并分数（用于BPE合并）
    int vocab_size;   // 词表大小
    unsigned int max_token_length;   // 最大token长度
    unsigned int bos_token_id;  // 开始token ID
    unsigned int eos_token_id;  // 结束token ID
    char prompt_template[1024]; // 提示模板
    char system_prompt_template[1024];  // 系统提示模板
} Tokenizer;

//4.2.加载提示模板函数
//根据参数加载不同类型的提示模板（带 / 不带系统提示、带 / 不带思考过程）
/*
参数说明：
checkpoint_path：模型检查点路径，用于拼接模板文件名
out_template：存储读取的模板内容的缓冲区
with_system_prompt：是否包含系统提示
enable_thinking：是否启用思考过程提示
*/
void load_prompt_template(char *checkpoint_path, char *out_template, int with_system_prompt, int enable_thinking) {
    char prompt_path[1024];

    //拼接提示模板文件路径
    strcpy(prompt_path, checkpoint_path);
    if (with_system_prompt)
      strcat(prompt_path, enable_thinking ? ".template.with-system-and-thinking" : ".template.with-system");
    else
      strcat(prompt_path, enable_thinking ? ".template.with-thinking" : ".template");

    //初始化输出模板缓冲区
    memset(out_template, 0, 1024);
    FILE *file = fopen(prompt_path, "rb");
    if (!file) { fprintf(stderr, "Couldn't load prompt template %s\n", prompt_path); exit(EXIT_FAILURE); }

    fread(out_template, 1024, 1, file);     //读取模板内容
    fclose(file);
}

//4.3.构建分词器
//功能：从文件加载分词器模型，包括词表、合并分数和提示模板
void build_tokenizer(Tokenizer *t, char *checkpoint_path, int vocab_size, int enable_thinking) {
    //加载词表文件
    char tokenizer_path[1024];

    // 拼接分词器文件路径
    strcpy(tokenizer_path, checkpoint_path);
    strcat(tokenizer_path, ".tokenizer");

    // 初始化分词器参数
    t->vocab_size = vocab_size;
    // malloc space to hold the scores and the strings
    t->vocab = (char **)malloc(vocab_size * sizeof(char *));
    t->merge_scores = (float *)malloc(vocab_size * sizeof(float));

    // 打开分词器文件
    FILE *file = fopen(tokenizer_path, "rb");
    if (!file) { fprintf(stderr, "Couldn't load tokenizer model %s\n", tokenizer_path); exit(EXIT_FAILURE); }
    // 读取分词器元数据
    fread(&t->max_token_length, sizeof(int), 1, file);
    fread(&t->bos_token_id, sizeof(int), 1, file);
    fread(&t->eos_token_id, sizeof(int), 1, file);

    int len;

    //读取每个token及其合并分数
    for (int i = 0; i < vocab_size; i++) {
        if (fread(t->merge_scores + i, sizeof(float), 1, file) != 1) {
          t->vocab[i] = (char *)malloc(1);
          t->vocab[i][0] = '\0'; //  添加字符串终止符
        } else {
          fread(&len, sizeof(int), 1, file);
          t->vocab[i] = (char *)malloc(len + 1);
          fread(t->vocab[i], 1, len, file);
          t->vocab[i][len] = '\0'; // 添加字符串终止符
        }
    }
    fclose(file);

    // 加载提示模板
    load_prompt_template(checkpoint_path, t->prompt_template, 0, enable_thinking);
    load_prompt_template(checkpoint_path, t->system_prompt_template, 1, enable_thinking);
}

//4.4.释放分词器资源
void free_tokenizer(Tokenizer *t) {
    // 释放每个token的字符串内存
    for (int i = 0; i < t->vocab_size; i++) { free(t->vocab[i]); }
    // 释放词表数组和合并分数数组
    free(t->vocab);
    free(t->merge_scores);
}

//4.5. 分词器核心功能1
// 解码：将token ID转换为字符串
char *decode(Tokenizer *t, int token) {
    return t->vocab[token];
}

//4.5.0.编码和解码的工具
// 简单的字符串查找
int str_lookup(char *str, char **vocab, int vocab_size) {
    //在词表中查找字符串，找到返回索引，否则返回-1
    for (int i = 0; i < vocab_size; i++) {
        if (!strcmp(str, vocab[i])) {   //遍历词表数组，使用strcmp比较字符串
            return i;
        }
    }
    return -1;
}

//4.5. 分词器核心功能2
//编码：将文本字符串转换为 token 序列，实现 BPE 编码算法
//BPE 算法：通过合并分数决定 token 合并顺序，平衡词表大小与语义表达
//特殊 token 处理：支持系统提示、结束符等特殊标记
void encode(Tokenizer *t, char *text, int *tokens, int *n_tokens) {
    // encode the string text (input) into an upper-bound preallocated tokens[] array

    //分配临时缓冲区 that will store merge candidates of always two consecutive tokens
    // *2 for concat, +1 for null terminator +2 for UTF8 (in case max_token_length is 1)
    char *str_buffer = malloc((t->max_token_length*2 +1 +2) * sizeof(char));
    char special_token[64 + 1];

    // start at 0 tokens    初始化token计数器
    *n_tokens = 0;

    // process the raw (UTF-8) byte sequence of the input string    处理输入文本的每个字符
    //贪心匹配词表
    for (char *c = text; *c != '\0'; c++) {
    //从单字符开始匹配，逐步合并
        int id, found_special_token = 0;

        //初始化当前字符缓冲区
        str_buffer[0] = *c;
        str_buffer[1] = '\0';

        // special tokens begin with < and end with >. If we find a substring beginning with ‘<’ and ending with ‘>’ and there's a token in the vocab for it, use that instead of parsing into shorter tokens
        //处理特殊token（<...>格式）
        if (*c == '<') {
        // 查找匹配特殊的token：解析形如<|endoftext|>的特殊token
          int end_of_token_pos = -1;
          found_special_token = 0;
          for (int k = 0; *c != '\0' && k < 64; k++) {
            if (c[k] == '>') {
              end_of_token_pos = k;
              break;
            }
          }

          if (end_of_token_pos != -1) {
            strncpy(special_token, c, end_of_token_pos + 1);
            special_token[end_of_token_pos + 1] = 0;

            id = str_lookup(special_token, t->vocab, t->vocab_size);
            if (id != -1) {
              c += end_of_token_pos;
              found_special_token = 1;
            }
          }
        }

        //非特殊token的处理：普通字符转为token
        if (!found_special_token) {
          // not a special token, just look up the single character
          id = str_lookup(str_buffer, t->vocab, t->vocab_size);
        }

        // 记录token或跳过未知字符
        if (id != -1) {
            // we found this codepoint in vocab, add it as a token
            tokens[(*n_tokens)++] = id;
        } else {
            printf("Warning: unknown character code point %d in input, skipping.\n", *str_buffer);
            (*n_tokens)++;
        }
    }

    //迭代合并最佳token对（基于合并分数的BPE合并
    while (1) {
        float best_score = -1e10;
        int best_id = -1;
        int best_idx = -1;

        // 寻找合并分数最高的相邻token对（最佳合并对）
        for (int i = 0; i < (*n_tokens - 1); i++) {
            // check if we can merge the pair (tokens[i], tokens[i+1])
            sprintf(str_buffer, "%s%s", t->vocab[tokens[i]], t->vocab[tokens[i + 1]]);
            int id = str_lookup(str_buffer, t->vocab, t->vocab_size);

            if (id != -1 && t->merge_scores[id] > best_score) {
                // this merge pair exists in vocab! record its score and position
                best_score = t->merge_scores[id];
                best_id = id;
                best_idx = i;
            }
        }

        // 无合并对时退出循环
        if (best_idx == -1) {
            break; 
        }

        // 执行合并操作
        // merge the consecutive pair (best_idx, best_idx+1) into new token best_id
        tokens[best_idx] = best_id;
        //移除后续token
        for (int i = best_idx + 1; i < (*n_tokens - 1); i++) {
            tokens[i] = tokens[i + 1];
        }
        (*n_tokens)--; // token length decreased      合并后减少token数量
    }

    free(str_buffer);
}

// ----------------------------------------------------------------------------
// The Sampler, which takes logits and returns a sampled token
// sampling can be done in a few ways: greedy argmax, sampling, top-p sampling
//3.采样器

//3.1.采样器数据结构
//功能：在进行 top-p 采样时，将概率值和对应的词汇表索引关联起来
typedef struct {
    float prob;     //负责存储某个 token 的概率值
    int index;      //用于记录该概率值在原始词汇表中对应的索引
} ProbIndex; 

//功能：该结构体用于存储文本生成过程中采样策略所需的参数
typedef struct {
    int vocab_size;     //词汇表的大小：限定概率分布的维度
    ProbIndex *probindex; //指向 ProbIndex 数组的指针。这个数组会在 top-p 采样时被当作缓冲区来使用。
    float temperature;  //温度参数：主要用于调整概率分布的形状。较低的温度会使分布更加集中，生成的文本更具确定性；较高的温度则会让分布变得更加分散，使生成的文本更具随机性。
    float topp;         //核采样概率阈值（p）。在 top-p 采样中，只从累积概率超过 p 的最小 token 集合里进行采样。
    unsigned long long rng_state;//随机数生成器的状态
} Sampler;

//3.2.贪心采样（确定性）
//特点：选择概率最高的 token
//应用场景：需要确定性输出的场景（如数学推理、代码生成）
int sample_argmax(float *probabilities, int n) {
    int max_i = 0;
    float max_p = probabilities[0];
    for (int i = 1; i < n; i++) {
        if (probabilities[i] > max_p) {
            max_i = i;
            max_p = probabilities[i];
        }
    }
    return max_i;   //返回概率最高的token
}

//3.3.多项式采样（基础随机采样）
//特点：根据概率分布随机采样
//应用场景：创意写作、对话系统
//数学原理：将 [0,1) 区间按概率分割，随机数落在哪个区间就选择对应 token
int sample_mult(float *probabilities, int n, float coin) {
    // sample index from probabilities (they must sum to 1!)
    // coin is a random number in [0, 1), usually from random_f32()
    float cdf = 0;
    for (int i = 0; i < n; i++) {
        cdf += probabilities[i];
        if (coin < cdf) {   // 根据累积概率分布采样
            return i;
        }
    }
    return n - 1; // in case of rounding errors  处理浮点数精度问题的后备方案
}

//比较逻辑（降序排序）
//这个比较函数被用于下面的 top-p 采样（核采样）
int compare(const void *a, const void *b) {
    //参数类型转换：将 void* 类型（支持任意类型）的参数强制转换为 ProbIndex* 类型，以便访问结构体中的 prob 成员（概率值）。
    ProbIndex *a_ = (ProbIndex *) a;
    ProbIndex *b_ = (ProbIndex *) b;

    //比较逻辑
    if (a_->prob > b_->prob) return -1; //表示 a 应排在 b 前面（即 a 的概率值更大）
    if (a_->prob < b_->prob) return 1;
    return 0;                           //表示 a 和 b 相等（概率值相同）
}

//3.4.top-p 采样（核采样）
//特点：在累积概率超过 topp 的集合中采样
//应用场景：通用场景，推荐默认使用
int sample_topp(float *probabilities, int n, float topp, ProbIndex *probindex, float coin) {
    // top-p sampling (or "nucleus sampling") samples from the smallest set of tokens that exceed probability topp. This way we never sample tokens that have very low probabilities and are less likely to go "off the rails".
    // coin is a random number in [0, 1), usually from random_f32()

    //筛选可能的候选token（预过滤）：过滤低概率token，筛选出概率较高的 token 集合
    int n0 = 0;
    // quicksort indices in descending order of probabilities values smaller than (1 - topp) / (n - 1) cannot be part of the result so for efficiency we crop these out as candidates before sorting
    const float cutoff = (1.0f - topp) / (n - 1);
    for (int i = 0; i < n; i++) {
        if (probabilities[i] >= cutoff) {
            probindex[n0].index = i;
            probindex[n0].prob = probabilities[i];
            n0++;
        }
    }

    //按概率降序排序
    qsort(probindex, n0, sizeof(ProbIndex), compare);

    //截断累积概率超过topp的token ：选择最小的 token 集合，使其累积概率超过topp
    float cumulative_prob = 0;
    int last_idx = n0 - 1; // in case of rounding errors consider all elements
    for (int i = 0; i < n0; i++) {
        cumulative_prob += probindex[i].prob;
        if (cumulative_prob > topp) {
            last_idx = i;
            break; // we've exceeded topp by including last_idx
        }
    }

    // 在截断后的集合中随机采样
    float r = coin * cumulative_prob;
    float cdf = 0;
    for (int i = 0; i <= last_idx; i++) {
        cdf += probindex[i].prob;
        if (r < cdf) {
            return probindex[i].index;
        }
    }
    return probindex[last_idx].index; // in case of rounding errors
}

//3.5.采样器初始化与资源管理
//3.5.1.初始化采样器参数并分配内存
void build_sampler(Sampler *sampler, int vocab_size, float temperature, float topp, unsigned long long rng_seed) {
    sampler->vocab_size = vocab_size;
    sampler->temperature = temperature;
    sampler->topp = topp;
    sampler->rng_state = rng_seed;
    // buffer only used with nucleus sampling; may not need but it's ~small
    sampler->probindex = malloc(sampler->vocab_size * sizeof(ProbIndex));
}

//3.5.2.释放采样器占用的内存
void free_sampler(Sampler *sampler) {
    free(sampler->probindex);
}

//随机数生成器
unsigned int random_u32(unsigned long long *state) {
    // xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
    // xorshift算法生成32位随机数
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    return (*state * 0x2545F4914F6CDD1Dull) >> 32;
}
float random_f32(unsigned long long *state) { // random float32 in [0,1)    转换为[0,1)浮点数
    return (random_u32(state) >> 8) / 16777216.0f;
}

//3.6.统一采样接口
//采样策略：支持贪心、多项式、top-p 三种采样方式
//温度参数：控制输出随机性，temperature=0 为完全确定（贪心采样）
int sample(Sampler *sampler, float *logits) {
    // sample the token given the logits and some hyperparameters
    int next;
    if (sampler->temperature == 0) {
        // greedy argmax sampling: take the token with the highest probability
        next = sample_argmax(logits, sampler->vocab_size);//贪心采样
    } 
    else {
        //应用温度参数
        for (int q=0; q<sampler->vocab_size; q++) { logits[q] /= sampler->temperature; }
        // apply softmax to the logits to get the probabilities for next token  应用softmax转换为概率分布
        softmax(logits, sampler->vocab_size);// 温度缩放

        // flip a (float) coin (this is our source of entropy for sampling)      随机数生成
        float coin = random_f32(&sampler->rng_state);

        // we sample from this distribution to get the next token   根据topp参数选择采样策略
        if (sampler->topp <= 0 || sampler->topp >= 1) {
            next = sample_mult(logits, sampler->vocab_size, coin);// 多项式采样
        } 
        else {
            //clamping the least likely tokens to zero
            next = sample_topp(logits, sampler->vocab_size, sampler->topp, sampler->probindex, coin);// top-p采样
        }
    }
    return next;
}

// ----------------------------------------------------------------------------
// generation loop

//4.7.文本生成函数
//功能：基于提示词生成文本，实现【单轮文本生成】，包含分词和 token 解码过程
void generate(Transformer *transformer, Tokenizer *tokenizer, Sampler *sampler, char *prompt) {
//参数：transformer：模型。tokenizer：分词器。sampler：采样器。prompt：输入提示词
    char *empty_prompt = "";
    if (prompt == NULL) { prompt = empty_prompt; }  //处理空提示词情况

    //编码提示词为token序列
    int num_prompt_tokens = 0;
    int *prompt_tokens = (int*)malloc((strlen(prompt)+3) * sizeof(int)); // +3 for '\0', ?BOS, ?EOS     分配内存存储编码后的 token
    encode(tokenizer, prompt, prompt_tokens, &num_prompt_tokens);   //使用encode函数将提示词转换为 token 序列
    if (num_prompt_tokens < 1) {
        fprintf(stderr, "Please provide a prompt using -i <string> on the command line.\n");
        exit(EXIT_FAILURE);//检查是否有有效的 token，否则终止程序
    }

    // start the main loop   生成主循环
    int next;        //存储下一个token（in the sequence）
    int token = prompt_tokens[0]; //从提示词的第一个token开始
    int pos = 0;     // 当前序列位置

    while (pos < transformer->config.seq_len) {
        // 前向传播获取下一个token的logits
        float *logits = forward(transformer, token, pos);

        //处理提示词阶段：强制使用提示词中的token
        if (pos < num_prompt_tokens - 1) {
            // if we are still processing the input prompt, force the next prompt token
            next = prompt_tokens[pos + 1];
        } else {
            // otherwise sample the next token from the logits   提示词处理完后：通过采样生成token
            next = sample(sampler, logits);
        }
        pos++;

        //当生成的 token 是 BOS（序列开始）或 EOS（序列结束）时，终止生成
        if (pos >= num_prompt_tokens && (next == tokenizer->bos_token_id || next == tokenizer->eos_token_id)) { break; }

        printf("%s", decode(tokenizer, token));//使用decode函数将当前 token 转换为字符串，输出解码后的token
        fflush(stdout);//确保实时输出
        token = next;//更新当前 token 为预测的下一个 token
    }
    printf("\n");
    free(prompt_tokens);
}

//辅助函数：从标准输入读取
//功能：从标准输入读取一行文本；去除行末的换行符；用于4.8.交互式输入场景
void read_stdin(const char *guide, char *buffer, size_t bufsize) {
    //从stdin读取一行，不包含换行符 \n
    printf("%s", guide);
    if (fgets(buffer, bufsize, stdin) != NULL) {
        size_t len = strlen(buffer);
        if (len > 0 && buffer[len - 1] == '\n') {
            buffer[len - 1] = '\0'; // strip newline    去除换行符
        }
    }
}

// ----------------------------------------------------------------------------
// chat loop
//4.8. 对话循环
//功能：实现【多轮对话交互】，包含提示词渲染、分词和回复生成
//提示词模板：支持系统提示 + 用户输入的 Llama2 对话格式
//上下文管理：通过 pos 变量跟踪序列位置，实现多轮对话
//用户输入提示词->通过encode函数->输入token序列->生成->输出token序列->通过decode函数->输出字符串
void chat(Transformer *transformer, Tokenizer *tokenizer, Sampler *sampler, char *cli_user_prompt, char *system_prompt) {
    // buffers for reading the system prompt and user prompt from stdin
    char user_prompt[32768];
    char rendered_prompt[32768];
    int num_prompt_tokens = 0;
    int *prompt_tokens = (int *)malloc(32768 * sizeof(int));
    int user_idx;

    // start the main loop
    int8_t user_turn = 1; // user starts
    int next;        // will store the next token in the sequence
    int token;       // stores the current token to feed into the transformer
    int prev_token;
    int pos = 0;     // position in the sequence

    /*******************对话主循环******************/ 
    //交替生成用户/助手回复
    while (1) {
        // if context window is exceeded, clear it
        if (pos >= transformer->config.seq_len) {
            printf("\n\n");
            user_turn = 1;
            pos = 0;
        }

        //处理用户输入阶段
        if (user_turn) {
            // get the user prompt          读取用户输入并编码，或使用命令行参数中的提示词
            if (pos == 0 && cli_user_prompt != NULL) {
                // user prompt for position 0 was passed in, use it
                strcpy(user_prompt, cli_user_prompt);
            } else {
                if (cli_user_prompt != NULL) {
                    break;
                }
                // otherwise get user prompt from stdin
                read_stdin("> ", user_prompt, sizeof(user_prompt));
                if (!user_prompt[0]) {
                    // Terminate if user enters a blank prompt
                    break;
                }
            }
            // render user/system prompts into the Llama 2 Chat schema        渲染提示词（应用Llama2对话格式）
            if (pos == 0 && system_prompt) {
                sprintf(rendered_prompt, tokenizer->system_prompt_template, system_prompt, user_prompt);
            } else {
                sprintf(rendered_prompt, tokenizer->prompt_template, user_prompt);
            }

            //编码渲染后的提示词为token序列
            encode(tokenizer, rendered_prompt, prompt_tokens, &num_prompt_tokens);
            user_idx = 0; // reset the user index
            user_turn = 0;
        }

        // determine the token to pass into the transformer next    确定输入模型的token
        if (user_idx < num_prompt_tokens) {
            // if we are still processing the input prompt, force the next prompt token
            token = prompt_tokens[user_idx++];
        } else {
            // otherwise use the next token sampled from previous turn
            token = next;
        }
        // EOS token ends the Assistant turn
        if (user_idx >= num_prompt_tokens && (token == tokenizer->bos_token_id || token == tokenizer->eos_token_id)) { user_turn = 1; }

        // forward the transformer to get logits for the next token         模型生成回复
        float *logits = forward(transformer, token, pos);
        next = sample(sampler, logits);
        pos++;

        //输出助手回复（解码token为字符串）
        if (user_idx >= num_prompt_tokens && next != tokenizer->bos_token_id && next != tokenizer->eos_token_id) {
            printf("%s", decode(tokenizer, next));
            fflush(stdout);
        }
        if (user_idx >= num_prompt_tokens && (next == tokenizer->bos_token_id || next == tokenizer->eos_token_id)) { printf("\n"); }
    }
    free(prompt_tokens);
}


// ----------------------------------------------------------------------------
// 7.CLI命令行接口

//7.1.错误提示与使用说明函数
void error_usage() {
    //输出程序使用格式
    fprintf(stderr, "Usage:   runq <checkpoint> [options]\n");
    fprintf(stderr, "Example: runq Qwen3-4B.bin -r 1\n");
    //输出可用选项及说明
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  -t <float>  temperature in [0,inf], default 1.0\n");
    fprintf(stderr, "  -p <float>  p value in top-p (nucleus) sampling in [0,1], default 0.9\n");
    fprintf(stderr, "  -s <int>    random seed, default time(NULL)\n");
    fprintf(stderr, "  -c <int>    context window size, 0 (default) = max_seq_len\n");
    fprintf(stderr, "  -m <string> mode: generate|chat, default: chat\n");
    fprintf(stderr, "  -i <string> input prompt\n");
    fprintf(stderr, "  -y <string> system prompt in chat mode, default is none\n");
    fprintf(stderr, "  -r <int>    reasoning mode, 0 (default) = no thinking, 1 = thinking\n");
    // 退出程序并返回错误状态
    exit(EXIT_FAILURE);
}

//7.2.主函数
int main(int argc, char *argv[]) {
    // default parameters
    // 初始化参数默认值
    char *checkpoint_path = NULL;  // e.g. out/model.bin    模型检查点路径（必须指定）
    float temperature = 1.0f;   // 0 = greedy deterministic. 1.0 = original. don't set higher   采样温度，控制输出随机性（0 表示确定性输出，值越大输出越随机）
    float topp = 0.9f;          // top-p in nucleus sampling. 1.0 = off. 0.9 works well, but slower     top-p采样阈值：top-p 采样时保留的最小累积概率
    char *prompt = NULL;        // prompt string    输入提示词
    unsigned long long rng_seed = 0; // seed rng with time by default   随机数种子
    char *mode = "chat";        // 运行模式：generate（单轮文本生成）或chat（多轮对话）
    char *system_prompt = NULL; // (optional)对话模式的系统提示
    int enable_thinking = 0;    // 1 enables thinking   是否启用思考模式
    int ctx_length = 0;         // context length   上下文窗口长度

    // 解析命令行参数
    if (argc >= 2) { 
        checkpoint_path = argv[1];  // 第一个参数为模型路径
    } 
    else {
         error_usage(); // 无参数时显示帮助
    }
    // 从第二个参数开始解析选项（每两个参数一组：-flag value）
    for (int i = 2; i < argc; i+=2) {
        //参数有效性验证
        if (i + 1 >= argc) { error_usage(); } //确保每个选项后有值
        if (argv[i][0] != '-') { error_usage(); } //选项必须以-开头
        if (strlen(argv[i]) != 2) { error_usage(); } // must be -x (one dash, one letter)   选项格式必须为-单字母
        // 读取参数值并转换类型
        if (argv[i][1] == 't') { temperature = atof(argv[i + 1]); }
        else if (argv[i][1] == 'p') { topp = atof(argv[i + 1]); }
        else if (argv[i][1] == 's') { rng_seed = atoi(argv[i + 1]); }
        else if (argv[i][1] == 'c') { ctx_length = atoi(argv[i + 1]); }
        else if (argv[i][1] == 'i') { prompt = argv[i + 1]; }
        else if (argv[i][1] == 'm') { mode = argv[i + 1]; }
        else if (argv[i][1] == 'y') { system_prompt = argv[i + 1]; }
        else if (argv[i][1] == 'r') { enable_thinking = atoi(argv[i + 1]); }
        else { error_usage(); }     //参数格式错误时直接调用error_usage退出
    }

    // parameter validation/overrides   验证参数并设置合理默认值
    if (rng_seed <= 0) rng_seed = (unsigned int)time(NULL);     // 随机数种子默认使用当前时间
    if (temperature < 0) temperature = 0;       // 温度不能为负
    if (topp < 0 || 1.0 < topp) topp = 0.9;     // top-p值限制在[0,1]

    // build the Transformer via the model .bin file    初始化Transformer模型
    Transformer transformer;
    build_transformer(&transformer, checkpoint_path, ctx_length);//调用build_transformer加载模型权重并设置上下文长度

    // build the Tokenizer via the tokenizer .bin file       初始化分词器
    Tokenizer tokenizer;
    build_tokenizer(&tokenizer, checkpoint_path, transformer.config.vocab_size, enable_thinking);//调用build_tokenizer加载词表，根据enable_thinking设置提示模板

    // build the Sampler    初始化采样器
    Sampler sampler;
    build_sampler(&sampler, transformer.config.vocab_size, temperature, topp, rng_seed);//调用build_sampler设置采样策略参数（温度、top-p 等）

    // run!
    //根据模式选择运行函数
    if (strcmp(mode, "generate") == 0) {
        generate(&transformer, &tokenizer, &sampler, prompt);   //generate函数接收提示词用于生成
    } else if (strcmp(mode, "chat") == 0) {
        chat(&transformer, &tokenizer, &sampler, prompt, system_prompt);    //chat函数额外接收《系统提示》用于对话格式构建
    } else {
        fprintf(stderr, "Unknown mode: %s\n", mode);
        error_usage();
    }

    free_sampler(&sampler);             //先释放采样器的临时缓冲区
    free_tokenizer(&tokenizer);         //再释放分词器的词表内存
    free_transformer(&transformer);     //最后释放模型的权重和运行时缓存
    //重要性：避免内存泄漏，确保程序退出时正确释放所有资源
    return 0;
}
