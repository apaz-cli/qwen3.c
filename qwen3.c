/* Inference for Qwen-3 Transformer model in pure C, int8 quantized forward pass. */

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <unistd.h>
#include <sys/mman.h>   // Both are Unix-like system header files

#include "prompts.h"


#define ub_assert(x) if (!(x)) { fprintf(stderr, "Assertion failed: %s\n", #x); exit(1); }

typedef struct {
    int magic_number;
    int dim;
    int hidden_dim;
    int n_layers;
    int n_heads;
    int n_kv_heads;
    int vocab_size;
    int seq_len;
    int head_dim;
    int shared_classifier; // For Qwen3-0.6B - 4B, the classifier is shared like in GPT2. For 14B, 32B, it is not.
    int qgroup_size; // quantization group size (export.py uses 64) - balances precision and storage
} Config;

typedef struct {
    int8_t *q; // quantized valuess
    float *s;  // scaling factors
} Int8Tensor;

typedef struct {
    // token embedding table - model's basic mapping for input tokens
    Int8Tensor *q_embedding_table; // (vocab_size, dim) - token embedding table (quantized)
    float *token_embedding_table; // same, but dequantized - token embedding table (dequantized)

    // RMSNorm weights
    float *rms_att_weight; // (layer, dim)
    float *rms_ffn_weight; // (layer, dim)
    float *rms_final_weight; // (dim,)

    // Weights for attention matmuls.
    // Note dim == n_heads * head_size - attention weight matrices (QKV and output matrices) under multi-query attention q, k, v head counts can differ
    Int8Tensor *wq; // (layer, dim, n_heads * head_size)
    Int8Tensor *wk; // (layer, dim, n_kv_heads * head_size)
    Int8Tensor *wv; // (layer, dim, n_kv_heads * head_size)
    Int8Tensor *wo; // (layer, n_heads * head_size, dim)

    // Qwen-specific QK-RMSNorm weights
    float *q_ln_weights;
    float *k_ln_weights;

    // weights for ffn - FFN weight matrices, key support for SwiGLU structure
    Int8Tensor *w1; // (layer, hidden_dim, dim)
    Int8Tensor *w2; // (layer, dim, hidden_dim)
    Int8Tensor *w3; // (layer, hidden_dim, dim)

    Int8Tensor *wcls; // (optional) classifier weights for the logits (optionally shared with token embedding)
} Qwen3Weights;

// 1.3. Runtime state and cache
// Activation flow: Starting from token embedding x, through attention and FFN computation, activations are passed and transformed between caches like xb, hb, finally outputting logits
typedef struct {
    // Activation buffers
    float *x; // activation at current time stamp (dim,) - current activation values
    float *xb; // same, but inside a residual branch (dim,)
    float *xb2; // an additional buffer just for convenience (dim,)

    // Hidden layer buffers
    float *hb; // buffer for hidden dimension in the ffn (hidden_dim,)
    float *hb2; // buffer for hidden dimension in the ffn (hidden_dim,)

    // Quantization buffers - quantized activation values, some computations may be accelerated based on quantized state
    Int8Tensor xq; // quantized x (dim,)
    Int8Tensor hq; // quantized hb (hidden_dim,)

    // Query, key, value vectors - related to self-attention mechanism
    float *q; // query (dim,)
    float *k; // key (dim,)
    float *v; // value (dim,)

    float *att; // buffer for scores/attention values (n_heads, seq_len) - attention scores

    float *logits; // output logits - raw probability distribution of model inference results

    // KV cache - stores historical key-value pairs, avoids redundant computation during inference, key to acceleration
    float *key_cache;   // (layer, seq_len, dim) - key cache
    float *value_cache; // (layer, seq_len, dim) - value cache
} RunState;

/* Core management unit for Qwen-3 model inference */
typedef struct {
    // Integrate model core components - package configuration, weights, runtime state together, allowing convenient access to all resources during model inference, achieving one-stop management from model loading (reading config, weights) to inference (using state to store intermediate results)
    Config config; // the hyperparameters of the architecture (the blueprint) - hyperparameter blueprint
    Qwen3Weights weights; // model weights
    RunState state; // buffers for the "wave" of activations in the forward pass - inference runtime state

    // Memory mapping advantages - Large model weight files are big, using mmap to map to memory avoids loading everything into memory, allows inference to run on machines with limited memory
    float *data; // memory-mapped data pointer - efficiently loads large model weights
    ssize_t file_size; // model file size - assists memory mapping management
} Transformer;

void malloc_run_state(RunState* s, Config *p) {
    int QGS = p->qgroup_size;

    int all_heads_dim = p->n_heads * p->head_dim;
    int kv_dim = p->n_kv_heads * p->head_dim;

    // Allocate memory for caches and intermediate variables in runtime state one by one
    size_t x_sz = p->dim * sizeof(float);
    size_t xb_sz = all_heads_dim * sizeof(float);
    size_t xb2_sz = all_heads_dim * sizeof(float);
    size_t hb_sz = p->hidden_dim * sizeof(float);
    size_t hb2_sz = p->hidden_dim * sizeof(float);
    size_t xqq_sz = all_heads_dim * sizeof(int8_t);
    size_t xqs_sz = all_heads_dim / QGS * sizeof(float);
    size_t hqq_sz = p->hidden_dim * sizeof(int8_t);
    size_t hqs_sz = p->hidden_dim / QGS * sizeof(float);
    size_t q_sz = all_heads_dim * sizeof(float);
    size_t att_sz = p->n_heads * p->seq_len * sizeof(float);
    size_t logits_sz = p->vocab_size * sizeof(float);
    size_t key_cache_sz = p->n_layers * (uint64_t)p->seq_len * kv_dim * sizeof(float);
    size_t value_cache_sz = p->n_layers * (uint64_t)p->seq_len * kv_dim * sizeof(float);

    char* buffer = malloc(x_sz + xb_sz + xb2_sz + hb_sz + hb2_sz
        + xqq_sz + xqs_sz + hqq_sz + hqs_sz
        + q_sz + att_sz + logits_sz
        + key_cache_sz + value_cache_sz);
    char* cur = buffer;

    if (!buffer) {
        fprintf(stderr, "malloc failed!\n");
        exit(1);
    }

    s->x = ((float*)cur); cur += x_sz;
    s->xb = ((float*)cur); cur += xb_sz;
    s->xb2 = ((float*)cur); cur += xb2_sz;
    s->hb = ((float*)cur); cur += hb_sz;
    s->hb2 = ((float*)cur); cur += hb2_sz;

    int8_t * xqq = ((int8_t*)cur); cur += xqq_sz;
    float* xqs = ((float*)cur); cur += xqs_sz;
    int8_t * hqq = ((int8_t*)cur); cur += hqq_sz;
    float* hqs = ((float*)cur); cur += hqs_sz;
    s->xq = (Int8Tensor) { .q = xqq, .s = xqs };
    s->hq = (Int8Tensor) { .q = hqq, .s = hqs };

    s->q = ((float*)cur); cur += q_sz;
    s->att = ((float*)cur); cur += att_sz;
    s->logits = ((float*)cur); cur += logits_sz;

    s->key_cache = ((float*)cur); cur += key_cache_sz;
    s->value_cache = ((float*)cur); cur += value_cache_sz;
}

void dequantize(Int8Tensor *qx, float *x, int n, Config *config) {
    int QGS = config->qgroup_size;
    ub_assert(QGS % 2 == 0);
    ub_assert(n % QGS == 0);
    for (int i = 0; i < n; i++) {
        x[i] = qx->q[i] * qx->s[i / QGS];
    }
}

void quantize(Int8Tensor *qx, float *x, int n, Config *config) {
    int QGS = config->qgroup_size;
    int num_groups = n / QGS;
    float Q_MAX = 127.0f;

    // Find maximum absolute value within each group, calculate scaling factor
    for (int group = 0; group < num_groups; group++) {
        // Calculate maximum value within group
        float wmax = 0;
        for (int i = 0; i < QGS; i++) {
            float val = fabs(x[group * QGS + i]);
            if (val > wmax) {
                wmax = val;
            }
        }

        // Calculate scaling factor
        float scale = wmax / Q_MAX;
        qx->s[group] = scale;

        // Quantize values and store
        for (int i = 0; i < QGS; i++) {
            float quant_value = x[group * QGS + i] / scale; // scale
            int8_t quantized = (int8_t) round(quant_value); // round and clamp
            qx->q[group * QGS + i] = quantized;
        }
    }
}

// 2.2.1. Quantized tensor initialization
// Zero-copy design: directly map memory blocks to QuantizedTensor structure. Avoid data copying, improve loading speed
/* initialize `n` x quantized tensor (with `size_each` elements), starting from memory pointed at *ptr */
Int8Tensor *init_quantized_tensors(void **ptr, int n, int size_each, Config *config) {
    int QGS = config->qgroup_size;
    void *p = *ptr;
    Int8Tensor *res = malloc(n * sizeof(Int8Tensor));

    for (int i = 0; i < n; i++) {
        // map quantized int8 values
        res[i].q = (int8_t*)p;
        p = (int8_t*)p + size_each;
        // Map scaling factors
        res[i].s = (float*)p;
        p = (float*)p + size_each / QGS;
    }
    *ptr = p; // Update pointer position
    return res;
}

void mmap_weights(Qwen3Weights *w, Config *p, void *ptr) {
    // first are the parameters that are kept in fp32 (the rmsnorm (1D) weights) - load FP32 RMSNorm weights
    float *fptr = (float*) ptr;

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

    // now read all the quantized weights - load quantized weights (through init_quantized_tensors function)
    ptr = (void *)fptr; // now cast the pointer back to void*
    w->q_embedding_table = init_quantized_tensors(&ptr, 1, p->vocab_size * p->dim, p);

    // dequantize token embedding table - special handling: token embedding table needs dequantization
    w->token_embedding_table = malloc(p->vocab_size * p->dim * sizeof(float));
    dequantize(w->q_embedding_table, w->token_embedding_table, p->vocab_size * p->dim, p);

    // Load attention layer and FFN layer quantized weights
    w->wq = init_quantized_tensors(&ptr, p->n_layers, p->dim * (p->n_heads * p->head_dim), p);
    w->wk = init_quantized_tensors(&ptr, p->n_layers, p->dim * (p->n_kv_heads * p->head_dim), p);
    w->wv = init_quantized_tensors(&ptr, p->n_layers, p->dim * (p->n_kv_heads * p->head_dim), p);
    w->wo = init_quantized_tensors(&ptr, p->n_layers, (p->n_heads * p->head_dim) * p->dim, p);

    w->w1 = init_quantized_tensors(&ptr, p->n_layers, p->dim * p->hidden_dim, p);
    w->w2 = init_quantized_tensors(&ptr, p->n_layers, p->hidden_dim * p->dim, p);
    w->w3 = init_quantized_tensors(&ptr, p->n_layers, p->dim * p->hidden_dim, p);

    w->wcls = p->shared_classifier ? w->q_embedding_table : init_quantized_tensors(&ptr, 1, p->dim * p->vocab_size, p);
}

void read_checkpoint(char *checkpoint, Config *config, Qwen3Weights* weights, float** data, ssize_t* file_size, int ctx_length) {
    // Open file and get size
    FILE *file = fopen(checkpoint, "rb");
    if (!file) { fprintf(stderr, "Couldn't open checkpoint %s\n", checkpoint); exit(1); }

    fseek(file, 0, SEEK_END); // move file pointer to end of file
    *file_size = ftell(file); // get the file size, in bytes

    // Use mmap for memory mapping
    /*
    mmap advantages:
    Zero-copy loading: directly map file to process address space
    Lazy loading mechanism: only actually accessed memory pages are loaded into physical memory
    Save memory bandwidth: avoid data copying of traditional fread+fwrite
    */
    *data = mmap(NULL, *file_size, PROT_READ, MAP_PRIVATE, fileno(file), 0);
    if (*data == MAP_FAILED) { fprintf(stderr, "mmap failed!\n"); exit(1); }
    fclose(file);// File can be closed after mapping

    // checkpoint format is header, and then the model weights
    memcpy(config, *data, sizeof(Config));
    if (config->magic_number != 0x6E657771) { fprintf(stderr, "File %s is not a qwen3.c checkpoint\n", checkpoint); exit(1); }// Verify file signature

    if (ctx_length != 0 && ctx_length <= config->seq_len)
        config->seq_len = ctx_length;

    printf("hidden_size=%d, intermediate_size=%d, num_hidden_layers=%d, num_attention_heads=%d, num_kv_heads=%d, head_dim=%d, ctx_length=%d, vocab_size=%d, shared_classifier=%d, quantization_block_size=%d\n\n", config->dim, config->hidden_dim, config->n_layers, config->n_heads, config->n_kv_heads, config->head_dim, config->seq_len, config->vocab_size, config->shared_classifier, config->qgroup_size);


    // Map weights
    void *weights_ptr = ((char *)*data) + 256; // Skip header (256 bytes)
    mmap_weights(weights, config, weights_ptr);
}

// 5. Model construction and release
// 5.1. Build Transformer model
// Initialization process: first load static weights (through memory mapping), then allocate dynamic runtime cache (KV cache, activation values, etc.)
void build_transformer(Transformer *t, char *checkpoint_path, int ctx_length) {
    // Read checkpoint (configuration + weights)
    read_checkpoint(checkpoint_path, &t->config, &t->weights, &t->data, &t->file_size, ctx_length);
    // Through ctx_length parameter, can dynamically adjust maximum sequence length, support smaller context length than training time (save memory)

    // Allocate runtime state
    malloc_run_state(&t->state, &t->config);
}

void free_transformer(Transformer *t) {
    free(t->weights.q_embedding_table);
    free(t->weights.token_embedding_table);
    free(t->weights.wq);
    free(t->weights.wk);
    free(t->weights.wv);
    free(t->weights.wo);
    free(t->weights.w1);
    free(t->weights.w2);
    free(t->weights.w3);
    if(t->weights.wcls != t->weights.q_embedding_table) { free(t->weights.wcls); }

    // close the memory mapping - unmap memory (avoid dangling pointers)
    if (t->data != MAP_FAILED) { munmap(t->data, t->file_size); }

    // Release runtime state
    free(t->state.x);
}

void rmsnorm(float *o, float *x, float *weight, int size) {
    // Calculate root mean square
    float ss = 0;
    for (int j = 0; j < size; j++) {
        ss += x[j] * x[j];
    }
    ss = 1.0f / sqrtf((ss / size) + 1e-6f);// Add 1e-6 to prevent division by zero error

    // Normalize and scale
    for (int j = 0; j < size; j++) {
        o[j] = weight[j] * (ss * x[j]);
    }
}

void softmax(float *x, int size) {
    float max_val = 0;
    for (int i = 0; i < size; i++) {
        if (x[i] > max_val) {
            max_val = x[i];
        }
    }
    float sum = 0;
    for (int i = 0; i < size; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    for (int i = 0; i < size; i++) {
        x[i] /= sum;
    }
}

// 6.3. Quantized matrix multiplication (core computation of the model)
// Uses OpenMP parallelization
void matmul(float *xout, Int8Tensor *x, Int8Tensor *w, int n, int d, Config *config) {
    int QGS = config->qgroup_size;
    // W (d,n) @ x (n,) -> xout (d,)
    // by far the most amount of time is spent inside this little function
    // inputs to this function are both quantized

    #pragma omp parallel for
    for (int i = 0; i < d; i++) {
        float val = 0;
        int in = i * n;

        // Group quantized matrix multiplication by QGS: improve cache hit rate
        for (int j = 0; j <= n - QGS; j += QGS) {
            int32_t ival = 0;
            for (int k = 0; k < QGS; k++) {
                ival += x->q[j + k] * w->q[in + j + k];
            }
            val += ((float) ival) * w->s[(in + j) / QGS] * x->s[j / QGS];
            // int8 multiplication accumulation followed by floating point operations: reduce number of floating point operations
        }

        xout[i] = val;
    }
}

float *forward(Transformer *transformer, int token, int pos) {
    Config *p = &transformer->config;
    Qwen3Weights* w = &transformer->weights;
    RunState* s = &transformer->state;
    float *x = s->x;
    int dim = p->dim;
    int kv_dim = p->n_kv_heads * p->head_dim;
    int kv_mul = p->n_heads / p->n_kv_heads; // Multi-query attention KV head reuse multiplier
    int hidden_dim =  p->hidden_dim;
    int all_heads_dim = p->n_heads * p->head_dim;
    int half_head_dim = p->head_dim / 2;

    // copy the token embedding into x - load current token's word embedding vector
    memcpy(x, w->token_embedding_table + token*dim, dim * sizeof(float));

    // forward all the layers - layer-by-layer computation
    for(int l = 0; l < p->n_layers; l++) {
        // Set KV cache pointers
        // KV cache: only compute current position KV for each inference, historical KV read directly from cache
        uint64_t loff = l * (uint64_t)p->seq_len * kv_dim; // kv cache layer offset for convenience
        s->k = s->key_cache + loff + pos * kv_dim;// Current position's K vector cache location
        s->v = s->value_cache + loff + pos * kv_dim;// Current position's V vector cache location

        // Attention layer: RMSNorm normalization
        rmsnorm(s->xb, x, w->rms_att_weight + l*dim, dim);

        // Calculate QKV vectors - quantized matrix multiplication (core performance bottleneck)
        // Matrix multiplication optimization: matmul function uses OpenMP parallel computation, optimizes memory access by groups
        quantize(&s->xq, s->xb, dim, p);
        matmul(s->q, &s->xq, w->wq + l, dim, all_heads_dim, p);// Query matrix
        matmul(s->k, &s->xq, w->wk + l, dim, kv_dim, p);       // Key matrix
        matmul(s->v, &s->xq, w->wv + l, dim, kv_dim, p);       // Value matrix

        float *gq = w->q_ln_weights + l * p->head_dim;   // 128 floats
        float *gk = w->k_ln_weights + l * p->head_dim;   // 128 floats

        /* ------------ Q-RMSNorm + rotate each query head ------------- */
        // Q-RMSNorm + RoPE rotary position encoding (query heads)
        for (int h = 0; h < p->n_heads; h++) {
            float *q = s->q + h * p->head_dim;
            rmsnorm(q, q, gq, p->head_dim);

            // RoPE rotation (implemented with complex multiplication)
            for (int j = 0; j < p->head_dim/2; j++) {
                float freq = powf(1e6, -(float)j / half_head_dim);
                float cos_freq = cosf(pos * freq), sin_freq = sinf(pos * freq);

                float x = q[j]; // real part
                float y = q[j + p->head_dim/2]; // imag part

                // Trigonometric function implementation of rotation
                q[j] = x * cos_freq - y * sin_freq; // Real part rotation
                q[j + p->head_dim/2] = x * sin_freq + y * cos_freq; // Imaginary part rotation
            }
        }

        /* ------------ K-RMSNorm + rotate each key head ------------ */
        // K-RMSNorm + RoPE rotary position encoding (key heads)
        for (int h = 0; h < p->n_kv_heads; h++) {
            float *k = s->k + h * p->head_dim;
            rmsnorm(k, k, gk, p->head_dim);

            // Same RoPE rotation (uses same frequency as query heads but different position)
            for (int j = 0; j < p->head_dim/2; j++) {
                float freq = powf(1e6, -(float)j / half_head_dim);
                float cos_freq = cosf(pos * freq), sin_freq = sinf(pos * freq);

                float x = k[j];
                float y = k[j + p->head_dim/2];

                k[j] = x * cos_freq - y * sin_freq;
                k[j + p->head_dim/2] = x * sin_freq + y * cos_freq;
            }
        }

        // Multi-head attention computation. iterate over all heads
        #pragma omp parallel for    // OpenMP parallel acceleration
        for (int h = 0; h < p->n_heads; h++) {
            // get the query vector for this head
            float *q = s->q + h * p->head_dim;
            // attention scores for this head
            float *att = s->att + h * p->seq_len;

            // Calculate attention scores between current query and all historical keys
            for (int t = 0; t <= pos; t++) {
                // get the key vector for this head and at this timestep
                float *k = s->key_cache + loff + t * kv_dim + (h / kv_mul) * p->head_dim;
                // calculate the attention score as the dot product of q and k
                float score = 0;
                for (int i = 0; i < p->head_dim; i++) {
                    score += q[i] * k[i];// Dot product calculates similarity
                }
                score /= sqrtf(p->head_dim);// Scale to prevent gradient vanishing
                // save the score to the attention buffer
                att[t] = score;
            }

            // softmax the scores to get attention weights, from 0..pos inclusively
            // Apply softmax normalization to attention scores
            softmax(att, pos + 1);

            // weighted sum of the values, store back into xb - aggregate value vectors according to attention weights
            float *xb = s->xb + h * p->head_dim;
            memset(xb, 0, p->head_dim * sizeof(float));
            for (int t = 0; t <= pos; t++) {
                // get the value vector for this head and at this timestep
                float *v = s->value_cache + loff + t * kv_dim + (h / kv_mul) * p->head_dim;
                // get the attention weight for this timestep
                float a = att[t];
                // accumulate the weighted value into xb
                for (int i = 0; i < p->head_dim; i++) {
                    xb[i] += a * v[i];// Weighted sum
                }
            }
        }

        // final matmul to get the output of the attention - attention output linear transformation
        quantize(&s->xq, s->xb, all_heads_dim, p);
        matmul(s->xb2, &s->xq, w->wo + l, all_heads_dim, dim, p);

        // residual connection back into x
        for (int i = 0; i < dim; i++) {
            x[i] += s->xb2[i];
        }

        // FFN RMSNorm
        rmsnorm(s->xb, x, w->rms_ffn_weight + l*dim, dim);

        // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
        // first calculate self.w1(x) and self.w3(x)
        quantize(&s->xq, s->xb, dim, p);
        // Calculate two linear transformations of FFN: w1(x) and w3(x)
        matmul(s->hb, &s->xq, w->w1 + l, dim, hidden_dim, p);
        matmul(s->hb2, &s->xq, w->w3 + l, dim, hidden_dim, p);

        // SwiGLU non-linearity - FFN layer computation (SwiGLU activation)
        // SwiGLU activation function: f(x) = silu(w1(x)) * w3(x), compared to ReLU, improves model expressiveness while maintaining computational efficiency
        // ReLU linear rectification function: f(x)=max(0,x), only keeps positive values, assigns zero to negative values
        for (int i = 0; i < hidden_dim; i++) {
            float val = s->hb[i];
            // silu(x)=x*σ(x), where σ(x) is the logistic sigmoid
            val *= (1.0f / (1.0f + expf(-val))); // silu activation function
            // elementwise multiply with w3(x)
            val *= s->hb2[i]; // Element-wise multiplication (gating mechanism)
            s->hb[i] = val;
        }

        // final matmul to get the output of the ffn - FFN output linear transformation
        quantize(&s->hq, s->hb, hidden_dim, p);
        matmul(s->xb, &s->hq, w->w2 + l, hidden_dim, dim, p);// Final projection

        // residual connection
        for (int i = 0; i < dim; i++) {
            x[i] += s->xb[i];
        }
    }

    // final rmsnorm - final normalization
    rmsnorm(x, x, w->rms_final_weight, dim);

    // classifier into logits - final classifier
    quantize(&s->xq, x, dim, p);
    matmul(s->logits, &s->xq, w->wcls, dim, p->vocab_size, p);

    return s->logits;
}

typedef struct {
    char **vocab;
    float *merge_scores;
    int vocab_size;
    unsigned int max_token_length;
    unsigned int bos_token_id;
    unsigned int eos_token_id;
} Tokenizer;

void build_tokenizer(Tokenizer *t, char *checkpoint_path, int vocab_size, int enable_thinking) {

    char tokenizer_path[1024];
    strcpy(tokenizer_path, checkpoint_path);
    strcat(tokenizer_path, ".tokenizer");

    t->vocab_size = vocab_size;
    t->vocab = (char **)malloc(vocab_size * sizeof(char *));
    t->merge_scores = (float *)malloc(vocab_size * sizeof(float));

    // Open tokenizer file
    FILE *file = fopen(tokenizer_path, "rb");
    if (!file) {
        fprintf(stderr, "Couldn't load tokenizer model %s\n", tokenizer_path);
        exit(1);
    }

    // Read tokenizer metadata
    fread(&t->max_token_length, sizeof(int), 1, file);
    fread(&t->bos_token_id, sizeof(int), 1, file);
    fread(&t->eos_token_id, sizeof(int), 1, file);

    // Read each token and its merge score
    int len;
    for (int i = 0; i < vocab_size; i++) {
        if (fread(t->merge_scores + i, sizeof(float), 1, file) != 1) {
          t->vocab[i] = (char *)malloc(1);
          t->vocab[i][0] = '\0'; //  Add terminators
        } else {
          fread(&len, sizeof(int), 1, file);
          t->vocab[i] = (char *)malloc(len + 1);
          fread(t->vocab[i], 1, len, file);
          t->vocab[i][len] = '\0'; // Add terminators
        }
    }
    fclose(file);
}


void free_tokenizer(Tokenizer *t) {
    for (int i = 0; i < t->vocab_size; i++) {
        free(t->vocab[i]);
    }
    free(t->vocab);
    free(t->merge_scores);
}

char *decode(Tokenizer *t, int token) {
    return t->vocab[token];
}

int str_lookup(char *str, char **vocab, int vocab_size) {
    // Search for string in vocabulary, return index if found, otherwise return -1
    for (int i = 0; i < vocab_size; i++) {
        if (!strcmp(str, vocab[i])) {   // Traverse vocabulary array, use strcmp to compare strings
            return i;
        }
    }
    return -1;
}

// Encoding: Convert text string to token sequence, implement BPE encoding algorithm
// BPE algorithm: Determine token merge order through merge scores, balance vocabulary size and semantic expression
// Special token handling: Support special tokens like system prompts, end tokens, etc.
void encode(Tokenizer *t, char *text, int *tokens, int *n_tokens) {
    // encode the string text (input) into an upper-bound preallocated tokens[] array

    // Allocate temporary buffer that will store merge candidates of always two consecutive tokens
    // *2 for concat, +1 for null terminator +2 for UTF8 (in case max_token_length is 1)
    char *str_buffer = malloc((t->max_token_length*2 +1 +2) * sizeof(char));
    char special_token[64 + 1];

    // start at 0 tokens - initialize token counter
    *n_tokens = 0;

    // process the raw (UTF-8) byte sequence of the input string - process each character of input text
    // Greedy matching vocabulary
    for (char *c = text; *c != '\0'; c++) {
    // Start matching from single character, gradually merge
        int id, found_special_token = 0;

        // Initialize current character buffer
        str_buffer[0] = *c;
        str_buffer[1] = '\0';

        // special tokens begin with < and end with >. If we find a substring beginning with ‘<’ and ending with ‘>’ and there's a token in the vocab for it, use that instead of parsing into shorter tokens
        // Handle special tokens (<...> format)
        if (*c == '<') {
        // Find matching special token: parse special tokens like <|endoftext|>
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

        // Handle non-special tokens: convert ordinary characters to tokens
        if (!found_special_token) {
          // not a special token, just look up the single character
          id = str_lookup(str_buffer, t->vocab, t->vocab_size);
        }

        // Record token or skip unknown characters
        if (id != -1) {
            // we found this codepoint in vocab, add it as a token
            tokens[(*n_tokens)++] = id;
        } else {
            printf("Warning: unknown character code point %d in input, skipping.\n", *str_buffer);
            (*n_tokens)++;
        }
    }

    // Iteratively merge best token pairs (BPE merging based on merge scores)
    while (1) {
        float best_score = -1e10;
        int best_id = -1;
        int best_idx = -1;

        // Find adjacent token pair with highest merge score (best merge pair)
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

        // Exit loop when no merge pairs available
        if (best_idx == -1) {
            break;
        }

        // Execute merge operation
        // merge the consecutive pair (best_idx, best_idx+1) into new token best_id
        tokens[best_idx] = best_id;
        // Remove subsequent token
        for (int i = best_idx + 1; i < (*n_tokens - 1); i++) {
            tokens[i] = tokens[i + 1];
        }
        (*n_tokens)--; // token length decreased - reduce token count after merging
    }

    free(str_buffer);
}


// sampling can be done in a few ways: greedy argmax, sampling, top-p sampling

typedef struct {
    float prob;     // Responsible for storing probability value of a certain token
    int vocab_idx;      // Used to record the index corresponding to this probability value in the original vocabulary
} ProbIndex;

// Function: This structure is used to store parameters required for sampling strategies during text generation
typedef struct {
    int vocab_size;       // Vocabulary size: limits the dimension of probability distribution
    ProbIndex *probindex; // Pointer to ProbIndex array. This array will be used as buffer during top-p sampling.
    float temperature;    // Temperature parameter: mainly used to adjust the shape of probability distribution. Lower temperature makes distribution more concentrated, generating more deterministic text; higher temperature makes distribution more dispersed, making generated text more random.
    float top_p;          // Nucleus sampling probability threshold (p). In top-p sampling, only sample from the smallest token set whose cumulative probability exceeds p.
    unsigned long long rng_state;// Random number generator state
} Sampler;

// 3.2. Greedy sampling (deterministic)
// Characteristics: Select token with highest probability
// Application scenarios: Scenarios requiring deterministic output (such as mathematical reasoning, code generation)
int sample_argmax(float *probabilities, int n) {
    int max_i = 0;
    float max_p = probabilities[0];
    for (int i = 1; i < n; i++) {
        if (probabilities[i] > max_p) {
            max_i = i;
            max_p = probabilities[i];
        }
    }
    return max_i;   // Return token with highest probability
}

// 3.3. Multinomial sampling (basic random sampling)
// Characteristics: Random sampling according to probability distribution
// Application scenarios: Creative writing, dialogue systems
// Mathematical principle: Divide [0,1) interval by probability, select corresponding token based on which interval the random number falls into
int sample_mult(float *probabilities, int n, float coin) {
    // sample index from probabilities (they must sum to 1!)
    // coin is a random number in [0, 1), usually from random_f32()
    float cdf = 0;
    for (int i = 0; i < n; i++) {
        cdf += probabilities[i];
        if (coin < cdf) {   // Sample according to cumulative probability distribution
            return i;
        }
    }
    return n - 1; // in case of rounding errors - fallback solution for floating point precision issues
}

// Comparison logic (descending sort)
// This comparison function is used for top-p sampling (nucleus sampling) below
int compare(const void *a, const void *b) {
    // Parameter type conversion: Cast void* type parameters (supporting any type) to ProbIndex* type to access the prob member (probability value) in the structure.
    ProbIndex *a_ = (ProbIndex *) a;
    ProbIndex *b_ = (ProbIndex *) b;

    // Comparison logic
    if (a_->prob > b_->prob) return -1; // Indicates a should be placed before b (i.e., a has higher probability value)
    if (a_->prob < b_->prob) return 1;
    return 0;                           // Indicates a and b are equal (same probability value)
}

// 3.4. top-p sampling (nucleus sampling)
// Characteristics: Sample from set where cumulative probability exceeds topp
// Application scenarios: General scenarios, recommended for default use
int sample_topp(float *probabilities, int n, float topp, ProbIndex *probindex, float coin) {
    // top-p sampling (or "nucleus sampling") samples from the smallest set of tokens that exceed probability topp. This way we never sample tokens that have very low probabilities and are less likely to go "off the rails".
    // coin is a random number in [0, 1), usually from random_f32()

    // Filter possible candidate tokens (pre-filtering): filter low-probability tokens, select token set with higher probabilities
    int n0 = 0;
    // quicksort indices in descending order of probabilities values smaller than (1 - topp) / (n - 1) cannot be part of the result so for efficiency we crop these out as candidates before sorting
    const float cutoff = (1.0f - topp) / (n - 1);
    for (int i = 0; i < n; i++) {
        if (probabilities[i] >= cutoff) {
            probindex[n0].vocab_idx = i;
            probindex[n0].prob = probabilities[i];
            n0++;
        }
    }

    // Sort by probability in descending order
    qsort(probindex, n0, sizeof(ProbIndex), compare);

    // Truncate tokens where cumulative probability exceeds topp: select smallest token set such that its cumulative probability exceeds topp
    float cumulative_prob = 0;
    int last_idx = n0 - 1; // in case of rounding errors consider all elements
    for (int i = 0; i < n0; i++) {
        cumulative_prob += probindex[i].prob;
        if (cumulative_prob > topp) {
            last_idx = i;
            break; // we've exceeded topp by including last_idx
        }
    }

    // Random sampling within truncated set
    float r = coin * cumulative_prob;
    float cdf = 0;
    for (int i = 0; i <= last_idx; i++) {
        cdf += probindex[i].prob;
        if (r < cdf) {
            return probindex[i].vocab_idx;
        }
    }
    return probindex[last_idx].vocab_idx; // in case of rounding errors
}

// 3.5. Sampler initialization and resource management
// 3.5.1. Initialize sampler parameters and allocate memory
void build_sampler(Sampler *sampler, int vocab_size, float temperature, float topp, unsigned long long rng_seed) {
    sampler->vocab_size = vocab_size;
    sampler->temperature = temperature;
    sampler->top_p = topp;
    sampler->rng_state = rng_seed;
    // buffer only used with nucleus sampling; may not need but it's ~small
    sampler->probindex = malloc(sampler->vocab_size * sizeof(ProbIndex));
}

// 3.5.2. Release memory occupied by sampler
void free_sampler(Sampler *sampler) {
    free(sampler->probindex);
}

// random float32 in [0,1)
float xorshift_f32(unsigned long long *state) {
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    unsigned int randn = (*state * 0x2545F4914F6CDD1Dull) >> 32;
    return (randn >> 8) / 16777216.0f;
}

// Sampling strategies: Support greedy, multinomial, top-p three sampling methods
// Temperature parameter: Control output randomness, temperature=0 for completely deterministic (greedy sampling)
int sample(Sampler *sampler, float *logits) {
    // sample the token given the logits and some hyperparameters
    int next;
    if (sampler->temperature == 0) {
        // greedy argmax sampling: take the token with the highest probability
        next = sample_argmax(logits, sampler->vocab_size);// Greedy sampling
    }
    else {
        // Apply temperature
        for (int q=0; q<sampler->vocab_size; q++) {
            logits[q] /= sampler->temperature;
        }
        // apply softmax to the logits to get the probabilities for next token - apply softmax to convert to probability distribution
        softmax(logits, sampler->vocab_size);// Temperature scaling

        // flip a (float) coin (this is our source of entropy for sampling)
        float coin = xorshift_f32(&sampler->rng_state);

        // we sample from this distribution to get the next token - select sampling strategy based on topp parameter
        if (sampler->top_p <= 0 || sampler->top_p >= 1) {
            next = sample_mult(logits, sampler->vocab_size, coin);// Multinomial sampling
        }
        else {
            // clamp the least likely tokens to zero
            next = sample_topp(logits, sampler->vocab_size, sampler->top_p, sampler->probindex, coin);// top-p sampling
        }
    }
    return next;
}

typedef struct {
    const char* template;
    int uses_system;
} PromptTemplate;

void render_prompt_template(PromptTemplate* prompt_template, char* system_prompt, char* user_prompt, char* output) {
    ub_assert(prompt_template != NULL);
    ub_assert(user_prompt != NULL);
    ub_assert(output != NULL);
    ub_assert((!prompt_template->uses_system && system_prompt == NULL) || (prompt_template->uses_system && system_prompt != NULL));
    if (prompt_template->uses_system && system_prompt) {
        sprintf(output, prompt_template->template, system_prompt, user_prompt);
    } else {
        sprintf(output, prompt_template->template, user_prompt);
    }
}

void generate(Transformer *transformer, Tokenizer *tokenizer, Sampler *sampler, char *prompt) {
// Parameters: transformer: model. tokenizer: tokenizer. sampler: sampler. prompt: input prompt
    char *empty_prompt = "";
    if (prompt == NULL) { prompt = empty_prompt; }  // Handle empty prompt case

    // Encode prompt to token sequence
    int num_prompt_tokens = 0;
    int *prompt_tokens = (int*)malloc((strlen(prompt)+3) * sizeof(int)); // +3 for '\0', ?BOS, ?EOS - allocate memory to store encoded tokens
    encode(tokenizer, prompt, prompt_tokens, &num_prompt_tokens);   // Use encode function to convert prompt to token sequence
    if (num_prompt_tokens < 1) {
        fprintf(stderr, "Please provide a prompt using -i <string> on the command line.\n");
        exit(1);// Check if there are valid tokens, otherwise terminate program
    }

    // start the main loop - generation main loop
    int next;        // Store next token (in the sequence)
    int token = prompt_tokens[0]; // Start from first token of prompt
    int pos = 0;     // Current sequence position

    while (pos < transformer->config.seq_len) {
        // Forward propagation to get logits for next token
        float *logits = forward(transformer, token, pos);

        // Handle prompt phase: force use of tokens in prompt
        if (pos < num_prompt_tokens - 1) {
            // if we are still processing the input prompt, force the next prompt token
            next = prompt_tokens[pos + 1];
        } else {
            // otherwise sample the next token from the logits - after prompt processing: generate tokens through sampling
            next = sample(sampler, logits);
        }
        pos++;

        // When generated token is BOS (beginning of sequence) or EOS (end of sequence), terminate generation
        if (pos >= num_prompt_tokens && (next == tokenizer->bos_token_id || next == tokenizer->eos_token_id)) { break; }

        printf("%s", tokenizer->vocab[token]); // To convert a token to string, just do the lookup
        fflush(stdout);
        token = next;// Update current token to predicted next token
    }
    printf("\n");
    free(prompt_tokens);
}

void chat(Transformer *transformer, Tokenizer *tokenizer, Sampler *sampler, PromptTemplate* prompt_template, char *system_prompt, char *cli_user_prompt) {
    char user_prompt[32768];
    char rendered_prompt[32768];
    int num_prompt_tokens = 0;
    int *prompt_tokens = (int *)malloc(32768 * sizeof(int));
    int user_idx;

    int8_t user_turn = 1;
    int next_token;
    int prev_token;
    int ctx_pos = 0;
    while (1) {
        // if context window is exceeded, clear it
        if (ctx_pos >= transformer->config.seq_len) {
            printf("\n\nCONTEXT WINDOW EXCEEDED, RESETTING\n\n");
            user_turn = 1;
            ctx_pos = 0;
        }

        // Handle user input phase.
        // If a cli prompt is passed, we catch that here and immediately respond.
        // Otherwise, we ask the user for input.
        if (user_turn) {
            // get the user prompt - read user input and encode, or use prompt from command line arguments
            if (ctx_pos == 0 && cli_user_prompt != NULL) {
                // user prompt for position 0 was passed in, use it
                strcpy(user_prompt, cli_user_prompt);
            } else {
                if (cli_user_prompt != NULL) {
                    break;
                }
                // otherwise get user prompt from stdin
                // Read one line from stdin, not including newline character \n
                printf("%s", "> ");
                if (fgets(user_prompt, sizeof(user_prompt), stdin) != NULL) {
                    size_t len = strlen(user_prompt);
                    if (len && user_prompt[len - 1] == '\n') {
                        user_prompt[len - 1] = '\0';
                    }
                }
                if (!user_prompt[0] || !strcmp(user_prompt, "exit")) {
                    break;
                }
            }

            if (!ctx_pos) {
                render_prompt_template(prompt_template, system_prompt, user_prompt, rendered_prompt);
            }


            // Encode rendered prompt to token sequence
            encode(tokenizer, rendered_prompt, prompt_tokens, &num_prompt_tokens);
            user_idx = 0; // reset the user index
            user_turn = 0;
        }

        // Process the full user prompt first (one at a time through the outer loop)
        // Then, do the reply
        int current_token = (user_idx < num_prompt_tokens) ? prompt_tokens[user_idx++] : next_token;

        // EOS token ends the Assistant turn
        if (user_idx >= num_prompt_tokens && (current_token == tokenizer->bos_token_id || current_token == tokenizer->eos_token_id)) {
            user_turn = 1;
        }

        // forward the transformer to get logits for the next token - model generates reply
        float *logits = forward(transformer, current_token, ctx_pos);
        next_token = sample(sampler, logits);
        ctx_pos++;

        // Output assistant reply (decode token to string)
        if (user_idx >= num_prompt_tokens && next_token != tokenizer->bos_token_id && next_token != tokenizer->eos_token_id) {
            printf("%s", tokenizer->vocab[next_token]); // To convert a token to string, just do the lookup
            fflush(stdout);
        }
        if (user_idx >= num_prompt_tokens && (next_token == tokenizer->bos_token_id || next_token == tokenizer->eos_token_id)) { printf("\n"); }
    }
    free(prompt_tokens);
}

void error_usage() {
    fprintf(stderr, "Usage:   qwen3 <checkpoint> [options]\n");
    fprintf(stderr, "Example: qwen3 Qwen3-4B.bin -r 1\n");
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  -t <float>  temperature in [0,inf], default 1.0\n");
    fprintf(stderr, "  -p <float>  p value in top-p (nucleus) sampling in [0,1], default 0.9\n");
    fprintf(stderr, "  -s <int>    random seed, default time(NULL)\n");
    fprintf(stderr, "  -c <int>    context window size, 0 (default) = max_seq_len\n");
    fprintf(stderr, "  -m <string> mode: generate|chat, default: chat\n");
    fprintf(stderr, "  -i <string> input prompt\n");
    fprintf(stderr, "  -y <string> system prompt in chat mode, default is none\n");
    fprintf(stderr, "  -r <int>    reasoning mode, 0 (default) = no thinking, 1 = thinking\n");
    exit(1);
}

// 7.2. Main function
int main(int argc, char *argv[]) {
    // default parameters
    // Initialize parameter default values
    char *checkpoint_path = NULL;  // e.g. out/model.bin - model checkpoint path (must be specified)
    float temperature = 1.0f;   // 0 = greedy deterministic. 1.0 = original. don't set higher - sampling temperature, controls output randomness (0 means deterministic output, higher values mean more random output)
    float topp = 0.9f;          // top-p in nucleus sampling. 1.0 = off. 0.9 works well, but slower - top-p sampling threshold: minimum cumulative probability retained during top-p sampling
    char *cli_user_prompt = NULL;        // prompt string - input prompt
    unsigned long long rng_seed = 0; // seed rng with time by default - random number seed
    char *mode = "chat";        // Running mode: generate (single-round text generation) or chat (multi-turn dialogue)
    char *system_prompt = NULL; // (optional) system prompt for dialogue mode
    int enable_thinking = 0;    // 1 enables thinking - whether to enable thinking mode
    int ctx_length = 0;         // context length - context window length

    // Parse command line arguments
    if (argc >= 2) {
        checkpoint_path = argv[1];  // First parameter is model path
    }
    else {
         error_usage(); // Show help when no parameters
    }
    // Parse options starting from second parameter (every two parameters as a group: -flag value)
    for (int i = 2; i < argc; i+=2) {
        // Parameter validity verification
        if (i + 1 >= argc) { error_usage(); } // Ensure each option has a value
        if (argv[i][0] != '-') { error_usage(); } // Options must start with -
        if (strlen(argv[i]) != 2) { error_usage(); } // must be -x (one dash, one letter) - option format must be -single letter
        // Read parameter values and convert types
        if (argv[i][1] == 't') { temperature = atof(argv[i + 1]); }
        else if (argv[i][1] == 'p') { topp = atof(argv[i + 1]); }
        else if (argv[i][1] == 's') { rng_seed = atoi(argv[i + 1]); }
        else if (argv[i][1] == 'c') { ctx_length = atoi(argv[i + 1]); }
        else if (argv[i][1] == 'i') { cli_user_prompt = argv[i + 1]; }
        else if (argv[i][1] == 'm') { mode = argv[i + 1]; }
        else if (argv[i][1] == 'y') { system_prompt = argv[i + 1]; }
        else if (argv[i][1] == 'r') { enable_thinking = atoi(argv[i + 1]); }
        else { error_usage(); }     // Call error_usage to exit when parameter format is wrong
    }

    // parameter validation/overrides - validate parameters and set reasonable default values
    if (rng_seed <= 0) rng_seed = (unsigned int)time(NULL);     // Random number seed defaults to current time
    if (temperature < 0) temperature = 0;       // Temperature cannot be negative
    if (topp < 0 || 1.0 < topp) topp = 0.9;     // top-p value limited to [0,1]

    // build the Transformer via the model .bin file - initialize Transformer model
    Transformer transformer;
    build_transformer(&transformer, checkpoint_path, ctx_length);// Call build_transformer to load model weights and set context length

    // build the Tokenizer via the tokenizer .bin file - initialize tokenizer
    Tokenizer tokenizer;
    build_tokenizer(&tokenizer, checkpoint_path, transformer.config.vocab_size, enable_thinking);// Call build_tokenizer to load vocabulary, set prompt template based on enable_thinking

    // build the Sampler - initialize sampler
    Sampler sampler;
    build_sampler(&sampler, transformer.config.vocab_size, temperature, topp, rng_seed);// Call build_sampler to set sampling strategy parameters (temperature, top-p, etc.)

    PromptTemplate prompt_template;
    prompt_template.uses_system = (system_prompt != NULL);
    prompt_template.template = enable_thinking
                                ? ((system_prompt)
                                    ? system_thinking_prompt_template
                                    : thinking_prompt_template)
                                : ((system_prompt)
                                    ? system_prompt_template
                                    : default_prompt_template);

    if (strcmp(mode, "generate") == 0) {
        generate(&transformer, &tokenizer, &sampler, cli_user_prompt);
    } else if (strcmp(mode, "chat") == 0) {
        chat(&transformer, &tokenizer, &sampler, &prompt_template, system_prompt, cli_user_prompt);
    } else {
        fprintf(stderr, "Unknown mode: %s\n", mode);
        error_usage();
    }

    free_sampler(&sampler);             // First release sampler's temporary buffer
    free_tokenizer(&tokenizer);         // Then release tokenizer's vocabulary memory
    free_transformer(&transformer);     // Finally release model's weights and runtime cache
    // Importance: avoid memory leaks, ensure all resources are properly released when program exits
    return 0;
}
