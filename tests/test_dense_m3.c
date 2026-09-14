#define _DARWIN_C_SOURCE
#include "ds4_gpu.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

typedef struct {
    uint16_t d;
    int8_t qs[32];
} q8_0_block;

typedef struct {
    uint32_t in_dim;
    uint32_t out_dim;
} shape;

static uint32_t rng = 0x4d325f33u;

static uint32_t next_u32(void) {
    rng ^= rng << 13;
    rng ^= rng >> 17;
    rng ^= rng << 5;
    return rng;
}

static int run_shape(const shape *s, void *model, uint64_t model_size) {
    const uint64_t row_bytes = (uint64_t)(s->in_dim / 32u) * sizeof(q8_0_block);
    const uint64_t weight_bytes = (uint64_t)s->out_dim * row_bytes;
    const uint64_t input_bytes = 3u * s->in_dim * sizeof(float);
    const uint64_t output_bytes = 3u * s->out_dim * sizeof(float);
    float *input = malloc((size_t)input_bytes);
    float *legacy = malloc((size_t)output_bytes);
    float *dense = malloc((size_t)output_bytes);
    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc(input_bytes);
    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc(output_bytes);
    int ok = input && legacy && dense && x && out && weight_bytes <= model_size;
    if (!ok) goto cleanup;

    q8_0_block *weights = model;
    for (uint64_t i = 0; i < weight_bytes / sizeof(*weights); i++) {
        weights[i].d = 0x3c00;
        for (unsigned k = 0; k < 32; k++)
            weights[i].qs[k] = (int8_t)((next_u32() % 31u) - 15);
    }
    for (uint64_t i = 0; i < input_bytes / sizeof(*input); i++)
        input[i] = (float)((int32_t)(next_u32() % 2001u) - 1000) / 1024.0f;
    ok = ds4_gpu_set_model_map(model, model_size) != 0 &&
         ds4_gpu_tensor_write(x, 0, input, input_bytes) != 0;
    for (unsigned repeat = 0; ok && repeat < 3; repeat++) {
        unsetenv("DS4_METAL_V41_VERIFY_DENSE_M3");
        ok = ds4_gpu_matmul_q8_0_decode_rows_exact_tensor(
            out, model, model_size, 0, s->in_dim, s->out_dim, x, 3) != 0 &&
             ds4_gpu_tensor_read(out, 0, legacy, output_bytes) != 0;
        setenv("DS4_METAL_V41_VERIFY_DENSE_M3", "1", 1);
        ok = ok && ds4_gpu_matmul_q8_0_decode_rows_exact_tensor(
            out, model, model_size, 0, s->in_dim, s->out_dim, x, 3) != 0 &&
             ds4_gpu_tensor_read(out, 0, dense, output_bytes) != 0;
        if (ok && memcmp(legacy, dense, (size_t)output_bytes) != 0) {
            fprintf(stderr, "shape %ux%u repeat %u: memcmp FAIL\n",
                    s->in_dim, s->out_dim, repeat);
            ok = 0;
        }
    }
    fprintf(stderr, "shape %ux%u x3: %s\n", s->in_dim, s->out_dim,
            ok ? "PASS" : "FAIL");

cleanup:
    unsetenv("DS4_METAL_V41_VERIFY_DENSE_M3");
    ds4_gpu_tensor_free(out);
    ds4_gpu_tensor_free(x);
    free(dense);
    free(legacy);
    free(input);
    return ok;
}

int main(void) {
    const shape shapes[] = {
        {5120, 1280},
        {1280, 32768},
        {5120, 512},
        {4096, 8192},
        {8192, 5120},
    };
    uint64_t max_weight_bytes = 0;
    for (unsigned i = 0; i < sizeof(shapes) / sizeof(shapes[0]); i++) {
        const uint64_t row_bytes = (uint64_t)(shapes[i].in_dim / 32u) *
            sizeof(q8_0_block);
        const uint64_t bytes = (uint64_t)shapes[i].out_dim * row_bytes;
        if (bytes > max_weight_bytes) max_weight_bytes = bytes;
    }
    const uint64_t page = (uint64_t)getpagesize();
    const uint64_t model_size = (max_weight_bytes + page - 1u) / page * page;
    void *model = NULL;
    if (posix_memalign(&model, (size_t)page, (size_t)model_size) != 0 ||
        !model || !ds4_gpu_init()) {
        fprintf(stderr, "dense M3 test setup: FAIL\n");
        free(model);
        return 1;
    }
    memset(model, 0, (size_t)model_size);
    int ok = 1;
    for (unsigned i = 0; ok && i < sizeof(shapes) / sizeof(shapes[0]); i++)
        ok = run_shape(&shapes[i], model, model_size);
    ds4_gpu_cleanup();
    free(model);
    return ok ? 0 : 1;
}
