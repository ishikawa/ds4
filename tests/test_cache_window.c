#include "../ds4_cache_window.h"
#include <assert.h>
#include <stdio.h>
#include <string.h>

int main(void) {
    enum { N = 12 };
    uint8_t reserved[N] = {0}, free_bits[N] = {0};
    uint32_t stack[N], count = 0;
    uint64_t addresses[N];
    assert(ds4_window_supported(true, 4, N));
    assert(!ds4_window_supported(false, 4, N));
    assert(!ds4_window_supported(true, 6, N));
    uint64_t stride;
    uint32_t slots;
    assert(ds4_window_layout(5327906865ull, 9953280, 16384, &stride, &slots));
    assert(stride == 9961472 && slots == 535);
    assert(!ds4_window_layout(UINT64_MAX, UINT64_MAX, 16384, &stride, &slots));
    assert(!ds4_window_layout(1, 1, 0, &stride, &slots));
    for (unsigned cycle = 0; cycle < 100; cycle++) {
        for (unsigned i = 0; i < N; i++) {
            addresses[i] = 1000 + i;
            (void)ds4_window_free_push(i, N, reserved, free_bits, stack, &count);
        }
        assert(count == N);
        ds4_window_reserve(0, 4, reserved, free_bits, stack, &count);
        ds4_window_reserve(4, 4, reserved, free_bits, stack, &count);
        assert(count == 4);
        for (unsigned i = 0; i < 8; i++) {
            /* Entry invalidation zeros addresses before any pread/seed. */
            addresses[i] = 0;
            assert(!ds4_window_free_push(i, N, reserved, free_bits, stack, &count));
        }
        for (unsigned i = 0; i < count; i++) assert(!reserved[stack[i]]);
        /* Normal and cancellation cleanup share the same idempotent return. */
        ds4_window_return(0, 4, N, reserved, free_bits, stack, &count);
        ds4_window_return(4, 4, N, reserved, free_bits, stack, &count);
        ds4_window_return(0, 8, N, reserved, free_bits, stack, &count);
        assert(count == N);
        unsigned seen[N] = {0};
        for (unsigned i = 0; i < count; i++) assert(++seen[stack[i]] == 1);
        for (unsigned i = 0; i < 8; i++) assert(addresses[i] == 0 && !reserved[i]);
    }
    puts("cache window sizing, reserve/return, seed exclusion and duplicate prevention: PASS");
    return 0;
}
