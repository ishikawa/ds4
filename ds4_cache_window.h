#ifndef DS4_CACHE_WINDOW_H
#define DS4_CACHE_WINDOW_H
#include <stdbool.h>
#include <stdint.h>

static inline bool ds4_window_supported(bool slabs, uint32_t slots, uint32_t capacity) {
    return slabs && slots != 0 && slots < capacity / 2u + capacity % 2u;
}

/* Pure ownership bookkeeping; reserved slots must never enter the free stack. */
static inline bool ds4_window_free_push(uint32_t slot, uint32_t capacity,
        const uint8_t *reserved, uint8_t *free_bits, uint32_t *stack, uint32_t *count) {
    if (slot >= capacity || reserved[slot] || free_bits[slot] || *count >= capacity)
        return false;
    free_bits[slot] = 1;
    stack[(*count)++] = slot;
    return true;
}
static inline void ds4_window_reserve(uint32_t first, uint32_t count,
        uint8_t *reserved, uint8_t *free_bits, uint32_t *stack, uint32_t *nfree) {
    for (uint32_t i = first; i < first + count; i++) {
        reserved[i] = 1;
        free_bits[i] = 0;
    }
    uint32_t keep = 0;
    for (uint32_t i = 0; i < *nfree; i++)
        if (!reserved[stack[i]]) stack[keep++] = stack[i];
    *nfree = keep;
}
static inline void ds4_window_return(uint32_t first, uint32_t count, uint32_t capacity,
        uint8_t *reserved, uint8_t *free_bits, uint32_t *stack, uint32_t *nfree) {
    for (uint32_t i = first; i < first + count; i++) {
        if (!reserved[i]) continue;
        reserved[i] = 0;
        (void)ds4_window_free_push(i, capacity, reserved, free_bits, stack, nfree);
    }
}
static inline bool ds4_window_layout(uint64_t packed, uint64_t expert,
        uint64_t page, uint64_t *stride, uint32_t *slots) {
    if (!packed || !expert || !page || expert > UINT64_MAX - (page - 1)) return false;
    *stride = (expert + page - 1) / page * page;
    const uint64_t n = packed / *stride + (packed % *stride != 0);
    if (n > UINT32_MAX || n > UINT64_MAX / *stride) return false;
    *slots = (uint32_t)n;
    return true;
}
#endif
