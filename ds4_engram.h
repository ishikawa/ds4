#ifndef DS4_ENGRAM_H
#define DS4_ENGRAM_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

enum {
    DS4_ENGRAM_LAYERS = 2,
    DS4_ENGRAM_NGRAM = 4,
    DS4_ENGRAM_HEADS = 8,
    DS4_ENGRAM_COLS = 24,
    DS4_ENGRAM_DIM = 256,
    DS4_ENGRAM_FP8_ROW_BYTES = 264,
    DS4_ENGRAM_Q4_K_ROW_BYTES = 144,
    DS4_ENGRAM_MAX_ROW_BYTES = DS4_ENGRAM_FP8_ROW_BYTES,
    DS4_ENGRAM_DEAD = -1
};

typedef struct {
    const uint32_t *token_map;
    uint32_t vocab_size, compressed_vocab_size, pad_id;
    uint32_t rows[DS4_ENGRAM_LAYERS];
    uint64_t multipliers[DS4_ENGRAM_LAYERS][DS4_ENGRAM_NGRAM];
    uint32_t primes[DS4_ENGRAM_LAYERS][DS4_ENGRAM_COLS];
} ds4_engram_layout;

/* Newest first; copying this value also snapshots the complete hash state. */
typedef struct {
    int32_t tail[DS4_ENGRAM_NGRAM - 1];
} ds4_engram_history;

bool ds4_engram_layout_valid(const ds4_engram_layout *layout);
void ds4_engram_history_reset(ds4_engram_history *history);
/* Validate the layout once at model load. Mask zero breaks all n-grams that
 * cross that position. Output is [token][layer][column], including masked rows;
 * the graph must suppress Engram at masked positions, as in the reference. */
bool ds4_engram_hash(const ds4_engram_layout *layout,
                     ds4_engram_history *history,
                     const int *tokens, const uint8_t *mask, size_t count,
                     uint32_t *rows);

typedef struct {
    int fd;
    uint64_t offset;
    uint32_t rows;
    uint32_t row_bytes;
} ds4_engram_table;

/* A separate uncached file descriptor, never an mmap or Metal model view.
 * exact_size is used for standalone sidecars; embedded tables may have data
 * after their extent in the same GGUF. */
bool ds4_engram_table_open(ds4_engram_table *table, const char *path,
                           uint64_t offset, uint32_t rows, uint32_t row_bytes,
                           bool exact_size);
void ds4_engram_table_close(ds4_engram_table *table);
/* Output uses F32 storage; FP8 rows retain the reference's BF16 rounding while
 * Q4_K rows are directly dequantized. No whole-table allocation. */
bool ds4_engram_read(const ds4_engram_table *table, const uint32_t *rows,
                     size_t count, float *out);
/* Read COLS rows per token, restoring token order after deduplicated disk reads.
 * Input stride is in row IDs; output is packed [token][COLS][DIM]. Temporary
 * storage is bounded to 384 KiB, independent of the table and prefix size.
 * On macOS, large batches use bounded concurrent pread readers. */
bool ds4_engram_read_batch(const ds4_engram_table *table, const uint32_t *rows,
                           size_t tokens, size_t stride, float *out);

#endif
