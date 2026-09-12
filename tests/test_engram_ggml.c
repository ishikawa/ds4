#define _DARWIN_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#include "ds4_engram.h"

#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

void dequantize_row_q4_K(const void *blocks, float *out, int64_t count);

int main(int argc, char **argv) {
    enum { SAMPLE_ROWS = 16 };
    if (argc != 2) {
        fprintf(stderr, "usage: %s ENGRAM_Q4K_SIDECAR\n", argv[0]);
        return 2;
    }
    struct stat st;
    assert(stat(argv[1], &st) == 0);
    assert(st.st_size > 0 &&
           st.st_size % DS4_ENGRAM_Q4_K_ROW_BYTES == 0);
    const uint64_t rows64 = (uint64_t)st.st_size /
                            DS4_ENGRAM_Q4_K_ROW_BYTES;
    assert(rows64 >= SAMPLE_ROWS && rows64 <= UINT32_MAX);

    ds4_engram_table table;
    assert(ds4_engram_table_open(&table, argv[1], 0, (uint32_t)rows64,
                                 DS4_ENGRAM_Q4_K_ROW_BYTES, true));
    uint8_t packed[SAMPLE_ROWS][DS4_ENGRAM_Q4_K_ROW_BYTES];
    float actual[SAMPLE_ROWS][DS4_ENGRAM_DIM];
    float expected[SAMPLE_ROWS][DS4_ENGRAM_DIM];
    uint32_t rows[SAMPLE_ROWS];
    for (uint32_t i = 0; i < SAMPLE_ROWS; i++) rows[i] = i;
    assert(pread(table.fd, packed, sizeof(packed), 0) ==
           (ssize_t)sizeof(packed));
    assert(ds4_engram_read(&table, rows, SAMPLE_ROWS, actual[0]));
    dequantize_row_q4_K(packed, expected[0],
                        (int64_t)SAMPLE_ROWS * DS4_ENGRAM_DIM);
    assert(!memcmp(actual, expected, sizeof(actual)));
    ds4_engram_table_close(&table);
    printf("Q4_K sidecar first %d rows match ggml: PASS\n", SAMPLE_ROWS);
    return 0;
}
