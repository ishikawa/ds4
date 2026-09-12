#define _DARWIN_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#include "ds4_engram.h"

#include <assert.h>
#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static ds4_engram_layout layout(void) {
    static uint32_t map[256];
    for (int i = 0; i < 256; i++) map[i] = i / 2;
    ds4_engram_layout l = {.token_map = map, .vocab_size = 256,
                           .compressed_vocab_size = 128, .pad_id = 1};
    for (int layer = 0; layer < 2; layer++) {
        for (int j = 0; j < 4; j++)
            l.multipliers[layer][j] = 35184372088831ull - 2 * (j + 4 * layer);
        for (int j = 0; j < 24; j++) {
            l.primes[layer][j] = 16000057;
            l.rows[layer] += l.primes[layer][j];
        }
    }
    return l;
}

/* Independent full-history formulation: no rolling state or reuse of hashes. */
static void reference(const ds4_engram_layout *l, const int *tokens,
                      const uint8_t *mask, int count, uint32_t *out) {
    for (int pos = 0; pos < count; pos++) {
        for (int layer = 0; layer < 2; layer++) {
            uint32_t offset = 0;
            for (int col = 0; col < 24; col++) {
                __uint128_t hash = 0;
                bool blocked = false;
                for (int shift = 0; shift < col / 8 + 2; shift++) {
                    int p = pos - shift;
                    blocked |= p < 0 || (mask && !mask[p]);
                    uint32_t id = blocked ? l->pad_id : l->token_map[tokens[p]];
                    hash ^= (__uint128_t)id * l->multipliers[layer][shift];
                }
                *out++ = (uint32_t)(hash % l->primes[layer][col]) + offset;
                offset += l->primes[layer][col];
            }
        }
    }
}

static void test_hash(void) {
    enum {N = 513, WIDTH = 48};
    ds4_engram_layout l = layout();
    assert(ds4_engram_layout_valid(&l));
    ds4_engram_layout bad = l;
    bad.multipliers[1][3] = UINT64_MAX;
    assert(!ds4_engram_layout_valid(&bad));
    bad = l;
    bad.rows[0]--;
    assert(!ds4_engram_layout_valid(&bad));
    bad = l;
    bad.primes[0][0] = 0;
    assert(!ds4_engram_layout_valid(&bad));
    int tokens[N];
    uint8_t mask[N];
    uint32_t expected[N * WIDTH], actual[N * WIDTH];
    for (int i = 0; i < N; i++) {
        tokens[i] = (i * 97 + i / 3) % 256;
        mask[i] = (i % 17 != 0 && (i < 125 || i > 131));
    }
    for (int masked = 0; masked < 2; masked++) {
        const uint8_t *m = masked ? mask : NULL;
        reference(&l, tokens, m, N, expected);
        for (int chunk = 1; chunk <= N; chunk++) {
            ds4_engram_history h;
            ds4_engram_history_reset(&h);
            for (int i = 0; i < N; i += chunk) {
                int n = N - i < chunk ? N - i : chunk;
                ds4_engram_history snapshot = h;
                assert(ds4_engram_hash(&l, &h, tokens + i, m ? m + i : NULL,
                                       n, actual + i * WIDTH));
                ds4_engram_history after = h;
                h = snapshot;
                assert(ds4_engram_hash(&l, &h, tokens + i, m ? m + i : NULL,
                                       n, actual + i * WIDTH));
                assert(memcmp(&h, &after, sizeof(h)) == 0);
            }
            assert(memcmp(actual, expected, sizeof(actual)) == 0);
        }
    }
    ds4_engram_history h, before;
    ds4_engram_history_reset(&h);
    before = h;
    int invalid[] = {0, 256};
    assert(!ds4_engram_hash(&l, &h, invalid, NULL, 2, actual));
    assert(memcmp(&h, &before, sizeof(h)) == 0);
    invalid[1] = -1;
    assert(!ds4_engram_hash(&l, &h, invalid, NULL, 2, actual));
    assert(ds4_engram_hash(&l, &h, NULL, NULL, 0, NULL));
    assert(!ds4_engram_hash(&l, &h, tokens, NULL, SIZE_MAX, actual));
    h.tail[0] = 128;
    assert(!ds4_engram_hash(&l, &h, tokens, NULL, 1, actual));
}

static void test_rows(void) {
    char path[] = "/tmp/ds4-engram-XXXXXX";
    int fd = mkstemp(path);
    assert(fd >= 0);
    /* Exercise offsets beyond 32 bits without allocating a large file. */
    const uint64_t offset = (1ull << 33) + 32;
    uint8_t raw[3][DS4_ENGRAM_FP8_ROW_BYTES];
    for (int r = 0; r < 3; r++) {
        for (int i = 0; i < 256; i++) raw[r][i] = i;
        raw[r][127] = 0;
        raw[r][255] = 128;
        for (int i = 0; i < 8; i++) raw[r][256 + i] = 126 + r;
    }
    assert(pwrite(fd, raw, sizeof(raw), offset) == sizeof(raw));
    ds4_engram_table t;
    assert(ds4_engram_table_open(&t, path, offset, 3,
                                 DS4_ENGRAM_FP8_ROW_BYTES, false));
    assert(fcntl(t.fd, F_GETFD) & FD_CLOEXEC);
    uint32_t rows[] = {2, 0, 2, 1};
    float out[4 * 256];
    assert(ds4_engram_read(&t, rows, 4, out));
    for (int r = 0; r < 4; r++) {
        for (int i = 0; i < 256; i++) {
            int code = raw[rows[r]][i], exponent = (code >> 3) & 15;
            double v = exponent ? (1 + (code & 7) / 8.) * pow(2., exponent - 7) :
                                  (code & 7) / 512.;
            if (code & 128) v = -v;
            v *= pow(2., (int)rows[r] - 1);
            assert(out[r * 256 + i] == (float)v);
            if (!v) assert(!!signbit(out[r * 256 + i]) == !!(code & 128));
        }
    }
    enum { TOKENS = 2051, STRIDE = 48, WIDTH = DS4_ENGRAM_COLS * DS4_ENGRAM_DIM };
    uint32_t *batch_ids = malloc((size_t)TOKENS * STRIDE * sizeof(*batch_ids));
    float *batch = malloc(((size_t)TOKENS * WIDTH + 1) * sizeof(*batch));
    assert(batch_ids && batch);
    for (size_t i = 0; i < (size_t)TOKENS * STRIDE; i++) batch_ids[i] = UINT32_MAX;
    for (size_t i = 0; i < TOKENS; i++)
        for (size_t j = 0; j < DS4_ENGRAM_COLS; j++)
            batch_ids[i * STRIDE + j] = (i * 7 + j * 11) % 3;
    const size_t sizes[] = {1, 2, 31, 65, 257, 2047, 2048, 2049, TOKENS};
    float expected[3][DS4_ENGRAM_DIM];
    const uint32_t all_rows[] = {0, 1, 2};
    assert(ds4_engram_read(&t, all_rows, 3, &expected[0][0]));
    for (size_t n = 0; n < sizeof(sizes) / sizeof(*sizes); n++) {
        size_t count = sizes[n];
        batch[count * WIDTH] = 123456.0f;
        assert(ds4_engram_read_batch(&t, batch_ids, count, STRIDE, batch));
        for (size_t i = 0; i < count; i++) {
            for (size_t j = 0; j < DS4_ENGRAM_COLS; j++) {
                assert(memcmp(batch + i * WIDTH + j * DS4_ENGRAM_DIM,
                    expected[batch_ids[i * STRIDE + j]], sizeof(expected[0])) == 0);
            }
        }
        assert(batch[count * WIDTH] == 123456.0f);
    }
    assert(ds4_engram_read_batch(&t, NULL, 0, 0, NULL));
    assert(!ds4_engram_read_batch(&t, batch_ids, 1, 23, batch) && errno == EINVAL);
    assert(!ds4_engram_read_batch(&t, batch_ids, 2, SIZE_MAX, batch) && errno == EINVAL);
    assert(!ds4_engram_read_batch(&t, batch_ids, SIZE_MAX, STRIDE, batch) && errno == EINVAL);
    batch_ids[(TOKENS - 1) * STRIDE] = 3;
    batch[0] = 123456.0f;
    assert(!ds4_engram_read_batch(&t, batch_ids, TOKENS, STRIDE, batch) && errno == EINVAL);
    assert(batch[0] == 123456.0f);
    batch_ids[(TOKENS - 1) * STRIDE] = 0;
    uint32_t bad = 3;
    errno = 0;
    assert(!ds4_engram_read(&t, &bad, 1, out) && errno == EINVAL);
    assert(ds4_engram_read(&t, NULL, 0, NULL));
    uint8_t nan = 127;
    assert(pwrite(fd, &nan, 1, offset) == 1);
    bad = 0;
    assert(!ds4_engram_read(&t, &bad, 1, out) && errno == EDOM);
    assert(!ds4_engram_read_batch(&t, batch_ids, 1, STRIDE, batch) && errno == EDOM);
    assert(pwrite(fd, raw, sizeof(raw), offset) == sizeof(raw));
    nan = 255;
    assert(pwrite(fd, &nan, 1, offset + 256) == 1);
    assert(!ds4_engram_read(&t, &bad, 1, out) && errno == EDOM);
    assert(!ds4_engram_read_batch(&t, batch_ids, 1, STRIDE, batch) && errno == EDOM);
    assert(ftruncate(fd, offset + 260) == 0);
    assert(!ds4_engram_read(&t, &bad, 1, out) && errno == EIO);
    assert(!ds4_engram_read_batch(&t, batch_ids, 1, STRIDE, batch) && errno == EIO);
    free(batch_ids);
    free(batch);
    ds4_engram_table_close(&t);
    ds4_engram_table_close(&t);
    assert(!ds4_engram_table_open(&t, path, offset, 1,
                                  DS4_ENGRAM_FP8_ROW_BYTES, false));
    assert(!ds4_engram_table_open(&t, path, UINT64_MAX - 1, 3,
                                  DS4_ENGRAM_FP8_ROW_BYTES, false));
    assert(!ds4_engram_table_open(&t, path, offset, 0,
                                  DS4_ENGRAM_FP8_ROW_BYTES, false));
    close(fd);
    assert(unlink(path) == 0);
}

static void test_all_scaled_values(void) {
    char path[] = "/tmp/ds4-engram-values-XXXXXX";
    const int fd = mkstemp(path);
    assert(fd >= 0);
    assert(unlink(path) == 0);
    ds4_engram_table table = {.fd = fd, .rows = 1,
                              .row_bytes = DS4_ENGRAM_FP8_ROW_BYTES};
    const uint32_t row = 0;
    uint8_t raw[DS4_ENGRAM_FP8_ROW_BYTES];
    float out[DS4_ENGRAM_DIM];
    for (uint32_t code = 0; code < 256; code++) {
        memset(raw, code, DS4_ENGRAM_DIM);
        for (uint32_t scale = 0; scale < 256; scale++) {
            memset(raw + DS4_ENGRAM_DIM, scale,
                   DS4_ENGRAM_FP8_ROW_BYTES - DS4_ENGRAM_DIM);
            assert(pwrite(fd, raw, sizeof(raw), 0) == sizeof(raw));
            const int exponent = (code >> 3) & 15;
            double value = exponent ? (1.0 + (code & 7u) / 8.0) * pow(2.0, exponent - 7) :
                                      (code & 7u) / 512.0;
            if (code & 128u) value = -value;
            float expected = (float)(value * pow(2.0, (int)scale - 127));
            uint32_t bits;
            memcpy(&bits, &expected, sizeof(bits));
            bits = (bits + 0x7fffu + ((bits >> 16) & 1u)) & 0xffff0000u;
            memcpy(&expected, &bits, sizeof(expected));
            const bool valid = (code & 127u) != 127u && scale != 255u && isfinite(expected);
            errno = 0;
            assert(ds4_engram_read(&table, &row, 1, out) == valid);
            if (valid) {
                for (uint32_t j = 0; j < DS4_ENGRAM_DIM; j++)
                    assert(!memcmp(out + j, &expected, sizeof(expected)));
            } else assert(errno == EDOM);
        }
    }
    close(fd);
}

static float reference_f16(uint16_t h) {
    const int sign = h & 0x8000u ? -1 : 1;
    const int exponent = (h >> 10) & 31;
    const int mantissa = h & 1023;
    if (!exponent) return sign * ldexpf((float)mantissa, -24);
    if (exponent == 31) return mantissa ? NAN : sign * INFINITY;
    return sign * ldexpf((float)(1024 + mantissa), exponent - 25);
}

static void reference_q4_k(const uint8_t raw[DS4_ENGRAM_Q4_K_ROW_BYTES],
                           float out[DS4_ENGRAM_DIM]) {
    uint16_t dh, mh;
    memcpy(&dh, raw, 2);
    memcpy(&mh, raw + 2, 2);
    const float d = reference_f16(dh), dmin = reference_f16(mh);
    for (int group = 0; group < 8; group++) {
        uint8_t scale, min;
        if (group < 4) {
            scale = raw[4 + group] & 63u;
            min = raw[8 + group] & 63u;
        } else {
            scale = (raw[8 + group] & 15u) | ((raw[group] >> 6) << 4);
            min = (raw[8 + group] >> 4) | ((raw[4 + group] >> 6) << 4);
        }
        const uint8_t *q = raw + 16 + (group / 2) * 32;
        for (int i = 0; i < 32; i++) {
            const uint8_t code = group & 1 ? q[i] >> 4 : q[i] & 15u;
            out[group * 32 + i] = d * scale * code - dmin * min;
        }
    }
}

static void test_q4_k_rows(void) {
    char path[] = "/tmp/ds4-engram-q4k-XXXXXX";
    const int fd = mkstemp(path);
    assert(fd >= 0);
    const uint64_t offset = 37;
    uint8_t raw[2][DS4_ENGRAM_Q4_K_ROW_BYTES];
    for (int row = 0; row < 2; row++) {
        memset(raw[row], 0, sizeof(raw[row]));
        const uint16_t d = row ? 0x3800u : 0x3c00u;
        const uint16_t dmin = row ? 0x3400u : 0x3a00u;
        memcpy(raw[row], &d, 2);
        memcpy(raw[row] + 2, &dmin, 2);
        for (int i = 0; i < 12; i++) raw[row][4 + i] = (uint8_t)(17 * i + 13 * row);
        for (int i = 0; i < 128; i++) raw[row][16 + i] = (uint8_t)(29 * i + 7 * row);
    }
    assert(pwrite(fd, raw, sizeof(raw), offset) == sizeof(raw));
    assert(ftruncate(fd, offset + sizeof(raw) + 19) == 0);

    ds4_engram_table table;
    assert(ds4_engram_table_open(&table, path, offset, 2,
                                 DS4_ENGRAM_Q4_K_ROW_BYTES, false));
    const uint32_t ids[] = {1, 0};
    float actual[2][DS4_ENGRAM_DIM], expected[2][DS4_ENGRAM_DIM];
    reference_q4_k(raw[1], expected[0]);
    reference_q4_k(raw[0], expected[1]);
    assert(ds4_engram_read(&table, ids, 2, actual[0]));
    assert(!memcmp(actual, expected, sizeof(actual)));
    uint32_t batch_ids[DS4_ENGRAM_COLS];
    float batch[DS4_ENGRAM_COLS][DS4_ENGRAM_DIM];
    for (int i = 0; i < DS4_ENGRAM_COLS; i++) batch_ids[i] = i & 1;
    assert(ds4_engram_read_batch(&table, batch_ids, 1, DS4_ENGRAM_COLS, batch[0]));
    for (int i = 0; i < DS4_ENGRAM_COLS; i++)
        assert(!memcmp(batch[i], expected[batch_ids[i] ? 0 : 1], sizeof(batch[i])));
    ds4_engram_table_close(&table);

    assert(ftruncate(fd, offset + sizeof(raw)) == 0);
    assert(ds4_engram_table_open(&table, path, offset, 2,
                                 DS4_ENGRAM_Q4_K_ROW_BYTES, true));
    assert(ds4_engram_read(&table, ids, 2, actual[0]));
    ds4_engram_table_close(&table);
    assert(ftruncate(fd, offset + sizeof(raw) - 1) == 0);
    assert(!ds4_engram_table_open(&table, path, offset, 2,
                                  DS4_ENGRAM_Q4_K_ROW_BYTES, true));
    assert(ftruncate(fd, offset + sizeof(raw) + 1) == 0);
    assert(!ds4_engram_table_open(&table, path, offset, 2,
                                  DS4_ENGRAM_Q4_K_ROW_BYTES, true));
    assert(ftruncate(fd, offset + sizeof(raw)) == 0);
    const uint16_t infinity = 0x7c00u;
    assert(pwrite(fd, &infinity, 2, offset) == 2);
    assert(ds4_engram_table_open(&table, path, offset, 2,
                                 DS4_ENGRAM_Q4_K_ROW_BYTES, true));
    const uint32_t first = 0;
    assert(!ds4_engram_read(&table, &first, 1, actual[0]) && errno == EDOM);
    ds4_engram_table_close(&table);
    close(fd);
    assert(unlink(path) == 0);
}

int main(void) {
    test_hash();
    test_rows();
    test_all_scaled_values();
    test_q4_k_rows();
    puts("Engram hashes, history and FP8/Q4_K disk rows: PASS");
    return 0;
}
