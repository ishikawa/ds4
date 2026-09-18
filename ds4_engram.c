#define _DARWIN_C_SOURCE
#define _POSIX_C_SOURCE 200809L

#include "ds4_engram.h"

#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>
#ifdef __APPLE__
#include <dispatch/dispatch.h>
#else
#include <pthread.h>
#endif

static bool path_has_parent(const char *path, size_t len) {
    size_t start = 0;
    while (start < len) {
        while (start < len && path[start] == '/') start++;
        size_t end = start;
        while (end < len && path[end] != '/') end++;
        if (end - start == 2 && path[start] == '.' && path[start + 1] == '.') return true;
        start = end;
    }
    return false;
}

char *ds4_engram_resolve_path(const char *gguf_path, const char *path,
                              size_t path_len, bool allow_outside) {
    if (!gguf_path || !path || !path_len || memchr(path, '\0', path_len)) return NULL;
    const char *slash = strrchr(gguf_path, '/');
    const size_t directory = slash ? (size_t)(slash - gguf_path + 1) : 0;
    const bool absolute = path[0] == '/';
    if (!allow_outside && (absolute || path_has_parent(path, path_len))) return NULL;
    if (path_len > SIZE_MAX - (absolute ? 1 : directory + 1)) return NULL;
    char *result = malloc((absolute ? 0 : directory) + path_len + 1);
    if (!result) return NULL;
    size_t pos = 0;
    if (!absolute && directory) { memcpy(result, gguf_path, directory); pos = directory; }
    memcpy(result + pos, path, path_len); result[pos + path_len] = '\0';
    if (!allow_outside) {
        char directory_path[PATH_MAX], resolved_directory[PATH_MAX], resolved[PATH_MAX];
        if (directory >= sizeof(directory_path)) { free(result); return NULL; }
        if (directory) { memcpy(directory_path, gguf_path, directory); directory_path[directory] = '\0'; }
        else strcpy(directory_path, ".");
        if (!realpath(directory_path, resolved_directory) || !realpath(result, resolved)) {
            free(result); return NULL;
        }
        size_t n = strlen(resolved_directory);
        if (strncmp(resolved, resolved_directory, n) ||
            (resolved[n] != '\0' && resolved[n] != '/')) { free(result); return NULL; }
    }
    return result;
}

bool ds4_engram_layout_valid(const ds4_engram_layout *l) {
    if (!l || !l->token_map || !l->vocab_size ||
        !l->compressed_vocab_size || l->compressed_vocab_size > INT32_MAX ||
        l->pad_id >= l->compressed_vocab_size) return false;
    for (uint32_t i = 0; i < l->vocab_size; i++)
        if (l->token_map[i] >= l->compressed_vocab_size) return false;
    for (int layer = 0; layer < DS4_ENGRAM_LAYERS; layer++) {
        for (int i = 0; i < DS4_ENGRAM_NGRAM; i++) {
            uint64_t m = l->multipliers[layer][i];
            if (!(m & 1) || m > (uint64_t)INT64_MAX / l->compressed_vocab_size)
                return false;
        }
        uint64_t total = 0;
        for (int i = 0; i < DS4_ENGRAM_COLS; i++) {
            if (l->primes[layer][i] < 2) return false;
            total += l->primes[layer][i];
        }
        if (total != l->rows[layer]) return false;
    }
    return true;
}

void ds4_engram_history_reset(ds4_engram_history *h) {
    for (int i = 0; i < DS4_ENGRAM_NGRAM - 1; i++) h->tail[i] = DS4_ENGRAM_DEAD;
}

bool ds4_engram_hash(const ds4_engram_layout *l, ds4_engram_history *h,
                     const int *tokens, const uint8_t *mask, size_t count,
                     uint32_t *rows) {
    if (!l || !h || !l->token_map || (count && (!tokens || !rows)) ||
        count > SIZE_MAX / (DS4_ENGRAM_LAYERS * DS4_ENGRAM_COLS * sizeof(*rows)))
        return false;
    for (int i = 0; i < DS4_ENGRAM_NGRAM - 1; i++) {
        if (h->tail[i] < DS4_ENGRAM_DEAD ||
            (h->tail[i] >= 0 && (uint32_t)h->tail[i] >= l->compressed_vocab_size))
            return false;
    }
    for (size_t i = 0; i < count; i++) {
        if (tokens[i] < 0 || (uint32_t)tokens[i] >= l->vocab_size) return false;
    }
    for (size_t i = 0; i < count; i++) {
        int32_t current = mask && !mask[i] ? DS4_ENGRAM_DEAD :
                          (int32_t)l->token_map[tokens[i]];
        uint32_t ids[DS4_ENGRAM_NGRAM];
        bool blocked = false;
        for (int j = 0; j < DS4_ENGRAM_NGRAM; j++) {
            int32_t id = j ? h->tail[j - 1] : current;
            blocked |= id == DS4_ENGRAM_DEAD;
            ids[j] = blocked ? l->pad_id : (uint32_t)id;
        }
        for (int layer = 0; layer < DS4_ENGRAM_LAYERS; layer++) {
            uint64_t hash = (uint64_t)ids[0] * l->multipliers[layer][0];
            uint32_t offset = 0;
            for (int j = 1; j < DS4_ENGRAM_NGRAM; j++) {
                hash ^= (uint64_t)ids[j] * l->multipliers[layer][j];
                for (int head = 0; head < DS4_ENGRAM_HEADS; head++) {
                    int col = (j - 1) * DS4_ENGRAM_HEADS + head;
                    uint32_t prime = l->primes[layer][col];
                    *rows++ = (uint32_t)(hash % prime) + offset;
                    offset += prime;
                }
            }
        }
        for (int j = DS4_ENGRAM_NGRAM - 2; j > 0; j--) h->tail[j] = h->tail[j - 1];
        h->tail[0] = current;
    }
    return true;
}

bool ds4_engram_table_open(ds4_engram_table *t, const char *path,
                           uint64_t offset, uint32_t rows, uint32_t row_bytes,
                           bool exact_size) {
    if (!t) return false;
    *t = (ds4_engram_table){.fd = -1};
    uint64_t bytes = (uint64_t)rows * row_bytes;
    if (!path || !rows ||
        (row_bytes != DS4_ENGRAM_FP8_ROW_BYTES &&
         row_bytes != DS4_ENGRAM_Q4_K_ROW_BYTES) ||
        offset > INT64_MAX || bytes > INT64_MAX - offset) {
        errno = EINVAL;
        return false;
    }
    int fd = open(path, O_RDONLY | O_CLOEXEC);
    if (fd < 0) return false;
    struct stat st;
    if (fstat(fd, &st) != 0) goto fail;
    if (!S_ISREG(st.st_mode) || st.st_size < 0 ||
        offset + bytes > (uint64_t)st.st_size ||
        (exact_size && offset + bytes != (uint64_t)st.st_size)) {
        errno = EINVAL;
        goto fail;
    }
#ifdef __APPLE__
    if (fcntl(fd, F_NOCACHE, 1) != 0 || fcntl(fd, F_RDAHEAD, 0) != 0) goto fail;
#endif
    *t = (ds4_engram_table){.fd = fd, .offset = offset, .rows = rows,
                            .row_bytes = row_bytes};
    return true;
fail: {
        int saved = errno;
        close(fd);
        errno = saved;
        return false;
    }
}

void ds4_engram_table_close(ds4_engram_table *t) {
    if (!t) return;
    if (t->fd >= 0) close(t->fd);
    *t = (ds4_engram_table){.fd = -1};
}

static bool read_row(int fd, uint64_t offset, uint8_t *row, size_t row_bytes) {
    size_t done = 0;
    while (done < row_bytes) {
        ssize_t n = pread(fd, row + done, row_bytes - done,
                          (off_t)(offset + done));
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) {
            if (n == 0) errno = EIO;
            return false;
        }
        done += (size_t)n;
    }
    return true;
}

typedef struct {
    uint32_t state[8];
    uint64_t bytes;
    uint8_t block[64];
    size_t used;
} engram_sha256;

static const uint32_t engram_sha256_k[64] = {
    0x428a2f98u,0x71374491u,0xb5c0fbcfu,0xe9b5dba5u,0x3956c25bu,0x59f111f1u,0x923f82a4u,0xab1c5ed5u,
    0xd807aa98u,0x12835b01u,0x243185beu,0x550c7dc3u,0x72be5d74u,0x80deb1feu,0x9bdc06a7u,0xc19bf174u,
    0xe49b69c1u,0xefbe4786u,0x0fc19dc6u,0x240ca1ccu,0x2de92c6fu,0x4a7484aau,0x5cb0a9dcu,0x76f988dau,
    0x983e5152u,0xa831c66du,0xb00327c8u,0xbf597fc7u,0xc6e00bf3u,0xd5a79147u,0x06ca6351u,0x14292967u,
    0x27b70a85u,0x2e1b2138u,0x4d2c6dfcu,0x53380d13u,0x650a7354u,0x766a0abbu,0x81c2c92eu,0x92722c85u,
    0xa2bfe8a1u,0xa81a664bu,0xc24b8b70u,0xc76c51a3u,0xd192e819u,0xd6990624u,0xf40e3585u,0x106aa070u,
    0x19a4c116u,0x1e376c08u,0x2748774cu,0x34b0bcb5u,0x391c0cb3u,0x4ed8aa4au,0x5b9cca4fu,0x682e6ff3u,
    0x748f82eeu,0x78a5636fu,0x84c87814u,0x8cc70208u,0x90befffau,0xa4506cebu,0xbef9a3f7u,0xc67178f2u,
};

static uint32_t engram_rotr32(uint32_t x, unsigned n) {
    return (x >> n) | (x << (32u - n));
}

static void engram_sha256_block(engram_sha256 *s, const uint8_t block[64]) {
    uint32_t w[64];
    for (unsigned i = 0; i < 16; i++)
        w[i] = (uint32_t)block[4*i] << 24 | (uint32_t)block[4*i+1] << 16 |
               (uint32_t)block[4*i+2] << 8 | block[4*i+3];
    for (unsigned i = 16; i < 64; i++) {
        uint32_t a = engram_rotr32(w[i-15],7) ^ engram_rotr32(w[i-15],18) ^ (w[i-15] >> 3);
        uint32_t b = engram_rotr32(w[i-2],17) ^ engram_rotr32(w[i-2],19) ^ (w[i-2] >> 10);
        w[i] = w[i-16] + a + w[i-7] + b;
    }
    uint32_t a=s->state[0],b=s->state[1],c=s->state[2],d=s->state[3];
    uint32_t e=s->state[4],f=s->state[5],g=s->state[6],h=s->state[7];
    for (unsigned i = 0; i < 64; i++) {
        uint32_t s1=engram_rotr32(e,6)^engram_rotr32(e,11)^engram_rotr32(e,25);
        uint32_t ch=(e&f)^(~e&g), t1=h+s1+ch+engram_sha256_k[i]+w[i];
        uint32_t s0=engram_rotr32(a,2)^engram_rotr32(a,13)^engram_rotr32(a,22);
        uint32_t t2=s0+((a&b)^(a&c)^(b&c));
        h=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
    }
    s->state[0]+=a;s->state[1]+=b;s->state[2]+=c;s->state[3]+=d;
    s->state[4]+=e;s->state[5]+=f;s->state[6]+=g;s->state[7]+=h;
}

static void engram_sha256_init(engram_sha256 *s) {
    const uint32_t init[8]={0x6a09e667u,0xbb67ae85u,0x3c6ef372u,0xa54ff53au,
        0x510e527fu,0x9b05688cu,0x1f83d9abu,0x5be0cd19u};
    memcpy(s->state,init,sizeof(init)); s->bytes=0; s->used=0;
}

static void engram_sha256_update(engram_sha256 *s, const void *data, size_t len) {
    const uint8_t *p=data; s->bytes+=len;
    while (len) { size_t n=64-s->used; if(n>len)n=len; memcpy(s->block+s->used,p,n);
        s->used+=n;p+=n;len-=n;if(s->used==64){engram_sha256_block(s,s->block);s->used=0;} }
}

static void engram_sha256_final(engram_sha256 *s, uint8_t out[32]) {
    uint64_t bits=s->bytes*8u;s->block[s->used++]=0x80;
    if(s->used>56){memset(s->block+s->used,0,64-s->used);engram_sha256_block(s,s->block);s->used=0;}
    memset(s->block+s->used,0,56-s->used);for(unsigned i=0;i<8;i++)s->block[63-i]=(uint8_t)(bits>>(8*i));
    engram_sha256_block(s,s->block);for(unsigned i=0;i<8;i++){out[4*i]=(uint8_t)(s->state[i]>>24);
        out[4*i+1]=(uint8_t)(s->state[i]>>16);out[4*i+2]=(uint8_t)(s->state[i]>>8);out[4*i+3]=(uint8_t)s->state[i];}
}

bool ds4_engram_table_sample_sha256(const ds4_engram_table *t,
                                    const char expected_hex[65]) {
    if (!t || t->fd < 0 || t->row_bytes != DS4_ENGRAM_Q4_K_ROW_BYTES ||
        !expected_hex || strlen(expected_hex) != 64) { errno = EINVAL; return false; }
    engram_sha256 sha; engram_sha256_init(&sha);
    uint8_t row[DS4_ENGRAM_Q4_K_ROW_BYTES];
    const uint32_t count = t->rows <= 384 ? t->rows : 384;
    uint32_t previous = UINT32_MAX;
    for (uint32_t i = 0; i < count; i++) {
        uint32_t index;
        if (t->rows <= 384) index = i;
        else if (i < 64) index = i;
        else if (i < 320) index = 64u + (uint32_t)(((uint64_t)(2u*(i-64u)+1u)*(t->rows-128u))/512u);
        else index = t->rows - 64u + i - 320u;
        if (index == previous) continue;
        previous = index;
        if (!read_row(t->fd, t->offset + (uint64_t)index*t->row_bytes, row, sizeof(row))) return false;
        engram_sha256_update(&sha,row,sizeof(row));
    }
    uint8_t digest[32]; engram_sha256_final(&sha,digest);
    static const char hex[]="0123456789abcdef"; char actual[65];
    for(unsigned i=0;i<32;i++){actual[2*i]=hex[digest[i]>>4];actual[2*i+1]=hex[digest[i]&15];}
    actual[64]='\0';
    if (memcmp(actual,expected_hex,65)) { errno=EINVAL; return false; }
    return true;
}

static float e4m3(uint8_t byte) {
    int exponent = (byte >> 3) & 15, mantissa = byte & 7;
    float value = exponent ? ldexpf((float)(8 + mantissa), exponent - 10) :
                             ldexpf((float)mantissa, -9);
    return byte & 128 ? -value : value;
}

static float f16(uint16_t half) {
    const uint32_t sign = (uint32_t)(half & 0x8000u) << 16;
    uint32_t exponent = (half >> 10) & 31u, mantissa = half & 1023u, bits;
    if (!exponent) {
        if (!mantissa) bits = sign;
        else {
            exponent = 113;
            while (!(mantissa & 1024u)) {
                mantissa <<= 1;
                exponent--;
            }
            bits = sign | (exponent << 23) | ((mantissa & 1023u) << 13);
        }
    } else if (exponent == 31) {
        bits = sign | 0x7f800000u | (mantissa << 13);
    } else {
        bits = sign | ((exponent + 112u) << 23) | (mantissa << 13);
    }
    float value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

static void q4_k_scale_min(int index, const uint8_t *packed,
                           uint8_t *scale, uint8_t *min) {
    if (index < 4) {
        *scale = packed[index] & 63u;
        *min = packed[index + 4] & 63u;
    } else {
        *scale = (packed[index + 4] & 15u) | ((packed[index - 4] >> 6) << 4);
        *min = (packed[index + 4] >> 4) | ((packed[index] >> 6) << 4);
    }
}

static bool dequantize_q4_k(const uint8_t raw[DS4_ENGRAM_Q4_K_ROW_BYTES],
                            float out[DS4_ENGRAM_DIM]) {
    uint16_t dh, mh;
    memcpy(&dh, raw, sizeof(dh));
    memcpy(&mh, raw + 2, sizeof(mh));
    const float d = f16(dh), dmin = f16(mh);
    if (!isfinite(d) || !isfinite(dmin)) {
        errno = EDOM;
        return false;
    }
    const uint8_t *scales = raw + 4, *q = raw + 16;
    for (int group = 0; group < 4; group++) {
        uint8_t sc, min;
        q4_k_scale_min(2 * group, scales, &sc, &min);
        const float d1 = d * sc, m1 = dmin * min;
        q4_k_scale_min(2 * group + 1, scales, &sc, &min);
        const float d2 = d * sc, m2 = dmin * min;
        for (int i = 0; i < 32; i++) out[group * 64 + i] = d1 * (q[i] & 15u) - m1;
        for (int i = 0; i < 32; i++) out[group * 64 + 32 + i] = d2 * (q[i] >> 4) - m2;
        q += 32;
    }
    return true;
}

bool ds4_engram_read(const ds4_engram_table *t, const uint32_t *rows,
                     size_t count, float *out) {
    if (!t || t->fd < 0 || (count && (!rows || !out)) ||
        count > SIZE_MAX / (DS4_ENGRAM_DIM * sizeof(*out))) {
        errno = EINVAL;
        return false;
    }
    for (size_t i = 0; i < count; i++) {
        if (rows[i] >= t->rows) {
            errno = EINVAL;
            return false;
        }
    }
    uint8_t raw[DS4_ENGRAM_MAX_ROW_BYTES];
    for (size_t i = 0; i < count; i++) {
        if (!read_row(t->fd, t->offset + (uint64_t)rows[i] * t->row_bytes,
                      raw, t->row_bytes)) return false;
        if (t->row_bytes == DS4_ENGRAM_Q4_K_ROW_BYTES) {
            if (!dequantize_q4_k(raw, out + i * DS4_ENGRAM_DIM)) return false;
            continue;
        }
        if (t->row_bytes != DS4_ENGRAM_FP8_ROW_BYTES) {
            errno = EINVAL;
            return false;
        }
        for (int j = 0; j < DS4_ENGRAM_DIM; j++) {
            uint8_t code = raw[j], scale = raw[DS4_ENGRAM_DIM + j / 32];
            if ((code & 127) == 127 || scale == 255) {
                errno = EDOM;
                return false;
            }
            float value = ldexpf(e4m3(code), (int)scale - 127);
            uint32_t bits;
            memcpy(&bits, &value, sizeof(bits));
            bits = (bits + 0x7fffu + ((bits >> 16) & 1u)) & 0xffff0000u;
            memcpy(&value, &bits, sizeof(value));
            if (!isfinite(value)) {
                errno = EDOM;
                return false;
            }
            out[i * DS4_ENGRAM_DIM + j] = value;
        }
    }
    return true;
}

typedef struct {
    uint32_t row, output;
} engram_request;

static int request_order(const void *a, const void *b) {
    const engram_request *x = a, *y = b;
    return (x->row > y->row) - (x->row < y->row);
}

enum { ENGRAM_READERS = 16 };

typedef struct {
    const ds4_engram_table *table;
    const engram_request *request;
    float *out;
    size_t count, readers;
    int error[ENGRAM_READERS];
} engram_batch;

static void read_batch_part(void *context, size_t part) {
    engram_batch *batch = context;
    const engram_request *request = batch->request;
    const size_t begin = batch->count * part / batch->readers;
    const size_t end = batch->count * (part + 1) / batch->readers;
    const float *previous = NULL;
    for (size_t i = begin; i < end; i++) {
        float *dst = batch->out + (size_t)request[i].output * DS4_ENGRAM_DIM;
        if (i > begin && request[i].row == request[i - 1].row) {
            memcpy(dst, previous, DS4_ENGRAM_DIM * sizeof(*dst));
        } else {
            if (!ds4_engram_read(batch->table, &request[i].row, 1, dst)) {
                batch->error[part] = errno ? errno : EIO;
                return;
            }
            previous = dst;
        }
    }
}

#ifndef __APPLE__
typedef struct {
    engram_batch *batch;
    size_t part;
} engram_reader;

static void *read_batch_thread(void *context) {
    engram_reader *reader = context;
    read_batch_part(reader->batch, reader->part);
    return NULL;
}
#endif

bool ds4_engram_read_batch(const ds4_engram_table *t, const uint32_t *rows,
                           size_t tokens, size_t stride, float *out) {
    if (!t || t->fd < 0 || (tokens && (!rows || !out || stride < DS4_ENGRAM_COLS)) ||
        tokens > SIZE_MAX / (DS4_ENGRAM_COLS * DS4_ENGRAM_DIM * sizeof(*out)) ||
        (tokens && tokens - 1 > (SIZE_MAX / sizeof(*rows) - DS4_ENGRAM_COLS) / stride)) {
        errno = EINVAL;
        return false;
    }
    for (size_t i = 0; i < tokens; i++) {
        for (size_t j = 0; j < DS4_ENGRAM_COLS; j++) {
            if (rows[i * stride + j] >= t->rows) {
                errno = EINVAL;
                return false;
            }
        }
    }
    if (!tokens) return true;
    enum { BATCH_TOKENS = 2048 };
    const size_t cap = tokens < BATCH_TOKENS ? tokens : BATCH_TOKENS;
    engram_request *request = malloc(cap * DS4_ENGRAM_COLS * sizeof(*request));
    if (!request) return false;
    bool ok = true;
    for (size_t start = 0; ok && start < tokens; start += cap) {
        const size_t n = tokens - start < cap ? tokens - start : cap;
        const size_t count = n * DS4_ENGRAM_COLS;
        for (size_t i = 0; i < count; i++) {
            request[i] = (engram_request){
                rows[(start + i / DS4_ENGRAM_COLS) * stride + i % DS4_ENGRAM_COLS],
                (uint32_t)i
            };
        }
        qsort(request, count, sizeof(*request), request_order);
        engram_batch batch = {.table = t, .request = request, .count = count,
            .out = out + start * DS4_ENGRAM_COLS * DS4_ENGRAM_DIM, .readers = 1};
        /* Fixed concurrency hides random-read latency without caching the table.
         * Each worker owns disjoint output rows; all finish before GPU use. */
        if (count >= 256) {
            batch.readers = ENGRAM_READERS;
#ifdef __APPLE__
            dispatch_apply_f(batch.readers,
                dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), &batch, read_batch_part);
#else
            pthread_t threads[ENGRAM_READERS - 1];
            engram_reader readers[ENGRAM_READERS - 1];
            size_t started = 0;
            for (size_t part = 1; part < batch.readers; part++) {
                readers[started] = (engram_reader){&batch, part};
                if (pthread_create(&threads[started], NULL, read_batch_thread,
                                   &readers[started])) break;
                started++;
            }
            read_batch_part(&batch, 0);
            /* Thread exhaustion only reduces concurrency, not correctness. */
            for (size_t part = started + 1; part < batch.readers; part++)
                read_batch_part(&batch, part);
            for (size_t part = 0; part < started; part++)
                if (pthread_join(threads[part], NULL)) abort();
#endif
        } else
        read_batch_part(&batch, 0);
        for (size_t i = 0; i < batch.readers; i++) {
            if (batch.error[i]) {
                errno = batch.error[i];
                ok = false;
                break;
            }
        }
    }
    int saved = errno;
    free(request);
    errno = saved;
    return ok;
}
