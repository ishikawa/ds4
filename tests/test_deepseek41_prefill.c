/* Run model checks only on dedicated GPU hosts with enough memory. */
#include "../ds4.c"
#include <assert.h>
#include <sys/resource.h>

#define CHECK(x) do { if (!(x)) { \
    fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x); goto done; \
} } while (0)

static int check_dispatch(void) {
    int rc = 1;
    g_ds4_shape = DS4_SHAPE_FLASH41;
    ds4_weights weights = {0};
    ds4_tensor gate = {.type = DS4_TENSOR_IQ2_XXS}, down = {.type = DS4_TENSOR_Q2_K};
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        weights.layer[il].ffn_gate_exps = weights.layer[il].ffn_up_exps = &gate;
        weights.layer[il].ffn_down_exps = &down;
    }
    ds41_gpu_graph g = {.ctx = 131072, .prefill_cap = 8192,
        .carry_cap = 32768, .streaming = true, .tp_world = 1};
    const uint32_t half = DS4_N_LAYER * DS4_N_EXPERT / 2u;
    const uint32_t saved = ds4_gpu_stream_expert_cache_configured_count();
    const char *seed_cap_value = getenv("DS4_METAL_V41_MAX_PREFILL_CACHE_SEED_EXPERTS_PER_LAYER");
    char *saved_seed_cap = seed_cap_value ? strdup(seed_cap_value) : NULL;
    CHECK(!seed_cap_value || saved_seed_cap);
    CHECK(unsetenv("DS4_METAL_V41_MAX_PREFILL_CACHE_SEED_EXPERTS_PER_LAYER") == 0);
    CHECK(!ds41_prefill_seed_tile(0, 0, 4096, 8193));
    CHECK(ds41_prefill_seed_tile(0, 4096, 4096, 8193));
    CHECK(!ds41_prefill_seed_tile(0, 8192, 1, 8193));
    CHECK(ds41_prefill_seed_tile(8192, 8192, 1, 8193));
    CHECK(!ds41_prefill_seed_tile(0, 0, 4096, 12288));
    CHECK(ds41_prefill_seed_tile(0, 8192, 4096, 12288));
    ds4_gpu_set_ssd_streaming(true);
    CHECK(ds41_prefill_seed_target(DS4_N_LAYER * 64u) == 64u);
    CHECK(setenv("DS4_METAL_V41_MAX_PREFILL_CACHE_SEED_EXPERTS_PER_LAYER", "24", 1) == 0);
    CHECK(ds41_prefill_seed_target(DS4_N_LAYER * 64u) == 24u);
    CHECK(setenv("DS4_METAL_V41_MAX_PREFILL_CACHE_SEED_EXPERTS_PER_LAYER", "0", 1) == 0);
    CHECK(ds41_prefill_seed_target(DS4_N_LAYER * 64u) == 0u);
    CHECK(setenv("DS4_METAL_V41_MAX_PREFILL_CACHE_SEED_EXPERTS_PER_LAYER", "invalid", 1) == 0);
    CHECK(ds41_prefill_seed_target(DS4_N_LAYER * 64u) == 64u);
    CHECK(unsetenv("DS4_METAL_V41_MAX_PREFILL_CACHE_SEED_EXPERTS_PER_LAYER") == 0);
    const uint32_t remaining[] = {1, 31, 32, 33, 127, 128, 255, 256, 257, 511, 512, 513, 1023, 1024,
        2047, 2048, 2049, 4095, 4096, 4097, 8191, 8192, 8193,
        16383, 16384, 16385, 32767, 32768, 32769, 40959, 40960, 40961, 65536};
    const uint32_t cold[] = {1, 1, 1, 1, 1, 1, 1, 256, 257, 511, 512, 513, 1023, 1024,
        2047, 2048, 2049, 4095, 4096, 4097, 8191, 8192, 8193,
        16383, 16384, 16385, 32767, 32768, 24577, 32767, 32768, 32768, 32768};
    for (uint32_t cache = half - 1; cache <= half; cache++) {
        ds4_gpu_set_streaming_expert_cache_budget(cache);
        for (uint32_t warm = 0; warm < 2; warm++) {
            g.pos = warm;
            for (size_t i = 0; i < sizeof(remaining) / sizeof(*remaining); i++) {
                uint32_t expected = cold[i];
#ifdef __APPLE__
                if (warm && cache == half && remaining[i] < 1024) expected = 1;
#elif !defined(DS4_ROCM_BUILD)
                if (remaining[i] > 2048 && remaining[i] < 8192 && remaining[i] % 2048 >= 256)
                    expected = remaining[i];
#endif
                if (ds41_prefill_count(&g, remaining[i]) != expected)
                    fprintf(stderr, "dispatch cache=%u configured=%u warm=%u remaining=%u expected=%u actual=%u\n",
                        cache, ds4_gpu_stream_expert_cache_configured_count(), warm,
                        remaining[i], expected, ds41_prefill_count(&g, remaining[i]));
                CHECK(ds41_prefill_count(&g, remaining[i]) == expected);
                uint32_t small = 0;
#ifndef __APPLE__
                if (remaining[i] >= 2 && remaining[i] < 256)
                    small = remaining[i] < 8 ? remaining[i] : 8;
#endif
                CHECK(ds41_short_prefill_count(&g, &weights, remaining[i]) == small);
            }
        }
    }
    g.pos = 0;
#if !defined(__APPLE__) && !defined(DS4_ROCM_BUILD)
    CHECK(ds41_prefill_count(&g, 2303) == 2048);
    CHECK(ds41_prefill_count(&g, 2304) == 2304);
    CHECK(ds41_prefill_count(&g, 3241) == 3241);
    CHECK(setenv("DS4_CUDA_DISABLE_SSD_MEDIUM_SWEEP", "1", 1) == 0);
    CHECK(ds41_prefill_count(&g, 3241) == 2048);
    CHECK(unsetenv("DS4_CUDA_DISABLE_SSD_MEDIUM_SWEEP") == 0);
    g.prefill_cap = 1024;
    CHECK(ds41_prefill_count(&g, 3241) == 1024);
    g.prefill_cap = 8192;
#endif
    g.tp_world = 2;
    g.streaming = false;
    for (size_t i = 0; i < sizeof(remaining) / sizeof(*remaining); i++) {
        uint32_t expected = cold[i], small = 0;
#ifdef __APPLE__
        if (remaining[i] >= 32 && remaining[i] < 256) expected = remaining[i];
#else
        if (remaining[i] >= 2 && remaining[i] < 256)
            small = remaining[i] < 8 ? remaining[i] : 8;
#endif
        CHECK(ds41_prefill_count(&g, remaining[i]) == expected);
        CHECK(ds41_short_prefill_count(&g, &weights, remaining[i]) == small);
    }
    CHECK(setenv("DS4_METAL_DISABLE_V41_TP_SMALL_PREFILL", "1", 1) == 0);
    CHECK(ds41_prefill_count(&g, 255) == 1);
    CHECK(ds41_short_prefill_count(&g, &weights, 255) == 0);
    CHECK(unsetenv("DS4_METAL_DISABLE_V41_TP_SMALL_PREFILL") == 0);
    const char *ablations[] = {"DS4_METAL_DISABLE_V41_BATCH_ATTN",
        "DS4_METAL_DISABLE_V41_BATCH_CORE", "DS4_METAL_DISABLE_V41_BATCH_MOE",
        "DS4_METAL_DISABLE_V41_BATCH_HC", "DS4_METAL_DISABLE_V41_LAYER_PREFILL"};
    for (size_t i = 0; i < sizeof(ablations) / sizeof(*ablations); i++) {
        CHECK(setenv(ablations[i], "1", 1) == 0);
        CHECK(ds41_prefill_count(&g, 65536) == 1);
        CHECK(unsetenv(ablations[i]) == 0);
    }
    g.tp_world = 1;
    CHECK(ds41_prefill_count(&g, 7) == 1);
    CHECK(ds41_prefill_count(&g, 8) == 8);
    for (size_t i = 0; i < sizeof(remaining) / sizeof(*remaining); i++)
        CHECK(ds41_prefill_count(&g, remaining[i]) ==
            (remaining[i] >= 8 && remaining[i] < 256 ? remaining[i] : cold[i]));
    ds4_imatrix_collector imatrix = {0};
    g.imatrix = &imatrix;
    CHECK(ds41_prefill_count(&g, 65536) == 1);
    g.imatrix = NULL;
    g.carry_cap = 0;
    CHECK(ds41_prefill_count(&g, 65536) == 2048);
    g.prefill_cap = 1024;
    CHECK(ds41_prefill_count(&g, 4096) == 1024);
    g.prefill_cap = 8192;
    g.carry_cap = 22528;
    CHECK(ds41_prefill_count(&g, 2486) == 2486);
    CHECK(setenv("DS4_METAL_V41_WIDE_PREFILL_MIN", "4096", 1) == 0);
    CHECK(ds41_prefill_count(&g, 2486) == 2048);
    CHECK(unsetenv("DS4_METAL_V41_WIDE_PREFILL_MIN") == 0);
    CHECK(ds41_prefill_count(&g, 21255) == 21255);
    CHECK(ds41_prefill_count(&g, 22528) == 22528);
    CHECK(ds41_prefill_count(&g, 22529) == 22528);
    g.carry_cap = 22529;
    CHECK(ds41_prefill_count(&g, 22530) == 22528);
    g.carry_cap = ds41_carry_cap(131072);
    CHECK(g.carry_cap >= 21255);
    CHECK(ds41_prefill_count(&g, 21255) == 21255);
    CHECK(setenv("DS4_METAL_DISABLE_V41_DEFER_TAIL_REBALANCE", "1", 1) == 0);
    CHECK(ds41_prefill_count(&g, g.carry_cap + 1u) == g.carry_cap);
    CHECK(unsetenv("DS4_METAL_DISABLE_V41_DEFER_TAIL_REBALANCE") == 0);
    CHECK(ds41_encoder_chunk_cap(&g, 8191) == 2048);
    CHECK(ds41_encoder_chunk_cap(&g, 8192) == 4096);
    CHECK(ds41_encoder_chunk_cap(&g, 16383) == 4096);
    CHECK(ds41_encoder_chunk_cap(&g, 16384) == 8192);
    puts("V4.1 cold/warm and TP prefill dispatch, tile boundaries and debug/imatrix fallbacks: PASS");
    rc = 0;
done:
    if (saved_seed_cap) {
        setenv("DS4_METAL_V41_MAX_PREFILL_CACHE_SEED_EXPERTS_PER_LAYER", saved_seed_cap, 1);
        free(saved_seed_cap);
    } else {
        unsetenv("DS4_METAL_V41_MAX_PREFILL_CACHE_SEED_EXPERTS_PER_LAYER");
    }
    ds4_gpu_set_streaming_expert_cache_budget(saved);
    ds4_gpu_set_ssd_streaming(false);
    return rc;
}

static int check_decoder_suffix_plans(void) {
    int rc = 1;
    g_ds4_shape = DS4_SHAPE_FLASH41;
    CHECK(!ds41_short_decoder_suffix_requested());
    CHECK(setenv("DS4_METAL_V41_SHORT_DECODER_SUFFIX", "0", 1) == 0);
    CHECK(!ds41_short_decoder_suffix_requested());
    CHECK(setenv("DS4_METAL_V41_SHORT_DECODER_SUFFIX", "01", 1) == 0);
    CHECK(!ds41_short_decoder_suffix_requested());
    CHECK(setenv("DS4_METAL_V41_SHORT_DECODER_SUFFIX", "true", 1) == 0);
    CHECK(!ds41_short_decoder_suffix_requested());
    CHECK(setenv("DS4_METAL_V41_SHORT_DECODER_SUFFIX", "1", 1) == 0);
    CHECK(ds41_short_decoder_suffix_requested());
    CHECK(setenv("DS4_METAL_DISABLE_V41_DECODER_SUFFIX", "1", 1) == 0);
    CHECK(!ds41_short_decoder_suffix_requested());
    unsetenv("DS4_METAL_DISABLE_V41_DECODER_SUFFIX");
    CHECK(ds41_short_decoder_suffix_requested());
    ds41_gpu_graph eligibility = {.carry_cap = 8192u, .tp_world = 1u};
    CHECK(!ds41_short_decoder_suffix_enabled(&eligibility, 2048u));
    CHECK(ds41_short_decoder_suffix_enabled(&eligibility, 4096u));
    CHECK(ds41_short_decoder_suffix_enabled(&eligibility, 8191u));
    CHECK(!ds41_short_decoder_suffix_enabled(&eligibility, 8192u));
    eligibility.tp_world = 2u;
    CHECK(!ds41_short_decoder_suffix_enabled(&eligibility, 4096u));
    eligibility.tp_world = 1u;
    CHECK(setenv("DS4_METAL_DISABLE_V41_DECODER_SUFFIX", "1", 1) == 0);
    CHECK(!ds41_short_decoder_suffix_enabled(&eligibility, 4096u));
    unsetenv("DS4_METAL_DISABLE_V41_DECODER_SUFFIX");

    const uint32_t totals[] = {127u, 128u, 129u, 2048u, 4096u, 8192u};
    for (uint32_t il = 20u; il < DS4_N_LAYER; il++) {
        const uint32_t dependency = 1u + (DS4_N_LAYER - 1u - il) * 127u;
        const uint32_t frontier_dependency =
            128u + (DS4_N_LAYER - 1u - il) * 127u;
        for (size_t i = 0; i < sizeof(totals) / sizeof(*totals); i++) {
            const uint32_t needed = dependency < totals[i] ? dependency : totals[i];
            const ds41_decoder_suffix_plan plan =
                ds41_decoder_suffix_make_plan(totals[i], il);
            CHECK(plan.first == totals[i] - needed);
            CHECK(plan.warm_count == (plan.first < 127u ? plan.first : 127u));
            CHECK(plan.warm_offset + plan.warm_count == plan.first);
            uint32_t frontier_needed = frontier_dependency < 512u ?
                512u : frontier_dependency;
            if (frontier_needed > totals[i]) frontier_needed = totals[i];
            const ds41_decoder_suffix_plan frontier_plan =
                ds41_short_decoder_suffix_make_plan(totals[i], il);
            CHECK(frontier_plan.first == totals[i] - frontier_needed);
            CHECK(frontier_plan.warm_count ==
                  (frontier_plan.first < 127u ? frontier_plan.first : 127u));
            CHECK(frontier_plan.warm_offset + frontier_plan.warm_count ==
                  frontier_plan.first);
        }
        for (uint32_t boundary = 127u; boundary <= 129u; boundary++) {
            const uint32_t total = dependency + boundary;
            const ds41_decoder_suffix_plan plan =
                ds41_decoder_suffix_make_plan(total, il);
            CHECK(plan.first == boundary);
            CHECK(plan.warm_count == 127u);
            CHECK(plan.warm_offset == boundary - 127u);
            const uint32_t frontier_total =
                (frontier_dependency < 512u ? 512u : frontier_dependency) + boundary;
            const ds41_decoder_suffix_plan frontier_plan =
                ds41_short_decoder_suffix_make_plan(frontier_total, il);
            CHECK(frontier_plan.first == boundary);
            CHECK(frontier_plan.warm_count == 127u);
            CHECK(frontier_plan.warm_offset == boundary - 127u);
        }
        const ds41_decoder_suffix_plan wide_plan =
            ds41_decoder_suffix_align_plan(
                ds41_short_decoder_suffix_make_plan(4096u, il), 2048u);
        CHECK(wide_plan.first == 0u || wide_plan.first == 2048u);
        CHECK(wide_plan.first % 2048u == 0u);
        CHECK(wide_plan.warm_count ==
              (wide_plan.first < 127u ? wide_plan.first : 127u));
        CHECK(wide_plan.warm_offset + wide_plan.warm_count == wide_plan.first);
    }
    CHECK(ds41_decoder_suffix_make_plan(2048u, 19u).first == 0u);
    CHECK(ds41_decoder_suffix_make_plan(2048u, 40u).first == 0u);
    CHECK(ds41_short_decoder_suffix_make_plan(2048u, 19u).first == 0u);
    CHECK(ds41_short_decoder_suffix_make_plan(2048u, 40u).first == 0u);
    puts("V4.1 short decoder suffix 4096-8191 eligibility, disable override, "
         "and logit/frontier 127/128/129 plans for every decoder layer: PASS");
    rc = 0;
done:
    unsetenv("DS4_METAL_DISABLE_V41_DECODER_SUFFIX");
    unsetenv("DS4_METAL_V41_SHORT_DECODER_SUFFIX");
    return rc;
}

typedef struct {
    uint64_t state;
    uint64_t logits;
    uint64_t history;
    uint64_t dspark;
    uint64_t dspark_last_hash[DS4_DSPARK_MAX_TARGET_LAYERS];
    uint64_t dspark_batch_hash[DS4_DSPARK_MAX_TARGET_LAYERS];
    uint64_t decode_state;
    uint64_t decode_logits;
    uint64_t append_state[3];
    uint64_t append_logits[3];
    uint64_t append_history[3];
    uint64_t append_dspark[3];
    uint32_t append_pos[3];
    uint32_t append_capture_start[3];
    uint32_t append_capture_tokens[3];
    bool append_capture_valid[3];
    uint32_t pos;
    uint32_t capture_start;
    uint32_t capture_tokens;
    bool capture_valid;
    uint32_t span_count;
    uint64_t span_offset[64];
    uint64_t span_bytes[64];
    uint64_t span_hash[64];
    uint8_t *state_copy;
    uint64_t state_copy_len;
} short_suffix_digest;

static uint64_t suffix_hash(uint64_t hash, const void *ptr, uint64_t bytes) {
    const uint8_t *p = ptr;
    for (uint64_t i = 0; i < bytes; i++) {
        hash ^= p[i];
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

static bool short_suffix_capture_digest(ds4_session *s, short_suffix_digest *out,
                                        bool copy_state) {
    if (!s || !out || !s->ds41_graph.valid || !ds4_gpu_synchronize()) return false;
    ds41_gpu_graph *g = &s->ds41_graph;
    ds41_state_span spans[64];
    const uint32_t n = ds41_state_spans(g, g->pos, spans);
    if (n > sizeof(spans) / sizeof(*spans)) return false;
    uint64_t state = UINT64_C(1469598103934665603);
    uint64_t state_bytes = 0;
    for (uint32_t i = 0; i < n; i++) {
        const void *contents = ds4_gpu_tensor_contents(spans[i].tensor);
        if (!contents) return false;
        out->span_offset[i] = state_bytes;
        out->span_bytes[i] = spans[i].bytes;
        out->span_hash[i] = suffix_hash(UINT64_C(1469598103934665603),
                                        contents, spans[i].bytes);
        state = suffix_hash(state, contents, spans[i].bytes);
        state_bytes += spans[i].bytes;
    }
    out->span_count = n;
    if (copy_state) {
        out->state_copy = malloc((size_t)state_bytes);
        if (!out->state_copy) return false;
        out->state_copy_len = state_bytes;
        for (uint32_t i = 0; i < n; i++)
            memcpy(out->state_copy + out->span_offset[i],
                   ds4_gpu_tensor_contents(spans[i].tensor),
                   (size_t)spans[i].bytes);
    }
    out->state = state;
    out->logits = suffix_hash(UINT64_C(1469598103934665603), s->logits,
                              (uint64_t)DS4_N_VOCAB * sizeof(float));
    out->history = suffix_hash(UINT64_C(1469598103934665603), &g->history,
                               sizeof(g->history));
    out->pos = g->pos;
    out->dspark = UINT64_C(1469598103934665603);
    if (g->dspark) {
        ds4_gpu_graph *capture = g->dspark;
        out->capture_valid = capture->dspark_capture_valid &&
            capture->dspark_capture_batch_valid;
        out->capture_start = capture->dspark_capture_batch_start;
        out->capture_tokens = capture->dspark_capture_batch_tokens;
        const uint64_t row_bytes = (uint64_t)DS4_N_EMBD * sizeof(float);
        const uint64_t last_bytes =
            (uint64_t)capture->dspark_target_layer_count * row_bytes;
        const uint8_t *last = ds4_gpu_tensor_contents(capture->dspark_target_hidden);
        const uint8_t *batch =
            ds4_gpu_tensor_contents(capture->dspark_target_hidden_batch);
        if ((last_bytes && !last) ||
            (capture->dspark_target_layer_count && !batch))
            return false;
        out->dspark = suffix_hash(out->dspark, last, last_bytes);
        for (uint32_t slot = 0; slot < capture->dspark_target_layer_count; slot++) {
            const uint8_t *base = batch;
            base += (uint64_t)slot * capture->prefill_cap * row_bytes;
            out->dspark_last_hash[slot] = suffix_hash(
                UINT64_C(1469598103934665603), last + slot * row_bytes, row_bytes);
            out->dspark_batch_hash[slot] = suffix_hash(
                UINT64_C(1469598103934665603), base,
                (uint64_t)out->capture_tokens * row_bytes);
            out->dspark = suffix_hash(out->dspark, base,
                (uint64_t)out->capture_tokens * row_bytes);
        }
    }
    return true;
}

static bool short_suffix_run(ds4_engine *engine, const ds4_tokens *tokens,
                             uint32_t count, bool candidate,
                             short_suffix_digest *out) {
    ds4_session *s = NULL;
    bool ok = false;
    if (candidate) setenv("DS4_METAL_V41_SHORT_DECODER_SUFFIX", "1", 1);
    else unsetenv("DS4_METAL_V41_SHORT_DECODER_SUFFIX");
    if (ds4_session_create(&s, engine, 9216) != 0) goto done;
    if (!ds41_graph_prefill(&s->ds41_graph, &engine->model, &engine->weights,
            tokens->v, count, NULL, NULL, (int)count, NULL, NULL) ||
        !ds41_graph_logits(&s->ds41_graph, &engine->model, &engine->weights, s->logits) ||
        !short_suffix_capture_digest(s, out, true)) goto done;
    const int token = sample_argmax(s->logits, DS4_N_VOCAB);
    if (!ds41_graph_step(&s->ds41_graph, &engine->model, &engine->weights,
                         token, s->logits)) goto done;
    short_suffix_digest decoded = {0};
    if (!short_suffix_capture_digest(s, &decoded, false)) goto done;
    out->decode_state = decoded.state;
    out->decode_logits = decoded.logits;
    ds4_session_free(s);
    s = NULL;

    /* Exercise warm layer-major continuation independently of the decode
     * check above: one 127-row append, then the 128/129 boundary rows. */
    if (ds4_session_create(&s, engine, 9216) != 0) goto done;
    if (!ds41_graph_prefill(&s->ds41_graph, &engine->model, &engine->weights,
                            tokens->v, count, NULL, NULL, (int)count, NULL, NULL))
        goto done;
    const uint32_t append_counts[] = {127u, 1u, 1u};
    uint32_t appended = 0;
    for (uint32_t boundary = 0; boundary < 3u; boundary++) {
        const uint32_t rows = append_counts[boundary];
        if (!ds41_graph_prefill(&s->ds41_graph, &engine->model, &engine->weights,
                                tokens->v + count + appended, rows,
                                NULL, NULL, (int)rows, NULL, NULL) ||
            !ds41_graph_logits(&s->ds41_graph, &engine->model, &engine->weights,
                               s->logits))
            goto done;
        appended += rows;
        short_suffix_digest warm = {0};
        if (!short_suffix_capture_digest(s, &warm, false)) goto done;
        out->append_state[boundary] = warm.state;
        out->append_logits[boundary] = warm.logits;
        out->append_history[boundary] = warm.history;
        out->append_dspark[boundary] = warm.dspark;
        out->append_pos[boundary] = warm.pos;
        out->append_capture_start[boundary] = warm.capture_start;
        out->append_capture_tokens[boundary] = warm.capture_tokens;
        out->append_capture_valid[boundary] = warm.capture_valid;
    }
    ok = true;
done:
    ds4_session_free(s);
    unsetenv("DS4_METAL_V41_SHORT_DECODER_SUFFIX");
    return ok;
}

static const char *short_suffix_span_name(uint32_t span, char name[32]) {
    if (span < 40u) snprintf(name, 32, "window[%u]", span);
    else {
        uint32_t cursor = 40u;
        for (uint32_t owner = 0; owner < 4u; owner++) {
            if (span == cursor++) { snprintf(name, 32, "compressed[%u]", owner); return name; }
            if (span == cursor++) { snprintf(name, 32, "index_cache[%u]", owner); return name; }
            /* Even test frontiers have no unfinished-pair spans. */
        }
        snprintf(name, 32, "state[%u]", span);
    }
    return name;
}

static bool short_suffix_digests_equal(const short_suffix_digest *control,
                                       const short_suffix_digest *candidate,
                                       uint32_t rows) {
    bool exact = true;
    if (control->state != candidate->state) {
        exact = false;
        fprintf(stderr, "rows=%u state digest differs control=%016llx candidate=%016llx\n",
                rows, (unsigned long long)control->state,
                (unsigned long long)candidate->state);
        if (control->span_count != candidate->span_count ||
            control->state_copy_len != candidate->state_copy_len) {
            fprintf(stderr, "  span shape differs count=%u/%u bytes=%llu/%llu\n",
                    control->span_count, candidate->span_count,
                    (unsigned long long)control->state_copy_len,
                    (unsigned long long)candidate->state_copy_len);
        } else {
            for (uint32_t i = 0; i < control->span_count; i++) {
                if (control->span_bytes[i] == candidate->span_bytes[i] &&
                    control->span_hash[i] == candidate->span_hash[i]) continue;
                char name[32];
                uint64_t first = 0;
                const uint64_t bytes = control->span_bytes[i] < candidate->span_bytes[i] ?
                    control->span_bytes[i] : candidate->span_bytes[i];
                const uint8_t *a = control->state_copy + control->span_offset[i];
                const uint8_t *b = candidate->state_copy + candidate->span_offset[i];
                while (first < bytes && a[first] == b[first]) first++;
                fprintf(stderr,
                    "  span=%u %s bytes=%llu/%llu hash=%016llx/%016llx "
                    "first_diff=%llu values=%02x/%02x\n",
                    i, short_suffix_span_name(i, name),
                    (unsigned long long)control->span_bytes[i],
                    (unsigned long long)candidate->span_bytes[i],
                    (unsigned long long)control->span_hash[i],
                    (unsigned long long)candidate->span_hash[i],
                    (unsigned long long)first,
                    first < bytes ? a[first] : 0, first < bytes ? b[first] : 0);
            }
        }
    }
#define SHORT_SUFFIX_COMPARE(field, format, cast) do { \
        if (control->field != candidate->field) { \
            exact = false; \
            fprintf(stderr, "rows=%u " #field " differs control=" format \
                    " candidate=" format "\n", rows, \
                    cast control->field, cast candidate->field); \
        } \
    } while (0)
    SHORT_SUFFIX_COMPARE(logits, "%016llx", (unsigned long long));
    SHORT_SUFFIX_COMPARE(history, "%016llx", (unsigned long long));
    SHORT_SUFFIX_COMPARE(dspark, "%016llx", (unsigned long long));
    for (uint32_t slot = 0; slot < DS4_DSPARK_MAX_TARGET_LAYERS; slot++) {
        if (control->dspark_last_hash[slot] != candidate->dspark_last_hash[slot] ||
            control->dspark_batch_hash[slot] != candidate->dspark_batch_hash[slot]) {
            exact = false;
            fprintf(stderr,
                "rows=%u dspark slot=%u differs last=%016llx/%016llx "
                "batch=%016llx/%016llx\n",
                rows, slot,
                (unsigned long long)control->dspark_last_hash[slot],
                (unsigned long long)candidate->dspark_last_hash[slot],
                (unsigned long long)control->dspark_batch_hash[slot],
                (unsigned long long)candidate->dspark_batch_hash[slot]);
        }
    }
    SHORT_SUFFIX_COMPARE(pos, "%u", (unsigned));
    SHORT_SUFFIX_COMPARE(capture_start, "%u", (unsigned));
    SHORT_SUFFIX_COMPARE(capture_tokens, "%u", (unsigned));
    SHORT_SUFFIX_COMPARE(capture_valid, "%u", (unsigned));
    SHORT_SUFFIX_COMPARE(decode_state, "%016llx", (unsigned long long));
    SHORT_SUFFIX_COMPARE(decode_logits, "%016llx", (unsigned long long));
    for (uint32_t boundary = 0; boundary < 3u; boundary++) {
        if (control->append_state[boundary] != candidate->append_state[boundary] ||
            control->append_logits[boundary] != candidate->append_logits[boundary] ||
            control->append_history[boundary] != candidate->append_history[boundary] ||
            control->append_dspark[boundary] != candidate->append_dspark[boundary] ||
            control->append_pos[boundary] != candidate->append_pos[boundary] ||
            control->append_capture_start[boundary] != candidate->append_capture_start[boundary] ||
            control->append_capture_tokens[boundary] != candidate->append_capture_tokens[boundary] ||
            control->append_capture_valid[boundary] != candidate->append_capture_valid[boundary]) {
            exact = false;
            fprintf(stderr,
                "rows=%u append=%u differs state=%016llx/%016llx logits=%016llx/%016llx "
                "history=%016llx/%016llx dspark=%016llx/%016llx pos=%u/%u "
                "capture=%u,%u,%u/%u,%u,%u\n",
                rows, 127u + boundary,
                (unsigned long long)control->append_state[boundary],
                (unsigned long long)candidate->append_state[boundary],
                (unsigned long long)control->append_logits[boundary],
                (unsigned long long)candidate->append_logits[boundary],
                (unsigned long long)control->append_history[boundary],
                (unsigned long long)candidate->append_history[boundary],
                (unsigned long long)control->append_dspark[boundary],
                (unsigned long long)candidate->append_dspark[boundary],
                control->append_pos[boundary], candidate->append_pos[boundary],
                control->append_capture_start[boundary],
                control->append_capture_tokens[boundary],
                control->append_capture_valid[boundary],
                candidate->append_capture_start[boundary],
                candidate->append_capture_tokens[boundary],
                candidate->append_capture_valid[boundary]);
        }
    }
#undef SHORT_SUFFIX_COMPARE
    return exact;
}

static int check_short_decoder_suffix_model(const char *model_path,
                                            const char *prompt_path,
                                            const char *mtp_path) {
    int rc = 1;
    ds4_engine *engine = NULL;
    ds4_tokens tokens = {0};
    char *prompt = NULL;
    size_t prompt_bytes = 0;
    ds4_engine_options opt = {.model_path = model_path, .mtp_path = mtp_path,
        .backend = DS4_BACKEND_METAL, .context_size = 9216, .power_percent = 100,
        .ssd_streaming = true, .ssd_streaming_cache_bytes = UINT64_C(42) << 30};
    if (mtp_path) setenv("DS4_V41_DSPARK_SPEC", "1", 1);
    CHECK(imatrix_read_text_file(prompt_path, &prompt, &prompt_bytes));
    CHECK(ds4_engine_open(&engine, &opt) == 0);
    ds4_encode_chat_prompt(engine, NULL, prompt, DS4_THINK_NONE, &tokens);
    CHECK(tokens.len >= 8192 + 129);
    const uint32_t counts[] = {2048u, 4096u, 8192u};
    const char *count_filter = getenv("DS4_TEST_SHORT_SUFFIX_COUNT");
    const unsigned long filter = count_filter ? strtoul(count_filter, NULL, 10) : 0u;
    CHECK(!count_filter || filter == 2048u || filter == 4096u || filter == 8192u);
    for (size_t i = 0; i < sizeof(counts) / sizeof(*counts); i++) {
        if (count_filter && filter != counts[i]) continue;
        short_suffix_digest control = {0}, candidate = {0};
        CHECK(short_suffix_run(engine, &tokens, counts[i], false, &control));
        CHECK(short_suffix_run(engine, &tokens, counts[i], true, &candidate));
        const bool exact = control.pos == counts[i] &&
            short_suffix_digests_equal(&control, &candidate, counts[i]);
        free(control.state_copy);
        free(candidate.state_copy);
        CHECK(exact);
        fprintf(stderr,
            "V4.1 short suffix rows=%u frontier=%016llx logits=%016llx "
            "dspark=%016llx decode=%016llx/%016llx "
            "append127/128/129=%016llx/%016llx/%016llx: exact PASS\n",
            counts[i], (unsigned long long)control.state,
            (unsigned long long)control.logits, (unsigned long long)control.dspark,
            (unsigned long long)control.decode_state,
            (unsigned long long)control.decode_logits,
            (unsigned long long)control.append_state[0],
            (unsigned long long)control.append_state[1],
            (unsigned long long)control.append_state[2]);
    }
    puts("V4.1 short decoder suffix 2048/4096/8192 frontier, Engram history, "
         "DSpark capture, logits, immediate decode and 127/128/129 warm append: PASS");
    rc = 0;
done:
    unsetenv("DS4_V41_DSPARK_SPEC");
    unsetenv("DS4_METAL_V41_SHORT_DECODER_SUFFIX");
    ds4_tokens_free(&tokens);
    free(prompt);
    ds4_engine_close(engine);
    return rc;
}

static int bench_short_decoder_suffix(const char *model_path,
                                      const char *prompt_path,
                                      uint32_t count,
                                      bool candidate,
                                      const char *mtp_path) {
    int rc = 1;
    ds4_engine *engine = NULL;
    ds4_session *session = NULL;
    ds4_tokens tokens = {0};
    char *prompt = NULL;
    size_t prompt_bytes = 0;
    ds4_engine_options opt = {.model_path = model_path, .mtp_path = mtp_path,
        .backend = DS4_BACKEND_METAL, .context_size = 9216, .power_percent = 100,
        .ssd_streaming = true, .ssd_streaming_cache_bytes = UINT64_C(42) << 30};
    if (candidate) setenv("DS4_METAL_V41_SHORT_DECODER_SUFFIX", "1", 1);
    else unsetenv("DS4_METAL_V41_SHORT_DECODER_SUFFIX");
    CHECK(count >= 2048u && count <= 8192u);
    CHECK(imatrix_read_text_file(prompt_path, &prompt, &prompt_bytes));
    CHECK(ds4_engine_open(&engine, &opt) == 0);
    ds4_encode_chat_prompt(engine, NULL, prompt, DS4_THINK_NONE, &tokens);
    CHECK(tokens.len >= (int)count);
    CHECK(ds4_session_create(&session, engine, 9216) == 0);
    const double begin = now_sec();
    CHECK(ds41_graph_prefill(&session->ds41_graph, &engine->model, &engine->weights,
                             tokens.v, count, NULL, NULL, (int)count, NULL, NULL));
    CHECK(ds41_graph_logits(&session->ds41_graph, &engine->model, &engine->weights,
                            session->logits));
    CHECK(ds4_gpu_synchronize());
    const double elapsed = now_sec() - begin;
    short_suffix_digest digest = {0};
    const bool dump_digest = getenv("DS4_TEST_DUMP_SHORT_SUFFIX_DIGEST") != NULL;
    CHECK(short_suffix_capture_digest(session, &digest, dump_digest));
    struct rusage usage = {0};
    CHECK(getrusage(RUSAGE_SELF, &usage) == 0);
#if defined(__APPLE__)
    const double rss_mib = (double)usage.ru_maxrss / 1048576.0;
#else
    const double rss_mib = (double)usage.ru_maxrss / 1024.0;
#endif
    fprintf(stdout,
        "SHORT_SUFFIX_BENCH rows=%u mode=%s seconds=%.6f prefill_tps=%.3f "
        "max_rss_mib=%.3f pos=%u state=%016llx logits=%016llx "
        "history=%016llx dspark=%016llx capture=%u,%u,%u\n",
        count, candidate ? "candidate" : "control", elapsed,
        (double)count / elapsed, rss_mib, session->ds41_graph.pos,
        (unsigned long long)digest.state,
        (unsigned long long)suffix_hash(UINT64_C(1469598103934665603),
            session->logits, (uint64_t)DS4_N_VOCAB * sizeof(float)),
        (unsigned long long)digest.history,
        (unsigned long long)digest.dspark,
        digest.capture_start, digest.capture_tokens, digest.capture_valid);
    if (dump_digest) {
        for (uint32_t i = 0; i < digest.span_count; i++) {
            char name[32];
            fprintf(stdout, "SHORT_SUFFIX_SPAN index=%u name=%s bytes=%llu hash=%016llx\n",
                    i, short_suffix_span_name(i, name),
                    (unsigned long long)digest.span_bytes[i],
                    (unsigned long long)digest.span_hash[i]);
        }
        for (uint32_t slot = 0; slot < DS4_DSPARK_MAX_TARGET_LAYERS; slot++) {
            if (!digest.dspark_last_hash[slot] && !digest.dspark_batch_hash[slot]) continue;
            fprintf(stdout, "SHORT_SUFFIX_DSPARK slot=%u last=%016llx batch=%016llx\n",
                    slot, (unsigned long long)digest.dspark_last_hash[slot],
                    (unsigned long long)digest.dspark_batch_hash[slot]);
        }
    }
    free(digest.state_copy);
    rc = 0;
done:
    unsetenv("DS4_METAL_V41_SHORT_DECODER_SUFFIX");
    ds4_session_free(session);
    ds4_tokens_free(&tokens);
    free(prompt);
    ds4_engine_close(engine);
    return rc;
}

typedef struct {
    ds4_session *session;
    int target, current, frontier;
    unsigned callbacks, scalar, batches, short_batches, deferred, displays;
    bool partial_checked, final_checked;
    double begin, first_display;
} prefill_progress;

static void progress_note(void *ud, const char *event, int current, int total) {
    prefill_progress *p = ud;
    ds4_session *s = p->session;
    assert(total == p->target && current >= p->current && current <= total);
    p->current = current;
    p->callbacks++;
    if (!strcmp(event, "prefill_display")) {
        if (!p->displays++) p->first_display = now_sec() - p->begin;
        assert(!s->ds41_graph.valid);
        if (!p->partial_checked) {
            ds4_session_snapshot snap = {0};
            char err[256];
            assert(ds4_session_save_snapshot(s, &snap, err, sizeof(err)) != 0);
            ds4_session_snapshot_free(&snap);
            p->partial_checked = true;
        }
    } else {
        assert(!strcmp(event, "prefill_chunk"));
        if (current != p->frontier) {
            const int count = current - p->frontier;
            ds41_gpu_graph before = s->ds41_graph;
            before.pos = (uint32_t)p->frontier;
            const uint32_t small = ds41_short_prefill_count(&before, &s->engine->weights,
                (uint32_t)(total - p->frontier));
            assert((uint32_t)count == (small ? small : ds41_prefill_count(&before,
                (uint32_t)(total - p->frontier))));
            if (count == 1) p->scalar++;
            else if (small) p->short_batches++;
            else p->batches++;
            if (!s->ds41_graph.valid) p->deferred++;
            p->frontier = current;
        }
        if (s->checkpoint_valid) {
            assert(current == total && s->ds41_graph.valid);
            p->final_checked = true;
        }
    }
}

static bool cancel_after_display(void *ud) {
    return ((prefill_progress *)ud)->displays >= 2;
}

static bool state_equal(ds4_session *a, ds4_session *b) {
    if (!a->checkpoint_valid || !b->checkpoint_valid ||
        !a->ds41_graph.valid || !b->ds41_graph.valid ||
        ds4_session_pos(a) != ds4_session_pos(b) ||
        memcmp(&a->ds41_graph.history, &b->ds41_graph.history,
            sizeof(a->ds41_graph.history))) return false;
    ds41_state_span sa[64], sb[64];
    uint32_t n = ds41_state_spans(&a->ds41_graph, a->ds41_graph.pos, sa);
    if (n != ds41_state_spans(&b->ds41_graph, b->ds41_graph.pos, sb)) return false;
    for (uint32_t i = 0; i < n; i++) {
        const size_t bytes = (size_t)sa[i].bytes;
        void *left = malloc(bytes), *right = malloc(bytes);
        const bool readable = left && right && sa[i].bytes == sb[i].bytes &&
            ds4_gpu_tensor_read(sa[i].tensor, 0, left, bytes) &&
            ds4_gpu_tensor_read(sb[i].tensor, 0, right, bytes);
        const bool equal = readable && memcmp(left, right, bytes) == 0;
        if (!equal) {
            fprintf(stderr, "frontier=%d cache span=%u differs (%llu bytes)\n",
                ds4_session_pos(a), i, (unsigned long long)sa[i].bytes);
            if (readable) {
                unsigned different = 0;
                float worst = 0;
                const float *x = left, *y = right;
                for (size_t j = 0; j < bytes / sizeof(float); j++) {
                    if (x[j] == y[j]) continue;
                    if (different++ < 4) fprintf(stderr, "  cache[%zu] %.9g != %.9g\n", j, x[j], y[j]);
                    const float gap = fabsf(x[j] - y[j]);
                    if (!isfinite(gap) || gap > worst) worst = gap;
                }
                fprintf(stderr, "  differing=%u worst=%g\n", different, worst);
            }
            free(left); free(right);
            return false;
        }
        free(left); free(right);
    }
    for (uint32_t i = 0; i < DS4_N_VOCAB; i++) {
        if (!isfinite(a->logits[i]) || a->logits[i] != b->logits[i]) {
            fprintf(stderr, "frontier=%d logit=%u control=%.9g mixed=%.9g\n",
                ds4_session_pos(a), i, a->logits[i], b->logits[i]);
            return false;
        }
    }
    return true;
}

typedef enum {
    PREFILL_METAL, PREFILL_CUDA, PREFILL_CUDA_LONG,
    PREFILL_CUDA_DEFERRED, PREFILL_CUDA_SMALL
} prefill_test_mode;

static int check_mixed(const char *model, const char *prompt_path,
                       const ds4_tp_options *tp_opt, bool resident, prefill_test_mode mode) {
    const bool cuda = mode != PREFILL_METAL;
    const bool cuda_long = mode == PREFILL_CUDA_LONG || mode == PREFILL_CUDA_DEFERRED;
    const bool deferred_only = mode == PREFILL_CUDA_DEFERRED;
    const bool cuda_small = mode == PREFILL_CUDA_SMALL;
    ds4_engine *engine = NULL;
    ds4_tp *tp = NULL;
    ds4_session *control = NULL, *mixed = NULL;
    ds4_session_snapshot snap = {0};
    ds4_tokens tokens = {0};
    char *prompt = NULL, err[256] = {0};
    size_t bytes;
    int rc = 1;
    ds4_engine_options opt = {.model_path = model,
        .backend = cuda ? DS4_BACKEND_CUDA : DS4_BACKEND_METAL,
        .context_size = cuda ? (cuda_long ? 65536 : 16384) : 131072, .power_percent = 100,
        .ssd_streaming = !tp_opt && !resident,
        .ssd_streaming_cache_bytes = tp_opt || resident ? 0 : UINT64_C(64) << 30};
    if (tp_opt) opt.tp = *tp_opt;
    CHECK(imatrix_read_text_file(prompt_path, &prompt, &bytes));
    CHECK(ds4_engine_open(&engine, &opt) == 0);
    if (tp_opt) {
        ds4_tp_identity id = {
            .gguf_bytes = ds4_engine_model_bytes(engine),
            .model_id = (uint32_t)ds4_engine_model_id(engine),
            .n_layer = (uint32_t)ds4_engine_layer_count(engine),
            .n_embd = (uint32_t)ds4_engine_embd_dim(engine),
            .n_vocab = (uint32_t)ds4_engine_vocab_size(engine),
            .quant_bits = (uint32_t)ds4_engine_routed_quant_bits(engine),
            .ctx_size = (uint32_t)opt.context_size,
        };
        ds4_engine_tp_gate_schedule(engine, &id.gate_slot_start,
            &id.gate_slot_step, &id.gates_per_token, id.gate_slot_mask);
        CHECK(ds4_tp_create(&tp, tp_opt, &id, err, sizeof(err)));
        CHECK(ds4_engine_tp_bind(engine, tp, err, sizeof(err)));
    }
    ds4_encode_chat_prompt(engine, NULL, prompt, DS4_THINK_NONE, &tokens);
    CHECK(tokens.len > (cuda ? opt.context_size : resident ? 131072 : 120000));
    CHECK(ds4_session_create(&control, engine, opt.context_size) == 0);
    CHECK(ds4_session_create(&mixed, engine, opt.context_size) == 0);
    const int ordinary_appends[] = {127, 1, 255, 256, 257, 1023, 1024, 4095, 4096,
        8191, 8192, 16383, 16384, 49153, 129, 4096, 16383};
    const int small_appends[] = {7, 1, 8, 9, 15, 16, 17, 31, 1, 32, 63, 64, 65, 127, 128, 129,
        255, 256, 257, 511, 512, 513, 1023, 1024};
    const int *appends = cuda_small ? small_appends : ordinary_appends;
    const size_t n_appends = cuda_small ? sizeof(small_appends) / sizeof(*small_appends) :
        cuda ? (cuda_long ? 14u : 9u) :
        sizeof(ordinary_appends) / sizeof(*ordinary_appends) - (resident ? 0u : 1u);
    unsigned scalar = 0, batches = 0, deferred = 0;
    for (size_t i = deferred_only ? 13u : 0u; i < n_appends; i++) {
        /* A single 49K append crosses the deferred-decoder threshold. Reset
         * both sessions to cover it without allocating two 128K graphs. */
        if (cuda_long && i == 13) {
            ds4_session_invalidate(control);
            ds4_session_invalidate(mixed);
        }
        const int start = ds4_session_pos(mixed);
        tokens.len = start + appends[i];
        /* The normal worker mirrors batch ordering. Keep TP partitions equal,
         * but compare queued decode against synchronous layer submission. */
        const char *ablation = tp ? "DS4_METAL_DISABLE_V41_TP_DECODE_QUEUE" :
            "DS4_METAL_DISABLE_V41_DEFER_DECODER";
        CHECK(setenv(ablation, "1", 1) == 0);
        if (cuda && !tp) {
            CHECK(setenv("DS4_CUDA_SESSION_BATCH_MOE", "0", 1) == 0);
            CHECK(setenv("DS4_CUDA_DISABLE_SSD_PREFETCH", "1", 1) == 0);
            CHECK(setenv("DS4_CUDA_DISABLE_SSD_MEDIUM_SWEEP", "1", 1) == 0);
        }
        if (cuda && tp && appends[i] < 256) {
            /* Mirror scalar control execution on the worker too. */
            ds4_tokens prefix = tokens;
            for (prefix.len = start + 1; prefix.len <= tokens.len; prefix.len++)
                CHECK(ds4_session_sync(control, &prefix, err, sizeof(err)) == 0);
        } else {
            CHECK(ds4_session_sync(control, &tokens, err, sizeof(err)) == 0);
        }
        if (cuda && !tp) {
            CHECK(unsetenv("DS4_CUDA_SESSION_BATCH_MOE") == 0);
            CHECK(unsetenv("DS4_CUDA_DISABLE_SSD_PREFETCH") == 0);
            CHECK(unsetenv("DS4_CUDA_DISABLE_SSD_MEDIUM_SWEEP") == 0);
        }
        CHECK(unsetenv(ablation) == 0);
        prefill_progress p = {.session = mixed, .frontier = start,
            .current = start, .target = tokens.len, .begin = now_sec()};
        ds4_session_set_progress(mixed, progress_note, &p);
        CHECK(ds4_session_sync(mixed, &tokens, err, sizeof(err)) == 0);
        const double seconds = now_sec() - p.begin;
        CHECK(p.current == tokens.len && p.final_checked);
        CHECK(!p.batches || (p.displays && p.partial_checked));
        scalar += p.scalar; batches += p.batches + p.short_batches; deferred += p.deferred;
        CHECK(state_equal(control, mixed));
        const unsigned callbacks = p.callbacks;
        CHECK(ds4_session_sync(mixed, &tokens, err, sizeof(err)) == 0);
        CHECK(callbacks == p.callbacks && state_equal(control, mixed));
        ds4_session_set_progress(mixed, NULL, NULL);
        CHECK(ds4_session_save_snapshot(mixed, &snap, err, sizeof(err)) == 0);
        const int token = tokens.v[tokens.len];
        CHECK(setenv("DS4_METAL_DISABLE_V41_TP_DECODE_QUEUE", "1", 1) == 0);
        CHECK(ds4_session_eval(control, token, err, sizeof(err)) == 0);
        CHECK(unsetenv("DS4_METAL_DISABLE_V41_TP_DECODE_QUEUE") == 0);
        CHECK(ds4_session_eval(mixed, token, err, sizeof(err)) == 0);
        CHECK(state_equal(control, mixed));
        if (!tp) {
            CHECK(ds4_session_load_snapshot(mixed, &snap, err, sizeof(err)) == 0);
            CHECK(ds4_session_eval(mixed, token, err, sizeof(err)) == 0);
            CHECK(state_equal(control, mixed));
        } else if (i + 1 == n_appends) {
            /* TP snapshots rebuild both ranks from tokens. Compare against
             * the same one-shot replay, not different batched reductions. */
            ds4_session_invalidate(control);
            CHECK(ds4_session_sync(control, &tokens, err, sizeof(err)) == 0);
            CHECK(ds4_session_load_snapshot(mixed, &snap, err, sizeof(err)) == 0);
            CHECK(state_equal(control, mixed));
            CHECK(ds4_session_eval(control, token, err, sizeof(err)) == 0);
            CHECK(ds4_session_eval(mixed, token, err, sizeof(err)) == 0);
            CHECK(state_equal(control, mixed));
            puts("TP snapshot rebuild, both-rank replay and next decode: PASS");
        }
        ds4_session_snapshot_free(&snap);
        fprintf(stderr, "mixed start=%d append=%d scalar=%u batches=%u short_batches=%u deferred=%u "
            "display=%u first_display=%.3fs time=%.3fs rate=%.2f: exact state/decode PASS\n",
            start, appends[i], p.scalar, p.batches, p.short_batches, p.deferred, p.displays,
            p.first_display, seconds, appends[i] / seconds);
        if (cuda && i == 3) {
            CHECK(ds4_session_save_snapshot(mixed, &snap, err, sizeof(err)) == 0);
            const int frontier = ds4_session_pos(mixed);
            tokens.len = frontier + 1024;
            prefill_progress cancelled = {.session = mixed, .frontier = frontier,
                .current = frontier, .target = tokens.len, .begin = now_sec()};
            ds4_session_set_progress(mixed, progress_note, &cancelled);
            ds4_session_set_cancel(mixed, cancel_after_display, &cancelled);
            CHECK(ds4_session_sync(mixed, &tokens, err, sizeof(err)) == DS4_SESSION_SYNC_INTERRUPTED);
            CHECK(cancelled.partial_checked && !mixed->checkpoint_valid);
            /* TP invalidates both ranks and resets to an empty valid graph;
             * the local path leaves its unfinished graph explicitly invalid. */
            CHECK(tp ? (mixed->ds41_graph.valid && mixed->ds41_graph.pos == 0 &&
                         mixed->checkpoint.len == 0) : !mixed->ds41_graph.valid);
            ds4_session_set_cancel(mixed, NULL, NULL);
            ds4_session_set_progress(mixed, NULL, NULL);
            if (tp) {
                /* TP restore replays the prefix; compare the same partition. */
                ds4_session_invalidate(control);
                tokens.len = frontier;
                CHECK(ds4_session_sync(control, &tokens, err, sizeof(err)) == 0);
            }
            CHECK(ds4_session_load_snapshot(mixed, &snap, err, sizeof(err)) == 0);
            CHECK(state_equal(control, mixed));
            CHECK(ds4_session_eval(control, tokens.v[frontier], err, sizeof(err)) == 0);
            CHECK(ds4_session_eval(mixed, tokens.v[frontier], err, sizeof(err)) == 0);
            CHECK(state_equal(control, mixed));
            ds4_session_snapshot_free(&snap);
            puts("V4.1 cancelled layer prefill, restore and next decode: exact PASS");
        }
    }
    CHECK(scalar && batches && ((cuda && !cuda_long) || deferred));
    puts(deferred_only ? "V4.1 deferred decoder, progress and restore: PASS" :
        "V4.1 mixed small/large continued prefill, dispatch, progress and restore: PASS");
    rc = 0;
done:
    if (cuda) unsetenv("DS4_CUDA_SESSION_BATCH_MOE");
    unsetenv("DS4_METAL_DISABLE_V41_DEFER_DECODER");
    unsetenv("DS4_METAL_DISABLE_V41_TP_DECODE_QUEUE");
    if (rc) fprintf(stderr, "mixed prefill failure: %s\n", err);
    ds4_session_snapshot_free(&snap);
    ds4_tokens_free(&tokens); free(prompt);
    ds4_session_free(mixed); ds4_session_free(control);
    if (tp) ds4_tp_send_stop(tp);
    ds4_engine_close(engine); ds4_tp_free(tp);
    return rc;
}

int main(int argc, char **argv) {
    if (argc == 2 && !strcmp(argv[1], "--dispatch")) return check_dispatch();
    if (argc == 2 && !strcmp(argv[1], "--decoder-suffix-plans"))
        return check_decoder_suffix_plans();
    if ((argc == 4 || argc == 5) && !strcmp(argv[1], "--short-decoder-suffix"))
        return check_short_decoder_suffix_model(argv[2], argv[3],
                                                argc == 5 ? argv[4] : NULL);
    if ((argc == 6 || argc == 7) && !strcmp(argv[1], "--short-decoder-suffix-bench")) {
        const unsigned long count = strtoul(argv[4], NULL, 10);
        const bool candidate = !strcmp(argv[5], "candidate");
        if ((!candidate && strcmp(argv[5], "control")) || count > UINT32_MAX) return 2;
        return bench_short_decoder_suffix(argv[2], argv[3], (uint32_t)count,
                                          candidate, argc == 7 ? argv[6] : NULL);
    }
    if (argc == 3) return check_mixed(argv[1], argv[2], NULL, false);
    if (argc == 4 && !strcmp(argv[1], "--resident"))
        return check_mixed(argv[2], argv[3], NULL, true, PREFILL_METAL);
    if (argc == 4 && !strcmp(argv[1], "--cuda"))
        return check_mixed(argv[2], argv[3], NULL, false, PREFILL_CUDA);
    if (argc == 4 && !strcmp(argv[1], "--cuda-small"))
        return check_mixed(argv[2], argv[3], NULL, false, PREFILL_CUDA_SMALL);
    if (argc == 4 && !strcmp(argv[1], "--cuda-long"))
        return check_mixed(argv[2], argv[3], NULL, false, PREFILL_CUDA_LONG);
    if (argc == 4 && !strcmp(argv[1], "--cuda-deferred"))
        return check_mixed(argv[2], argv[3], NULL, false, PREFILL_CUDA_DEFERRED);
    if (argc == 8 && (!strcmp(argv[1], "--tensor-parallel") ||
                     !strcmp(argv[1], "--tensor-parallel-cuda") ||
                     !strcmp(argv[1], "--tensor-parallel-cuda-small"))) {
        ds4_tp_options tp = {.role = DS4_TP_LEADER, .requested = true,
            .listen_host = argv[4], .listen_port = atoi(argv[5]),
            .transport = DS4_TP_TRANSPORT_RDMA, .rdma_device = argv[6],
            .rdma_gid_index = atoi(argv[7]), .rdma_gid_index_set = true};
        return check_mixed(argv[2], argv[3], &tp, false,
            !strcmp(argv[1], "--tensor-parallel-cuda-small") ? PREFILL_CUDA_SMALL :
            !strcmp(argv[1], "--tensor-parallel-cuda") ? PREFILL_CUDA : PREFILL_METAL);
    }
    fprintf(stderr, "usage: %s --dispatch | --decoder-suffix-plans | "
        "--short-decoder-suffix MODEL LONG_PROMPT_FILE [MTP_MODEL] | MODEL LONG_PROMPT_FILE | "
        "--short-decoder-suffix-bench MODEL LONG_PROMPT_FILE ROWS control|candidate [MTP_MODEL] | "
        "--resident MODEL LONG_PROMPT_FILE | "
        "--tensor-parallel MODEL LONG_PROMPT_FILE LISTEN_HOST PORT RDMA_DEVICE GID\n", argv[0]);
    return 2;
}
