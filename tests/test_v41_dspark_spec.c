/* Exercise the V4.1 DSpark frontier functions with real Metal tensors. */
#include "../ds4.c"
#include <assert.h>
#include <stdio.h>
#include <string.h>

enum {
    TEST_STATE_WORDS = 512,
    TEST_ROW_BYTES = TEST_STATE_WORDS * sizeof(float),
    TEST_CTX = 16,
    TEST_POS = 4,
};

typedef struct {
    uint32_t target_pos;
    uint32_t support_next_pos;
    uint32_t append_calls;
} cycle_commit_fixture;

static bool cycle_fixture_restore_prefix(cycle_commit_fixture *fixture,
                                          uint32_t start,
                                          uint32_t accepted_rows) {
    if (!fixture || accepted_rows == 0 || accepted_rows > DS41_SPEC_ROWS)
        return false;
    fixture->target_pos = start + accepted_rows;
    return true;
}

static bool cycle_fixture_append_target_row(cycle_commit_fixture *fixture,
                                            uint32_t pos) {
    if (!fixture || pos != fixture->support_next_pos) return false;
    fixture->support_next_pos++;
    fixture->append_calls++;
    return true;
}

static void test_forced_outcome_contract(void) {
    assert(ds41_dspark_forced_accepted_draft("full", 0) == 2u);
    assert(ds41_dspark_forced_accepted_draft("reject1", 2) == 0u);
    assert(ds41_dspark_forced_accepted_draft("reject2", 0) == 1u);
    assert(ds41_dspark_forced_accepted_draft("unknown", 2) == 2u);
    assert(ds41_dspark_forced_accepted_draft(NULL, 1) == 1u);
}

static void test_cycle_commit_invariants(void) {
    const uint32_t start = 32u;
    for (uint32_t accepted_draft = 0; accepted_draft <= 2u;
         accepted_draft++) {
        cycle_commit_fixture fixture = {
            .target_pos = start,
            .support_next_pos = start,
        };
        const uint32_t accepted_rows = 1u + accepted_draft;
        assert(cycle_fixture_restore_prefix(
            &fixture, start, accepted_rows));
        for (uint32_t row = 0; row < accepted_rows; row++)
            assert(cycle_fixture_append_target_row(&fixture, start + row));
        assert(fixture.target_pos == fixture.support_next_pos);
        assert(fixture.target_pos == start + accepted_rows);
        assert(fixture.append_calls == accepted_rows);
        assert(!cycle_fixture_append_target_row(&fixture, start));
        assert(fixture.append_calls == accepted_rows);
    }

    cycle_commit_fixture error = {
        .target_pos = start + DS41_SPEC_ROWS,
        .support_next_pos = start,
    };
    assert(cycle_fixture_restore_prefix(&error, start, 0) == false);
    error.target_pos = start;
    assert(cycle_fixture_append_target_row(&error, start));
    error.target_pos = start + 1u;
    assert(error.target_pos == error.support_next_pos);
    assert(error.append_calls == 1u);
    assert(!cycle_fixture_append_target_row(&error, start));
}

static void fill_tensor(ds4_gpu_tensor *tensor, float value) {
    float values[TEST_STATE_WORDS];
    for (uint32_t i = 0; i < TEST_STATE_WORDS; i++) values[i] = value;
    assert(ds4_gpu_tensor_write(tensor, 0, values, sizeof(values)) != 0);
}

static void fill_tensor_range(ds4_gpu_tensor *tensor, uint64_t offset,
                              float value) {
    float values[TEST_STATE_WORDS];
    for (uint32_t i = 0; i < TEST_STATE_WORDS; i++) values[i] = value;
    assert(ds4_gpu_tensor_write(tensor, offset, values, sizeof(values)) != 0);
}

static void expect_tensor(ds4_gpu_tensor *tensor, float value) {
    float values[TEST_STATE_WORDS];
    assert(ds4_gpu_tensor_read(tensor, 0, values, sizeof(values)) != 0);
    for (uint32_t i = 0; i < TEST_STATE_WORDS; i++) assert(values[i] == value);
}

static void state_fill(ds41_gpu_graph *g, uint32_t owner, uint32_t kind,
                       float value) {
    assert(owner < 4u && kind < 2u);
    fill_tensor(kind ? g->previous_score[owner] : g->previous_kv[owner], value);
}

static void state_expect(ds41_gpu_graph *g, uint32_t owner, uint32_t kind,
                         float value) {
    assert(owner < 4u && kind < 2u);
    expect_tensor(kind ? g->previous_score[owner] : g->previous_kv[owner], value);
}

static void window_fill(ds41_gpu_graph *g, uint32_t layer, uint32_t row,
                        float value) {
    assert(layer < DS4_N_LAYER && row < DS41_SPEC_ROWS);
    fill_tensor_range(g->window[layer],
        (uint64_t)((g->pos + row) % 128u) * TEST_ROW_BYTES, value);
}

static void window_expect(ds41_gpu_graph *g, uint32_t pos, uint32_t layer,
                          uint32_t row, float value) {
    float values[TEST_STATE_WORDS];
    assert(layer < DS4_N_LAYER && row < DS41_SPEC_ROWS);
    assert(ds4_gpu_tensor_read(g->window[layer],
        (uint64_t)((pos + row) % 128u) * TEST_ROW_BYTES,
        values, sizeof(values)) != 0);
    for (uint32_t i = 0; i < TEST_STATE_WORDS; i++) assert(values[i] == value);
}

static void graph_fixture_init(ds41_gpu_graph *g) {
    memset(g, 0, sizeof(*g));
    g->ctx = TEST_CTX;
    g->pos = TEST_POS;
    g->valid = true;
    g->history.tail[0] = 17;
    g->history.tail[1] = 18;
    g->history.tail[2] = 19;
    g->dspark_capture_generation = 23;
    g->dspark_capture_committed_generation = 21;
    g->dspark_capture_pos = 20;
    g->spec_raw_backup = ds4_gpu_tensor_alloc(
        (uint64_t)DS4_N_LAYER * DS41_SPEC_ROWS * TEST_ROW_BYTES);
    g->spec_state_initial = ds4_gpu_tensor_alloc(8u * TEST_ROW_BYTES);
    g->spec_state_prefix1 = ds4_gpu_tensor_alloc(8u * TEST_ROW_BYTES);
    g->spec_state_prefix2 = ds4_gpu_tensor_alloc(8u * TEST_ROW_BYTES);
    assert(g->spec_raw_backup && g->spec_state_initial &&
           g->spec_state_prefix1 && g->spec_state_prefix2);
    for (uint32_t layer = 0; layer < DS4_N_LAYER; layer++) {
        g->window[layer] = ds4_gpu_tensor_alloc(128u * TEST_ROW_BYTES);
        assert(g->window[layer]);
        for (uint32_t row = 0; row < DS41_SPEC_ROWS; row++)
            window_fill(g, layer, row, 1000.0f + layer * 10.0f + row);
    }
    for (uint32_t owner = 0; owner < 4u; owner++) {
        g->previous_kv[owner] = ds4_gpu_tensor_alloc(TEST_ROW_BYTES);
        g->previous_score[owner] = ds4_gpu_tensor_alloc(TEST_ROW_BYTES);
        assert(g->previous_kv[owner] && g->previous_score[owner]);
        state_fill(g, owner, 0, 2000.0f + owner);
        state_fill(g, owner, 1, 2100.0f + owner);
    }
}

static void graph_fixture_free(ds41_gpu_graph *g) {
    for (uint32_t layer = 0; layer < DS4_N_LAYER; layer++)
        ds4_gpu_tensor_free(g->window[layer]);
    for (uint32_t owner = 0; owner < 4u; owner++) {
        ds4_gpu_tensor_free(g->previous_kv[owner]);
        ds4_gpu_tensor_free(g->previous_score[owner]);
    }
    ds4_gpu_tensor_free(g->spec_raw_backup);
    ds4_gpu_tensor_free(g->spec_state_initial);
    ds4_gpu_tensor_free(g->spec_state_prefix1);
    ds4_gpu_tensor_free(g->spec_state_prefix2);
}

static void save_owner_prefixes(ds41_gpu_graph *g) {
    const uint32_t source_layers[] = {2u, 8u, 14u, 20u};
    for (uint32_t i = 0; i < 4u; i++) {
        const uint32_t owner = ds41_owner_for_layer(source_layers[i]);
        assert(owner == i);
        state_fill(g, owner, 0, 3000.0f + owner);
        state_fill(g, owner, 1, 3100.0f + owner);
        assert(ds4_gpu_begin_commands());
        assert(ds41_spec_copy_state_owner(g, g->spec_state_prefix1,
                                           owner, true));
        assert(ds4_gpu_end_commands());
    }
    for (uint32_t owner = 0; owner < 4u; owner++) {
        state_fill(g, owner, 0, 5000.0f + owner);
        state_fill(g, owner, 1, 5100.0f + owner);
        assert(ds4_gpu_begin_commands());
        assert(ds41_spec_copy_state_owner(g, g->spec_state_prefix2,
                                           owner, true));
        assert(ds4_gpu_end_commands());
    }
    for (uint32_t owner = 0; owner < 4u; owner++) {
        float values[TEST_STATE_WORDS];
        const uint64_t kv_offset = (uint64_t)(owner * 2u) * TEST_ROW_BYTES;
        const uint64_t score_offset = kv_offset + TEST_ROW_BYTES;
        assert(ds4_gpu_tensor_read(g->spec_state_prefix1, kv_offset,
                                    values, sizeof(values)) != 0);
        for (uint32_t i = 0; i < TEST_STATE_WORDS; i++)
            assert(values[i] == 3000.0f + owner);
        assert(ds4_gpu_tensor_read(g->spec_state_prefix1, score_offset,
                                    values, sizeof(values)) != 0);
        for (uint32_t i = 0; i < TEST_STATE_WORDS; i++)
            assert(values[i] == 3100.0f + owner);
        assert(ds4_gpu_tensor_read(g->spec_state_prefix2, kv_offset,
                                    values, sizeof(values)) != 0);
        for (uint32_t i = 0; i < TEST_STATE_WORDS; i++)
            assert(values[i] == 5000.0f + owner);
        assert(ds4_gpu_tensor_read(g->spec_state_prefix2, score_offset,
                                    values, sizeof(values)) != 0);
        for (uint32_t i = 0; i < TEST_STATE_WORDS; i++)
            assert(values[i] == 5100.0f + owner);
    }
}

static void mutate_rows_and_state(ds41_gpu_graph *g, float base) {
    for (uint32_t layer = 0; layer < DS4_N_LAYER; layer++)
        for (uint32_t row = 0; row < DS41_SPEC_ROWS; row++)
            window_fill(g, layer, row, base + layer * 10.0f + row);
    for (uint32_t owner = 0; owner < 4u; owner++) {
        state_fill(g, owner, 0, base + 100.0f + owner);
        state_fill(g, owner, 1, base + 200.0f + owner);
    }
}

static void test_frontier_restore(void) {
    ds41_gpu_graph g;
    ds41_spec_frontier frontier;
    graph_fixture_init(&g);
    assert(ds41_spec_frontier_snapshot(&frontier, &g, DS41_SPEC_ROWS));
    assert(frontier.valid && frontier.pos == TEST_POS &&
           frontier.n_rows == DS41_SPEC_ROWS);
    save_owner_prefixes(&g);

    mutate_rows_and_state(&g, 6000.0f);
    assert(ds41_spec_frontier_restore(&frontier, &g, 1));
    assert(g.pos == TEST_POS + 1u);
    window_expect(&g, TEST_POS, 0, 0, 6000.0f);
    window_expect(&g, TEST_POS, 0, 1, 1001.0f);
    window_expect(&g, TEST_POS, 0, 2, 1002.0f);
    for (uint32_t owner = 0; owner < 4u; owner++) {
        state_expect(&g, owner, 0, 3000.0f + owner);
        state_expect(&g, owner, 1, 3100.0f + owner);
    }

    assert(ds41_spec_frontier_restore_initial(&frontier, &g));
    assert(g.pos == TEST_POS);
    mutate_rows_and_state(&g, 7000.0f);
    assert(ds41_spec_frontier_restore(&frontier, &g, 2));
    assert(g.pos == TEST_POS + 2u);
    window_expect(&g, TEST_POS, 0, 0, 7000.0f);
    window_expect(&g, TEST_POS, 0, 1, 7001.0f);
    window_expect(&g, TEST_POS, 0, 2, 1002.0f);
    for (uint32_t owner = 0; owner < 4u; owner++) {
        state_expect(&g, owner, 0, 5000.0f + owner);
        state_expect(&g, owner, 1, 5100.0f + owner);
    }

    assert(ds41_spec_frontier_restore_initial(&frontier, &g));
    mutate_rows_and_state(&g, 8000.0f);
    assert(ds41_spec_frontier_restore(&frontier, &g, 3));
    assert(g.pos == TEST_POS + 3u);
    window_expect(&g, TEST_POS, 0, 0, 8000.0f);
    window_expect(&g, TEST_POS, 0, 1, 8001.0f);
    window_expect(&g, TEST_POS, 0, 2, 8002.0f);
    for (uint32_t owner = 0; owner < 4u; owner++) {
        state_expect(&g, owner, 0, 8100.0f + owner);
        state_expect(&g, owner, 1, 8200.0f + owner);
    }

    assert(ds41_spec_frontier_restore_initial(&frontier, &g));
    assert(g.pos == TEST_POS && g.history.tail[0] == 17 &&
           g.history.tail[1] == 18 && g.history.tail[2] == 19 &&
           g.dspark_capture_generation == 23 && g.dspark_capture_pos == 20);
    window_expect(&g, TEST_POS, 0, 0, 1000.0f);
    window_expect(&g, TEST_POS, 0, 1, 1001.0f);
    window_expect(&g, TEST_POS, 0, 2, 1002.0f);
    for (uint32_t owner = 0; owner < 4u; owner++) {
        state_expect(&g, owner, 0, 2000.0f + owner);
        state_expect(&g, owner, 1, 2100.0f + owner);
    }
    mutate_rows_and_state(&g, 9000.0f);
    assert(!ds41_spec_frontier_restore(&frontier, &g, 0));
    assert(!ds41_spec_frontier_restore(&frontier, &g, 4));
    assert(ds41_spec_frontier_restore_initial(&frontier, &g));

    g.pos = TEST_CTX - 2u;
    for (uint32_t layer = 0; layer < DS4_N_LAYER; layer++) {
        window_fill(&g, layer, 0, 9500.0f + layer);
        window_fill(&g, layer, 1, 9600.0f + layer);
    }
    ds41_spec_frontier legacy;
    assert(ds41_spec_frontier_snapshot(&legacy, &g, DS41_SPEC_LEGACY_ROWS));
    assert(legacy.n_rows == DS41_SPEC_LEGACY_ROWS);
    mutate_rows_and_state(&g, 9700.0f);
    assert(ds41_spec_frontier_restore(&legacy, &g, DS41_SPEC_LEGACY_ROWS));
    assert(g.pos == TEST_CTX);
    state_expect(&g, 0, 0, 9800.0f);
    g.pos = TEST_CTX - 2u;
    assert(!ds41_spec_frontier_snapshot(&legacy, &g, DS41_SPEC_ROWS));
    graph_fixture_free(&g);
}

int main(void) {
    test_forced_outcome_contract();
    test_cycle_commit_invariants();
    assert(ds4_gpu_init());
    test_frontier_restore();
    ds4_gpu_cleanup();
    puts("V4.1 DSpark cycle invariants and real snapshot/restore PASS");
    return 0;
}
