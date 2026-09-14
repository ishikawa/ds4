#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

enum {
    FIXTURE_LAYERS = 2,
    FIXTURE_ROWS = 3,
};

typedef struct {
    uint32_t pos;
    uint32_t history;
    int raw[FIXTURE_LAYERS][FIXTURE_ROWS];
    int state_initial;
    int state_prefix1;
    int state_prefix2;
    int state_current;
    bool valid;
} spec_fixture;

static void fixture_snapshot(spec_fixture *f, const spec_fixture *source) {
    *f = *source;
    f->state_initial = source->state_current;
    f->state_prefix1 = f->state_initial;
    f->state_prefix2 = f->state_initial;
}

static void fixture_verify(spec_fixture *f) {
    for (int layer = 0; layer < FIXTURE_LAYERS; layer++) {
        for (int row = 0; row < FIXTURE_ROWS; row++) {
            f->raw[layer][row] = 100 + layer * 10 + row;
        }
    }
    f->state_prefix1 = 201;
    f->state_prefix2 = 202;
    f->state_current = 203;
    f->history = 303;
    f->pos += FIXTURE_ROWS;
}

static bool fixture_restore_initial(spec_fixture *f, const spec_fixture *snapshot) {
    if (!f || !snapshot || !snapshot->valid) return false;
    for (int layer = 0; layer < FIXTURE_LAYERS; layer++) {
        for (int row = 0; row < FIXTURE_ROWS; row++) {
            f->raw[layer][row] = snapshot->raw[layer][row];
        }
    }
    f->pos = snapshot->pos;
    f->history = snapshot->history;
    f->state_current = snapshot->state_initial;
    f->valid = true;
    return true;
}

static bool fixture_restore_rows(spec_fixture *f,
                                 const spec_fixture *snapshot,
                                 uint32_t accepted_rows) {
    if (!f || !snapshot || !snapshot->valid ||
        accepted_rows < 1u || accepted_rows > FIXTURE_ROWS) return false;
    for (int layer = 0; layer < FIXTURE_LAYERS; layer++) {
        for (uint32_t row = accepted_rows; row < FIXTURE_ROWS; row++) {
            f->raw[layer][row] = snapshot->raw[layer][row];
        }
    }
    if (accepted_rows == 1u) f->state_current = f->state_prefix1;
    else if (accepted_rows == 2u) f->state_current = f->state_prefix2;
    f->pos = snapshot->pos + accepted_rows;
    f->valid = true;
    return true;
}

static bool fixture_restore_legacy(spec_fixture *f,
                                   const spec_fixture *snapshot,
                                   uint32_t accepted_drafts) {
    if (accepted_drafts > 2u) return false;
    if (accepted_drafts == 0u) return fixture_restore_initial(f, snapshot);
    return fixture_restore_rows(f, snapshot, accepted_drafts);
}

static bool check_rows(uint32_t accepted_rows, int expected_state) {
    spec_fixture source = {
        .pos = 40,
        .history = 7,
        .raw = {{10, 11, 12}, {20, 21, 22}},
        .state_current = 100,
        .valid = true,
    };
    spec_fixture snapshot;
    fixture_snapshot(&snapshot, &source);
    spec_fixture actual = source;
    fixture_verify(&actual);
    if (!fixture_restore_rows(&actual, &snapshot, accepted_rows)) return false;
    if (actual.pos != source.pos + accepted_rows ||
        actual.state_current != expected_state || actual.history != 303) return false;
    for (int layer = 0; layer < FIXTURE_LAYERS; layer++) {
        for (uint32_t row = 0; row < FIXTURE_ROWS; row++) {
            const int expected = row < accepted_rows ?
                100 + layer * 10 + (int)row : source.raw[layer][row];
            if (actual.raw[layer][row] != expected) return false;
        }
    }
    return true;
}

static bool check_error_fallback(void) {
    spec_fixture source = {
        .pos = 40,
        .history = 7,
        .raw = {{10, 11, 12}, {20, 21, 22}},
        .state_current = 100,
        .valid = true,
    };
    spec_fixture snapshot;
    fixture_snapshot(&snapshot, &source);
    spec_fixture actual = source;
    fixture_verify(&actual);
    if (!fixture_restore_legacy(&actual, &snapshot, 0u) ||
        memcmp(actual.raw, source.raw, sizeof(source.raw)) != 0 ||
        actual.pos != source.pos || actual.history != source.history ||
        actual.state_current != source.state_current) return false;
    if (fixture_restore_rows(&actual, &snapshot, 0u) ||
        fixture_restore_rows(&actual, &snapshot, 4u) ||
        fixture_restore_legacy(&actual, &snapshot, 3u)) return false;
    return true;
}

int main(void) {
    if (!check_rows(1u, 201) || !check_rows(2u, 202) ||
        !check_rows(3u, 203) || !check_error_fallback()) {
        fprintf(stderr, "V4.1 DSpark snapshot/restore fixture failed\n");
        return 1;
    }
    puts("V4.1 DSpark snapshot/restore fixture passed");
    return 0;
}
