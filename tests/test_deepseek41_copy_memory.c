/* Metadata-only memory admission regression; never initializes Metal. */
#include "../ds4.c"

#define REQUIRE(x) do { if (!(x)) { \
    fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x); goto done; \
} } while (0)

/* Metadata-only: no GGUF, mmap, Metal initialization, or GPU allocation. */
static int check_resident_copy_memory(void) {
    int rc = 1;
    const uint64_t gib = UINT64_C(1073741824);
    g_ds4_shape = DS4_SHAPE_FLASH41;
    ds4_tensor gate = {.type = DS4_TENSOR_IQ2_XXS, .ndim = 3,
        .dim = {5120, 2304, 384}, .abs_offset = 4224};
    ds4_tensor up = gate, down = {.type = DS4_TENSOR_Q2_K, .ndim = 3,
        .dim = {2304, 5120, 384}};
    gate.bytes = routed_expert_row_bytes(&gate) * gate.dim[1] * gate.dim[2];
    up.bytes = gate.bytes;
    up.abs_offset = gate.abs_offset + gate.bytes + 96u;
    down.bytes = routed_expert_row_bytes(&down) * down.dim[1] * down.dim[2];
    down.abs_offset = up.abs_offset + up.bytes + 96u;
    ds4_engine e = {.backend = DS4_BACKEND_METAL, .ssd_streaming = true};
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        e.weights.layer[il].ffn_gate_exps = &gate;
        e.weights.layer[il].ffn_up_exps = &up;
        e.weights.layer[il].ffn_down_exps = &down;
    }
    uint64_t max_bytes = 0, expected = 0;
    const ds4_tensor *tensors[] = {&gate, &up, &down};
    for (uint32_t i = 0; i < 3; i++)
        expected = ((expected + 4095u) & ~UINT64_C(4095)) +
                   (tensors[i]->abs_offset & 4095u) + tensors[i]->bytes;
    REQUIRE(metal_graph_v41_prefill_layer_copy_max_bytes(&e.weights, &max_bytes));
    REQUIRE(max_bytes == expected);
    /* Include an outlier at the last layer, not just the first layer's size. */
    ds4_tensor tail = down;
    tail.bytes += 8192u;
    e.weights.layer[39].ffn_down_exps = &tail;
    REQUIRE(metal_graph_v41_prefill_layer_copy_max_bytes(&e.weights, &max_bytes));
    REQUIRE(max_bytes == expected + 8192u);
    e.weights.layer[39].ffn_down_exps = &down;
    uint64_t expert = 0;
    REQUIRE(ds4_streaming_routed_expert_bytes(&e.weights, &expert));
    REQUIRE(expert == 9953280u);
    const uint64_t reserve = 2u * 384u * expert;
    e.ssd_streaming_cache_bytes = 42u * gib;
    REQUIRE(ds4_engine_configure_streaming_cache_budget(&e));
    REQUIRE(e.ssd_streaming_prefill_headroom_bytes == reserve);
    e.v41_prefill_layer_copy_bytes = 2u * max_bytes;
    e.ssd_streaming_cache_bytes = 42u * gib;
    REQUIRE(ds4_engine_configure_streaming_cache_budget(&e));
    REQUIRE(e.ssd_streaming_prefill_headroom_bytes == 0);
    e.quality = true;
    e.ssd_streaming_cache_bytes = 42u * gib;
    REQUIRE(ds4_engine_configure_streaming_cache_budget(&e));
    REQUIRE(e.ssd_streaming_prefill_headroom_bytes == reserve);
    e.quality = false;
    REQUIRE(setenv("DS4_METAL_ENABLE_STREAMING_FULL_EXPERT_ADDR_TABLE", "1", 1) == 0);
    e.ssd_streaming_cache_bytes = 42u * gib;
    REQUIRE(ds4_engine_configure_streaming_cache_budget(&e));
    REQUIRE(e.ssd_streaming_prefill_headroom_bytes == reserve);
    unsetenv("DS4_METAL_ENABLE_STREAMING_FULL_EXPERT_ADDR_TABLE");
    e.ssd_streaming_prefill_headroom_bytes = 0;
    /* Admission accounts for permanent copies once, including later sessions. */
    ds4_tensor fixed = {.type = DS4_TENSOR_F16, .bytes = 9u * gib,
                       .abs_offset = 100u * gib};
    e.weights.layer[0].attn_norm = &fixed;
    REQUIRE(setenv("DS4_V41_MEMORY_BUDGET_GB", "56", 1) == 0);
    const uint64_t graph = ds41_graph_bytes(32768);
    e.ssd_streaming_cache_experts = 40u * 384u;
    REQUIRE(ds41_memory_admit_for_host(&e, graph, true, 128u * gib, 110u * gib));
    const uint64_t total = fixed.bytes + graph + 2u * gib +
        e.v41_prefill_layer_copy_bytes + e.ssd_streaming_cache_bytes;
    REQUIRE(total <= 56u * gib && total + expert > 56u * gib);
    const uint32_t fitted = e.ssd_streaming_cache_experts;
    REQUIRE(!ds41_memory_admit_for_host(&e, graph + expert, false, 128u * gib, 110u * gib));
    REQUIRE(e.ssd_streaming_cache_experts == fitted);
    /* Shared storage is charged once at physical slot stride, guard intact. */
    REQUIRE(ds4_window_layout(max_bytes, expert, 16384,
        &e.v41_copy_window_stride, &e.v41_copy_window_slots));
    e.ssd_streaming_cache_experts = 40u * 384u;
    REQUIRE(ds41_memory_admit_for_host(&e, graph, true, 128u * gib, 110u * gib));
    REQUIRE(e.ssd_streaming_cache_bytes ==
        (uint64_t)e.ssd_streaming_cache_experts * e.v41_copy_window_stride);
    REQUIRE(fixed.bytes + graph + 2u * gib + e.ssd_streaming_cache_bytes <= 56u * gib);
    REQUIRE(fixed.bytes + graph + 2u * gib + e.ssd_streaming_cache_bytes +
        e.v41_copy_window_stride > 56u * gib);
    REQUIRE(e.ssd_streaming_cache_experts > fitted);
    REQUIRE(e.ssd_streaming_cache_experts > 2u * e.v41_copy_window_slots);
    REQUIRE(!ds41_memory_admit_for_host(&e, graph + e.v41_copy_window_stride,
        false, 128u * gib, 110u * gib));
    e.ssd_streaming_cache_experts = 2u * e.v41_copy_window_slots;
    REQUIRE(!ds41_memory_admit_for_host(&e, graph, true, 128u * gib, 110u * gib));
    /* Computed examples anchored to reported rounded 32768 plan components.
     * This fixture has no GGUF metadata; exact file bytes may shift a slot. */
    const uint64_t reported_copy = (uint64_t)(9.924 * (double)gib);
    fixed.bytes = 54u * gib - graph - 2893u * expert - reported_copy;
    e.v41_prefill_layer_copy_bytes = reported_copy;
    REQUIRE(ds4_window_layout(reported_copy / 2u, expert, 16384,
        &e.v41_copy_window_stride, &e.v41_copy_window_slots));
    const uint32_t contexts[] = {32768, 131072};
    for (uint32_t i = 0; i < 2; i++) {
        const uint64_t runtime = ds41_graph_bytes(contexts[i]);
        e.ssd_streaming_cache_bytes = 42u * gib;
        REQUIRE(ds4_engine_configure_streaming_cache_budget(&e));
        REQUIRE(ds41_memory_admit_for_host(&e, runtime, true, 128u * gib, 110u * gib));
        fprintf(stderr, "computed fixture ctx=%u graph=%.6f static=%.6f cache=%.6f planned=%.6f prefill=%u decode=%u window_slots=%u\n",
            contexts[i], ds4_bytes_to_gib(runtime), ds4_bytes_to_gib(fixed.bytes),
            ds4_bytes_to_gib(e.ssd_streaming_cache_bytes),
            ds4_bytes_to_gib(fixed.bytes + runtime + e.ssd_streaming_cache_bytes),
            e.ssd_streaming_cache_experts - 2u * e.v41_copy_window_slots,
            e.ssd_streaming_cache_experts, e.v41_copy_window_slots);
        e.ssd_streaming_static_decode_map = true;
        e.ssd_streaming_decode_map_bytes = fixed.bytes;
        ds4_engine_print_startup_memory(&e, (int)contexts[i]);
    }
    puts("V4.1 resident copy packing, reserve replacement and memory admission: PASS");
    rc = 0;
done:
    unsetenv("DS4_METAL_ENABLE_STREAMING_FULL_EXPERT_ADDR_TABLE");
    unsetenv("DS4_V41_MEMORY_BUDGET_GB");
    return rc;
}

int main(void) { return check_resident_copy_memory(); }
