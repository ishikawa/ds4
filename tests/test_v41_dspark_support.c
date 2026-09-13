#include "ds4.h"

#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    if (argc != 4) {
        fprintf(stderr,
                "usage: %s SUPPORT.gguf BASE_REVISION RESIDENT_BUDGET_BYTES\n",
                argv[0]);
        return 2;
    }
    char *end = NULL;
    uint64_t budget = strtoull(argv[3], &end, 10);
    if (!end || *end != '\0') {
        fprintf(stderr, "invalid resident budget: %s\n", argv[3]);
        return 2;
    }

    ds4_test_v41_dspark_support_result result;
    int rc = ds4_test_v41_dspark_support_model(
        argv[1], argv[2], budget, &result);
    printf("v41 dspark support: tensors=%" PRIu64 " bytes=%" PRIu64
           " stages=%u experts=%u top_k=%u targets=%u markov_rank=%u "
           "hc_head=%s missing=%u invalid=%u metadata_errors=%u\n",
           result.tensor_count, result.file_bytes, result.stages,
           result.n_expert, result.n_expert_used, result.target_layer_count,
           result.markov_rank, result.has_hc_head ? "yes" : "no",
           result.missing_tensors, result.invalid_tensors,
           result.metadata_errors);
    return rc;
}
