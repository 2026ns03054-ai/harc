/*
 * test_engine.c - checks the properties HARC relies on and measures prefill speed-up.
 *   1. batched prefill gives bit-identical logits/KV to token-by-token forward()
 *   2. resume from token log (fresh engine, batched re-prefill) -> identical output
 *   3. resume from KV checkpoint (fresh engine, KV import)       -> identical output
 *   4. prefill speed: sequential vs batched for N prompt tokens  -> speed-up factor
 * usage: test_engine model.bin tokenizer.bin [gen_tokens] [cut_at] [bench_tokens]
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "engine.h"

static double now_s(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }

/* generate G tokens after prompt; if start_prefix>0, resume from prefix using batched prefill */
static int generate(Engine *e, const int *prompt, int np, const int *prefix, int nprefix, int G, uint64_t seed, int *out, int batched) {
    int m = np + nprefix, pos;
    int *seqv = malloc(sizeof(int) * (m + G + 1));
    memcpy(seqv, prompt, np * sizeof(int)); memcpy(seqv + np, prefix, nprefix * sizeof(int));
    float *logits;
    if (batched) logits = engine_forward_batch(e, seqv, m, 0);
    else { for (pos = 0; pos < m; pos++) logits = engine_forward_one(e, seqv[pos], pos); }
    int g = nprefix;
    memcpy(out, prefix, nprefix * sizeof(int));
    for (pos = m; g < G; pos++) {
        int t = engine_sample(e, logits, seed, g);
        out[g++] = t;
        if (g == G) break;
        logits = engine_forward_one(e, t, pos);
    }
    free(seqv);
    return g;
}

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s model.bin tokenizer.bin [gen] [cut] [bench]\n", argv[0]); return 1; }
    int G = argc > 3 ? atoi(argv[3]) : 64, cut = argc > 4 ? atoi(argv[4]) : 25, B = argc > 5 ? atoi(argv[5]) : 128;
    uint64_t seed = 42;
    Engine *e = engine_load(argv[1], argv[2], 1.0f, 0.9f);
    int prompt[512];
    int np = engine_encode(e, "Once upon a time, there was a little girl who loved to play outside in the park with her friends.", prompt, 512);
    int *ref = malloc(sizeof(int) * G), *a = malloc(sizeof(int) * G), *b = malloc(sizeof(int) * G);
    int fails = 0;

    /* reference: everything token by token */
    generate(e, prompt, np, NULL, 0, G, seed, ref, 0);
    /* 1. batched prefill, same engine */
    generate(e, prompt, np, NULL, 0, G, seed, a, 1);
    int ok1 = memcmp(ref, a, sizeof(int) * G) == 0; fails += !ok1;
    printf("[1] batched prefill == sequential:            %s\n", ok1 ? "PASS" : "FAIL");

    /* 2. token-log resume on a fresh engine */
    Engine *e2 = engine_load(argv[1], argv[2], 1.0f, 0.9f);
    generate(e2, prompt, np, ref, cut, G, seed, b, 1);
    int ok2 = memcmp(ref, b, sizeof(int) * G) == 0; fails += !ok2;
    printf("[2] resume from token log (cut at %3d) == ref: %s\n", cut, ok2 ? "PASS" : "FAIL");

    /* 3. KV-checkpoint resume: positions [0, np+cut-1) exported from e, imported into fresh e3 */
    generate(e, prompt, np, NULL, 0, cut, seed, a, 0);       /* e now holds KV for np+cut-1 positions */
    int kvpos = np + cut - 1;
    size_t bytes = engine_kv_bytes_per_pos(e) * kvpos;
    void *kv = malloc(bytes);
    engine_kv_export(e, 0, kvpos, kv);
    Engine *e3 = engine_load(argv[1], argv[2], 1.0f, 0.9f);
    engine_kv_import(e3, 0, kvpos, kv);
    float *logits = engine_forward_one(e3, ref[cut - 1], kvpos);
    memcpy(b, ref, cut * sizeof(int));
    int g = cut;
    for (int pos = kvpos + 1; g < G; pos++) { int t = engine_sample(e3, logits, seed, g); b[g++] = t; if (g == G) break; logits = engine_forward_one(e3, t, pos); }
    int ok3 = memcmp(ref, b, sizeof(int) * G) == 0; fails += !ok3;
    printf("[3] resume from KV checkpoint (%zu B) == ref:  %s\n", bytes, ok3 ? "PASS" : "FAIL");
    printf("    KV bytes per position: %zu\n", engine_kv_bytes_per_pos(e));

    /* 3b. KV checkpoint that lags the token log: import [0, kvpos-d), re-prefill d+1 tokens in one batch */
    int ok3b = 1;
    for (int d = 1; d <= 6; d++) {
        Engine *e4 = engine_load(argv[1], argv[2], 1.0f, 0.9f);
        int kp = kvpos - d;
        engine_kv_import(e4, 0, kp, kv);                   /* position-major, so a prefix of the buffer */
        int *sq = malloc(sizeof(int) * (np + cut));
        memcpy(sq, prompt, np * sizeof(int)); memcpy(sq + np, ref, cut * sizeof(int));
        logits = engine_forward_batch(e4, sq + kp, np + cut - kp, kp);
        memcpy(b, ref, cut * sizeof(int)); g = cut;
        for (int pos = np + cut; g < G; pos++) { int t = engine_sample(e4, logits, seed, g); b[g++] = t; if (g == G) break; logits = engine_forward_one(e4, t, pos); }
        if (memcmp(ref, b, sizeof(int) * G)) ok3b = 0;
        free(sq); engine_free(e4);
    }
    fails += !ok3b;
    printf("[3b] lagging KV checkpoint + batched catch-up == ref: %s\n", ok3b ? "PASS" : "FAIL");

    /* 4. prefill speed */
    int *toks = malloc(sizeof(int) * B);
    for (int i = 0; i < B; i++) toks[i] = prompt[i % np];
    double t0 = now_s(); for (int p = 0; p < B; p++) engine_forward_one(e, toks[p], p); double ts = now_s() - t0;
    t0 = now_s(); engine_forward_batch(e, toks, B, 0); double tb = now_s() - t0;
    printf("[4] prefill %d tokens: sequential %.1f ms (%.3f ms/token), batched %.1f ms (%.3f ms/token), speed-up %.2fx\n",
           B, ts * 1e3, ts * 1e3 / B, tb * 1e3, tb * 1e3 / B, ts / tb);
    printf("%s\n", fails ? "SOME TESTS FAILED" : "ALL EXACTNESS TESTS PASSED");
    return fails ? 1 : 0;
}
