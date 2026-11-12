/*
 * engine.h - thin inference engine on top of llama2.c (MIT, A. Karpathy)
 *  - batched prefill (matrix-matrix) that is bit-identical to token-by-token forward()
 *  - counter-based sampling: coin for generated token g = H(seed, g)
 *  - KV cache export / import for checkpoint recovery
 */
#ifndef HARC_ENGINE_H
#define HARC_ENGINE_H
#include <stdint.h>
#include <stddef.h>

typedef struct Engine Engine;

Engine *engine_load(const char *checkpoint, const char *tokenizer_path, float temperature, float topp);
void    engine_free(Engine *e);

int  engine_seq_len(Engine *e);
int  engine_n_layers(Engine *e);
int  engine_kv_dim(Engine *e);
size_t engine_kv_bytes_per_pos(Engine *e);          /* bytes of K+V for one position, all layers */

/* tokenizer helpers */
int  engine_encode(Engine *e, const char *text, int *out, int max);   /* with BOS */
const char *engine_decode(Engine *e, int prev, int tok);

/* forward passes; both write the KV cache and return logits of the last position */
float *engine_forward_one(Engine *e, int token, int pos);
float *engine_forward_batch(Engine *e, const int *tokens, int n, int pos0);

/* counter-based sampling: deterministic given (seed, g) */
int  engine_sample(Engine *e, float *logits, uint64_t seed, int g);

/* KV cache checkpoint for positions [a, b), position-major (appendable): for each pos: for each layer: K row, V row */
size_t engine_kv_export(Engine *e, int a, int b, void *buf);
void   engine_kv_import(Engine *e, int a, int b, const void *buf);

uint64_t fnv1a_tokens(const int *t, int n);
int engine_sample_stateful(Engine *e, float *logits, uint64_t *state);
#endif
