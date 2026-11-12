/*
 * engine.c - see engine.h. Includes llama2.c's run.c as a library (TESTING
 * hides its main()). Compile WITHOUT -fopenmp so every reduction runs in a
 * fixed order; this is what makes resumption bit-exact (Proposition 2).
 */
#define TESTING
#include "../third_party/llama2.c/run.c"
#include "engine.h"

struct Engine {
    Transformer t;
    Tokenizer tok;
    Sampler smp;
    /* batch buffers, sized for seq_len rows */
    float *X, *XB, *XB2, *Q, *HB, *HB2;
};

static void *xcalloc(size_t n, size_t s) { void *p = calloc(n, s); if (!p) { fprintf(stderr, "out of memory\n"); exit(1); } return p; }

Engine *engine_load(const char *ckpt, const char *tok_path, float temperature, float topp) {
    Engine *e = xcalloc(1, sizeof *e);
    build_transformer(&e->t, (char *)ckpt);
    build_tokenizer(&e->tok, (char *)tok_path, e->t.config.vocab_size);
    build_sampler(&e->smp, e->t.config.vocab_size, temperature, topp, 0);
    Config *p = &e->t.config;
    size_t S = p->seq_len;
    e->X = xcalloc(S * p->dim, sizeof(float));
    e->XB = xcalloc(S * p->dim, sizeof(float));
    e->XB2 = xcalloc(S * p->dim, sizeof(float));
    e->Q = xcalloc(S * p->dim, sizeof(float));
    e->HB = xcalloc(S * p->hidden_dim, sizeof(float));
    e->HB2 = xcalloc(S * p->hidden_dim, sizeof(float));
    return e;
}

void engine_free(Engine *e) {
    free(e->X); free(e->XB); free(e->XB2); free(e->Q); free(e->HB); free(e->HB2);
    free_sampler(&e->smp); free_tokenizer(&e->tok); free_transformer(&e->t); free(e);
}

int engine_seq_len(Engine *e) { return e->t.config.seq_len; }
int engine_n_layers(Engine *e) { return e->t.config.n_layers; }
int engine_kv_dim(Engine *e) { Config *p = &e->t.config; return (p->dim * p->n_kv_heads) / p->n_heads; }
size_t engine_kv_bytes_per_pos(Engine *e) { return (size_t)2 * engine_n_layers(e) * engine_kv_dim(e) * sizeof(float); }

int engine_encode(Engine *e, const char *text, int *out, int max) {
    int n = 0;
    int *tmp = xcalloc(strlen(text) + 3, sizeof(int));
    encode(&e->tok, (char *)text, 1, 0, tmp, &n);
    if (n > max) n = max;
    memcpy(out, tmp, n * sizeof(int));
    free(tmp);
    return n;
}
const char *engine_decode(Engine *e, int prev, int tok) { return decode(&e->tok, prev, tok); }

float *engine_forward_one(Engine *e, int token, int pos) { return forward(&e->t, token, pos); }

/* Y(n x d) = X(n x k) * W(d x k)^T ; same per-element summation order as matmul() */
static void matmul_batch(float *Y, const float *X, const float *W, int n, int k, int d) {
    /* four independent accumulators share each weight load; every accumulator
       still sums j = 0..k-1 in order, so results equal matmul() bit for bit */
    for (int i = 0; i < d; i++) {
        const float *wr = W + (size_t)i * k;
        int b = 0;
        for (; b + 4 <= n; b += 4) {
            const float *x0 = X + (size_t)b * k, *x1 = x0 + k, *x2 = x1 + k, *x3 = x2 + k;
            float v0 = 0.0f, v1 = 0.0f, v2 = 0.0f, v3 = 0.0f;
            for (int j = 0; j < k; j++) { float wj = wr[j]; v0 += wj * x0[j]; v1 += wj * x1[j]; v2 += wj * x2[j]; v3 += wj * x3[j]; }
            Y[(size_t)b * d + i] = v0; Y[(size_t)(b + 1) * d + i] = v1; Y[(size_t)(b + 2) * d + i] = v2; Y[(size_t)(b + 3) * d + i] = v3;
        }
        for (; b < n; b++) {
            const float *xr = X + (size_t)b * k;
            float val = 0.0f;
            for (int j = 0; j < k; j++) val += wr[j] * xr[j];
            Y[(size_t)b * d + i] = val;
        }
    }
}

float *engine_forward_batch(Engine *e, const int *tokens, int n, int pos0) {
    if (n <= 0) return NULL;
    if (n == 1) return forward(&e->t, tokens[0], pos0);
    Config *p = &e->t.config; TransformerWeights *w = &e->t.weights; RunState *s = &e->t.state;
    int dim = p->dim, hidden = p->hidden_dim, kv_dim = engine_kv_dim(e);
    int kv_mul = p->n_heads / p->n_kv_heads, head_size = dim / p->n_heads;
    for (int b = 0; b < n; b++) memcpy(e->X + (size_t)b * dim, w->token_embedding_table + (size_t)tokens[b] * dim, dim * sizeof(float));

    for (unsigned long long l = 0; l < (unsigned long long)p->n_layers; l++) {
        size_t loff = l * p->seq_len * kv_dim;
        for (int b = 0; b < n; b++) rmsnorm(e->XB + (size_t)b * dim, e->X + (size_t)b * dim, w->rms_att_weight + l * dim, dim);
        matmul_batch(e->Q, e->XB, w->wq + l * dim * dim, n, dim, dim);
        /* K and V go straight into the cache rows pos0..pos0+n-1 (contiguous) */
        matmul_batch(s->key_cache + loff + (size_t)pos0 * kv_dim, e->XB, w->wk + l * dim * kv_dim, n, dim, kv_dim);
        matmul_batch(s->value_cache + loff + (size_t)pos0 * kv_dim, e->XB, w->wv + l * dim * kv_dim, n, dim, kv_dim);
        for (int b = 0; b < n; b++) {          /* RoPE, identical to forward() */
            int pos = pos0 + b;
            float *q = e->Q + (size_t)b * dim, *k = s->key_cache + loff + (size_t)pos * kv_dim;
            for (int i = 0; i < dim; i += 2) {
                int head_dim = i % head_size;
                float freq = 1.0f / powf(10000.0f, head_dim / (float)head_size);
                float val = pos * freq, fcr = cosf(val), fci = sinf(val);
                int rotn = i < kv_dim ? 2 : 1;
                for (int v = 0; v < rotn; v++) {
                    float *vec = v == 0 ? q : k;
                    float v0 = vec[i], v1 = vec[i + 1];
                    vec[i] = v0 * fcr - v1 * fci; vec[i + 1] = v0 * fci + v1 * fcr;
                }
            }
        }
        for (int b = 0; b < n; b++) {          /* causal attention per position */
            int pos = pos0 + b;
            for (int h = 0; h < p->n_heads; h++) {
                float *q = e->Q + (size_t)b * dim + h * head_size, *att = s->att + h * p->seq_len;
                for (int t = 0; t <= pos; t++) {
                    float *k = s->key_cache + loff + (size_t)t * kv_dim + (h / kv_mul) * head_size;
                    float score = 0.0f;
                    for (int i = 0; i < head_size; i++) score += q[i] * k[i];
                    att[t] = score / sqrtf(head_size);
                }
                softmax(att, pos + 1);
                float *xb = e->XB + (size_t)b * dim + h * head_size;
                memset(xb, 0, head_size * sizeof(float));
                for (int t = 0; t <= pos; t++) {
                    float *v = s->value_cache + loff + (size_t)t * kv_dim + (h / kv_mul) * head_size;
                    float a = att[t];
                    for (int i = 0; i < head_size; i++) xb[i] += a * v[i];
                }
            }
        }
        matmul_batch(e->XB2, e->XB, w->wo + l * dim * dim, n, dim, dim);
        for (size_t i = 0; i < (size_t)n * dim; i++) e->X[i] += e->XB2[i];
        for (int b = 0; b < n; b++) rmsnorm(e->XB + (size_t)b * dim, e->X + (size_t)b * dim, w->rms_ffn_weight + l * dim, dim);
        matmul_batch(e->HB, e->XB, w->w1 + l * dim * hidden, n, dim, hidden);
        matmul_batch(e->HB2, e->XB, w->w3 + l * dim * hidden, n, dim, hidden);
        for (size_t i = 0; i < (size_t)n * hidden; i++) {
            float val = e->HB[i];
            val *= (1.0f / (1.0f + expf(-val)));
            val *= e->HB2[i];
            e->HB[i] = val;
        }
        matmul_batch(e->XB, e->HB, w->w2 + l * dim * hidden, n, hidden, dim);
        for (size_t i = 0; i < (size_t)n * dim; i++) e->X[i] += e->XB[i];
    }
    /* logits only for the last position, exactly as forward() would produce */
    float *xl = e->X + (size_t)(n - 1) * dim;
    memcpy(s->x, xl, dim * sizeof(float));
    rmsnorm(s->x, s->x, w->rms_final_weight, dim);
    matmul(s->logits, s->x, w->wcls, p->dim, p->vocab_size);
    return s->logits;
}

static uint64_t splitmix64(uint64_t z) {
    z += 0x9E3779B97F4A7C15ull;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

int engine_sample(Engine *e, float *logits, uint64_t seed, int g) {
    Sampler *sm = &e->smp;
    if (sm->temperature == 0.0f) return sample_argmax(logits, sm->vocab_size);
    for (int q = 0; q < sm->vocab_size; q++) logits[q] /= sm->temperature;
    softmax(logits, sm->vocab_size);
    uint64_t r = splitmix64(seed ^ splitmix64((uint64_t)g + 1));
    float coin = (float)(r >> 40) / 16777216.0f;           /* 24 bits -> [0,1) */
    if (sm->topp <= 0 || sm->topp >= 1) return sample_mult(logits, sm->vocab_size, coin);
    return sample_topp(logits, sm->vocab_size, sm->topp, sm->probindex, coin);
}

/* position-major layout so checkpoints can be appended: for p in [a,b): for l: K[l][p], V[l][p] */
size_t engine_kv_export(Engine *e, int a, int b, void *buf) {
    Config *p = &e->t.config; RunState *s = &e->t.state;
    int kv_dim = engine_kv_dim(e); size_t row = (size_t)kv_dim * sizeof(float); char *o = buf;
    for (int pos = a; pos < b; pos++)
        for (int l = 0; l < p->n_layers; l++) {
            size_t off = (size_t)l * p->seq_len * kv_dim + (size_t)pos * kv_dim;
            memcpy(o, s->key_cache + off, row); o += row;
            memcpy(o, s->value_cache + off, row); o += row;
        }
    return (size_t)(o - (char *)buf);
}

void engine_kv_import(Engine *e, int a, int b, const void *buf) {
    Config *p = &e->t.config; RunState *s = &e->t.state;
    int kv_dim = engine_kv_dim(e); size_t row = (size_t)kv_dim * sizeof(float); const char *in = buf;
    for (int pos = a; pos < b; pos++)
        for (int l = 0; l < p->n_layers; l++) {
            size_t off = (size_t)l * p->seq_len * kv_dim + (size_t)pos * kv_dim;
            memcpy(s->key_cache + off, in, row); in += row;
            memcpy(s->value_cache + off, in, row); in += row;
        }
}

uint64_t fnv1a_tokens(const int *t, int n) {
    uint64_t h = 1469598103934665603ull;
    for (int i = 0; i < n; i++) { uint32_t v = (uint32_t)t[i]; for (int k = 0; k < 4; k++) { h ^= (v >> (8 * k)) & 0xff; h *= 1099511628211ull; } }
    return h;
}

/* ablation for RQ5: conventional stateful RNG (state restarts from the seed on every attempt) */
int engine_sample_stateful(Engine *e, float *logits, uint64_t *state) {
    Sampler *sm = &e->smp;
    if (sm->temperature == 0.0f) return sample_argmax(logits, sm->vocab_size);
    for (int q = 0; q < sm->vocab_size; q++) logits[q] /= sm->temperature;
    softmax(logits, sm->vocab_size);
    float coin = random_f32((unsigned long long *)state);
    if (sm->topp <= 0 || sm->topp >= 1) return sample_mult(logits, sm->vocab_size, coin);
    return sample_topp(logits, sm->vocab_size, sm->topp, sm->probindex, coin);
}
