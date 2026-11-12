/*
 * client.c - workload generator for HARC experiments
 *
 * Sends N inference requests (JOB) to a master and records each RESULT.
 * Request i always gets the same prompt, length and seed for a given --seed,
 * so runs under different policies can be compared request by request.
 *
 * usage: client <master_ip:port> [--n 40] [--mix 256:1.0 | --mix 128:0.5,512:0.5]
 *               [--concurrency 4 | --rate R] [--seed 1] [--timeout 900] [--out client.csv]
 */
#include "net.h"
#include <math.h>
#include <semaphore.h>

static const char *PROMPTS[] = {
    "Once upon a time, there was a little girl named Lily who loved to play in the park.",
    "Tom had a red ball. One day he kicked it over the fence and",
    "The sun was shining and the birds were singing when Ben found a small box.",
    "Sara wanted to bake a cake for her mom, but she did not have any eggs.",
    "In a big forest lived a kind old bear who liked to share his honey.",
    "Max the dog was very sad because he could not find his favourite toy.",
    "One rainy day, Anna and her brother built a fort out of blankets.",
    "There was a tiny boat on the lake. A frog jumped onto it and said",
};
#define NPROMPTS (int)(sizeof PROMPTS / sizeof PROMPTS[0])

static char g_host[64]; static int g_port, g_timeout = 900;
static FILE *g_out; static pthread_mutex_t g_mx = PTHREAD_MUTEX_INITIALIZER;
static sem_t g_slots; static int g_closed = 1;
static int g_ok = 0, g_fail = 0;

typedef struct { int i, L; unsigned long long seed; } job_t;

static uint64_t mix64(uint64_t x) { x += 0x9E3779B97F4A7C15ULL; x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL; x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL; return x ^ (x >> 31); }

static void *run_job(void *a) {
    job_t *j = a; char cid[32]; snprintf(cid, sizeof cid, "q%04d", j->i);
    double t0 = wall_s(); int fd = connect_to(g_host, g_port, 3000, g_timeout);
    char line[LINE_LEN]; int ok = 0; int G = 0, fails = 0, att = 0; unsigned long long h = 0; double mlat = 0, qms = 0;
    if (fd >= 0 && send_line(fd, "JOB %s %d %llu", cid, j->L, j->seed) == 0 &&
        send_line(fd, "PROMPT %s", PROMPTS[j->i % NPROMPTS]) == 0 && recv_line(fd, line, sizeof line) > 0) {
        char rid[64];
        ok = sscanf(line, "RESULT %63s %d %llu %d %d %lf %lf", rid, &G, &h, &fails, &att, &mlat, &qms) == 7;
    }
    if (fd >= 0) close(fd);
    double t1 = wall_s();
    pthread_mutex_lock(&g_mx);
    fprintf(g_out, "%s,%d,%llu,%.6f,%.6f,%.3f,%d,%d,%llu,%d,%d,%.3f,%.3f\n", cid, j->L, j->seed, t0, t1, (t1 - t0) * 1e3, ok, G, h, fails, att, mlat, qms);
    fflush(g_out);
    if (ok) g_ok++; else g_fail++;
    printf("[client] %s L=%d %s G=%d failures=%d latency=%.0f ms\n", cid, j->L, ok ? "ok" : "FAILED", G, fails, (t1 - t0) * 1e3);
    pthread_mutex_unlock(&g_mx);
    if (g_closed) sem_post(&g_slots);
    free(j); return NULL;
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <master_ip:port> [--n N] [--mix L:p,...] [--concurrency C | --rate R] [--seed S] [--timeout s] [--out file]\n", argv[0]); return 1; }
    signal(SIGPIPE, SIG_IGN); setvbuf(stdout, NULL, _IONBF, 0);
    if (parse_hostport(argv[1], g_host, sizeof g_host, &g_port) < 0) { fprintf(stderr, "bad master address\n"); return 1; }
    int n = 40, conc = 4; double rate = 0; unsigned long long seed = 1; char out[256] = "client.csv", mix[256] = "256:1.0";
    for (int i = 2; i < argc; i++) {
        if (!strcmp(argv[i], "--n") && i + 1 < argc) n = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--mix") && i + 1 < argc) snprintf(mix, sizeof mix, "%s", argv[++i]);
        else if (!strcmp(argv[i], "--concurrency") && i + 1 < argc) conc = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--rate") && i + 1 < argc) { rate = atof(argv[++i]); g_closed = 0; }
        else if (!strcmp(argv[i], "--seed") && i + 1 < argc) seed = strtoull(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--timeout") && i + 1 < argc) g_timeout = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--out") && i + 1 < argc) snprintf(out, sizeof out, "%s", argv[++i]);
    }
    int Ls[16]; double Ps[16]; int nm = 0; double ptot = 0;
    for (char *sv, *tok = strtok_r(mix, ",", &sv); tok && nm < 16; tok = strtok_r(NULL, ",", &sv)) {
        Ls[nm] = atoi(tok); char *c = strchr(tok, ':'); Ps[nm] = c ? atof(c + 1) : 1.0; ptot += Ps[nm]; nm++;
    }
    g_out = fopen(out, "w");
    fprintf(g_out, "cid,L,seed,t_submit,t_done,latency_ms,ok,G,hash,failures,attempts,master_latency_ms,queue_ms\n");
    sem_init(&g_slots, 0, conc > 0 ? conc : 1);
    pthread_t *th = calloc(n, sizeof *th);
    uint64_t arr = mix64(seed * 7919 + 17);
    double t0 = wall_s();
    for (int i = 0; i < n; i++) {
        job_t *j = malloc(sizeof *j); j->i = i;
        j->seed = mix64(seed * 1000003ULL + (unsigned long long)i);
        double u = (mix64(j->seed ^ 0xABCDEFULL) >> 11) * (1.0 / 9007199254740992.0) * ptot, acc = 0; j->L = Ls[nm - 1];
        for (int m = 0; m < nm; m++) { acc += Ps[m]; if (u < acc) { j->L = Ls[m]; break; } }
        if (g_closed) sem_wait(&g_slots);
        else if (i > 0) { arr = mix64(arr); double v = ((arr >> 11) + 0.5) * (1.0 / 9007199254740992.0); usleep((useconds_t)(-log(v) / rate * 1e6)); }
        pthread_create(&th[i], NULL, run_job, j);
    }
    for (int i = 0; i < n; i++) pthread_join(th[i], NULL);
    printf("[client] %d ok, %d failed, makespan %.1f s\n", g_ok, g_fail, wall_s() - t0);
    fclose(g_out);
    return g_fail ? 2 : 0;
}
