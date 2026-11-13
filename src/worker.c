/*
 * worker.c - HARC inference worker
 *
 * Registers with the master, sends heartbeats, and serves TASKs one at a time.
 * A TASK carries the prompt, the tokens already committed, an optional KV
 * checkpoint, a commit interval k_c and a recovery mode. The worker resumes
 * from that state, streams COMMIT (and KV) messages every k_c tokens, and
 * finishes with DONE. Sampling uses a counter-based RNG so that resumed output
 * is identical to an uninterrupted run.
 *
 * usage: worker <name> <my_ip> <my_port> <master_ip:port> <model.bin> <tokenizer.bin>
 *               [--hb-ms 500] [--log work_<name>.csv] [--rng counter|stateful]
 */
#include "net.h"
#include "engine.h"

enum { MODE_RESTART = 0, MODE_TOKENS = 1, MODE_KV = 2 };

static char g_name[64], g_ip[64], g_mhost[64];
static int g_port, g_mport, g_hb_ms = 500, g_stateful = 0;
static double g_tp_ms, g_td_ms;
static Engine *g_e;
static FILE *g_log;
static volatile sig_atomic_t g_stop = 0;
static void on_signal(int s) { (void)s; g_stop = 1; }

static int master_msg(const char *fmt, ...) {
    char line[LINE_LEN]; va_list ap; va_start(ap, fmt); vsnprintf(line, sizeof line, fmt, ap); va_end(ap);
    int fd = connect_to(g_mhost, g_mport, 1000, 3);
    if (fd < 0) return -1;
    send_line(fd, "%s", line);
    char r[LINE_LEN]; int ok = recv_line(fd, r, sizeof r) > 0 && !strncmp(r, "OK", 2);
    close(fd); return ok ? 0 : -1;
}

static int do_register(void) {
    return master_msg("REGISTER %s %s %d %.4f %.4f %zu", g_name, g_ip, g_port, g_tp_ms, g_td_ms, engine_kv_bytes_per_pos(g_e));
}

/* heartbeats; if the master declared us down (e.g. heartbeats delayed on a congested link), join again */
static void *hb_thread(void *a) {
    (void)a;
    while (!g_stop) {
        int fd = connect_to(g_mhost, g_mport, 1000, 3);
        if (fd >= 0) {
            char r[LINE_LEN] = "";
            if (send_line(fd, "HB %s", g_name) == 0 && recv_line(fd, r, sizeof r) > 0 && !strncmp(r, "REJOIN", 6) && !g_stop) {
                close(fd); fd = -1;
                if (do_register() == 0) logmsg("declared down by the master; joined again as a new session");
            }
            if (fd >= 0) close(fd);
        }
        usleep(g_hb_ms * 1000);
    }
    return NULL;
}

static void wlog(const char *rid, int attempt, const char *phase, int a, int b) {
    if (!g_log) return;
    fprintf(g_log, "%.6f,%s,%s,%d,%s,%d,%d\n", wall_s(), g_name, rid, attempt, phase, a, b);
    fflush(g_log);                     /* survives kill -9: data is in the kernel already */
}

/* measure ms/token for batched prefill and for decode */
static void calibrate(void) {
    int toks[64]; for (int i = 0; i < 64; i++) toks[i] = 100 + i;
    double t0 = mono_s(); engine_forward_batch(g_e, toks, 64, 0); g_tp_ms = (mono_s() - t0) * 1e3 / 64;
    t0 = mono_s(); for (int p = 64; p < 96; p++) engine_forward_one(g_e, toks[p % 64], p); g_td_ms = (mono_s() - t0) * 1e3 / 32;
}

static void serve_task(int fd, char *hdr) {
    char rid[64]; int L, kc, mode, nprefix, kvpos, attempt; unsigned long long seed; long kvbytes;
    if (sscanf(hdr, "TASK %63s %d %llu %d %d %d %d %ld %d", rid, &L, &seed, &kc, &mode, &nprefix, &kvpos, &kvbytes, &attempt) != 9) { send_line(fd, "ERR bad task"); return; }
    char pline[LINE_LEN * 2];
    if (recv_line(fd, pline, sizeof pline) <= 0 || strncmp(pline, "PROMPT ", 7)) { send_line(fd, "ERR no prompt"); return; }
    int S = engine_seq_len(g_e);
    int *seqv = calloc(S + 8, sizeof(int)), *gen = calloc(S + 8, sizeof(int));
    int np = engine_encode(g_e, pline + 7, seqv, S);
    int got = recv_tokens(fd, gen, S);
    if (got < 0 || got != nprefix) { send_line(fd, "ERR prefix"); free(seqv); free(gen); return; }
    if (np + L > S) L = S - np;
    memcpy(seqv + np, gen, nprefix * sizeof(int));
    int m = np + nprefix;
    if (kvbytes > 0) {
        char *kv = malloc(kvbytes);
        if (recv_exact(fd, kv, kvbytes) < 0) { free(kv); free(seqv); free(gen); return; }
        engine_kv_import(g_e, 0, kvpos, kv); free(kv);
    } else kvpos = 0;
    if (kvpos > m - 1) kvpos = m - 1;
    logmsg("TASK %s attempt %d: L=%d prefix=%d kv_pos=%d mode=%d k_c=%d", rid, attempt, L, nprefix, kvpos, mode, kc);

    double t0 = mono_s();
    float *logits = engine_forward_batch(g_e, seqv + kvpos, m - kvpos, kvpos);
    wlog(rid, attempt, "prefill", kvpos, m);
    double tprefill = mono_s() - t0;

    uint64_t rstate = seed ? seed : 1;
    int g = nprefix, pos = m, last_commit = nprefix, kv_done = kvpos, ok = 1;
    double commit_s = 0; int ncommits = 0;
    size_t kvrow = engine_kv_bytes_per_pos(g_e);
    char *kvbuf = mode == MODE_KV ? malloc(kvrow * S) : NULL;
    double td0 = mono_s();
    while (g < L && ok && !g_stop) {
        int t = g_stateful ? engine_sample_stateful(g_e, logits, &rstate) : engine_sample(g_e, logits, seed, g);
        gen[g++] = t;
        wlog(rid, attempt, "decode", g - 1, g);
        if (mode != MODE_RESTART && (g - last_commit >= kc || g == L)) {
            double c0 = mono_s();
            char head[96]; snprintf(head, sizeof head, "COMMIT %s %d %d", rid, g, g - last_commit);
            ok = send_head_tokens(fd, head, gen + last_commit, g - last_commit) == 0;
            if (ok && mode == MODE_KV && np + g - 1 > kv_done) {
                int b = np + g - 1;
                size_t nb = engine_kv_export(g_e, kv_done, b, kvbuf);
                ok = send_line(fd, "KV %s %d %d %zu", rid, kv_done, b, nb) == 0 && send_all(fd, kvbuf, nb) == 0;
                kv_done = b;
            }
            commit_s += mono_s() - c0; ncommits++;
            last_commit = g;
        }
        if (g == L) break;
        logits = engine_forward_one(g_e, t, pos++);
    }
    double tdec = mono_s() - td0;
    if (ok && !g_stop) {
        send_line(fd, "DONE %s %d %llu %.3f %.3f %.4f %d", rid, g, (unsigned long long)fnv1a_tokens(gen, g),
                  tprefill * 1e3 / (m - kvpos > 0 ? m - kvpos : 1), tdec * 1e3 / (g - nprefix > 0 ? g - nprefix : 1),
                  ncommits ? commit_s * 1e3 / ncommits : 0.0, ncommits);
        send_tokens(fd, gen, g);
        logmsg("DONE %s: %d tokens (%d new) in %.0f ms", rid, g, g - nprefix, (tprefill + tdec) * 1e3);
    }
    free(kvbuf); free(seqv); free(gen);
}

int main(int argc, char **argv) {
    if (argc < 7) {
        fprintf(stderr, "usage: %s <name> <my_ip> <my_port> <master_ip:port> <model.bin> <tokenizer.bin> [--hb-ms N] [--log file] [--rng counter|stateful]\n", argv[0]);
        return 1;
    }
    setvbuf(stdout, NULL, _IONBF, 0);
    snprintf(g_name, sizeof g_name, "%s", argv[1]); snprintf(g_ip, sizeof g_ip, "%s", argv[2]);
    g_port = atoi(argv[3]);
    if (parse_hostport(argv[4], g_mhost, sizeof g_mhost, &g_mport) < 0) { fprintf(stderr, "bad master address\n"); return 1; }
    char logpath[256]; snprintf(logpath, sizeof logpath, "work_%s.csv", g_name);
    for (int i = 7; i < argc; i++) {
        if (!strcmp(argv[i], "--hb-ms") && i + 1 < argc) g_hb_ms = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--log") && i + 1 < argc) snprintf(logpath, sizeof logpath, "%s", argv[++i]);
        else if (!strcmp(argv[i], "--rng") && i + 1 < argc) g_stateful = !strcmp(argv[++i], "stateful");
    }
    char tag[80]; snprintf(tag, sizeof tag, "WORKER %s", g_name); set_log_tag(tag);
    signal(SIGPIPE, SIG_IGN);
    struct sigaction sa; memset(&sa, 0, sizeof sa); sa.sa_handler = on_signal;
    sigaction(SIGINT, &sa, NULL); sigaction(SIGTERM, &sa, NULL);

    g_log = fopen(logpath, "a");
    g_e = engine_load(argv[5], argv[6], 1.0f, 0.9f);
    calibrate();
    logmsg("model loaded; prefill %.3f ms/token (batched), decode %.3f ms/token, rng=%s", g_tp_ms, g_td_ms, g_stateful ? "stateful" : "counter");
    int lfd = listen_on(g_port);
    while (!g_stop && do_register() < 0) { logmsg("master not reachable, retrying"); sleep(1); }
    logmsg("registered with master %s:%d", g_mhost, g_mport);
    pthread_t hb; pthread_create(&hb, NULL, hb_thread, NULL); pthread_detach(hb);

    while (!g_stop) {
        int cfd = accept(lfd, NULL, NULL);
        if (cfd < 0) { if (errno == EINTR) continue; continue; }
        set_recv_timeout(cfd, 30);
        char hdr[LINE_LEN];
        if (recv_line(cfd, hdr, sizeof hdr) > 0) {
            if (!strncmp(hdr, "TASK ", 5)) serve_task(cfd, hdr);
            else if (!strncmp(hdr, "PING", 4)) send_line(cfd, "PONG %s", g_name);
        }
        close(cfd);
    }
    master_msg("BYE %s", g_name);
    logmsg("left gracefully");
    return 0;
}
