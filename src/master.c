/*
 * master.c - HARC master (micro-cluster coordinator)
 *
 * Policies (--policy):
 *   restart  B1: a failed request starts again from the prompt; no commits
 *   yd       B2: token-log commits at the fixed Young/Daly period sqrt(2 c mu)
 *   contkv   B3: KV checkpoint streamed after every token (LUMEN-style)
 *   random   B4: HARC commits and recovery modes, but random worker placement
 *   harc     HARC: Weibull lifetime model, risk-matched placement (Lemma 1),
 *                  hazard-adaptive commit interval (Eq. 8), per-request recovery mode (Eqs. 9-11)
 *
 * Messages it accepts (one line each):
 *   REGISTER name ip port tp_ms td_ms [kv_bytes_per_pos] | HB name (-> OK, or REJOIN if it was declared down) | BYE name
 *   JOB cid L seed  + "PROMPT text" line      -> RESULT ... when finished (same connection)
 *   STATUS | LOAD
 *
 * usage: master <port> [--policy harc] [--tau-ms 1500] [--mu0 300] [--bw-mbps 100]
 *                      [--commit-ms 0.2] [--out results_dir] [--seed 1]
 */
#include "net.h"
#include <poll.h>
#include <math.h>
#include <sys/stat.h>

enum { P_RESTART, P_YD, P_CONTKV, P_RANDOM, P_HARC };
static const char *PNAME[] = { "restart", "yd", "contkv", "random", "harc" };
enum { W_DOWN, W_IDLE, W_BUSY };
enum { R_PENDING, R_RUNNING, R_DONE };
enum { MODE_RESTART = 0, MODE_TOKENS = 1, MODE_KV = 2 };

#define MAXW 64
#define MAXR 4096
#define MAXS 8192
#define MAXTOK 4096

typedef struct { char name[64], ip[64]; int port, state, sess, req; double tp, td, last_hb, start; long kvpp; } worker_t;
typedef struct { double start, end; } session_t;
typedef struct {
    char cid[64], prompt[2048]; int L, nprefix, kvpos, state, failures, attempts, worker, mode, kc, G, client_fd, np_est;
    unsigned long long seed, hash; int prefix[MAXTOK];
    char *kv; size_t kvbytes, kvcap;
    double submit, first_start, done;
} req_t;

static worker_t W[MAXW]; static int NW = 0;
static session_t SS[MAXS]; static int NS = 0;
static req_t *R[MAXR]; static int NR = 0;
static pthread_mutex_t M = PTHREAD_MUTEX_INITIALIZER;
static int g_policy = P_HARC;
static double g_tau = 1.5, g_mu0 = 300, g_bw = 100e6 / 8, g_commit = 0.0002;
static double g_k = 1, g_theta = 300;
static char g_out[256] = "results";
static FILE *g_ev, *g_res;
static double T0;

static void ev(const char *event, int wi, int ri, const char *fmt, ...) {
    char info[512] = ""; if (fmt) { va_list ap; va_start(ap, fmt); vsnprintf(info, sizeof info, fmt, ap); va_end(ap); }
    fprintf(g_ev, "%.4f,%s,%s,%s,%s\n", mono_s() - T0, event, wi >= 0 ? W[wi].name : "", ri >= 0 ? R[ri]->cid : "", info);
    fflush(g_ev);
}

/* ---------------- lifetime model (Eqs. 2, 6, 7) ---------------- */
static void fit_weibull(double now) {
    double tF[MAXS], tA[MAXS]; int nF = 0, nA = 0;
    for (int i = 0; i < NS; i++) {
        double t = (SS[i].end >= 0 ? SS[i].end : now) - SS[i].start;
        if (t < 0.5) t = 0.5;
        tA[nA++] = t; if (SS[i].end >= 0) tF[nF++] = t;
    }
    if (nF < 3) { g_k = 1.0; g_theta = g_mu0; return; }
    double mlnF = 0; for (int i = 0; i < nF; i++) mlnF += log(tF[i]); mlnF /= nF;
    double lo = 0.05, hi = 10;
    for (int it = 0; it < 80; it++) {
        double k = 0.5 * (lo + hi), s = 0, sl = 0;
        for (int i = 0; i < nA; i++) { double tk = pow(tA[i], k); s += tk; sl += tk * log(tA[i]); }
        double g = 1.0 / k + mlnF - sl / s;            /* decreasing in k */
        if (g > 0) lo = k; else hi = k;
    }
    g_k = 0.5 * (lo + hi);
    double s = 0; for (int i = 0; i < nA; i++) s += pow(tA[i], g_k);
    g_theta = pow(s / nF, 1.0 / g_k);
}
static double hazard(double a) { if (a < 1) a = 1; return (g_k / g_theta) * pow(a / g_theta, g_k - 1); }
static double p_fail(double a, double d) { if (a < 0) a = 0; return 1 - exp(pow(a / g_theta, g_k) - pow((a + d) / g_theta, g_k)); }
static double mean_life(void) {
    double s = 0; int n = 0;
    for (int i = 0; i < NS; i++) if (SS[i].end >= 0) { s += SS[i].end - SS[i].start; n++; }
    return n >= 3 ? s / n : g_mu0;
}

/* ---------------- per-assignment decisions ---------------- */
static double pred_D(req_t *r, worker_t *w) { return ((r->np_est + r->nprefix) * w->tp + (r->L - r->nprefix) * w->td) / 1000.0; }

static void decide(req_t *r, worker_t *w, double now, int *mode, int *kc, double *h_out, double *q_out) {
    int Lr = r->L - r->nprefix; if (Lr < 1) Lr = 1;
    double td = w->td / 1000.0, tp = w->tp / 1000.0, a = now - w->start, D = pred_D(r, w);
    double h = hazard(a), q = p_fail(a, D);
    *h_out = h; *q_out = q;
    if (g_policy == P_RESTART) { *mode = MODE_RESTART; *kc = r->L + 1; return; }
    if (g_policy == P_CONTKV) { *mode = MODE_KV; *kc = 1; return; }
    if (g_policy == P_YD) {
        double Wp = sqrt(2 * g_commit * mean_life());
        int k = (int)floor(Wp / td); *kc = k < 1 ? 1 : (k > Lr ? Lr : k); *mode = MODE_TOKENS; return;
    }
    /* HARC (and B4, which shares these decisions) */
    double Wp = sqrt(2 * g_commit / h);                                   /* Eq. 8 */
    int k = (int)floor(Wp / td); k = k < 1 ? 1 : (k > Lr ? Lr : k);
    double j = r->nprefix + Lr / 2.0, beta = w->kvpp > 0 ? (double)w->kvpp : 13824.0;
    double R0 = g_tau + r->np_est * tp + j * td;                          /* Eq. 9 */
    double RT = g_tau + (r->np_est + j) * tp + (k / 2.0) * td;            /* Eq. 10 */
    double RK = g_tau + j * beta / g_bw + (k / 2.0) * td;                 /* Eq. 11 */
    double CT = ceil((double)Lr / k) * g_commit, CK = CT + Lr * beta / g_bw;
    double E0 = q * R0, ET = CT + q * RT, EK = CK + q * RK;
    if (E0 <= ET && E0 <= EK) { *mode = MODE_RESTART; *kc = r->L + 1; }
    else if (ET <= EK) { *mode = MODE_TOKENS; *kc = k; }
    else { *mode = MODE_KV; *kc = k; }
}

/* ---------------- task execution ---------------- */
typedef struct { int ri, wi, attempt, sess; } targ_t;

static void end_session(int wi, double now) { if (W[wi].sess >= 0 && SS[W[wi].sess].end < 0) SS[W[wi].sess].end = now; W[wi].sess = -1; }

static void finish_client(req_t *r) {
    if (r->client_fd < 0) return;
    send_line(r->client_fd, "RESULT %s %d %llu %d %d %.3f %.3f", r->cid, r->G, r->hash, r->failures, r->attempts,
              (r->done - r->submit) * 1e3, (r->first_start - r->submit) * 1e3);
    close(r->client_fd); r->client_fd = -1;
}

static void *task_thread(void *p) {
    targ_t *ta = p; int ri = ta->ri, wi = ta->wi, attempt = ta->attempt, sess = ta->sess; free(ta);
    req_t *r = R[ri];
    char ip[64]; int port, L, kc, mode, nprefix, kvpos; size_t kvbytes; unsigned long long seed;
    pthread_mutex_lock(&M);
    snprintf(ip, sizeof ip, "%s", W[wi].ip); port = W[wi].port;
    L = r->L; kc = r->kc; mode = r->mode; nprefix = r->nprefix; seed = r->seed;
    kvpos = mode == MODE_KV || r->kvbytes ? r->kvpos : 0; kvbytes = kvpos ? r->kvbytes : 0;
    int *pre = malloc(sizeof(int) * (nprefix + 1)); memcpy(pre, r->prefix, sizeof(int) * nprefix);
    char *kv = kvbytes ? malloc(kvbytes) : NULL; if (kvbytes) memcpy(kv, r->kv, kvbytes);
    char prompt[2048]; snprintf(prompt, sizeof prompt, "%s", r->prompt);
    pthread_mutex_unlock(&M);

    int ok = 0, fd = connect_to(ip, port, 1000, 0);
    if (fd >= 0) {
        ok = send_line(fd, "TASK %s %d %llu %d %d %d %d %zu %d", r->cid, L, seed, kc, mode, nprefix, kvpos, kvbytes, attempt) == 0
          && send_line(fd, "PROMPT %s", prompt) == 0 && send_tokens(fd, pre, nprefix) == 0
          && (kvbytes == 0 || send_all(fd, kv, kvbytes) == 0);
    }
    free(pre); free(kv);
    int done = 0;
    char *line = malloc(65536);
    while (ok && !done) {
        struct pollfd pf = { fd, POLLIN, 0 };
        int pr = poll(&pf, 1, 200);
        pthread_mutex_lock(&M); int down = W[wi].state == W_DOWN || W[wi].sess != sess; pthread_mutex_unlock(&M);
        if (down) { ok = 0; break; }
        if (pr == 0) continue;
        if (recv_line_strict(fd, line, 65536) <= 0) { ok = 0; break; }   /* a cut-off line is never applied */
        if (!strncmp(line, "COMMIT ", 7)) {
            char *sv, *f; strtok_r(line, " ", &sv); strtok_r(NULL, " ", &sv);
            f = strtok_r(NULL, " ", &sv); int g = f ? atoi(f) : -1;
            f = strtok_r(NULL, " ", &sv); int n = f ? atoi(f) : -1;
            int base = g - n, tmp[MAXTOK], got = 0;
            if (n <= 0 || base < 0 || g > MAXTOK) { ok = 0; break; }
            for (char *t; got < n && (t = strtok_r(NULL, " ", &sv)); got++) tmp[got] = atoi(t);
            if (got != n) { ok = 0; break; }
            pthread_mutex_lock(&M);
            if (base <= r->nprefix) {                                     /* contiguous with what we hold */
                memcpy(r->prefix + base, tmp, sizeof(int) * n);
                if (g > r->nprefix) r->nprefix = g;
            }
            pthread_mutex_unlock(&M);
        } else if (!strncmp(line, "KV ", 3)) {
            char rid[64]; int a, b; size_t nb;
            if (sscanf(line, "KV %63s %d %d %zu", rid, &a, &b, &nb) != 4) { ok = 0; break; }
            char *buf = malloc(nb);
            if (recv_exact(fd, buf, nb) < 0) { free(buf); ok = 0; break; }
            pthread_mutex_lock(&M);
            if (a == r->kvpos) {
                if (r->kvbytes + nb > r->kvcap) { r->kvcap = (r->kvbytes + nb) * 2; r->kv = realloc(r->kv, r->kvcap); }
                memcpy(r->kv + r->kvbytes, buf, nb); r->kvbytes += nb; r->kvpos = b;
            }
            pthread_mutex_unlock(&M); free(buf);
        } else if (!strncmp(line, "DONE ", 5)) {
            char rid[64]; int G, nc; unsigned long long h; double tp, td, cms;
            if (sscanf(line, "DONE %63s %d %llu %lf %lf %lf %d", rid, &G, &h, &tp, &td, &cms, &nc) < 6) { ok = 0; break; }
            int *toks = malloc(sizeof(int) * (G + 1));
            int got = recv_tokens(fd, toks, G);
            pthread_mutex_lock(&M);
            r->G = G; r->hash = h; r->done = mono_s(); r->state = R_DONE;
            if (got == G) memcpy(r->prefix, toks, sizeof(int) * (G < MAXTOK ? G : MAXTOK));
            if (W[wi].sess == sess) { W[wi].state = W_IDLE; W[wi].req = -1; }
            if (td > 0) W[wi].td = 0.8 * W[wi].td + 0.2 * td;
            if (nc > 0 && cms > 0) g_commit = 0.9 * g_commit + 0.1 * (cms / 1000.0);
            fprintf(g_res, "%s,%d,%llu,%.4f,%.4f,%.4f,%.3f,%d,%llu,%d,%d,%s\n", r->cid, r->L, r->seed, r->submit - T0, r->first_start - T0,
                    r->done - T0, (r->done - r->submit) * 1e3, G, h, r->failures, r->attempts, PNAME[g_policy]);
            fflush(g_res);
            ev("done", wi, ri, "G=%d failures=%d kv_bytes=%zu", G, r->failures, r->kvbytes);
            free(r->kv); r->kv = NULL; r->kvbytes = r->kvcap = 0;
            finish_client(r);
            pthread_mutex_unlock(&M);
            free(toks); done = 1;
        }
    }
    free(line);
    if (fd >= 0) close(fd);
    if (!done) {
        pthread_mutex_lock(&M);
        double now = mono_s();
        if (W[wi].sess == sess) {
            if (W[wi].state != W_DOWN) { W[wi].state = W_DOWN; end_session(wi, now); ev("worker_down", wi, ri, "detected=connection"); }
            W[wi].req = -1;
        }
        /* committed tokens (and any KV checkpoint) stay valid: the next attempt resumes from them */
        r->failures++; r->state = R_PENDING;
        ev("request_failed", wi, ri, "attempt=%d resume_from=%d kvpos=%d", attempt, r->nprefix, r->kvpos);
        pthread_mutex_unlock(&M);
    }
    return NULL;
}

/* ---------------- scheduler (Algorithm 2) ---------------- */
static double g_now_sort;
static int cmp_worker_hz(const void *a, const void *b) {
    int x = *(const int *)a, y = *(const int *)b;
    double hx = hazard(g_now_sort - W[x].start), hy = hazard(g_now_sort - W[y].start);
    if (hx != hy) return hx < hy ? -1 : 1;
    return strcmp(W[x].name, W[y].name);
}
static int cmp_worker_name(const void *a, const void *b) { return strcmp(W[*(const int *)a].name, W[*(const int *)b].name); }
static int cmp_req_long(const void *a, const void *b) {
    req_t *x = R[*(const int *)a], *y = R[*(const int *)b];
    double dx = (x->np_est + x->nprefix) * 0.25 + (x->L - x->nprefix), dy = (y->np_est + y->nprefix) * 0.25 + (y->L - y->nprefix);
    if (dx != dy) return dx > dy ? -1 : 1;
    return (x->submit < y->submit) ? -1 : 1;
}
static int cmp_req_fifo(const void *a, const void *b) { return R[*(const int *)a]->submit < R[*(const int *)b]->submit ? -1 : 1; }

static void *scheduler(void *arg) {
    (void)arg;
    for (;;) {
        usleep(20000);
        pthread_mutex_lock(&M);
        double now = mono_s();
        for (int i = 0; i < NW; i++)
            if (W[i].state != W_DOWN && now - W[i].last_hb > g_tau) { W[i].state = W_DOWN; end_session(i, now); ev("worker_down", i, W[i].req, "detected=heartbeat"); }
        int idle[MAXW], ni = 0, pend[MAXR], np = 0;
        for (int i = 0; i < NW; i++) if (W[i].state == W_IDLE) idle[ni++] = i;
        for (int i = 0; i < NR; i++) if (R[i]->state == R_PENDING) pend[np++] = i;
        if (ni && np) {
            if (g_policy == P_HARC || g_policy == P_RANDOM || g_policy == P_YD) fit_weibull(now);
            g_now_sort = now;
            /* queue order decides WHICH requests run now; placement only decides WHERE each one runs */
            qsort(pend, np, sizeof(int), cmp_req_fifo);
            int n = ni < np ? ni : np;
            if (g_policy == P_HARC) {                       /* Lemma 1: lowest hazard <-> longest request */
                qsort(idle, ni, sizeof(int), cmp_worker_hz);
                qsort(pend, n, sizeof(int), cmp_req_long);
            } else if (g_policy == P_RANDOM) {
                for (int i = ni - 1; i > 0; i--) { int j = rand() % (i + 1), t = idle[i]; idle[i] = idle[j]; idle[j] = t; }
            } else qsort(idle, ni, sizeof(int), cmp_worker_name);
            for (int i = 0; i < n; i++) {
                int wi = idle[i], ri = pend[i]; req_t *r = R[ri]; double h, q;
                decide(r, &W[wi], now, &r->mode, &r->kc, &h, &q);
                r->state = R_RUNNING; r->worker = wi; r->attempts++;
                if (r->attempts == 1) r->first_start = now;
                W[wi].state = W_BUSY; W[wi].req = ri;
                ev("assign", wi, ri, "attempt=%d mode=%d kc=%d age=%.1f hazard=%.6f pfail=%.4f k=%.3f theta=%.1f prefix=%d",
                   r->attempts, r->mode, r->kc, now - W[wi].start, h, q, g_k, g_theta, r->nprefix);
                targ_t *ta = malloc(sizeof *ta); ta->ri = ri; ta->wi = wi; ta->attempt = r->attempts; ta->sess = W[wi].sess;
                pthread_t t; pthread_create(&t, NULL, task_thread, ta); pthread_detach(t);
            }
        }
        pthread_mutex_unlock(&M);
    }
    return NULL;
}

/* ---------------- connections ---------------- */
static void *handle(void *arg) {
    int fd = (int)(long)arg; char line[LINE_LEN * 3];
    set_recv_timeout(fd, 10);
    if (recv_line(fd, line, sizeof line) <= 0) { close(fd); return NULL; }
    char name[64], ip[64]; int port; double tp, td; long kvpp = 0;
    if (sscanf(line, "REGISTER %63s %63s %d %lf %lf %ld", name, ip, &port, &tp, &td, &kvpp) >= 5) {
        pthread_mutex_lock(&M);
        int i; for (i = 0; i < NW; i++) if (!strcmp(W[i].name, name)) break;
        if (i == NW && NW < MAXW) NW++;
        double now = mono_s();
        if (W[i].sess >= 0) end_session(i, now);          /* re-registration closes the previous session */
        snprintf(W[i].name, sizeof W[i].name, "%s", name); snprintf(W[i].ip, sizeof W[i].ip, "%s", ip);
        W[i].port = port; W[i].tp = tp; W[i].td = td; W[i].kvpp = kvpp; W[i].state = W_IDLE; W[i].req = -1;
        W[i].start = now; W[i].last_hb = now;
        if (NS < MAXS) { SS[NS].start = now; SS[NS].end = -1; W[i].sess = NS++; }
        ev("register", i, -1, "tp=%.3f td=%.3f", tp, td);
        pthread_mutex_unlock(&M);
        send_line(fd, "OK");
    } else if (sscanf(line, "HB %63s", name) == 1) {
        int rejoin = 0;
        pthread_mutex_lock(&M);
        for (int i = 0; i < NW; i++)
            if (!strcmp(W[i].name, name)) {
                if (W[i].state != W_DOWN) W[i].last_hb = mono_s();
                else { rejoin = 1; ev("rejoin_request", i, -1, NULL); }   /* falsely suspected: let it come back */
            }
        pthread_mutex_unlock(&M);
        send_line(fd, rejoin ? "REJOIN" : "OK");
    } else if (sscanf(line, "BYE %63s", name) == 1) {
        pthread_mutex_lock(&M);
        for (int i = 0; i < NW; i++) if (!strcmp(W[i].name, name) && W[i].state != W_DOWN) { W[i].state = W_DOWN; end_session(i, mono_s()); ev("worker_bye", i, -1, NULL); }
        pthread_mutex_unlock(&M);
        send_line(fd, "OK");
    } else if (!strncmp(line, "JOB ", 4) || !strncmp(line, "SUBMIT ", 7)) {
        req_t *r = calloc(1, sizeof *r);
        if (sscanf(strchr(line, ' ') + 1, "%63s %d %llu", r->cid, &r->L, &r->seed) != 3 || recv_line(fd, line, sizeof line) <= 0 || strncmp(line, "PROMPT ", 7)) {
            send_line(fd, "ERR bad submit"); free(r); close(fd); return NULL;
        }
        snprintf(r->prompt, sizeof r->prompt, "%s", line + 7);
        r->np_est = (int)strlen(r->prompt) / 4 + 2;
        set_recv_timeout(fd, 0);
        pthread_mutex_lock(&M);
        if (NR >= MAXR) { pthread_mutex_unlock(&M); send_line(fd, "ERR full"); free(r); close(fd); return NULL; }
        r->client_fd = fd; r->submit = mono_s(); r->state = R_PENDING; r->worker = -1;
        R[NR++] = r;
        ev("submit", -1, NR - 1, "L=%d", r->L);
        pthread_mutex_unlock(&M);
        return NULL;                                   /* fd stays open until RESULT */
    } else if (!strncmp(line, "LOAD", 4)) {
        pthread_mutex_lock(&M);
        int q = 0, alive = 0; double tds = 0;
        for (int i = 0; i < NR; i++) q += R[i]->state != R_DONE;
        for (int i = 0; i < NW; i++) if (W[i].state != W_DOWN) { alive++; tds += W[i].td; }
        pthread_mutex_unlock(&M);
        send_line(fd, "LOAD %d %d %.4f", q, alive, alive ? tds / alive : 0);
    } else if (!strncmp(line, "STATUS", 6)) {
        pthread_mutex_lock(&M);
        double now = mono_s();
        send_line(fd, "POLICY %s k=%.3f theta=%.1f sessions=%d commit_ms=%.4f", PNAME[g_policy], g_k, g_theta, NS, g_commit * 1e3);
        for (int i = 0; i < NW; i++)
            send_line(fd, "WORKER %s %s age=%.1f hazard=%.6f td=%.3f", W[i].name, W[i].state == W_DOWN ? "DOWN" : W[i].state == W_IDLE ? "IDLE" : "BUSY", now - W[i].start, hazard(now - W[i].start), W[i].td);
        int nd = 0; for (int i = 0; i < NR; i++) nd += R[i]->state == R_DONE;
        send_line(fd, "END requests=%d done=%d", NR, nd);
        pthread_mutex_unlock(&M);
    } else send_line(fd, "ERR unknown");
    close(fd);
    return NULL;
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <port> [--policy restart|yd|contkv|random|harc] [--tau-ms N] [--mu0 S] [--bw-mbps B] [--commit-ms C] [--out DIR] [--seed N]\n", argv[0]); return 1; }
    setvbuf(stdout, NULL, _IONBF, 0); signal(SIGPIPE, SIG_IGN);
    int port = atoi(argv[1]); unsigned seed = 1;
    for (int i = 2; i < argc; i++) {
        if (!strcmp(argv[i], "--policy") && i + 1 < argc) { i++; for (int p = 0; p < 5; p++) if (!strcmp(argv[i], PNAME[p])) g_policy = p; }
        else if (!strcmp(argv[i], "--tau-ms") && i + 1 < argc) g_tau = atof(argv[++i]) / 1000.0;
        else if (!strcmp(argv[i], "--mu0") && i + 1 < argc) g_mu0 = atof(argv[++i]);
        else if (!strcmp(argv[i], "--bw-mbps") && i + 1 < argc) g_bw = atof(argv[++i]) * 1e6 / 8;
        else if (!strcmp(argv[i], "--commit-ms") && i + 1 < argc) g_commit = atof(argv[++i]) / 1000.0;
        else if (!strcmp(argv[i], "--out") && i + 1 < argc) snprintf(g_out, sizeof g_out, "%s", argv[++i]);
        else if (!strcmp(argv[i], "--seed") && i + 1 < argc) seed = (unsigned)atoi(argv[++i]);
    }
    srand(seed); g_theta = g_mu0;
    mkdir(g_out, 0755);
    char p1[512], p2[512]; snprintf(p1, sizeof p1, "%s/events.csv", g_out); snprintf(p2, sizeof p2, "%s/results.csv", g_out);
    g_ev = fopen(p1, "w"); g_res = fopen(p2, "w");
    fprintf(g_ev, "t,event,worker,request,info\n");
    fprintf(g_res, "cid,L,seed,submit,first_start,done,latency_ms,G,hash,failures,attempts,policy\n");
    for (int i = 0; i < MAXW; i++) W[i].sess = -1;
    set_log_tag("MASTER"); T0 = mono_s();
    int lfd = listen_on(port);
    logmsg("listening on %d, policy=%s, tau=%.2fs, mu0=%.0fs, bw=%.0f Mb/s", port, PNAME[g_policy], g_tau, g_mu0, g_bw * 8 / 1e6);
    pthread_t st; pthread_create(&st, NULL, scheduler, NULL); pthread_detach(st);
    for (;;) {
        int cfd = accept(lfd, NULL, NULL);
        if (cfd < 0) { if (errno == EINTR) continue; continue; }
        pthread_t t; pthread_create(&t, NULL, handle, (void *)(long)cfd); pthread_detach(t);
    }
}
