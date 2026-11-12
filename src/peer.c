/*
 * peer.c - HyPEC tier-1 overlay node (Algorithm 1: load-aware discovery)
 *
 * Every node is a TCP server and a client and knows only its neighbours.
 * A node is a leaf (requester), a super-peer, or a cluster master acting as a super-peer.
 *   HELLO name ip port super     add a neighbour (sent by a joining node)
 *   BYE name                     remove a neighbour
 *   QUERY id ttl oip oport       flood with TTL and duplicate suppression; masters reply HIT to the origin
 *   HIT id cluster Q W td        reply to the requester (load fields are "-" in the flat overlay)
 *   DISCOVER L ttl delta_ms      make this node issue a query and pick a cluster (Eq. 5)
 *   SETLOAD Q W td | STATS | RESET | NEIGHBORS
 * In the HyPEC overlay only super-peers forward queries; in the flat overlay (--flat, baseline B5)
 * every node forwards and the requester takes the first HIT, since there are no load fields.
 *
 * usage: peer <name> <ip> <port> [--super] [--cluster NAME] [--master ip:port] [--flat]
 *             [--join ip:port,...] [--hop-ms 5]
 */
#include "net.h"

#define MAXN 128
#define SEEN 4096
typedef struct { char name[64], ip[64]; int port, super; } nb_t;
static nb_t NB[MAXN]; static int NNB = 0;
static char g_name[64], g_ip[64], g_cluster[64] = "", g_mhost[64] = ""; static int g_port, g_mport = 0, g_super = 0, g_flat = 0, g_hop_ms = 5;
static int g_Q = 0, g_W = 1; static double g_td = 30;
static char seen[SEEN][48]; static int nseen = 0;
static long c_sent = 0, c_recv = 0, c_dup = 0, c_hit = 0;
static pthread_mutex_t M = PTHREAD_MUTEX_INITIALIZER;

/* pending discovery at this node */
static char p_id[48] = ""; static double p_t0; static int p_n = 0;
static struct { char cl[64]; int Q, W; double td, ms; } p_hit[64];

static int one_msg(const char *ip, int port, const char *line) {
    int fd = connect_to(ip, port, 1000, 2); if (fd < 0) return -1;
    int r = send_line(fd, "%s", line); close(fd); return r;
}

typedef struct { char ip[64]; int port; char line[LINE_LEN]; } fw_t;
static void *fw_thread(void *a) {
    fw_t *f = a;
    if (g_hop_ms > 0) usleep(g_hop_ms * 1000);                 /* emulated per-hop link delay */
    if (one_msg(f->ip, f->port, f->line) == 0) { pthread_mutex_lock(&M); c_sent++; pthread_mutex_unlock(&M); }
    free(f); return NULL;
}
static void send_async(const char *ip, int port, const char *line) {
    fw_t *f = malloc(sizeof *f); snprintf(f->ip, sizeof f->ip, "%s", ip); f->port = port; snprintf(f->line, sizeof f->line, "%s", line);
    pthread_t t; pthread_create(&t, NULL, fw_thread, f); pthread_detach(t);
}

static void get_load(void) {
    if (!g_mport) return;
    int fd = connect_to(g_mhost, g_mport, 500, 1); if (fd < 0) return;
    char r[LINE_LEN];
    if (send_line(fd, "LOAD") == 0 && recv_line(fd, r, sizeof r) > 0) sscanf(r, "LOAD %d %d %lf", &g_Q, &g_W, &g_td);
    close(fd);
}

static void handle_line(char *line) {
    char a1[64], a2[64], a3[64]; int x, y, z;
    if (sscanf(line, "HELLO %63s %63s %d %d", a1, a2, &x, &y) == 4) {
        pthread_mutex_lock(&M);
        int i; for (i = 0; i < NNB; i++) if (!strcmp(NB[i].name, a1)) break;
        if (i == NNB && NNB < MAXN) NNB++;
        snprintf(NB[i].name, 64, "%s", a1); snprintf(NB[i].ip, 64, "%s", a2); NB[i].port = x; NB[i].super = y;
        pthread_mutex_unlock(&M);
    } else if (sscanf(line, "BYE %63s", a1) == 1) {
        pthread_mutex_lock(&M);
        for (int i = 0; i < NNB; i++) if (!strcmp(NB[i].name, a1)) { NB[i] = NB[--NNB]; break; }
        pthread_mutex_unlock(&M);
    } else if (sscanf(line, "QUERY %47s %d %63s %d %63s", a1, &x, a2, &y, a3) == 5) {
        pthread_mutex_lock(&M);
        c_recv++;
        int dup = 0; for (int i = 0; i < nseen && i < SEEN; i++) if (!strcmp(seen[i], a1)) { dup = 1; break; }
        if (dup) { c_dup++; pthread_mutex_unlock(&M); return; }
        snprintf(seen[nseen++ % SEEN], 48, "%s", a1);
        int is_master = g_cluster[0] != 0, fwd = (g_flat || g_super) && x > 1;
        nb_t nbs[MAXN]; int nn = NNB; memcpy(nbs, NB, sizeof(nb_t) * nn);
        pthread_mutex_unlock(&M);
        if (is_master) {
            get_load();
            char hit[LINE_LEN];
            if (g_flat) snprintf(hit, sizeof hit, "HIT %s %s - - -", a1, g_cluster);
            else snprintf(hit, sizeof hit, "HIT %s %s %d %d %.3f", a1, g_cluster, g_Q, g_W, g_td);
            send_async(a2, y, hit);
        }
        if (fwd) {
            char q[LINE_LEN]; snprintf(q, sizeof q, "QUERY %s %d %s %d %s", a1, x - 1, a2, y, g_name);
            for (int i = 0; i < nn; i++) {
                if (!strcmp(nbs[i].name, a3)) continue;                 /* not back to the sender */
                if (!g_flat && !nbs[i].super) continue;                 /* HyPEC: only super-peers forward */
                send_async(nbs[i].ip, nbs[i].port, q);
            }
        }
    } else if (!strncmp(line, "HIT ", 4)) {
        char id[48], cl[64], qs[32], ws[32], ts[32];
        if (sscanf(line, "HIT %47s %63s %31s %31s %31s", id, cl, qs, ws, ts) != 5) return;
        pthread_mutex_lock(&M);
        c_hit++;
        if (!strcmp(id, p_id) && p_n < 64) {
            int i = p_n++; snprintf(p_hit[i].cl, 64, "%s", cl);
            p_hit[i].Q = atoi(qs); p_hit[i].W = atoi(ws) > 0 ? atoi(ws) : 1; p_hit[i].td = atof(ts); p_hit[i].ms = (mono_s() - p_t0) * 1e3;
        }
        pthread_mutex_unlock(&M);
    } else if (sscanf(line, "SETLOAD %d %d %d", &x, &y, &z) == 3) {
        pthread_mutex_lock(&M); g_Q = x; g_W = y; g_td = z; pthread_mutex_unlock(&M);
    }
}

static void *conn(void *arg) {
    int fd = (int)(long)arg; char line[LINE_LEN];
    set_recv_timeout(fd, 5);
    if (recv_line(fd, line, sizeof line) <= 0) { close(fd); return NULL; }
    int L, ttl, delta;
    if (sscanf(line, "DISCOVER %d %d %d", &L, &ttl, &delta) == 3) {
        static int qn = 0; char q[LINE_LEN];
        pthread_mutex_lock(&M);
        snprintf(p_id, sizeof p_id, "%s-%d-%d", g_name, getpid(), ++qn); p_n = 0; p_t0 = mono_s();
        snprintf(seen[nseen++ % SEEN], 48, "%s", p_id);
        nb_t nbs[MAXN]; int nn = NNB; memcpy(nbs, NB, sizeof(nb_t) * nn);
        snprintf(q, sizeof q, "QUERY %s %d %s %d %s", p_id, ttl, g_ip, g_port, g_name);
        pthread_mutex_unlock(&M);
        for (int i = 0; i < nn; i++) send_async(nbs[i].ip, nbs[i].port, q);     /* in parallel */
        usleep(delta * 1000);                                                     /* collect HITs until delta */
        pthread_mutex_lock(&M);
        int best = -1; double bestT = 1e300;
        for (int i = 0; i < p_n; i++) {
            double T = g_flat ? p_hit[i].ms : p_hit[i].ms / 1e3 + (p_hit[i].Q + 1) * L * p_hit[i].td / 1e3 / p_hit[i].W;  /* Eq. 5 */
            if (T < bestT) { bestT = T; best = i; }
        }
        double first = p_n ? p_hit[0].ms : -1;
        char out[LINE_LEN * 2]; int o = snprintf(out, sizeof out, "SELECT %s %s %d %.3f %.3f", p_id, best >= 0 ? p_hit[best].cl : "-", p_n, first, best >= 0 ? p_hit[best].ms : -1.0);
        for (int i = 0; i < p_n && o < (int)sizeof out - 80; i++) o += snprintf(out + o, sizeof out - o, " %s:%.2f", p_hit[i].cl, p_hit[i].ms);
        p_id[0] = 0;
        pthread_mutex_unlock(&M);
        send_line(fd, "%s", out);
    } else if (!strncmp(line, "STATS", 5)) {
        pthread_mutex_lock(&M); send_line(fd, "STATS %ld %ld %ld %ld", c_sent, c_recv, c_dup, c_hit); pthread_mutex_unlock(&M);
    } else if (!strncmp(line, "RESET", 5)) {
        pthread_mutex_lock(&M); c_sent = c_recv = c_dup = c_hit = 0; pthread_mutex_unlock(&M); send_line(fd, "OK");
    } else if (!strncmp(line, "NEIGHBORS", 9)) {
        pthread_mutex_lock(&M); char out[LINE_LEN * 2]; int o = snprintf(out, sizeof out, "NEIGHBORS %d", NNB);
        for (int i = 0; i < NNB && o < (int)sizeof out - 80; i++) o += snprintf(out + o, sizeof out - o, " %s%s", NB[i].name, NB[i].super ? "*" : "");
        pthread_mutex_unlock(&M); send_line(fd, "%s", out);
    } else if (!strncmp(line, "HELLO ", 6)) { handle_line(line); send_line(fd, "WELCOME %s %d", g_name, g_super); }
    else { handle_line(line); send_line(fd, "OK"); }
    close(fd); return NULL;
}

int main(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "usage: %s <name> <ip> <port> [--super] [--cluster C] [--master ip:port] [--flat] [--join ip:port,...] [--hop-ms 5]\n", argv[0]); return 1; }
    signal(SIGPIPE, SIG_IGN); setvbuf(stdout, NULL, _IONBF, 0);
    snprintf(g_name, 64, "%s", argv[1]); snprintf(g_ip, 64, "%s", argv[2]); g_port = atoi(argv[3]);
    char join[2048] = "";
    for (int i = 4; i < argc; i++) {
        if (!strcmp(argv[i], "--super")) g_super = 1;
        else if (!strcmp(argv[i], "--flat")) g_flat = 1;
        else if (!strcmp(argv[i], "--cluster") && i + 1 < argc) { snprintf(g_cluster, 64, "%s", argv[++i]); g_super = 1; }
        else if (!strcmp(argv[i], "--master") && i + 1 < argc) parse_hostport(argv[++i], g_mhost, sizeof g_mhost, &g_mport);
        else if (!strcmp(argv[i], "--join") && i + 1 < argc) snprintf(join, sizeof join, "%s", argv[++i]);
        else if (!strcmp(argv[i], "--hop-ms") && i + 1 < argc) g_hop_ms = atoi(argv[++i]);
    }
    char tag[80]; snprintf(tag, sizeof tag, "PEER %s", g_name); set_log_tag(tag);
    int lfd = listen_on(g_port);
    /* join: introduce ourselves to each given node; the HELLO reply path makes links symmetric */
    for (char *sv, *t = strtok_r(join, ",", &sv); t; t = strtok_r(NULL, ",", &sv)) {
        char h[64]; int p; if (parse_hostport(t, h, sizeof h, &p) < 0) continue;
        char hello[LINE_LEN]; snprintf(hello, sizeof hello, "HELLO %s %s %d %d", g_name, g_ip, g_port, g_super);
        for (int tries = 0; tries < 20; tries++) {
            int fd = connect_to(h, p, 1000, 2); char r[LINE_LEN], nm[64]; int sp;
            if (fd < 0) { usleep(100000); continue; }
            if (send_line(fd, "%s", hello) == 0 && recv_line(fd, r, sizeof r) > 0 && sscanf(r, "WELCOME %63s %d", nm, &sp) == 2) {
                char add[LINE_LEN]; snprintf(add, sizeof add, "HELLO %s %s %d %d", nm, h, p, sp); handle_line(add);
                logmsg("joined via %s (%s)", nm, sp ? "super-peer" : "peer");
            }
            close(fd); break;
        }
    }
    for (;;) {
        int cfd = accept(lfd, NULL, NULL);
        if (cfd < 0) continue;
        pthread_t t; pthread_create(&t, NULL, conn, (void *)(long)cfd); pthread_detach(t);
    }
}
