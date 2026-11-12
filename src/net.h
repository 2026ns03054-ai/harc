/*
 * net.h - shared helpers (TCP sockets, line protocol, logging, timing)
 * adapted from the CCZG 526 Lab II prototype
 */
#ifndef HARC_NET_H
#define HARC_NET_H

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <unistd.h>
#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <signal.h>
#include <time.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/select.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>
#include <netdb.h>
#include <dirent.h>

#define LINE_LEN 1024

/* ---------- time ---------- */
static inline long long now_ms(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (long long)tv.tv_sec * 1000 + tv.tv_usec / 1000;
}

/* ---------- logging (thread safe, timestamped) ---------- */
static pthread_mutex_t g_log_mtx = PTHREAD_MUTEX_INITIALIZER;
static char g_log_tag[64] = "node";

static void set_log_tag(const char *t) { snprintf(g_log_tag, sizeof g_log_tag, "%s", t); }

static void logmsg(const char *fmt, ...) {
    struct timeval tv; gettimeofday(&tv, NULL);
    struct tm tmv; localtime_r(&tv.tv_sec, &tmv);
    char msg[LINE_LEN * 2];
    va_list ap; va_start(ap, fmt); vsnprintf(msg, sizeof msg, fmt, ap); va_end(ap);
    pthread_mutex_lock(&g_log_mtx);
    printf("[%02d:%02d:%02d.%03ld] [%s] %s\n", tmv.tm_hour, tmv.tm_min, tmv.tm_sec,
           (long)(tv.tv_usec / 1000), g_log_tag, msg);
    fflush(stdout);
    pthread_mutex_unlock(&g_log_mtx);
}

/* ---------- sockets ---------- */
static int listen_on(int port) {
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) { perror("socket"); exit(1); }
    int one = 1;
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
    struct sockaddr_in a; memset(&a, 0, sizeof a);
    a.sin_family = AF_INET; a.sin_addr.s_addr = htonl(INADDR_ANY); a.sin_port = htons(port);
    if (bind(fd, (struct sockaddr *)&a, sizeof a) < 0) { perror("bind"); exit(1); }
    if (listen(fd, 64) < 0) { perror("listen"); exit(1); }
    return fd;
}

static void set_recv_timeout(int fd, int seconds) {
    struct timeval tv = { seconds, 0 };
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv);
}

/* connect with a connect-timeout (ms) and a receive timeout (s); returns fd or -1 */
static int connect_to(const char *host, int port, int conn_timeout_ms, int recv_timeout_s) {
    struct addrinfo hints, *res = NULL;
    char ps[16]; snprintf(ps, sizeof ps, "%d", port);
    memset(&hints, 0, sizeof hints);
    hints.ai_family = AF_INET; hints.ai_socktype = SOCK_STREAM;
    if (getaddrinfo(host, ps, &hints, &res) != 0 || !res) return -1;

    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) { freeaddrinfo(res); return -1; }
    int fl = fcntl(fd, F_GETFL, 0);
    fcntl(fd, F_SETFL, fl | O_NONBLOCK);
    int rc = connect(fd, res->ai_addr, res->ai_addrlen);
    freeaddrinfo(res);
    if (rc < 0 && errno != EINPROGRESS) { close(fd); return -1; }
    if (rc < 0) {
        fd_set ws; FD_ZERO(&ws); FD_SET(fd, &ws);
        struct timeval tv = { conn_timeout_ms / 1000, (conn_timeout_ms % 1000) * 1000 };
        if (select(fd + 1, NULL, &ws, NULL, &tv) <= 0) { close(fd); return -1; }
        int err = 0; socklen_t len = sizeof err;
        getsockopt(fd, SOL_SOCKET, SO_ERROR, &err, &len);
        if (err) { close(fd); return -1; }
    }
    fcntl(fd, F_SETFL, fl);
    if (recv_timeout_s > 0) set_recv_timeout(fd, recv_timeout_s);
    int one = 1; setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
    return fd;
}

static int send_all(int fd, const void *buf, size_t n) {
    const char *p = buf;
    while (n > 0) {
        ssize_t k = send(fd, p, n, MSG_NOSIGNAL);
        if (k <= 0) { if (k < 0 && errno == EINTR) continue; return -1; }
        p += k; n -= (size_t)k;
    }
    return 0;
}

static int send_line(int fd, const char *fmt, ...) {
    char buf[LINE_LEN * 2];
    va_list ap; va_start(ap, fmt); int n = vsnprintf(buf, sizeof buf - 2, fmt, ap); va_end(ap);
    if (n < 0) return -1;
    if (n > (int)sizeof buf - 2) n = sizeof buf - 2;
    buf[n++] = '\n';
    return send_all(fd, buf, (size_t)n);
}

/* read one '\n'-terminated line (newline stripped). returns length, or -1 on EOF/error */
static int recv_line(int fd, char *buf, int max) {
    int i = 0;
    while (i < max - 1) {
        char c; ssize_t k = recv(fd, &c, 1, 0);
        if (k <= 0) { if (k < 0 && errno == EINTR) continue; return i > 0 ? i : -1; }
        if (c == '\n') break;
        if (c != '\r') buf[i++] = c;
    }
    buf[i] = 0;
    return i;
}

/* like recv_line, but a line cut off by EOF (peer died mid-message) is an error */
__attribute__((unused)) static int recv_line_strict(int fd, char *buf, int max) {
    int i = 0;
    for (;;) {
        char c; ssize_t k = recv(fd, &c, 1, 0);
        if (k <= 0) { if (k < 0 && errno == EINTR) continue; return -1; }
        if (c == '\n') break;
        if (i >= max - 1) return -1;
        if (c != '\r') buf[i++] = c;
    }
    buf[i] = 0;
    return i;
}

/* "<head> t1 t2 ... tn\n" in a single buffer, so a message is never split between two sends */
__attribute__((unused)) static int send_head_tokens(int fd, const char *head, const int *t, int n) {
    size_t hl = strlen(head), cap = hl + (size_t)n * 12 + 4; char *b = malloc(cap); size_t o = 0;
    memcpy(b, head, hl); o = hl;
    for (int i = 0; i < n; i++) o += snprintf(b + o, cap - o, " %d", t[i]);
    b[o++] = '\n'; int r = send_all(fd, b, o); free(b); return r;
}

__attribute__((unused)) static int recv_exact(int fd, char *buf, size_t n) {
    size_t got = 0;
    while (got < n) {
        ssize_t k = recv(fd, buf + got, n - got, 0);
        if (k <= 0) { if (k < 0 && errno == EINTR) continue; return -1; }
        got += (size_t)k;
    }
    return 0;
}

/* "10.1.1.4:5000" -> host, port */
__attribute__((unused)) static int parse_hostport(const char *s, char *host, int hostlen, int *port) {
    const char *c = strrchr(s, ':');
    if (!c) return -1;
    int hl = (int)(c - s);
    if (hl <= 0 || hl >= hostlen) return -1;
    memcpy(host, s, hl); host[hl] = 0;
    *port = atoi(c + 1);
    return *port > 0 ? 0 : -1;
}

/* monotonic seconds */
static inline double mono_s(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static inline double wall_s(void) { struct timeval tv; gettimeofday(&tv, NULL); return tv.tv_sec + tv.tv_usec * 1e-6; }

/* tokens <-> text line helpers ("-" means empty) */
static int send_tokens(int fd, const int *t, int n) {
    if (n <= 0) return send_line(fd, "-");
    size_t cap = (size_t)n * 12 + 4; char *b = malloc(cap); size_t o = 0;
    for (int i = 0; i < n; i++) o += snprintf(b + o, cap - o, i ? " %d" : "%d", t[i]);
    b[o++] = '\n'; int r = send_all(fd, b, o); free(b); return r;
}
/* reads a long token line; returns count or -1 */
static int recv_tokens(int fd, int *out, int max) {
    int n = 0, cur = 0, have = 0, neg = 0;
    for (;;) {
        char c; ssize_t k = recv(fd, &c, 1, 0);
        if (k <= 0) { if (k < 0 && errno == EINTR) continue; return -1; }
        if (c == '-' && !have) { neg = 1; continue; }
        if (c >= '0' && c <= '9') { cur = cur * 10 + (c - '0'); have = 1; continue; }
        if (have) { if (n < max) out[n++] = cur; cur = 0; have = 0; }
        if (c == '\n') return n;
        (void)neg;
    }
}
#endif
