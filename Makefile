# HARC / HyPEC prototype
#   make            build everything into build/
#   make test       exactness tests and prefill speed-up (needs models/, see tools/get_models.sh)
#   make PORTABLE=1 build without CPU-specific instructions and without FMA contraction,
#                   so that workers on different CPU models compute identical results (RQ5)
CC      ?= gcc
ifeq ($(PORTABLE),1)
ARCH    = -march=x86-64-v2 -ffp-contract=off
else
ARCH    = -march=native
endif
CFLAGS  ?= -O3 $(ARCH) -Wall -Wno-unknown-pragmas -Wno-unused-function -Wno-format-truncation
LDLIBS  = -lm -lpthread
LLAMA   = third_party/llama2.c/run.c

all: build/master build/worker build/client build/peer build/test_engine

build:
	mkdir -p build

build/master: src/master.c src/net.h | build
	$(CC) $(CFLAGS) -o $@ src/master.c $(LDLIBS)

build/worker: src/worker.c src/engine.c src/engine.h src/net.h $(LLAMA) | build
	$(CC) $(CFLAGS) -Wno-unused-variable -Wno-unused-but-set-variable -Wno-sign-compare -Wno-maybe-uninitialized -o $@ src/worker.c src/engine.c $(LDLIBS)

build/client: src/client.c src/net.h | build
	$(CC) $(CFLAGS) -o $@ src/client.c $(LDLIBS)

build/peer: src/peer.c src/net.h | build
	$(CC) $(CFLAGS) -o $@ src/peer.c $(LDLIBS)

build/test_engine: src/test_engine.c src/engine.c src/engine.h $(LLAMA) | build
	$(CC) $(CFLAGS) -Wno-unused-variable -Wno-unused-but-set-variable -Wno-sign-compare -Wno-maybe-uninitialized -o $@ src/test_engine.c src/engine.c $(LDLIBS)

MODEL ?= models/stories15M.bin
test: build/test_engine
	./build/test_engine $(MODEL) models/tokenizer.bin 120 40 128

clean:
	rm -rf build

.PHONY: all test clean
