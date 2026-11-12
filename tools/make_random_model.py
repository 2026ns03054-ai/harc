#!/usr/bin/env python3
"""Write a randomly initialised checkpoint in llama2.c format, for testing only.

Timing and exactness tests behave the same as with trained weights of the same
shape; generated text is meaningless. For the paper's experiments, download the
real TinyStories checkpoints released with llama2.c (see README).

usage: make_random_model.py out.bin [dim hidden layers heads kv_heads vocab seq_len]
default shape = stories15M (288 768 6 6 6 32000 256)
"""
import struct, sys
import numpy as np

out = sys.argv[1]
dim, hidden, layers, heads, kvh, vocab, seq = (int(x) for x in (sys.argv[2:9] if len(sys.argv) >= 9 else [288, 768, 6, 6, 6, 32000, 256]))
head = dim // heads
kv = dim * kvh // heads
rng = np.random.default_rng(1234)
def w(*shape, std=0.02):
    return (rng.standard_normal(int(np.prod(shape))) * std).astype(np.float32)
with open(out, 'wb') as f:
    f.write(struct.pack('7i', dim, hidden, layers, heads, kvh, vocab, seq))  # positive vocab = shared classifier
    for arr in [w(vocab, dim), np.ones(layers * dim, np.float32),
                w(layers, dim, dim), w(layers, kv, dim), w(layers, kv, dim), w(layers, dim, dim),
                np.ones(layers * dim, np.float32),
                w(layers, hidden, dim), w(layers, dim, hidden), w(layers, hidden, dim),
                np.ones(dim, np.float32),
                np.zeros(seq * head // 2, np.float32), np.zeros(seq * head // 2, np.float32)]:
        f.write(arr.tobytes())
print(f'wrote {out}: dim={dim} hidden={hidden} layers={layers} heads={heads} kv_heads={kvh} vocab={vocab} seq_len={seq}')
