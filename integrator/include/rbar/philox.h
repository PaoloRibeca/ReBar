/*
    philox.h -- (c) 2026 Paolo Ribeca, <paolo.ribeca@gmail.com>

    Philox4x32-10 counter-based RNG (Salmon, Moraes, Dror, Shaw, SC'11
    "Parallel random numbers: as easy as 1, 2, 3"; constants and round
    structure as in Random123).  Stateless: the 4x32 output depends only on
    (counter, key), so every sample of every class is reproducible and
    independent of the thread / block / GPU layout.

    Usage in this code base (stream scheme "salted-v1", the default since
    v0.3.0):
      key = rb_class_key(seed, salt) -- 64 bit; salt = FNV-1a 64 of
        "kind|L|Nickel index" (src/main.cpp), so every graph of every kind
        and loop order has its own key for any run seed
      counter = (sample_lo, sample_hi, stream, call) -- sample index (64
        bit); stream = kind << 26; call = 0,1,2,...
    A class's random numbers therefore depend only on (seed, kind, L,
    Nickel index, sample index): not on its position in the list, on how a
    loop order is split into runs, or on the other loop orders.
    (Legacy scheme of v0.1-0.2, --streams legacy: key = seed, stream =
    (kind << 26) | position in the list.)
    Each call yields 4 x 32 bits = two 64-bit words = two 53-bit uniforms
    in (0,1).

    This program was designed and developed by the author(s),
    with the assistance of the following AI tool(s):
      2026 Claude (Anthropic).
    The final logic and implementation were reviewed and verified in
    their entirety by the author(s).

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
*/

#pragma once
#include "config.h"

struct RbPhilox4 { uint32_t v[4]; };

RB_HDI void rb_mulhilo32(uint32_t a, uint32_t b, uint32_t *hi, uint32_t *lo) {
#if defined(__CUDA_ARCH__)
  *hi = __umulhi(a, b); *lo = a * b;
#else
  uint64_t p = (uint64_t)a * (uint64_t)b; *hi = (uint32_t)(p >> 32); *lo = (uint32_t)p;
#endif
}

RB_HDI RbPhilox4 rb_philox4x32_10(RbPhilox4 c, uint32_t k0, uint32_t k1) {
  const uint32_t M0 = 0xD2511F53u, M1 = 0xCD9E8D57u, W0 = 0x9E3779B9u, W1 = 0xBB67AE85u;
#if defined(__CUDA_ARCH__)
#pragma unroll
#endif
  for (int r = 0; r < 10; ++r) {
    uint32_t hi0, lo0, hi1, lo1;
    rb_mulhilo32(M0, c.v[0], &hi0, &lo0);
    rb_mulhilo32(M1, c.v[2], &hi1, &lo1);
    RbPhilox4 o;
    o.v[0] = hi1 ^ c.v[1] ^ k0; o.v[1] = lo1; o.v[2] = hi0 ^ c.v[3] ^ k1; o.v[3] = lo0;
    c = o;
    k0 += W0; k1 += W1;
  }
  return c;
}

/* splitmix64 finaliser (Steele, Lea, Flood 2014): a bijective 64-bit mixer */
RB_HDI uint64_t rb_mix64(uint64_t z) {
  z ^= z >> 30; z *= 0xbf58476d1ce4e5b9ULL;
  z ^= z >> 27; z *= 0x94d049bb133111ebULL;
  return z ^ (z >> 31);
}
/* Philox key of one class: mixes the run seed and the class salt (distinct seeds give unrelated
   keys) */
RB_HDI uint64_t rb_class_key(uint64_t seed, uint64_t salt) {
  return rb_mix64(rb_mix64(seed + 0x9e3779b97f4a7c15ULL) ^ salt);
}

/* 64-bit word -> double uniform in (0,1): 53 random bits, offset by half an ulp (never 0, never
   1) */
RB_HDI double rb_u01(uint64_t w) { return ((double)(w >> 11) + 0.5) * 1.1102230246251565404e-16; }

/* Sequential stream of uniforms for one sample: generates 2 uniforms per Philox call. */
struct RbStream {
  uint32_t k0, k1, c0, c1, c2, call;
  double buf[2];
  int left;
};
RB_HDI void rb_stream_init(RbStream *s, uint64_t key, uint32_t stream, uint64_t sample) {
  s->k0 = (uint32_t)key; s->k1 = (uint32_t)(key >> 32);
  s->c0 = (uint32_t)sample; s->c1 = (uint32_t)(sample >> 32); s->c2 = stream; s->call = 0;
  s->left = 0;
}
RB_HDI double rb_next(RbStream *s) {
  if (s->left == 0) {
    RbPhilox4 c; c.v[0] = s->c0; c.v[1] = s->c1; c.v[2] = s->c2; c.v[3] = s->call++;
    RbPhilox4 o = rb_philox4x32_10(c, s->k0, s->k1);
    s->buf[0] = rb_u01(((uint64_t)o.v[0] << 32) | o.v[1]);
    s->buf[1] = rb_u01(((uint64_t)o.v[2] << 32) | o.v[3]);
    s->left = 2;
  }
  return s->buf[2 - s->left--];
}
