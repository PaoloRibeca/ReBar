/*
    jsonr.h -- (c) 2026 Paolo Ribeca, <paolo.ribeca@gmail.com>

    Minimal JSON reader (no dependencies), enough to read back the run JSON
    this program writes (checkpoint / resume).  Numbers are parsed with
    strtod (correctly rounded: "%.17g" values come back bit for bit); null
    becomes a NUL value.  Values of the top-level keys listed in `skip` are
    parsed but not stored (the run JSON of a large order holds tens of MB
    of per-class diagnostics and launch logs that a resume does not need).

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
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <set>
#include <string>
#include <utility>
#include <vector>

struct JV {
  enum T { NUL, BOOL, NUM, STR, ARR, OBJ } t = NUL;
  bool b = false;
  double n = 0;
  std::string s;
  std::vector<JV> a;
  std::vector<std::pair<std::string, JV>> o;
  const JV *get(const std::string &k) const {
    for (auto &p : o) if (p.first == k) return &p.second;
    return nullptr;
  }
  std::string str(const std::string &k, const std::string &d = "") const {
    const JV *v = get(k);
    return v && v->t == STR ? v->s : d;
  }
  double num(const std::string &k, double d = 0) const {
    const JV *v = get(k);
    return v && v->t == NUM ? v->n : d;
  }
  bool boolean(const std::string &k, bool d = false) const {
    const JV *v = get(k);
    return v && v->t == BOOL ? v->b : d;
  }
};

class JsonReader {
 public:
  JsonReader(const char *p, const char *e, const std::set<std::string> &skip) : p_(p), e_(e),
      b0_(p), skip_(skip) {}
  bool parse(JV *out, std::string *err) {
    if (!value(out, 0, false)) {
      *err = "JSON parse error at byte " + std::to_string(pos_err_) + ": " + msg_;
      return false;
    }
    ws();
    if (p_ != e_) { *err = "trailing characters after the JSON value"; return false; }
    return true;
  }
  // Parser state and parsing functions.
 private:
  const char *p_, *e_, *b0_ = nullptr;
  const std::set<std::string> &skip_;
  size_t pos_err_ = 0;
  std::string msg_;
  bool fail(const char *m) { msg_ = m; pos_err_ = (size_t)(p_ - b0_); return false; }
  void ws() { while (p_ < e_ && (*p_ == ' ' || *p_ == '\n' || *p_ == '\r' || *p_ == '\t')) ++p_; }
  bool lit(const char *w) {
    size_t n = strlen(w);
    if ((size_t)(e_ - p_) < n || memcmp(p_, w, n)) return false;
    p_ += n;
    return true;
  }
  bool string(std::string *s) {
    if (p_ >= e_ || *p_ != '"') return fail("expected a string");
    ++p_;
    while (p_ < e_ && *p_ != '"') {
      if (*p_ == '\\') {
        if (++p_ >= e_) return fail("bad escape");
        char c = *p_++;
        switch (c) {
          case '"': case '\\': case '/': if (s) *s += c; break;
          case 'b': if (s) *s += '\b'; break;
          case 'f': if (s) *s += '\f'; break;
          case 'n': if (s) *s += '\n'; break;
          case 'r': if (s) *s += '\r'; break;
          case 't': if (s) *s += '\t'; break;
          case 'u': {
            if (e_ - p_ < 4) return fail("bad \\u escape");
            unsigned v = (unsigned)strtoul(std::string(p_, 4).c_str(), nullptr, 16);
            p_ += 4;
            // This program writes ASCII only
            if (s) { if (v < 0x80) *s += (char)v; else *s += '?'; }
            break;
          }
          default: return fail("bad escape");
        }
      } else {
        if (s) *s += *p_;
        ++p_;
      }
    }
    if (p_ >= e_) return fail("unterminated string");
    ++p_;
    return true;
  }
  // discard = true: parse and drop the value
  bool value(JV *v, int depth, bool discard) {
    if (depth > 64) return fail("nesting too deep");
    ws();
    if (p_ >= e_) return fail("unexpected end");
    const char c = *p_;
    if (c == '{') {
      ++p_;
      if (!discard) v->t = JV::OBJ;
      ws();
      if (p_ < e_ && *p_ == '}') { ++p_; return true; }
      for (;;) {
        ws();
        std::string k;
        if (!string(&k)) return false;
        ws();
        if (p_ >= e_ || *p_ != ':') return fail("expected ':'");
        ++p_;
        const bool drop = discard || (depth == 0 && skip_.count(k));
        if (drop) {
          JV dummy;
          if (!value(&dummy, depth + 1, true)) return false;
        } else {
          v->o.emplace_back(k, JV());
          if (!value(&v->o.back().second, depth + 1, false)) return false;
        }
        ws();
        if (p_ < e_ && *p_ == ',') { ++p_; continue; }
        if (p_ < e_ && *p_ == '}') { ++p_; return true; }
        return fail("expected ',' or '}'");
      }
    }
    if (c == '[') {
      ++p_;
      if (!discard) v->t = JV::ARR;
      ws();
      if (p_ < e_ && *p_ == ']') { ++p_; return true; }
      for (;;) {
        if (discard) { JV dummy; if (!value(&dummy, depth + 1, true)) return false; }
        else { v->a.emplace_back(); if (!value(&v->a.back(), depth + 1, false)) return false; }
        ws();
        if (p_ < e_ && *p_ == ',') { ++p_; continue; }
        if (p_ < e_ && *p_ == ']') { ++p_; return true; }
        return fail("expected ',' or ']'");
      }
    }
    if (c == '"') {
      if (!discard) v->t = JV::STR;
      return string(discard ? nullptr : &v->s);
    }
    if (lit("true")) { if (!discard) { v->t = JV::BOOL; v->b = true; } return true; }
    if (lit("false")) { if (!discard) { v->t = JV::BOOL; v->b = false; } return true; }
    if (lit("null")) { if (!discard) v->t = JV::NUL; return true; }
    {
      const char *q = p_;
      while (q < e_ && (isdigit((unsigned char)*q) || *q == '-' || *q == '+' || *q == '.'
          || *q == 'e' || *q == 'E'))
        ++q;
      if (q == p_) return fail("unexpected character");
      if (!discard) {
        v->t = JV::NUM;
        v->n = strtod(std::string(p_, q).c_str(), nullptr);
      }
      p_ = q;
      return true;
    }
  }
};

// Read and parse a whole file.  skip: top-level keys whose values are not stored.
inline bool json_read_file(const std::string &fn, JV *out, std::string *err,
    const std::set<std::string> &skip = {}) {
  FILE *f = fopen(fn.c_str(), "rb");
  if (!f) { *err = "cannot open " + fn; return false; }
  std::string text;
  char buf[1 << 16];
  size_t k;
  while ((k = fread(buf, 1, sizeof buf, f)) > 0) text.append(buf, k);
  fclose(f);
  JsonReader r(text.data(), text.data() + text.size(), skip);
  if (!r.parse(out, err)) { *err = fn + ": " + *err; return false; }
  return true;
}
