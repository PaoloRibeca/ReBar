/*
    json.h -- (c) 2026 Paolo Ribeca, <paolo.ribeca@gmail.com>

    Minimal JSON writer (no dependencies).  Non-finite numbers are written
    as null.

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
#include <cmath>
#include <cstdio>
#include <unistd.h>
#include <string>
#include <vector>

class JsonW {
 public:
  std::string s;
  void raw(const std::string &x) { comma(); s += x; }
  void key(const std::string &k) { comma(); s += quote(k) + ":"; need_comma = false; }
  void str(const std::string &v) { comma(); s += quote(v); }
  void num(double v) {
    comma();
    if (!std::isfinite(v)) { s += "null"; return; }
    char b[40]; snprintf(b, sizeof b, "%.17g", v); s += b;
  }
  void integer(long long v) { comma(); s += std::to_string(v); }
  void boolean(bool v) { comma(); s += v ? "true" : "false"; }
  void begin_obj() { comma(); s += "{"; need_comma = false; }
  void end_obj() { s += "}"; need_comma = true; }
  void begin_arr() { comma(); s += "["; need_comma = false; }
  void end_arr() { s += "]"; need_comma = true; }
  void kv(const std::string &k, double v) { key(k); num(v); }
  void kv(const std::string &k, const std::string &v) { key(k); str(v); }
  void kvi(const std::string &k, long long v) { key(k); integer(v); }
  void kvb(const std::string &k, bool v) { key(k); boolean(v); }
  void kvarr(const std::string &k, const std::vector<double> &v) {
    key(k);
    begin_arr();
    for (double x : v) num(x);
    end_arr();
  }
  static std::string quote(const std::string &x) {
    std::string r = "\"";
    for (char c : x) {
      if (c == '"' || c == '\\') { r += '\\'; r += c; }
      else if ((unsigned char)c < 0x20) { char b[8]; snprintf(b, sizeof b, "\\u%04x", c); r += b; }
      else r += c;
    }
    return r + "\"";
  }
  // Atomic replacement: write FILE.tmp, flush it to stable storage, rename over FILE.  On any write
  // error (disk full, quota) the old FILE is left untouched -- it may be the checkpoint a resume
  // needs.
  bool write(const std::string &fn) const {
    std::string tmp = fn + ".tmp";
    FILE *f = fopen(tmp.c_str(), "w");
    if (!f) return false;
    bool ok = fwrite(s.data(), 1, s.size(), f) == s.size() && fputc('\n', f) != EOF
        && fflush(f) == 0;
    ok = ok && fsync(fileno(f)) == 0;
    ok = (fclose(f) == 0) && ok;
    if (!ok) { remove(tmp.c_str()); return false; }
    return rename(tmp.c_str(), fn.c_str()) == 0;
  }
  // Comma bookkeeping.
 private:
  bool need_comma = false;
  void comma() { if (need_comma) s += ","; need_comma = true; }
};
