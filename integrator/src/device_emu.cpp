/*
    device_emu.cpp -- (c) 2026 Paolo Ribeca, <paolo.ribeca@gmail.com>

    An EMULATED GPU: runs the device program of rbar/batch.h (rb_block_run)
    on the host, with the device's data layout: packed, pointer-relocated
    class images, a views array indexed by class, interleaved
    (lane-strided) logP scratch shared by `blocks` persistent blocks, a
    launch-wide atomic chunk counter, and chunk results written to an
    output array pre-filled with NaN (so that a chunk that is never written
    is detected by the scheduler's sample-count check).  Lanes of a phase
    run one after the other (optionally in reverse order, to expose any
    dependence between lanes inside a phase, i.e. a missing barrier).
    Evicted classes get a zeroed view: a launch that touches a non-resident
    class is rejected here (residency check, as in the CUDA device), and
    with --emu-no-residency-check it reaches the in-kernel view guard
    instead (chunks come back with n = NaN).

    In the GPU build this file is compiled by nvcc (as CUDA source, host
    pass), i.e. with the same front end and preprocessing as the real
    kernel; in the CPU build by the host C++ compiler.  The header
    functions have internal linkage (rbar/config.h), so this object file
    really runs its own, nvcc-compiled, copy of rb_block_run.

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

#include <omp.h>
#include <cmath>
#include <cstring>
#include <map>
#include "backend.h"

namespace {
class EmuDevice : public Device {
  int id_;
  BackendOptions o_;
  int G_;
  int launches_ = 0;
  bool armed_ = false;
  std::vector<RbClassView> views_;
  std::map<int, std::vector<double>> img_;
  std::vector<double> scratch_;
  int natoms_max_ = 1;
  // Device interface.
 public:
  EmuDevice(int id, const BackendOptions &o) : id_(id), o_(o), G_(std::max(1, o.emu_blocks)) {}
  std::string label() const override { return "emu:" + std::to_string(id_); }
  bool open(int nclasses, int natoms_max, std::string *) override {
    RbClassView z;
    memset(&z, 0, sizeof z);
    views_.assign((size_t)std::max(1, nclasses), z);
    natoms_max_ = std::max(1, natoms_max);
    scratch_.assign((size_t)G_ * RB_BLOCK * natoms_max_, NAN);
    return true;
  }
  int blocks() const override { return G_; }
  double tick_hz() const override { return 1e9; }
  // Fault injection counts production launches only
  void arm() override { launches_ = 0; armed_ = true; }
  size_t mem_budget() const override {
    return o_.emu_mem > 0 ? (size_t)o_.emu_mem : ((size_t)1 << 62);
  }
  bool upload(int cls, const PackedClass &pc, std::string *err) override {
    if (cls < 0 || cls >= (int)views_.size()) { *err = "class slot out of range"; return false; }
    std::vector<double> &b = img_[cls];
    b = pc.buf;
    views_[cls] = view_at(pc.rel, b.data());
    return true;
  }
  bool evict(int cls, std::string *err) override {
    if (cls < 0 || cls >= (int)views_.size()) { *err = "class slot out of range"; return false; }
    img_.erase(cls);
    memset(&views_[cls], 0, sizeof(RbClassView));
    return true;
  }
  bool launch(const std::vector<RbWorkItem> &items, uint64_t nchunks, std::vector<RbChunkOut> &out,
      LaunchStats *st, double, std::string *err) override {
    if (armed_) ++launches_;
    const bool inject = armed_ && o_.emu_fail_device == id_ && launches_ == o_.emu_fail_after;
    if (inject && o_.emu_fail_mode == 0) { *err = "injected fault (emulator test)"; return false; }
    RbChunkOut poison;
    // NaN in every double: an unwritten chunk fails the count check
    memset(&poison, 0xff, sizeof poison);
    out.assign(nchunks, poison);
    if (inject && o_.emu_fail_mode == 1 && !items.empty() && items[0].cls >= 0
        && items[0].cls < (int)views_.size())
      // The kernel must refuse this view (rb_view_ok)
      memset(&views_[items[0].cls], 0, sizeof(RbClassView));
    else
      for (const RbWorkItem &it : items)
        if (it.cls < 0 || it.cls >= (int)views_.size() || !img_.count(it.cls)) {
          *err = "work item for a non-resident class";
          return false;
        }
    unsigned long long counter = 0;
    const int nt = std::max(1, omp_get_max_threads() / std::max(1, o_.emu_devices));
    const double t0 = rb_now_s();
#pragma omp parallel for schedule(dynamic, 1) num_threads(nt)
    for (int b = 0; b < G_; ++b) {
      // Lane t of block b: element a at [a * G*RB_BLOCK + b*RB_BLOCK + t] (in elements of the
      // evaluation type), exactly as in the CUDA kernel
      if (o_.fp32) {
        RbHostBlock hb(&counter, (float *)scratch_.data() + (size_t)b * RB_BLOCK, 1, G_ * RB_BLOCK,
          o_.emu_reverse);
        rb_block_run<RbExec, float>(hb.x, views_.data(), items.data(), (int)items.size(), nchunks,
          out.data());
      } else {
        RbHostBlock hb(&counter, scratch_.data() + (size_t)b * RB_BLOCK, 1, G_ * RB_BLOCK,
          o_.emu_reverse);
        rb_block_run<RbExec, double>(hb.x, views_.data(), items.data(), (int)items.size(), nchunks,
          out.data());
      }
    }
    st->elapsed_s = rb_now_s() - t0;
    st->ticks = 0;
    for (auto &c : out) st->ticks += c.ticks;
    return true;
  }
  bool fptest(std::vector<double> &fp, std::vector<uint32_t> &kat, uint64_t *signature,
      std::string *) override {
    fp.assign(4 * RB_NFPTEST, 0);
    kat.assign(4 * RB_NKAT, 0);
    for (int i = 0; i < RB_NFPTEST; ++i) rb_fptest(i, fp.data());
    for (int i = 0; i < RB_NKAT; ++i) rb_kat(i, kat.data());
    *signature = rb_build_signature(); // computed by THIS object file's compiler and -D settings
    return true;
  }
  bool debug_samples(int cls, const RbStreamId &id, uint64_t idx0, int n,
      std::vector<RbSampleDbg> &out, std::string *) override {
    out.assign(n, RbSampleDbg());
    std::vector<double> scr((size_t)n * natoms_max_, NAN); // interleaved, as on the device
    for (int i = 0; i < n; ++i)
      rb_sample_dbg(views_[cls], id.key, id.stream, idx0 + i, scr.data() + i, n, &out[i]);
    return true;
  }
  void info_json(JsonW &J) const override {
    J.begin_obj();
    J.kv("type", "emulated");
#if defined(__CUDACC__)
    J.kv("compiled_by", "nvcc (host pass)");
#else
    J.kv("compiled_by", "host C++ compiler");
#endif
    J.kvi("blocks", G_); J.kvi("lanes_per_block", RB_BLOCK);
    J.kv("mem_budget", (double)mem_budget()); J.kvb("reverse_lanes", o_.emu_reverse);
    J.kvi("fail_after", o_.emu_fail_device == id_ ? o_.emu_fail_after : 0);
    J.end_obj();
  }
};
} // namespace

std::unique_ptr<Device> make_emu_device(int id, const BackendOptions &o) {
  return std::unique_ptr<Device>(new EmuDevice(id, o));
}
