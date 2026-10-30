/*
    backend.h -- (c) 2026 Paolo Ribeca, <paolo.ribeca@gmail.com>

    Sampling backends behind the run driver.
      cpu: OpenMP on the host; one OpenMP thread = one block of the
        canonical batch algorithm (rbar/batch.h).
      cuda: every visible (or --devices-selected) GPU, one host thread per
        device, dynamic work queue of chunk-aligned units, per-class Neyman
        allocation done by the driver (src/main.cpp).
      emu: the same multi-device scheduler driving EMULATED devices: the
        device-side program (rb_block_run) is compiled for the host and run
        lane by lane on packed, pointer-relocated class images.  This is
        the test path for everything the GPU build does except the CUDA API
        calls themselves.
      auto: cuda if at least one device passes its start-up self-test, else
        cpu (reason recorded in the JSON).

    Determinism: a class's accumulator depends only on (Philox key, stream,
    sample range); results are bitwise the same for the cpu and emu
    backends and any number of threads / devices / blocks, and differ from
    a real GPU only in the last bits of the per-sample weights (device
    libm, FMA contraction).

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
#include <atomic>
#include <memory>
#include <string>
#include <vector>
#include "build.h"
#include "json.h"
#include "rbar/batch.h"

// Set by SIGINT / SIGTERM: finish the current launches, write the JSON, exit
extern std::atomic<bool> g_rb_stop;
// A device hung: skip CUDA teardown and leave with _Exit after the JSON
extern std::atomic<bool> g_rb_hung;
// TEST hooks (run --test-stop-after-units / --test-kill-after-units; -1 = off): after that many
// units (multi-device backend) or tasks (cpu backend) have been evaluated in this process, set
// g_rb_stop (as SIGTERM does) or SIGKILL the process (as a scheduler's hard kill does).  Used by
// the checkpoint / resume tests only.
extern std::atomic<long long> g_rb_test_stop_units, g_rb_test_kill_units;

// One sample range [idx0, idx0 + n) of one class.  init: the task's accumulator starts from it
// instead of zero (resume of a task interrupted after a prefix of its units: the canonical
// unit-by-unit merge order of the task is then continued exactly, so a resumed task gives the same
// bits as an uninterrupted one).
struct SampleTask { int cls; uint64_t idx0, n; RbAcc init{}; };
struct RbStreamId { uint64_t key; uint32_t stream; }; // Philox key and counter word 2 of one class
struct SampleOut {
  RbAcc acc; // = the task's init merged with the evaluated units, in unit order
  double cost_s = 0; // core-seconds (cpu) or device-seconds (cuda, emu) spent on this task
  uint64_t done = 0; // samples evaluated: always the prefix [idx0, idx0 + done) of the task
};

struct BackendOptions {
  std::string kind = "cpu"; // auto | cpu | emu | cuda
  std::string devices; // comma-separated CUDA ordinals (default: all visible)
  double launch_s = 0.5; // target duration of one GPU launch (time-box granularity)
  long long max_launch_chunks = 1 << 19;
  int selftest_classes = 3; // classes checked sample by sample against the host at start-up
  double selftest_tol = 1e-11; // |w_dev - w_host| <= tol * sum|terms| per sample
  bool selftest = true;
  double mem_frac = 0.7; // fraction of free device memory usable for class images
  // Minimum hang timeout of a launch (0: max(300, 50*launch_s)); the timeout of a calibrated launch
  // is max(hang_s, 20 x its expected duration)
  double hang_s = 0;
  bool fp32 = false; // evaluate in FP32 (bare kinds only; FP64 accumulation)
  // Emulator (test) options
  int emu_devices = 2, emu_blocks = 5;
  bool emu_reverse = false; // run the lanes of every phase in reverse order (race / barrier test)
  // Fault injection: that device fails at its n-th launch
  int emu_fail_device = -1, emu_fail_after = 0;
  double emu_mem = 0; // emulated device memory budget in bytes (0 = unlimited): eviction test
  // 0: that launch returns an error; 1: it runs on a ZEROED class view (as after an upload race),
  // bypassing the residency check: tests the in-kernel view guard
  int emu_fail_mode = 0;
};

class Backend {
 public:
  virtual ~Backend() {}
  virtual std::string name() const = 0;
  // All classes of the run (class id = index) and their Philox key / stream word
  virtual bool prepare(const std::vector<const ClassData *> &cls,
      const std::vector<RbStreamId> &ids, std::string *err) = 0;
  // Evaluate the tasks (any internal order); no new work is started after `deadline` (steady-clock
  // seconds) or once g_rb_stop is set; out[i].done is always a prefix of task i.
  virtual void run(const std::vector<SampleTask> &tasks, double deadline,
      std::vector<SampleOut> &out) = 0;
  virtual double capacity() const = 0; // cost-seconds per wall-clock second (threads or devices)
  virtual const char *cost_unit() const = 0; // "core-s" or "device-s"
  virtual void info_json(JsonW &J) const = 0; // the "backend_info" object
  virtual void round_report(FILE *f) const {} // one line per device after each round
};

std::unique_ptr<Backend> make_cpu_backend(bool fp32 = false);
// emu or cuda
std::unique_ptr<Backend> make_multi_backend(const BackendOptions &o, std::string *err);
double rb_now_s();

// CPU canonical evaluation of one sample range (OpenMP over chunks; any thread count gives the same
// bits); the unit accumulators are merged in order into *init when given (resumed task), else into
// zero
RbAcc cpu_run_range(const ClassData &cd, const RbStreamId &id, uint64_t idx0, uint64_t n,
    double *core_s, bool fp32 = false, const RbAcc *init = nullptr);

// Position-independent class image (what is uploaded to a device).
struct PackedClass {
  std::vector<double> buf; // 8-byte aligned storage; segments start at multiples of 16 bytes
  RbClassView rel; // pointer fields hold byte offsets into buf
  size_t bytes() const { return buf.size() * sizeof(double); }
};
PackedClass pack_class(const ClassData &cd);
size_t packed_bytes(const ClassData &cd); // = pack_class(cd).bytes(), without building the image
RbClassView view_at(const RbClassView &rel, const void *base); // relocate to an image at `base`

// One device (real GPU or emulated).
struct LaunchStats { double elapsed_s = 0; unsigned long long ticks = 0; };
class Device {
 public:
  virtual ~Device() {}
  virtual std::string label() const = 0;
  virtual bool open(int nclasses, int natoms_max, std::string *err) = 0;
  virtual void arm() {} // start-up diagnostics done: production launches follow
  // Called first by every host thread that drives the device
  virtual bool bind(std::string *err) { return true; }
  virtual int blocks() const = 0; // persistent blocks per launch (grid size)
  // RbChunkOut::ticks per second (SM clock on the GPU, 1e9 on the host)
  virtual double tick_hz() const = 0;
  virtual size_t mem_budget() const = 0; // bytes usable for class images
  virtual bool upload(int cls, const PackedClass &pc, std::string *err) = 0;
  virtual bool evict(int cls, std::string *err) = 0;
  // expected_s: the scheduler's estimate of the launch's duration (0 = unknown); sets the hang
  // timeout
  virtual bool launch(const std::vector<RbWorkItem> &items, uint64_t nchunks,
      std::vector<RbChunkOut> &out, LaunchStats *st, double expected_s, std::string *err) = 0;
  // Start-up diagnostics: rb_fptest values [4*RB_NFPTEST], Philox KAT outputs [4*RB_NKAT], build
  // signature
  virtual bool fptest(std::vector<double> &fp, std::vector<uint32_t> &kat, uint64_t *signature,
      std::string *err) = 0;
  virtual bool debug_samples(int cls, const RbStreamId &id, uint64_t idx0, int n,
      std::vector<RbSampleDbg> &out, std::string *err) = 0;
  // Static properties (name, capability, memory, kernel attributes)
  virtual void info_json(JsonW &J) const = 0;
};

struct CudaProbe {
  bool compiled = false;
  int count = 0, driver = 0, runtime = 0;
  std::string err;
};
CudaProbe cuda_probe(); // never fails; count = 0 when no usable driver/device
// nullptr without CUDA
std::unique_ptr<Device> make_cuda_device(int ordinal, const BackendOptions &o);
std::unique_ptr<Device> make_emu_device(int id, const BackendOptions &o);
