# ReBar binaries

This archive holds a binary release of ReBar, the code that computes the renormalization-group functions β, η and ν
of the three-dimensional O(N) φ⁴ model to ten loops: <https://github.com/PaoloRibeca/ReBar>. The version, the
build stamp and the compile-time limits of the binaries are in `VERSION` (the output of `rebar_cpu version`).

| File | What it is |
|---|---|
| `rebar_cpu`, `rebar_cpu-e18`, `rebar_cpu-e20` | the integrator for CPUs (OpenMP), for graphs of up to 16, 18 and 20 lines (`rebar_cpu` re-executes `rebar_cpu-e18` by itself when an order needs it; the ten-loop orders need `-e20`); statically linked, for any x86-64 Linux |
| `rebar_gpu`, `rebar_gpu-e18`, `rebar_gpu-e20` | the same for NVIDIA GPUs: machine code for compute capability 8.0 (A100) and 9.0 (H100, H200) and PTX for 9.0, which newer GPUs compile at load; the CUDA and C++ runtimes are linked in, so they need only an NVIDIA driver for CUDA 12 and glibc 2.34 or later (Ubuntu 22.04, RHEL 9 or newer); absent when the archive was built on a machine without CUDA |
| `topologies` | the graph generator (statically linked) |
| `lists/` | the graph lists of Γ⁽⁴⁾ to ten loops and Γ⁽²⁾ to ten loops in the form the integrator reads, with their README |
| `LICENSE` | GNU General Public License, version 3 or later |

    ./rebar_cpu version                                                          # version, build stamp, limits
    ./rebar_cpu run --kind b --loops 5 --gen --pilot 50000 --samples 1e8 --seed 1 --out b5.json
    ./rebar_cpu run --kind b --loops 6 --list lists/gamma4_L6.txt --drop-1vr --seconds 3600 --out b6.json
    ./rebar_gpu devices                                                          # the GPUs seen (JSON)
    ./rebar_gpu run --kind h --loops 9 --list lists/gamma2_L9.txt --drop-1vr --seconds 36000 --out h9.json
    ./topologies -E 4 -V 5 -N                                                    # the four-point graphs to 4 loops

The documentation of the options is in the repository (`integrator/README.md`, `integrator/TECHNICAL.md`,
`graphs/README.md`); the ledger of pooled results, the tools that merge runs into it and the paper are there too.
A run of the integrator can be contributed to the ledger: see `CONTRIBUTING.md` in the repository.
