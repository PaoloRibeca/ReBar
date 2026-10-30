# Releases

Binary releases of ReBar (the integrator for CPUs and for NVIDIA GPUs, the graph generator and the graph lists) are
built from a tagged commit by the GitHub workflow [`.github/workflows/build-binaries.yml`](../.github/workflows/build-binaries.yml)
and attached to the [releases page](https://github.com/PaoloRibeca/ReBar/releases) as
`ReBar-<version>-Linux-x86_64.tar.xz`. This directory holds what defines them.

| File | What it is |
|---|---|
| `CURRENT` | the version of the next release, `N.N.N`; the integrator compiles it in as its version |
| `MANIFEST` | what an archive holds: source in the tree and name in the archive, one per line |
| `ARCHIVE-README.md` | the README inside the archive |
| `ReBar-<version>-<os>-<arch>/`, `.tar.xz` | assembled packages (ignored by git) |

The tools are in [`../tools/`](../tools/):

    tools/release package [<ver>]   # build the binaries with the release flags and assemble the archive here
    tools/release ci-begin          # tag v<CURRENT> on HEAD and push it: the workflow builds and publishes the release
    tools/release ci-end            # wait for the workflow and download the archive it attached
    tools/build-info                # the build stamp of this tree: <release>-<count> <date> <commit>

## Versioning

The version that people cite is the one in `CURRENT`, set by hand when a release is made (`tools/release package
0.5.0` writes it; commit it before `ci-begin`). Every build is also stamped automatically by `tools/build-info`
with the number of files committed over the whole history, which grows with every commit, the date of the last
commit and its short hash (`+` when the tree had uncommitted changes): `rebar_cpu version` prints, for instance,
`rebar_cpu 0.4.3 build 0.4.3-4167 2026-10-28 63420a4 variant e16 (...)`, and every run JSON records the stamp as
`build` next to `version`. Two builds of one release are thus told apart, and a run can be traced to its commit.
The stamp is read at `make` time: after a commit, `make clean` before rebuilding gives the new one (`tools/release`
always rebuilds).

The ledger accepts runs only from known versions (`series/rebar_ledger.py`, `POLICY`), so a new version whose runs are
to be merged must be added there.

## Making a release

1. Set the version: `tools/release package 0.5.0` (or edit `CURRENT`), check the archive it assembles, and commit.
2. `tools/release ci-begin` tags the commit `v0.5.0` and pushes the tag. The workflow checks the repository out with
   its full history (for the stamp), installs the CUDA 12.4 compiler and runtime, builds the CPU binaries and the
   generator statically for any x86-64 Linux (`-march=x86-64-v2`), the GPU binaries for compute capabilities 8.0 and
   9.0 with PTX for 9.0 and the C++ runtime linked statically (they need the runner's glibc, 2.34, or newer), assembles the archive with `tools/release package`, runs the binaries once, and creates the
   GitHub release of the tag with the archive attached (a manual run of the workflow only uploads the archive as a
   workflow artifact).
3. `tools/release ci-end` waits for it and downloads the archive.

`tools/release package` on any machine assembles the same archive for that machine (without the GPU binaries where
there is no CUDA toolkit); `RELEASE_LDFLAGS=""` links dynamically, `RELEASE_CXXFLAGS` and `RELEASE_CUDA_ARCH` change
the flags.
