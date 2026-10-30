/*
    device_cuda_stub.cpp -- (c) 2026 Paolo Ribeca, <paolo.ribeca@gmail.com>

    Linked into the CPU-only build (bin/rebar_cpu) in place of
    cuda/device_cuda.cu.

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

#include "backend.h"

CudaProbe cuda_probe() {
  CudaProbe p;
  p.compiled = false;
  p.err = "this binary was built without CUDA (build bin/rebar_gpu with 'make gpu')";
  return p;
}
std::unique_ptr<Device> make_cuda_device(int, const BackendOptions &) { return nullptr; }
