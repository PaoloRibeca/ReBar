/*
    cache.h -- (c) 2026 Paolo Ribeca, <paolo.ribeca@gmail.com>

    The class cache: the built classes of a list (ClassData and the seconds
    their construction took), written once by `rebar build` and loaded by
    later runs through --class-cache FILE, so that the host-side forest
    expansion of an order is done once and not in every job. The file
    records the code version, the limit variant and the build options; a
    file built differently is refused with a message, and the run builds
    the classes itself.

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
#include <string>
#include <utility>
#include <vector>
#include "build.h"

struct CachedClass {
  ClassData cd;
  double build_s = 0;
};
// "version|RB_MAXE|RB_MAXLOOP|RB_MAXATOMS|kind|prune|vertex_disjoint|omega_shift|omega_scale|allow_negative_faces|tables"
std::string cache_fingerprint(const BuildOptions &bo, bool tables, const char *version);
bool cache_write(const std::string &fn, const std::string &fingerprint,
    const std::vector<CachedClass> &classes, std::string *err);
bool cache_read(const std::string &fn, const std::string &fingerprint,
    std::vector<CachedClass> &classes, std::string *err);
