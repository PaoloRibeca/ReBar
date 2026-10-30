/*
    handle.hpp -- (c) 2000-2026 Paolo Ribeca, <paolo.ribeca@gmail.com>

    A minimal reference-counted handle: a pointer to a heap object shared
    by all the copies of the handle and deleted with the last of them, so
    that topologies, which derive from it, can be passed and stored by
    value at the cost of a pointer. With HANDLE_PROFILE defined,
    nHandledObjects counts the objects alive.

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

#ifndef TOPOLOGIES_HANDLE_HPP
#define TOPOLOGIES_HANDLE_HPP

#include <iostream>

// Compile-time switches (set with -D):
//  HANDLE_PROFILE counts the handled objects alive in (nHandledObjects),
//  HANDLE_DEBUG also prints the count at every allocation and deallocation

inline long int nHandledObjects = 0;

template <class T> class handle {
  T *object;
  int *pRefCount;
public:
  // That is: when the handle is dereferenced, the pointer to the object is handed over directly: it
  //  is "as if" it were a pointer to the object
  T *operator->() { return this->object; }
  handle(T *pT) : object(pT), pRefCount(new int(1)) {
#ifdef HANDLE_PROFILE
    ++nHandledObjects;
#endif
#ifdef HANDLE_DEBUG
    std::cout << "<nHandledObjects>=" << nHandledObjects << std::endl;
#endif
  }
  handle(const handle &o) : object(o.object), pRefCount(o.pRefCount) {
    ++(*this->pRefCount);
  }
  handle &operator=(const handle &o) {
    // Everything is already fine as it is
    if (this->object == o.object) return *this;
    if (--(*this->pRefCount) == 0) {
      // If the count has just dropped to zero, then (object) was necessarily allocated
      delete this->object;
      delete this->pRefCount;
#ifdef HANDLE_PROFILE
      --nHandledObjects;
#endif
#ifdef HANDLE_DEBUG
      std::cout << "<nHandledObjects>=" << nHandledObjects << std::endl;
#endif
    }
    this->object = o.object; // Possibly null
    this->pRefCount = o.pRefCount; // Possibly null
    ++(*this->pRefCount);
    return *this;
  }
  ~handle() {
    if (--(*this->pRefCount) == 0) {
      delete this->object;
      delete this->pRefCount;
#ifdef HANDLE_PROFILE
      --nHandledObjects;
#endif
#ifdef HANDLE_DEBUG
      std::cout << "<nHandledObjects>=" << nHandledObjects << std::endl;
#endif
    }
  }
};

#endif
