/*
    permutations.hpp -- (c) 2000-2026 Paolo Ribeca, <paolo.ribeca@gmail.com>

    A non-recursive iterator, permutatorOverColors (permutator), over the
    permutations of a vector that preserve a colouring of its elements,
    i.e. the permutations within each colour class, among which
    getCanonical seeks the canonical form of a topology.

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

#ifndef TOPOLOGIES_PERMUTATIONS_HPP
#define TOPOLOGIES_PERMUTATIONS_HPP

#include <algorithm>
#include <cstdlib>
#include <iostream>

// Compile-time switch (set with -D):
//  DEBUG_PERMUTATIONS_OVER_COLORS prints each permuted copy before it is written back

// Given as input a vector of objects and a vector of colours, one wants to obtain _in a
//  non-recursive way_ the permutations allowed by the colours.
// For instance, given an object
// -------------
// |A|B|C|D|E|F|
// ------------- and the colours {1,2,1,3,1,2}
// one derives the vector of the classes
// {1,1,1,2,2,3} -> {0,3,5}
// and the base permutation {1,3,5,2,6,4}; at this point a permuted object is derived (and copied)
// -------------
// |A|C|E|B|F|D|
// -------------;
// by permuting such an object _separately within each class_ one obtains the permutations allowed
//  by the colours ---obviously it is necessary to apply every time the inverse of the base
//  permutation to the permutations of this object, and to copy everything back into the original
//  vector.

// WARNING: the colours must be numbers >=0

template <class obj> class permutatorOverColors {
  int order;
  int nClasses;
  int *classes;
  obj *original; // It is only a pointer to the user's data
  obj *permutedCopy;
  int *basePermutation;
public:
  permutatorOverColors(obj *first, obj *last, int *iColors) {
    this->order = last - first;
    this->original = first;
    int maxCol = 0;
    for (int i = 0; i < this->order; ++i) maxCol = (iColors[i] > maxCol) ? iColors[i] : maxCol;
    int *tempCol = (int *)calloc(maxCol + 1, sizeof(int));
    for (int i = 0; i < this->order; ++i) ++(tempCol[iColors[i]]);
    this->nClasses = 0;
    for (int i = 0; i < maxCol + 1; ++i)
      if (tempCol[i]) ++(this->nClasses);
    this->classes = new int[this->nClasses];
    this->basePermutation = new int[this->order];
    int currentClass = 0, curr = 0;
    this->classes[0] = 0;
    for (int i = 0; i < maxCol + 1; ++i)
      if (tempCol[i]) {
        if (currentClass < this->nClasses - 1)
          this->classes[currentClass + 1] = this->classes[currentClass] + tempCol[i];
        // The vector of the colours is scanned to find all the occurrences of colour (i)
        for (int j = 0; j < this->order; ++j)
          if (iColors[j] == i) {
            this->basePermutation[curr] = j;
            ++curr;
          }
        ++(currentClass);
      }
    this->permutedCopy = new obj[this->order];
    // Warning: the original gets _copied_, hence the function works (!) only for the types that
    //  have a working copy constructor (!!)
    for (int i = 0; i < this->order; ++i) this->permutedCopy[i] = first[this->basePermutation[i]];
    free(tempCol);
  }
  ~permutatorOverColors(void) {
    delete [] this->classes;
    delete [] this->permutedCopy;
    delete [] this->basePermutation;
  }
  bool nextPermutation(obj *out = 0) {
    if (!out) out = this->original;
    int currentClass = 0;
    while ((currentClass < this->nClasses) && !std::next_permutation(this->permutedCopy
        + this->classes[currentClass], this->permutedCopy
        + ((currentClass == (this->nClasses - 1)) ? this->order : this->classes[currentClass + 1])))
      ++currentClass;
    if (currentClass == this->nClasses) return false;
    else {
#ifdef DEBUG_PERMUTATIONS_OVER_COLORS
      std::cout << "[ ";
      for (int i = 0; i < this->order; ++i) std::cout << this->permutedCopy[i] << " ";
      std::cout << "]   ";
#endif
      for (int i = 0; i < this->order; ++i) out[this->basePermutation[i]] = this->permutedCopy[i];
      return true;
    }
  }
  // It would be most appropriate to implement along the same lines a (prevPermutation) function
};

typedef class permutatorOverColors<int> permutator;

#endif
