/*
    partitions.hpp -- (c) 2000-2026 Paolo Ribeca, <paolo.ribeca@gmail.com>

    Tables of the integer partitions of every n up to NUM_MAX_IN_PART into
    a given number of parts (zeros allowed, non-increasing order), built
    once by createPartitionTable before any topology is constructed, and
    the functions that count them (howManyPartitions) and find the index
    of a given one (findPartition): the index of the partition of its
    degree that a row of the adjacency matrix is, part of the colour of
    its vertex in getCanonical.

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

#ifndef TOPOLOGIES_PARTITIONS_HPP
#define TOPOLOGIES_PARTITIONS_HPP

#include <cstdlib>
#include <iostream>

// Compile-time switch (set with -D):
//  DEBUG_PARTITIONS prints the tables as they are built and every partition read

// The largest number that can be partitioned, i.e. the largest vertex degree the generators accept
constexpr int NUM_MAX_IN_PART = 12;

// Cursor into the table being filled by (enumeratePartitions): when null the partitions are only
//  counted, not written
inline int *partitionCursor = 0;

// Enumerates (and, when (partitionCursor) is set, writes) the partitions of (n) into (nParts) parts
//  between (min) and (max), in non-increasing order; (partial), (sum) and (pos) are the state of
//  the recursion.
// Warning: it does not work in the case n=0 [which (createPartitionTable) handles by hand]
inline long int enumeratePartitions(int n, int nParts, int max, int min, int *partial = 0,
    int sum = 0, int pos = 0) {
  int newSum, remainder, nEmpty;
  long int nPartitions = 0;
  if (pos == 0) partial = new int[nParts];
  if (pos == (nParts - 1)) {
    for (int i = max; i > min - 1; --i) {
      newSum = sum + i;
      if (newSum != n) continue;
      partial[pos] = i;
      ++nPartitions;
      if (partitionCursor)
        for (int j = 0; j < nParts; ++j) {
          *(partitionCursor) = partial[j];
          partitionCursor++;
        }
    }
  } else
    for (int i = max; i > min - 1; --i) {
      newSum = sum + i;
      remainder = n - newSum;
      nEmpty = nParts - 1 - pos;
      if ((nEmpty * i < remainder) || (nEmpty * min > remainder)) continue;
      partial[pos] = i;
      max = (remainder < i) ? remainder : i;
      if (max < min) continue;
      nPartitions += enumeratePartitions(n, nParts, max, min, partial, newSum, pos + 1);
    }
  if (pos == 0) delete [] partial;
  return nPartitions;
}

// The tables must be built by (createPartitionTable) before any topology is constructed [(main)
//  does it first]
inline int partitionCounts[NUM_MAX_IN_PART + 1][NUM_MAX_IN_PART + 1];
inline int *partitionTable[NUM_MAX_IN_PART + 1][NUM_MAX_IN_PART + 1];

// As far as the number of the partitions is concerned the scheme is
//               || into
// Partition of  ||   0 | 1 | 2 | 3 | 4 | ...
// --------------------------------------
// --------------------------------------
//             0 ||   0 | 1 | < | < | < |  Here everything is diagonal
// --------------------------------------  except 0 into 1
//             1 ||   0 | 1 | < | < | < |
// --------------------------------------
//             2 ||   0 | 1 | 2 | < | < |
// --------------------------------------
//             3 ||   0 | 1 | 2 | 3 | < |
// --------------------------------------
//           ...
// (< : the same as the diagonal entry of the row)

// Whereas for the effective lengths the scheme is
//               || into
// Partition of  ||   0 | 1 | 2 | 3 | 4 | ...
// --------------------------------------
// --------------------------------------
//             0 ||  NV | 0 | 0 | 0 | 0 |  Here everything is diagonal
// --------------------------------------  except 0 into 1
//             1 ||  NV | 1 | 1 | 1 | 1 |
// --------------------------------------
//             2 ||  NV | 1 | 2 | < | < |
// --------------------------------------
//             3 ||  NV | 1 | 2 | 3 | < |
// --------------------------------------
//           ...
// (NV : not valid)

// Builds the partition tables. It must be called before anything else
inline void createPartitionTable(void) {
#ifdef DEBUG_PARTITIONS
  std::cout << "(createPartitionTable[void]): Start." << std::endl;
#endif
  partitionCounts[0][0] = 0;
  partitionCounts[0][1] = 1;
  for (int n = 1; n <= NUM_MAX_IN_PART; ++n) {
    partitionCounts[n][0] = 0;
    partitionCounts[n][1] = 1;
    partitionTable[n][1] = new int;
    *(partitionTable[n][1]) = n;
    for (int nParts = 2; nParts <= n; ++nParts) {
      partitionCursor = 0; // To inhibit the writing of the partitions
      partitionCounts[n][nParts] = enumeratePartitions(n, nParts, n, 0);
      partitionTable[n][nParts] = new int[nParts * partitionCounts[n][nParts]];
      partitionCursor = partitionTable[n][nParts]; // To allow the writing
      enumeratePartitions(n, nParts, n, 0);
#ifdef DEBUG_PARTITIONS
      for (int i = 0; i < partitionCounts[n][nParts]; ++i) {
        for (int j = 0; j < nParts; ++j)
          std::cout << (*(partitionTable[n][nParts] + i * nParts + j)) << " ";
        std::cout << std::endl;
      }
#endif
    }
    for (int nParts = n + 1; nParts <= NUM_MAX_IN_PART; ++nParts)
      partitionCounts[n][nParts] = partitionCounts[n][n];
  }
#ifdef DEBUG_PARTITIONS
  std::cout << "(createPartitionTable): End." << std::endl;
#endif
  return;
}

// (howManyPartitions) returns the number of the partitions of (n) into (nParts);
inline int howManyPartitions(int n, int nParts) {
  if (n > NUM_MAX_IN_PART) {
    std::cerr << "(howManyPartitions): Number of elements out of range." << std::endl;
    exit(1);
  }
  if (nParts == 0) return 0;
  if (nParts == 1) return 1;
  return ((nParts > n) ? partitionCounts[n][n + ((!n) ? 1 : 0)] : partitionCounts[n][nParts]);
}

// Comparison for (qsort): non-increasing order
inline int intComp(const void *a, const void *b) {
  return ((*(int *)a < *(int *)b) ? 1 : (*(int *)a == *(int *)b) ? 0 : -1);
}

// (findPartition) returns the index of the partition of (n) into (nParts) whose parts are those of
//  (vec), in whatever order, or -1 if there is none
inline int findPartition(int n, int nParts, int *vec) {
  int *sorted = new int[nParts];
  for (int i = 0; i < nParts; ++i) sorted[i] = vec[i];
  qsort(sorted, nParts, sizeof(int), intComp);
  int effectiveLength;
  int *effectivePartitions;
  if (n == 0) {
    effectiveLength = 0;
  } else if (n == 1 || nParts == 1) {
    effectiveLength = 1;
    effectivePartitions = partitionTable[n][1];
  } else {
    effectiveLength = (nParts > n) ? n : nParts;
    effectivePartitions = (nParts > n) ? (partitionTable[n][n]) : (partitionTable[n][nParts]);
  }
  int i, j = 0;
  for (i = 0; i < howManyPartitions(n, nParts); ++i) {
    for (j = 0; j < effectiveLength; ++j)
      if (sorted[j] != effectivePartitions[i * effectiveLength + j]) break;
    if (j == effectiveLength) break;
  }
  delete [] sorted;
  return ((j == effectiveLength) ? i : -1);
}

#endif
