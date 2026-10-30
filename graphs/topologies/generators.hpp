/*
    generators.hpp -- (c) 2000-2026 Paolo Ribeca, <paolo.ribeca@gmail.com>

    The graph classes of the generator of 2003, kept for the compact
    output and the canonical relabelling of its lists. A topology (a
    handle to handledPermutatedTopology) is the adjacency matrix of a graph
    plus the number of self-loops of each vertex, read through a
    smartPermutation that relabels the vertices without moving data; it
    knows its lines and loops, whether it is connected and one-particle
    irreducible, finds its canonical form (getCanonical: the largest
    matrix over the permutations within classes of vertices of equal
    colour, i.e. degree, self-loops and partition) and writes itself in
    the compact format of graphs/lists. Here "tadpole" means a self-loop,
    a line joining a vertex to itself, not a QFT tadpole.

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

#ifndef TOPOLOGIES_GENERATORS_HPP
#define TOPOLOGIES_GENERATORS_HPP

#include <cstdlib>
#include <iostream>

#include "handle.hpp"
#include "partitions.hpp"
#include "permutations.hpp"

// Compile-time switch (set with -D):
//  SMART_PERMUTATION_DEBUG prints the permutations compared by (getCanonical)

// To be safe, in (smartPermutation) the matrix is _not_ assumed to be symmetrised
class smartPermutation {
  // I don't really move anything!
  int order;
  int **realObject;
  int *currentPermutation;
  int **permutatedObject; // A vector of pointers
public:
  smartPermutation(int iOrder, int **iRealObject) {
    this->order = iOrder;
    this->realObject = iRealObject;
    this->currentPermutation = new int[iOrder];
    // By default the identity permutation is created
    for (int i = 0; i < iOrder; ++i) this->currentPermutation[i] = i;
    this->permutatedObject = new int *[(iOrder * (iOrder - 1)) / 2];
    for (int i = 0; i < (iOrder - 1); ++i)
      for (int j = i + 1; j < iOrder; ++j)
        // The matrix is _not_ assumed to be symmetrised
        this->permutatedObject[(i * (2 * iOrder - i - 1)) / 2 + j - i - 1] = this->realObject[i]
            + j;
  }
  ~smartPermutation() {
    delete [] this->currentPermutation;
    delete [] this->permutatedObject;
  }
  // The copy constructor allocates new memory (the class is not handled)
  smartPermutation(const smartPermutation &o) : order(o.order), realObject(o.realObject) {
    this->currentPermutation = new int[o.order];
    // The permutation stored in (o) is copied
    for (int i = 0; i < o.order; ++i) this->currentPermutation[i] = o.currentPermutation[i];
    int len = (o.order * (o.order - 1)) / 2;
    this->permutatedObject = new int *[len];
    for (int i = 0; i < len; ++i) this->permutatedObject[i] = o.permutatedObject[i];
  }
  int *rCurrentPermutation(void) { return this->currentPermutation; }
  int rPermutatedObject(int num) { return this->permutatedObject[num][0]; }
  // Actually it also sets the permutation to what is requested
  smartPermutation *setAndGetPermutation(int *iPermutation) {
    for (int i = 0; i < this->order; ++i) this->currentPermutation[i] = iPermutation[i];
    for (int i = 0; i < (this->order - 1); ++i)
      for (int j = i + 1; j < this->order; ++j)
        // The matrix is _not_ assumed to be symmetrised
        this->permutatedObject[(i * (2 * this->order - i - 1)) / 2 + j - i - 1] =
            ((this->currentPermutation[i] < this->currentPermutation[j])
                ? (this->realObject[this->currentPermutation[i]] + this->currentPermutation[j])
                : (this->realObject[this->currentPermutation[j]] + this->currentPermutation[i]));
    return this;
  }
  // Resets the permutation to the identity
  smartPermutation *setAndGetPermutationIdentity(void) {
    for (int i = 0; i < this->order; ++i) this->currentPermutation[i] = i;
    for (int i = 0; i < (this->order - 1); ++i)
      for (int j = i + 1; j < this->order; ++j)
        // The matrix is _not_ assumed to be symmetrised
        this->permutatedObject[(i * (2 * this->order - i - 1)) / 2 + j - i - 1] =
            this->realObject[i] + j;
    return this;
  }
  // It is _necessary_ to pass the smartPermutations by reference, otherwise passing by copy
  //  automatically invokes the destructors and "consumes" the objects
  // ...but the references are to _const_ (the arguments are not modified)
  friend bool operator==(const smartPermutation &a, const smartPermutation &b) {
    if (a.order == b.order) {
      int len = (a.order * (a.order - 1)) / 2;
      for (int i = 0; i < len; ++i)
        if (*(a.permutatedObject[i]) != *(b.permutatedObject[i])) return false;
      return true;
    } else return false;
  }
  friend bool operator<(const smartPermutation &a, const smartPermutation &b) {
    if (a.order < b.order) return true;
    if (a.order == b.order) {
      int len = (a.order * (a.order - 1)) / 2;
      int i = 0;
      while (*(a.permutatedObject[i]) == *(b.permutatedObject[i])) {
        ++i;
        if (i == len) return false;
      }
      if (*(a.permutatedObject[i]) < *(b.permutatedObject[i])) return true;
      else return false;
    } else return false;
  }
  friend bool operator>(const smartPermutation &a, const smartPermutation &b) {
    if (a.order > b.order) return true;
    if (a.order == b.order) {
      int len = (a.order * (a.order - 1)) / 2;
      int i = 0;
      while (*(a.permutatedObject[i]) == *(b.permutatedObject[i])) {
        ++i;
        if (i == len) return false;
      }
      if (*(a.permutatedObject[i]) > *(b.permutatedObject[i])) return true;
      else return false;
    } else return false;
  }
  friend std::ostream &operator<<(std::ostream &s, const smartPermutation &a) {
    int len = (a.order * (a.order - 1)) / 2;
    if (&s == &std::cout) {
      s << "(";
      for (int i = 0; i < len; ++i) s << *(a.permutatedObject[i]) << ((i == (len - 1)) ? "" : " ");
      s << ")";
    } else {
      int numL = 0, numZ = 0, prev = 0;
      // First of all (first) serves to establish whether there is at least one non-null link
      bool first = false;
      for (int i = 0; i < len; ++i)
        if (a.permutatedObject[i][0]) {
          first = true;
          break;
        }
      if (first) {
        // (numZ) accumulates the number of the zeros following each link, (numL) the number of the
        //  consecutive links with identical valence
        for (int i = 0; i < len; ++i)
          if (first || a.permutatedObject[i][0] != prev) {
            if (!first) {
              // If the previous ones were zeros...
              if (numZ) s << numZ;
              // If the previous ones were numbers...
              else if (numL > 1) s << "^" << numL;
            }
            // If the next one is a number...
            if (a.permutatedObject[i][0]) {
              s << ((first && !numZ) ? "" : " ") << (a.permutatedObject[i][0]);
              numL = 1, numZ = 0;
              prev = a.permutatedObject[i][0];
            } else {
              s << ".";
              numZ = 1, numL = 0;
              prev = 0;
            }
            first = false;
          } else
            // Here it is necessarily (first==false)
            if (prev == 0) ++numZ;
            else ++numL;
        // The exit conditions _must_ be checked.
        // If the previous ones were zeros...
        if (numZ) s << numZ;
        // If the previous ones were numbers...
        else if (numL > 1) s << "^" << numL;
      }
    }
    return s;
  }
};

// To compute the number of the connected components of a topology
inline int numOfConnectedGroups(int **iMatrix, int iOrder) {
  // The allocation, kept by means of static objects, is [re-]done only when a different dimension
  //  is specified.
  // To compute the number of the connected components of the graph it suffices to colour the
  //  vertices according to the groups linked to each other, i.e. reachable from one another
  static int order = 0;
  static int *connectedGroups;
  static bool *connectedToSeed, *difference;
  static int nConnectedGroups, posSeed; // They might as well not be static
  if (iOrder != order) {
    if (order) {
      free(connectedGroups);
      delete [] connectedToSeed;
      delete [] difference;
    }
    order = iOrder;
    connectedGroups = (int *)calloc(iOrder, sizeof(int));
    connectedToSeed = new bool[iOrder];
    difference = new bool[iOrder];
  }
  if (order)
    for (int i = 0; i < iOrder; ++i) connectedGroups[i] = 0;
  nConnectedGroups = 0;
  posSeed = 0;
  while (posSeed < iOrder && !connectedGroups[posSeed]) {
    // Initial cleaning-initialisation
    ++nConnectedGroups;
    connectedGroups[posSeed] = nConnectedGroups;
    for (int i = 0; i < iOrder; ++i) connectedToSeed[i] = false;
    // The difference is computed between the elements of (connectedGroups) which equal
    //  (nConnectedGroups) and the elements of (connectedToSeed)
    connectedToSeed[posSeed] = true;
    for (int i = 0; i < iOrder; ++i) // i is the column
      // The matrix is not assumed to be symmetrised
      if ((posSeed < i) ? iMatrix[posSeed][i] : iMatrix[i][posSeed]) connectedToSeed[i] = true;
    bool changed;
    do {
      // Cleaning of the (difference)
      for (int i = 0; i < iOrder; ++i) difference[i] = false;
      // Computation of the (difference) and update of the list of the connected vertices
      for (int i = 0; i < iOrder; ++i)
        if (connectedToSeed[i] && (connectedGroups[i] != nConnectedGroups))
          difference[i] = true, connectedGroups[i] = nConnectedGroups;
      // Visit of the first neighbours of the vertices belonging to the (difference)
      for (int i = 0; i < iOrder; ++i)
        if (difference[i])
          for (int j = 0; j < iOrder; ++j) // j is the column
            // The matrix is not assumed to be symmetrised
            if ((i < j) ? iMatrix[i][j] : iMatrix[j][i]) connectedToSeed[j] = true;
      changed = false;
      for (int i = 0; i < iOrder; ++i) changed |= difference[i];
    } while (changed);
    while (connectedGroups[posSeed])
      if ((++posSeed) == iOrder) break;
  }
  return nConnectedGroups;
}

class handledPermutatedTopology {
  int order;
  int **matrix;
  int *tadpoles;
  int nLines;
  // The initial "colour" of the vertices is given by the two corresponding values in the
  //  (topologicList) and in the (partitionList)
  int *topologicList;
  int *partitionList;
  // It seems to me that these are not needed here, and that they could instead be replaced directly
  //  by the colours (to which however the tadpoles contribute too)
  int *colors;
  smartPermutation *permutation;
  // It tells whether the permutation currently stored is the one that canonicalises the topology.
  //  It is raised only after the execution of (getCanonical), and it is lowered as soon as a new
  //  permutation has been selected
  bool bCanonized;
  // The problem is: in my approach, must the symmetry factor take the tadpoles into account? I
  //  would say so...
  long int symmetryFactor;
  bool bConnected;
  int nLoops;
  bool bOPI;
public:
  // The symmetrisation is _not_ contained in the constructor... OR IS IT?
  handledPermutatedTopology(int iOrder, int **iMatrix, int *iTadpoles = 0, int *iTopologicList = 0,
      int *iPartitionList = 0) {
    this->order = iOrder;
    this->matrix = new int *[iOrder];
    for (int i = 0; i < iOrder; ++i) {
      this->matrix[i] = new int[iOrder];
      for (int j = 0; j < iOrder; ++j) this->matrix[i][j] = (j > i) ? iMatrix[i][j] : iMatrix[j][i];
    }
    if (iTadpoles) {
      this->tadpoles = new int[iOrder];
      for (int i = 0; i < iOrder; ++i) this->tadpoles[i] = iTadpoles[i];
    } else this->tadpoles = 0;
    nLines = 0;
    for (int i = 0; i < iOrder; ++i)
      for (int j = i + 1; j < iOrder; ++j)
        if (iMatrix[i][j]) nLines += iMatrix[i][j];
    this->topologicList = (int *)calloc(iOrder, sizeof(int));
    if (iTopologicList)
      for (int i = 0; i < iOrder; ++i) this->topologicList[i] = iTopologicList[i];
    else
      for (int i = 0; i < iOrder; ++i)
        for (int j = 0; j < iOrder; ++j)
          this->topologicList[i] += ((i < j) ? (iMatrix[i][j]) : (iMatrix[j][i]));
    this->partitionList = new int[iOrder];
    if (iPartitionList)
      for (int i = 0; i < iOrder; ++i) this->partitionList[i] = iPartitionList[i];
    else {
      // Here symmetrising is called for
      for (int i = 0; i < iOrder; ++i) {
        for (int j = 0; j < i; ++j) this->matrix[i][j] = iMatrix[j][i];
        this->partitionList[i] = findPartition(this->topologicList[i], iOrder, this->matrix[i]);
      }
    }
    this->permutation = new smartPermutation(iOrder, this->matrix);
    this->bCanonized = false;
    // Initial colouring
    this->colors = new int[iOrder];
    // Computation of the symmetry factor, trivial for now
    this->symmetryFactor = 0;
    // Check whether the topology is connected
    int nConnectedGroups = numOfConnectedGroups(iMatrix, iOrder);
    this->bConnected = (nConnectedGroups == 1);
    // Recall that the number of the loops is given by (l)-(p)+(n)+(t), where:
    //  -(l) is the number of the lines (nLines) of the graph,
    //  -(p) is the number of the vertices (order) of the graph,
    //  -(n) is the number of the connected components of the graph,
    //  -(t) is the number of the tadpoles of the graph.
    // Computation of the number of the loops
    this->nLoops = this->nLines - iOrder + nConnectedGroups;
    if (iTadpoles)
      for (int i = 0; i < iOrder; ++i) this->nLoops += iTadpoles[i];
    // Check whether the topology is OPI.
    // (31-5-2001) To see how many connected components there are in a graph it suffices to take the
    //  upper/lower triangle of the topology, to replace the links of every row with the same number
    //  and to regard every row as a vector of colours.
    //  At that point, by summing the colours corresponding to the various rows it is immediate to
    //  find what one is looking for, provided one _takes_an_or_ of the colours.
    //  Unfortunately, as things stand the class (color) only implements the and of the colours,
    //  which makes it unusable in this case.
    // So [for now?] the approach of replacing every 1 in turn with a 0 and recomputing the number
    //  of the connected components is used.
    // As the definition of OPR (one-particle reducible) one takes "a diagram whose number of
    //  connected components increases by 1 if a suitable single propagator is cut"
    {
      // Just not to make mistakes, for now another temporary copy is made
      int **tmpMatrix = new int *[iOrder];
      for (int i = 0; i < iOrder; ++i) {
        tmpMatrix[i] = new int[iOrder];
        for (int j = 0; j < iOrder; ++j) tmpMatrix[i][j] = (j > i) ? iMatrix[i][j] : iMatrix[j][i];
      }
      this->bOPI = true;
      // The _upper_ triangle of the topology matrix is scanned
      for (int i = 0; i < iOrder; ++i) {
        for (int j = i + 1; j < iOrder; ++j)
          if (tmpMatrix[i][j] == 1) {
            // The propagator is cut
            tmpMatrix[i][j] = 0, tmpMatrix[j][i] = 0;
            if (nConnectedGroups != numOfConnectedGroups(tmpMatrix, iOrder)) {
              this->bOPI = false;
              break;
            }
            // The propagator is reconnected, restoring the topology
            tmpMatrix[i][j] = 1, tmpMatrix[j][i] = 1;
          }
        if (this->bOPI == false) break;
      }
      for (int i = 0; i < iOrder; ++i) delete [] tmpMatrix[i];
      delete [] tmpMatrix;
    }
  }
  ~handledPermutatedTopology() {
    delete this->permutation; // Before deleting (this->matrix)...
    for (int i = 0; i < this->order; ++i) delete [] this->matrix[i];
    delete [] this->matrix;
    // Recall that both (tadpoles) and (topologicList) are possibly copied from the initial
    //  arguments
    if (this->tadpoles) delete [] this->tadpoles;
    free(this->topologicList);
    delete [] this->partitionList;
    delete [] this->colors;
  }
  // Here go the member functions (the ones which do not take topologies as arguments)
  int rOrder(void) { return this->order; }
  // Direct reading of the matrix is prevented: the topology exists only in terms of the permutation
  //  associated with it
  int rMatrix(int row, int col) {
    return ((row < col)
        ? this->permutation
            ->rPermutatedObject((row * (2 * this->order - row - 1)) / 2 + col - row - 1)
        : this->permutation
            ->rPermutatedObject((col * (2 * this->order - col - 1)) / 2 + row - col - 1));
  }
  // Self-loop count of vertex (num) in the original labelling
  int rawTad(int num) { return ((this->tadpoles) ? this->tadpoles[num] : 0); }
  // Direct reading of the tadpoles is prevented: the permuted vector is read directly
  int rTadpoles(int num) {
    return ((this->tadpoles) ? this->tadpoles[this->permutation->rCurrentPermutation()[num]] : 0);
  }
  int rNLines(void) { return this->nLines; }
  long int rSymmetryFactor(void) { return this->symmetryFactor; }
  int rNLoops(void) { return this->nLoops; }
  bool isConnected(void) { return this->bConnected; }
  bool isOPI(void) { return this->bOPI; }
  smartPermutation *rPermutation(void) { return this->permutation; }
  smartPermutation *setAndGetPermutation(int *iPermutation) {
    this->bCanonized = false; // As a precaution...
    return this->permutation->setAndGetPermutation(iPermutation);
  }
  smartPermutation *setAndGetPermutationIdentity(void) {
    this->bCanonized = false; // As a precaution...
    return this->permutation->setAndGetPermutationIdentity();
  }
  smartPermutation *getCanonical(void) {
    if (this->bCanonized) return this->permutation;
    // Creation and initialisation of the vectors of the indices and of the colours
    int *indices = new int[this->order];
    int *vertexColors = (int *)calloc(this->order, sizeof(int));
    int maxCol = 0;
    for (int i = 0; i < this->order; ++i) {
      for (int j = 1; j < this->topologicList[i]; ++j)
        vertexColors[i] += howManyPartitions(j, this->order);
      vertexColors[i] += this->partitionList[i];
      maxCol = (vertexColors[i] > maxCol) ? vertexColors[i] : maxCol;
    }
    { // The self-loop count is the most significant colour (without it two graphs
      //  differing only in where their self-loops sit could be taken for one)
      int base = maxCol + 1;
      for (int i = 0; i < this->order; ++i) {
        vertexColors[i] += this->rawTad(i) * base;
        maxCol = (vertexColors[i] > maxCol) ? vertexColors[i] : maxCol;
      }
    }
    // Initialisation of the indices and recolouring ---_to be improved_---
    int curr = 0;
    int *newVertexColors = new int[this->order];
    newVertexColors[0] = 0;
    bool first;
    for (int i = 0; i < maxCol + 1; ++i) {
      first = true;
      for (int j = 0; j < this->order; ++j)
        if (vertexColors[j] == i) {
          indices[curr] = j;
          if (curr)
            newVertexColors[curr] = (first) ? newVertexColors[curr - 1] + 1
                : newVertexColors[curr - 1];
          first = false;
          ++curr;
        }
    }
    for (int i = 0; i < this->order; ++i) vertexColors[i] = newVertexColors[i];
    delete [] newVertexColors;
    smartPermutation perm(this->order, this->matrix);
    permutator P(indices, indices + this->order, vertexColors);
    // If the indices differ from the identity the colours are not canonical: the current
    //  permutation must therefore not be regarded as valid; moreover, one _must_ take into account
    //  the first (identical) permutation of (indices)
    bool validCurrent = true;
    for (int i = 0; i < this->order; ++i)
      if (indices[i] != i) {
        validCurrent = false;
        break;
      }
    if (!validCurrent || (*(perm.setAndGetPermutation(indices)) > *(this->permutation)))
      this->setAndGetPermutation(indices);
    while (P.nextPermutation())
      // If the topology has not been canonicalised the initial permutation is present in
      //  (this->permutation)
      if (*(perm.setAndGetPermutation(indices)) > *(this->permutation))
#ifdef SMART_PERMUTATION_DEBUG
      {
        std::cout << "[ ";
        for (int i = 0; i < this->order; ++i) std::cout << indices[i] << " ";
        std::cout << "]:   ";
        std::cout << (perm) << " >" << std::endl << "[ ";
        for (int i = 0; i < this->order; ++i)
          std::cout << this->permutation->rCurrentPermutation()[i] << " ";
        std::cout << "]:   " << (*(this->permutation)) << " ?"
            << ((perm > (*(this->permutation))) ? "true" : "false") << std::endl << std::endl;
#endif
        this->setAndGetPermutation(indices);
#ifdef SMART_PERMUTATION_DEBUG
      }
#endif
    delete [] indices;
    free(vertexColors);
    this->bCanonized = true;
    return this->permutation;
  }
};

class topology : public handle<handledPermutatedTopology> {
public:
  topology(int iOrder, int **iMatrix, int *iTadpoles = 0, int *iTopologicList = 0,
      int *iPartitionList = 0)
      : handle<handledPermutatedTopology>(new handledPermutatedTopology(iOrder, iMatrix, iTadpoles,
        iTopologicList, iPartitionList)) { }
  topology(handledPermutatedTopology *pp) : handle<handledPermutatedTopology>(pp) { }
  // Here go all the friend functions which take at least one topology as an argument.
  // Isomorphism test
  friend bool operator==(topology a, topology b) {
    // The comparison is made between two smartPermutations; it holds for the tadpoles too, which
    //  are part of the initial colouring, so that the canonical form has them in order
    int ord = a->rOrder();
    if (b->rOrder() != ord) return false;
    a->getCanonical(), b->getCanonical();
    for (int i = 0; i < ord; ++i)
      if (a->rTadpoles(i) != b->rTadpoles(i)) return false;
    if (*(a->rPermutation()) == *(b->rPermutation())) return true;
    return false;
  }
  // (<) and (>) serve for the ordering inside the set or the map
  friend bool operator<(topology a, topology b) {
    // Here one has to compare the (permutation)s, but also the tadpoles; moreover, since in general
    //  the topologies can also be isomorphic, one has in general to compare their canonical forms:
    //  _SIDE_EFFECTS_?
    int ord = a->rOrder();
    if (b->rOrder() < ord) return false;
    if (b->rOrder() > ord) return true;
    // Having got here, the orders are equal: ambiguous case.
    // The canonicalisation must be done here, so as to allow the tadpoles to be compared in order
    a->getCanonical(), b->getCanonical();
    for (int i = 0; i < ord; ++i) {
      if (b->rTadpoles(i) > a->rTadpoles(i)) return false;
      if (b->rTadpoles(i) < a->rTadpoles(i)) return true;
    }
    // Having got here, the tadpoles are equal: ambiguous case
    if (*(a->rPermutation()) < *(b->rPermutation())) return true;
    return false;
  }
  friend bool operator>(topology a, topology b) {
    // Here one has to compare the (permutation)s, but also the tadpoles; moreover, since in general
    //  the topologies can also be isomorphic, one has in general to compare their canonical forms:
    //  _SIDE_EFFECTS_?
    int ord = a->rOrder();
    if (b->rOrder() > ord) return false;
    if (b->rOrder() < ord) return true;
    // Having got here, the orders are equal: ambiguous case.
    // The canonicalisation must be done here, so as to allow the tadpoles to be compared in order
    a->getCanonical(), b->getCanonical();
    for (int i = 0; i < ord; ++i) {
      if (b->rTadpoles(i) < a->rTadpoles(i)) return false;
      if (b->rTadpoles(i) > a->rTadpoles(i)) return true;
    }
    // Having got here, the tadpoles are equal: ambiguous case
    if (*(a->rPermutation()) > *(b->rPermutation())) return true;
    return false;
  }
  friend std::ostream &operator<<(std::ostream &s, topology a) {
    if (&s == &std::cout) {
      for (int i = 0; i < a->rOrder(); ++i) {
        for (int j = 0; j < a->rOrder(); ++j)
          if (j == i)
            // The case of the tadpoles!
            s << (2 * a->rTadpoles(i)) << " ";
          else s << a->rMatrix(i, j) << " ";
        s << std::endl;
      }
      s << "Symmetry factor=" << a->rSymmetryFactor() << ", loops=" << (a->rNLoops())
          << (a->isConnected() ? ", connected, " : ", disconnected, ")
          << (a->isOPI() ? "OPI." : "non-OPI.") << std::endl;
    } else {
      s << a->rOrder() << "(";
      // Tadpoles, if any
      int i;
      for (i = 0; i < a->rOrder(); ++i)
        if (a->rTadpoles(i)) break;
      if (i < a->rOrder()) {
        s << "[";
        // Since there is at least one non-null tadpole, the order is necessarily non-null.
        // nCons accumulates the number of the consecutive equal tadpole numbers
        int current = a->rTadpoles(0), nCons = 1;
        for (i = 1; i < a->rOrder() + 1; ++i)
          if (i == a->rOrder() || a->rTadpoles(i) != current) {
            if (current) {
              s << current;
              if (nCons > 1) s << "^" << nCons;
            } else s << "." << nCons;
            if (i != a->rOrder()) {
              current = a->rTadpoles(i);
              if (current) s << " ";
              nCons = 1;
            }
          } else ++nCons;
        s << "]";
      }
      s << *(a->rPermutation()) << ")" << std::endl;
    }
    return s;
  }
};

#endif
