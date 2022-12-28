#ifndef DISJOINTSET_H
#define DISJOINTSET_H

#include<vector>

typedef struct {
    int parent;
    int rank;
    int size;
} Node;

typedef std::vector <Node> Forest;

/**
 * Class representing Disjoint-set data structure (forest implementation).
 * Read more at https://en.wikipedia.org/wiki/Disjoint-set_data_structure
 */
class DisjointSet {
public:
    DisjointSet();

    DisjointSet(int num_elements);

    int find(int x);

    void merge(int x, int y);

    int size(int x) const;

private:
    Forest elements;
};

#endif // DISJOINTSET_H
