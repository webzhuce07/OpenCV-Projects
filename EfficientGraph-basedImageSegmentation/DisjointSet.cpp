#include "DisjointSet.h"

DisjointSet::DisjointSet() {

}

DisjointSet::DisjointSet(int num_elements) {
    elements.clear();
    elements.resize(num_elements);

    for (int i = 0; i < num_elements; i++) {
        elements[i].parent = i;
        elements[i].rank = 1;
        elements[i].size = 1;
    }
}

int DisjointSet::find(int x) {
    int y = x;

    while (y != elements[y].parent) {
        y = elements[y].parent;
    }

    elements[x].parent = y;

    return y;
}

void DisjointSet::merge(int x, int y) {
    x = find(x);
    y = find(y);

    if (x != y) {
        if (elements[x].rank < elements[y].rank) {
            elements[x].parent = y;
            elements[y].size += elements[x].size;
        } else {
            elements[y].parent = x;
            elements[x].size += elements[y].size;

            if (elements[x].rank == elements[y].rank) {
                elements[x].rank++;
            }
        }
    }
}

int DisjointSet::size(int x) const {
    return elements[x].size;
}
