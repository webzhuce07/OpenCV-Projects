#ifndef GRAPH_H_INCLUDED
#define GRAPH_H_INCLUDED

#include <opencv2/opencv.hpp>
#include "DisjointSet.h"

struct Edge {
    int a, b;
    float weight;

    bool operator<(const Edge &other) {
        return weight < other.weight;
    }
};

float diff(const cv::Mat &image, int x1, int y1, int x2, int y2);

std::vector <Edge> build_graph(const cv::Mat &image);

DisjointSet segment_graph(int num_vertices, std::vector <Edge> &edges, float c);

#endif // GRAPH_H_INCLUDED
