#ifndef EGBS_H
#define EGBS_H

#include <opencv2/opencv.hpp>
#include <unordered_map>
#include "graph.h"
#include "DisjointSet.h"

struct Component {
    int count;
    cv::Vec3f color_sum;

    struct Component &operator+=(const Component &rhs) {
        count += rhs.count;
        color_sum += rhs.color_sum;
        return *this;
    }

    cv::Vec3f getAverageColor() {
        return color_sum / count;
    }
};

class EGBS {
public:
    EGBS();

    void applySegmentation(cv::Mat &image, int c, int min_size);

private:
    void recolor(cv::Mat &image, DisjointSet &forest);
};

#endif // EGBS_H
