#include <iostream>
#include "EGBS.h"

using namespace std;

int main(int argc, const char *argv[]) {
    string imageName("1.jpg");

    if (argc > 1) {
        imageName = argv[1];
    }

    cv::Mat image = cv::imread(imageName.c_str());

    if (image.empty()) {
        cout << "Could not find or open the image" << endl;
        return -1;
    }

    EGBS egbs = EGBS();
    egbs.applySegmentation(image, 1500, 20);

    cv::imshow("Modified image", image);

    // Wait for a keystroke in the window (otherwise the program would end far too quickly)
    cv::waitKey(0);

    return 0;
}
