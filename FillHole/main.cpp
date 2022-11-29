#include <opencv2/opencv.hpp> 
#include <iostream>

using namespace cv;
using namespace std;

void FillHole(Mat& src, bool fillBackGround = false)
{
	assert(src.channels() == 1);
	int width = src.cols, height = src.rows;
	int color = fillBackGround == false ? 255 : 0;

	for (int y = 0; y < height; y++)
	{
		if (src.at<uchar>(y, 0) == color) 
			floodFill(src, cv::Point(0, y), Scalar(127));
		if (src.at<uchar>(y, width - 1) == color) 
			floodFill(src, cv::Point(width - 1, y), Scalar(127));
	}

	for (int x = 0; x < width; x++)
	{
		if (src.at<uchar>(0, x) == color)
			floodFill(src, cv::Point(x, 0), Scalar(127));
		if (src.at<uchar>(height - 1, x) == color)
			floodFill(src, cv::Point(x, height - 1), Scalar(127));
	}

	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			if (src.at<uchar>(y, x) == 127)
				src.at<uchar>(y, x) = color;
			else
				src.at<uchar>(y, x) = 255 - color;
		}
	}
}


int main(void)
{
	string testImage = "test.png";
	Mat1b src = imread(testImage, 0);

	if (src.empty()){
		cout << "The specified image '" << testImage << "' does not exists" << endl;
		exit(-1);
	}

	imshow("Origin", src);

	FillHole(src);

	imshow("Result", src);

	waitKey(0);

	return 0;
}