#include<opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main()
{
	Mat image = imread("../1.bmp");
	Mat gray;
	cvtColor(image, gray, COLOR_BGR2GRAY);
	Mat mean;
	medianBlur(gray, mean, 201);
	
	Mat diff;
	addWeighted(gray, -1, mean, 1, 0, diff);

	Mat thresh_low, thresh_high;
	threshold(diff, thresh_low, 41, 255, THRESH_BINARY);

	Mat imgOpen;
	Mat element = getStructuringElement(MORPH_ELLIPSE, Size(7, 7), Point(-1, -1));
	morphologyEx(thresh_low, imgOpen, MORPH_OPEN, element);

	Mat imgClose;
	Mat element1 = getStructuringElement(MORPH_ELLIPSE, Size(7, 7), Point(-1, -1));
	morphologyEx(imgOpen, imgClose, MORPH_CLOSE, element1);

	vector<vector<Point>>contours;
	vector<vector<Point>>contoursFinal;
	vector<Vec4i>hierachy;
	vector<Rect>rects;
	findContours(imgClose, contours, hierachy, RETR_EXTERNAL, CHAIN_APPROX_NONE);
	for (size_t t = 0; t < contours.size(); t++)
	{
		Rect rect = boundingRect(contours[t]);
		double area = contourArea(contours[t]);
		if(area < 1000)
			continue;
		rects.push_back(rect);
		contoursFinal.push_back(contours[t]);
		drawContours(image, contoursFinal, -1, Scalar(0, 0, 255), 2);
	}

	imshow("¸´ÔÓ±³¾°µÄÈ±ÏÝ¼ì²â", image);
	waitKey(0);
	printf("È±ÏÝ¸öÊý:%d\n", rects.size());
	printf("ÂÖÀª¸öÊý£º%d\n", contours.size());

	return 0;
}