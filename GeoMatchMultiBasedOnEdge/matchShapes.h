#pragma once
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

vector<Point> ImageTemplateContours(Mat img_template)
{
	//灰度化
	Mat gray_img_template;
	cvtColor(img_template, gray_img_template, COLOR_BGR2GRAY);

	//阈值分割
	Mat thresh_img_template;
	threshold(gray_img_template, thresh_img_template, 0, 255, THRESH_OTSU);
	//膨胀处理
	Mat ellipse = getStructuringElement(MORPH_ELLIPSE, Size(15, 15));
	Mat erode_img_template;
	//erode(thresh_img_template, erode_img_template, ellipse);
	morphologyEx(thresh_img_template, thresh_img_template, MORPH_OPEN, ellipse, Point(-1, -1), 1);

	//寻找边界
	vector<vector<Point>> contours_template;
	vector<Vec4i> hierarchy;
	findContours(thresh_img_template, contours_template, hierarchy, RETR_LIST, CHAIN_APPROX_NONE, Point());

	//绘制边界
	drawContours(img_template, contours_template, 0, Scalar(0, 0, 255), 1, 8, hierarchy);


	return contours_template[0];
}


vector<Point2d> ShapeTemplateMatch(Mat image, vector<Point> imgTemplatecontours, double minMatchValue)
{
	vector<Point2d> image_coordinates;
	//灰度化
	Mat gray_img;
	cvtColor(image, gray_img, COLOR_BGR2GRAY);

	//阈值分割
	Mat thresh_img;
	threshold(gray_img, thresh_img, 0, 255, THRESH_OTSU);

	//寻找边界
	vector<vector<Point>> contours_img;
	vector<Vec4i> hierarchy;
	findContours(thresh_img, contours_img, hierarchy, RETR_LIST, CHAIN_APPROX_NONE, Point());
	//根据形状模板进行匹配
	int min_pos = -1;
	double	min_value = minMatchValue;//匹配分值，小于该值则匹配成功
	for (int i = 0; i < contours_img.size(); i++)
	{
		//计算轮廓面积，筛选掉一些没必要的小轮廓
		if (contourArea(contours_img[i]) > 12)
		{
			//得到匹配分值 
			double value = matchShapes(contours_img[i], imgTemplatecontours, CONTOURS_MATCH_I3, 0.0);
			//将匹配分值与设定分值进行比较 
			if (value < min_value)
			{
				min_pos = i;
				//绘制目标边界
				//drawContours(image, contours_img, min_pos, Scalar(0, 0, 255), 1, 8, hierarchy, 0);

				//获取重心点
				Moments M;
				M = moments(contours_img[min_pos]);
				double cX = double(M.m10 / M.m00);
				double cY = double(M.m01 / M.m00);
				//显示目标中心并提取坐标点
				//circle(image, Point2d(cX, cY), 1, Scalar(0, 255, 0), 2, 8);
				//putText(image, "center", Point2d(cX - 20, cY - 20), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 1, 8);
							//将目标的重心坐标都存在数组中 
				image_coordinates.push_back(Point2d(cX, cY));//向数组中存放点的坐标
			}
		}
	}
	return image_coordinates;
}

