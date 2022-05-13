//*********Models.h*************
#pragma once
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;
using namespace concurrency;
 
struct Grad
{
	float x;
	float y;
	float g_rec;
};
 
 
Mat rotateImg(Mat src, double degree)
{
	//计算旋转后的图像的宽高
	double angle = degree  * CV_PI / 180.; // 弧度
	double a = sin(angle), b = cos(angle);
	int width = src.cols;
	int height = src.rows;
	int width_rotate = int(height * fabs(a) + width * fabs(b));
	int height_rotate = int(width * fabs(a) + height * fabs(b));
 
	//定义旋转中心，并根据旋转角度计算旋转矩阵
	CvPoint2D32f center = cvPoint2D32f(width / 2, height / 2);
	Mat r = getRotationMatrix2D(center, degree, 1);
 
	//加入旋转中心的偏移
	r.at<double>(0, 2) += (width_rotate - width) / 2;
	r.at<double>(1, 2) += (height_rotate - height) / 2;
 
	//利用反射变换得到旋转后的图像
	Mat src_rotate = Mat::zeros(height_rotate, width_rotate, src.type());
	warpAffine(src, src_rotate, r, Size(width_rotate, height_rotate), CV_INTER_LINEAR | CV_WARP_FILL_OUTLIERS);
	return src_rotate;
}
 
struct Models
{
	int numLevels;
	vector<float> angles;
 
	vector<Point> centers;
	vector<vector<Point>> positions;
	vector<vector<Grad>> grads;
	vector<int> pxCounts;
};
 
 
void createModels(Mat src, float angleStart, float angleEnd, float angleStep, int lowThresh, int highThresh, int numLevels, Models& models)
{
 
	//首先利用ppl多线程计算旋转模板
	int angleNum = (angleEnd - angleStart) / angleStep;
	int ksize = 2 * MIN(src.rows, src.cols) / 100 + 1;
	vector<Mat> edgeImages(angleNum);
	vector<Mat> gx(angleNum);
	vector<Mat> gy(angleNum);
	vector<Mat> g(angleNum);
	models.angles = vector<float>(angleNum);
	mutex mt;
	parallel_for(0, angleNum, [&](int t)
	{
		Mat blurImage, edgeImage;
		float angle = angleStart + t * angleStep;
		models.angles[t] = angle;
		blur(src, blurImage, Size(ksize, ksize));
		Canny(blurImage, edgeImage, lowThresh, highThresh);
		Mat rotateEdgeImage = rotateImg(edgeImage, angle);
		Mat rotateBlurImage = rotateImg(blurImage, angle);
 
		Mat _gx, _gy;
		threshold(rotateEdgeImage, rotateEdgeImage, 10, 255, THRESH_BINARY);
		edgeImages[t] = rotateEdgeImage;
		Sobel(rotateBlurImage, _gx, CV_8U, 1, 0, 3);
		Sobel(rotateBlurImage, _gy, CV_8U, 0, 1, 3);
		_gx = _gx & rotateEdgeImage;
		_gy = _gy & rotateEdgeImage;
		gx[t] = _gx;
		gy[t] = _gy;
	});
 
	//然后对所有旋转后的模板进行下采样
	models.numLevels = numLevels;
	int totalNum = numLevels * models.angles.size();
	models.pxCounts = vector<int>(totalNum);
	models.centers = vector<Point>(totalNum);
	models.grads = vector<vector<Grad>>(totalNum);
	models.positions = vector<vector<Point>>(totalNum);
 
	parallel_for(0, totalNum, [&](int k)
	{
		
		int pryLevel = k / models.angles.size();
		int rotateIndex =  k % models.angles.size();
		Mat edgeImage = edgeImages[rotateIndex];
		Mat _gx = gx[rotateIndex];
		Mat _gy = gy[rotateIndex];
		for (size_t i = pryLevel; i > 0; i--)
		{
			pyrDown(_gx, _gx);
			pyrDown(_gy, _gy);
			pyrDown(edgeImage, edgeImage);
		}
		threshold(edgeImage, edgeImage, 10, 255, THRESH_BINARY);
		//统计每个旋转模板的非零点个数，用于后面计算匹配度
		models.pxCounts[k] = countNonZero(edgeImage);
		models.centers[k] = Point(edgeImage.cols / 2, edgeImage.rows / 2);
		int length = edgeImage.rows* edgeImage.cols;
		Mat g(edgeImage.rows, edgeImage.cols, CV_8UC1);
		vector<Point> pos;
		vector<Grad> grads;
		Grad grad = { 0 };
		for (int i = 0; i < length; i++)
		{
			int y = i / edgeImage.cols;
			int x = i % edgeImage.cols;
			int igx = _gx.at<uchar>(y, x);
			int igy = _gy.at<uchar>(y, x);
			if (igx != 0 || igy != 0)
			{
				int ig = sqrt(float(igx*igx) + float(igy*igy));
				g.at<uchar>(y, x) = ig;
				pos.push_back(Point(x, y));
				if (ig==0)
					grad.g_rec = 0;
				else
					grad.g_rec = 1./ig;
				grad.x = igx;
				grad.y = igy;
				grads.push_back(grad);
			}
			else continue;
		}
		models.positions[k] = pos;
		models.grads[k] = grads;
	});
}
 
 
 
void calcGradientImage(Mat src, int ksize, int lowThresh, int highThresh,int numLevels, Mat& gx, Mat& gy, Mat& g_rec)
{
	blur(src, src, Size(ksize, ksize));//模糊图像，降低噪点干扰
	Mat edge;
	Canny(src, edge, lowThresh, highThresh);//利用canny提取边缘
	Sobel(src, gx, CV_8U, 1, 0, 3); //计算X方向梯度
	Sobel(src, gy, CV_8U, 0, 1, 3); //计算y方向梯度
	gx = gx & edge;
	gy = gy & edge;
 
	for (size_t i = 0; i < numLevels; i++)
	{
		pyrDown(edge, edge);
		pyrDown(gx, gx);
		pyrDown(gy, gy);
	}
 
	int rows = edge.rows;
	int cols = edge.cols;
	int cn = edge.channels();
	int length = rows * cols * cn;
	// 计算梯度大小
	Mat g = Mat::zeros(edge.size(), CV_8U);
	g_rec = Mat::zeros(edge.size(), CV_32F);
	for (int i = 0; i < length; i++)
	{
		int y = i / cols;
		int x = i % cols;
		uchar _gx = gx.at<uchar>(y, x);
		uchar _gy = gy.at<uchar>(y, x);
		if (_gx != 0 || _gy != 0)
		{
			uchar _g = sqrt(uchar(_gx*_gx) + uchar(_gy*_gy));
			if (_g==0)
				g_rec.at<float>(y, x) = 0;
			else
				g_rec.at<float>(y, x) = 1./_g;
		}
	}
}
 