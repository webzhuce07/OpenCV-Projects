// CoherenceFilter.cpp : 定义控制台应用程序的入口点。
//
 
#include <vector>
#include <opencv2/opencv.hpp>
 
using namespace std;
using namespace cv;
 
/* ==============================================
*   Coherence-Enhancing Shock Filters
*  Author:WinCoder@qq.com
*  inspired by
*  Joachim Weickert "Coherence-Enhancing Shock Filters"
*  http://www.mia.uni-saarland.de/Publications/weickert-dagm03.pdf
*
*   Paras:
*   @img        : input image ranging value from 0 to 255.
*   @sigma      : sobel kernel size.
*   @str_sigma  : neighborhood size,see detail in reference[2]
*   @belnd      : blending coefficient.default value 0.5.
*   @iter       : number of iteration.
*
*   Example:
*   Mat dst = CoherenceFilter(I,11,11,0.5,4);
*   imshow("shock filter",dst);
*/
 
Mat CoherenceFilter(Mat img, int sigma, int str_sigma, float blend, int iter)
{
	Mat I = img.clone();
	int height = I.rows;
	int width = I.cols;
 
	for (int i = 0; i <iter; i++)
	{
		Mat gray;
		cvtColor(I, gray, COLOR_BGR2GRAY);
		// 计算特征值和特征向量
		Mat eigen;
		cornerEigenValsAndVecs(gray, eigen, str_sigma, 3);
 
		vector<Mat> vec;
		split(eigen, vec);// vec[0]: λ1, vec[1]: λ2, vec[2]: x1, vec[3]: y1, vec[4]: x2,  vec[5]: y2
 
		// 主特征向量 w
		Mat x, y;
		x = vec[2];	// c
		y = vec[3];	// s
 
		// Sobel近似求解二阶导数
		Mat gxx, gxy, gyy;
		Sobel(gray, gxx, CV_32F, 2, 0, sigma);
		Sobel(gray, gxy, CV_32F, 1, 1, sigma);
		Sobel(gray, gyy, CV_32F, 0, 2, sigma);
 
		Mat ero;
		Mat dil;
		erode(I, ero, Mat());	// 最小值附近进行腐蚀
		dilate(I, dil, Mat());	// 最大值附近进行膨胀
 
		Mat img1 = ero;
		for (int nY = 0; nY<height; nY++)
		{
			for (int nX = 0; nX<width; nX++)
			{
				// 边界检测子v(ww)
				if (x.at<float>(nY, nX)* x.at<float>(nY, nX)* gxx.at<float>(nY, nX)
					+ 2 * x.at<float>(nY, nX)* y.at<float>(nY, nX)* gxy.at<float>(nY, nX)
					+ y.at<float>(nY, nX)* y.at<float>(nY, nX)* gyy.at<float>(nY, nX)<0)
				{
					img1.at<Vec3b>(nY, nX) = dil.at<Vec3b>(nY, nX);	// 最大值的影响区域 dil
				}
				// 否则是最小值的影响区域 ero
			}
		}
		// 和原图按比例混合
		I = I*(1.0 - blend) + img1*blend;
	}
	return I;
}
 
int main()
{
	Mat img = imread("1.png", 1);
	Mat result = CoherenceFilter(img, 11, 11, 0.5, 2);
	imshow("Source", img);
	imshow("Result", result);
	waitKey(0);
 
    return 0;
}