#include <opencv2/opencv.hpp> 
#include "time.h"  
using namespace std;
using namespace cv;

#define rnd( low,uper) ((int)(((double)rand()/(double)RAND_MAX)*((double)(uper)-(double)(low))+(double)(low)+0.5))  
/*************************************************************8888
粒子群算法变量的说明
******************************************************************************/
const int number = 20;
int antThreshold[number][2];//以阈值作为粒子  
int vect[number][2];//更新的速度  
float pbest[number] = { 0.0 };;//每个粒子历史最优解  
float gbest = 0.0;//全局历史最优解  
int pbestThreshold[number][2];//每个粒子的最优历史阈值  
int gbestThreshold[2];//全局粒子的最优阈值  

float w = 0.9;//惯性因子  
float c1 = 2.0;//加速因子1  
float c2 = 2.0;//加速因子2  

//histogram     
float histogram[256] = { 0 };
/*********************************************************************8888
函数名：GetAvgValue
参数类型：IplImage* src
实现功能：获得灰度图像的总平均灰度值
*****************************************************************************/
float GetAvgValue(Mat& src)
{
	int height = src.rows;
	int width = src.cols;


	for (int i = 0; i < height; i++)
	{
		unsigned char* p = (unsigned char*)src.data + src.step*i;
		for (int j = 0; j < width; j++)
		{
			histogram[*p++]++;
		}
	}
	//normalize histogram     
	int size = height * width;
	for (int i = 0; i < 256; i++) {
		histogram[i] = histogram[i] / size;
	}

	//average pixel value     
	float avgValue = 0;
	for (int i = 0; i < 256; i++) {
		avgValue += i * histogram[i];
	}
	return avgValue;
}
/*****************************************************************************
函数名：ThresholdOTSU
参数类型：int threshold1 , int threshold2 , float avgValue
功能：求得最大类间方差
**********************************************************************************/
float  ThresholdOTSU(int threshold1, int threshold2, float avgValue)
{

	int threshold;
	float maxVariance = 0;
	float w = 0, u = 0;
	for (int i = threshold1; i < threshold2; i++)
	{
		w += histogram[i];
		u += i * histogram[i];
	}

	float t = avgValue * w - u;
	float variance = t * t / (w*(1 - w));
	/* if(variance>maxVariance)
	 {
		 maxVariance=variance;
		 threshold=i;
	 }
	  */
	return variance;
}
/*****************************************************************
函数名：Init
参数类型：void
功能：初始化粒子群算法的粒子与速度
************************************************************************/
void Init()
{
	for (int index = 0; index < number; index++)
	{
		antThreshold[index][0] = rnd(10, 50);
		antThreshold[index][1] = antThreshold[index][0] + 50;
		if (antThreshold[index][1] > 255)
			antThreshold[index][1] = 255;
		vect[index][0] = rnd(3, 5);
		vect[index][1] = rnd(3, 5);
	}

}
/******************************************************************
函数名：Pso
参数类型：void
功能：粒子群算法的实现
***************************************************************************/

void Pso(float value)
{
	for (int index = 0; index < number; index++)
	{
		float variance;
		variance = ThresholdOTSU(antThreshold[index][0], antThreshold[index][1], value);
		if (variance > pbest[index])
		{
			pbest[index] = variance;
			pbestThreshold[index][0] = antThreshold[index][0];
			pbestThreshold[index][1] = antThreshold[index][1];
		}
		if (variance > gbest)
		{
			gbest = variance;
			gbestThreshold[0] = antThreshold[index][0];
			gbestThreshold[1] = antThreshold[index][1];
		}
	}
}
/***************************************************************************************88
函数名：updateData
参数类型：void
功能：更新粒子数据与速度
**************************************************************************************************/
void updateData()
{
	for (int index = 0; index < number; index++)
	{
		for (int i = 0; i < 2; i++)
		{
			vect[index][i] = w * vect[index][i] + c1 * ((double)(rand()) / (double)RAND_MAX)*(pbestThreshold[index][i] - antThreshold[index][i]) +
				c2 * c1*((double)(rand()) / (double)RAND_MAX)*(gbestThreshold[i] - antThreshold[index][i]);
			if (vect[index][i] > 5)
				vect[index][i] = 5;
			if (vect[index][i] < 3)
				vect[index][i] = 3;
			antThreshold[index][i] = vect[index][i] + antThreshold[index][i];
		}
		if (antThreshold[index][0] > antThreshold[index][1])
			antThreshold[index][1] = antThreshold[index][0] + 50;
		if (antThreshold[index][1] > 255)
			antThreshold[index][1] = 255;
		if (antThreshold[index][0] < 0)
			antThreshold[index][0] = 0;
	}

}
/**************************************************************8
函数名：Threshold
参数类型：IplImage *src , int lower , int higher
功能：利用算法得到的双阈值对图像进行阈值分割
***********************************************************************/
void Threshold(Mat& src, int lower, int higher)
{
	assert(src.channels() == 1);
	for (int h = 0; h < src.rows; h++)
		for (int w = 0; w < src.cols; w++)
		{
			if (*(src.data + h * src.step + w * src.elemSize()) < higher&&*(src.data + h * src.step + w * src.elemSize()) > lower)
				*(src.data + h * src.step + w * src.elemSize()) = 255;  
				//;
			else
				*(src.data + h * src.step + w * src.elemSize()) = 0;
		}
}
int main()
{
	srand((unsigned)time(NULL));
	Mat img = imread("1.png", 1);
	
	Mat ycrcb;
	cvtColor(img, ycrcb, COLOR_BGR2YCrCb);
	vector<Mat> color_planes;
	split(ycrcb, color_planes);
	Mat cb;

	medianBlur(color_planes[2], cb, 3);
	float avgValue = 0.0;
	avgValue = GetAvgValue(cb);
	Init();
	for (int i = 0; i < 3000; i++)
	{
		Pso(avgValue);
		updateData();
	}

	//cvThreshold(cb , cb , gbestThreshold[0] , gbestThreshold[1] , CV_THRESH_BINARY);  
	Threshold(cb, gbestThreshold[0], gbestThreshold[1]);
	printf("%d , %d\n", gbestThreshold[0], gbestThreshold[1]);
	imshow("cb", cb);
	imwrite("cb1.jpg", cb);
	waitKey(0);

	return 0;
}