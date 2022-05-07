#include<opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

Mat Rotate(Mat src, double angle, Point2f center, double scale = 1.0)
{
	Mat wrapMat =getRotationMatrix2D(center, angle, scale);
	Mat dst;
	warpAffine(src, dst, wrapMat, src.size(), INTER_CUBIC, BORDER_CONSTANT, Scalar(255, 255, 255));
	return dst;
}

double GetAngle(vector<Vec4i> lines)
{
	vector<double> angles;
	for each (auto line in lines)
	{
		double k = (line[1] - line[3]) * 1.0 / (line[0] - line[2]);
		k = atan(k);
		k = k * 180 / CV_PI;
		angles.push_back(k);
	}

	//��ֵ
	sort(angles.begin(), angles.end());
	double angle = 0;
	if (angles.size() % 2 != 0)
		angle = angles[angles.size() / 2];
	else 
		angle = (angles[angles.size() / 2] + angles[angles.size() / 2 - 1]) / 2;
	return angle;
}

//�����ɫ����
Mat ClearBackGround(Mat src) 
{
	int height = src.rows, width = src.cols;
	//ȥ����ɫ������seedPoint�����ʼ���ӣ������ĴΣ������ĸ��Ƕ���һ�Σ���ȥ������Χ�ĺڱ�
	floodFill(src, Point(0, 0), Scalar(255, 255, 255));
	floodFill(src, Point(0, height - 1), Scalar(255, 255, 255));
	floodFill(src, Point(width - 1, height - 1), Scalar(255, 255, 255));
	floodFill(src, Point(width - 1, 0), Scalar(255, 255, 255));

	return src;
}
	

Mat Correct2(Mat src)
{
	Mat gray = src;
	//�ҶȻ�
	if(src.channels() == 3)
		cvtColor(src, gray, COLOR_BGR2GRAY);
	//��ʴ������
	Mat element = getStructuringElement(MORPH_ELLIPSE, Size(5, 5), Point(-1, -1));
	erode(gray, gray, element);
	dilate(gray, gray, element);
	//��Ե���
	Mat canny;
	Canny(gray, canny, 50, 150);
	//����任�õ�����
	vector<Vec4i> lines;
	HoughLinesP(canny, lines, 0.8, CV_PI / 180, 90, 100, 10);

	//�Ƕ�
	double angle = GetAngle(lines);

	Mat dst = Rotate(src, angle, Point2f(src.cols / 2.0f, src.rows / 2.0f));
	//dst = ClearBackGround(dst);
	return dst;
}

int main()
{
	Mat image = imread("../640.png");
	Mat rotateImg = Correct2(image);
	imshow("rotateImg", rotateImg);
	waitKey(0);

	return 0;
}