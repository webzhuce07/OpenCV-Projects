#include <opencv2/opencv.hpp>
#include <iostream>
 
using namespace cv;
using namespace std;
 
Mat ReplaceAndBlend(Mat &frame, Mat &mask);
Mat g_Background;//����

int main(int argc, char** argv) 
{
	// start here...	
	g_Background = imread("../0.jpg");
	VideoCapture capture; //��Ƶץȡ
	capture.open("../test.mp4");
	if (!capture.isOpened()) {
		printf("could not find the video file...\n");
		return -1;
	}
	const char* title = "input video";
	const char* resultWin = "result video";
	namedWindow(title);
	namedWindow(resultWin);
	Mat frame, hsv, mask;
	int count = 0;
		
	while (capture.read(frame)) {//�ж϶�ȡ��Ƶ��֡�Ƿ�ɹ�����ȡ��֡ͼ���Ƹ�frame����
		if (frame.size() != g_Background.size())
			resize(g_Background, g_Background, Size(frame.cols, frame.rows), 0, 0, INTER_LINEAR);


		cvtColor(frame, hsv, COLOR_BGR2HSV);//ת��ΪHSV
		inRange(hsv, Scalar(35, 43, 46), Scalar(155, 255, 255), mask);
		// ��̬ѧ���������и�ʴ��Ȼ���˹����g��˹ģ�����߽粻��
		Mat k = getStructuringElement(MORPH_RECT, Size(3, 3), Point(-1, -1));
		morphologyEx(mask, mask, MORPH_CLOSE, k);
		morphologyEx(mask, mask, MORPH_DILATE, k);
		erode(mask, mask, k);
		GaussianBlur(mask, mask, Size(5, 5), 0, 0);
		//�����滻�ͻ��
		Mat result = ReplaceAndBlend(frame, mask);
		char c = waitKey(1);
		if (c == 27) {
			break;
		}
		imshow(resultWin, result);//��ʾ������video
		imshow(title, frame);//��ʾ����video
	}
 
	waitKey(0);
	return 0;
}
//�����滻�ͻ��
Mat ReplaceAndBlend(Mat &frame, Mat &mask) {
	Mat result = Mat::zeros(frame.size(), frame.type());
	int h = frame.rows;
	int w = frame.cols;
	int dims = frame.channels();
 
	// replace and blend
	int m = 0;
	double wt = 0;
 
	int r = 0, g = 0, b = 0;
	int r1 = 0, g1 = 0, b1 = 0;
	int r2 = 0, g2 = 0, b2 = 0;
 
	for (int row = 0; row < h; row++) {
		uchar* current = frame.ptr<uchar>(row);//��ǰ
		uchar* bgrow = g_Background.ptr<uchar>(row);//����2
		uchar* maskrow = mask.ptr<uchar>(row);//���� ��
		uchar* targetrow = result.ptr<uchar>(row);//Ŀ�� ��
		for (int col = 0; col < w; col++) {
			m = *maskrow++;
			if (m == 255) { // ��ֵΪ����
				*targetrow++ = *bgrow++;
				*targetrow++ = *bgrow++;
				*targetrow++ = *bgrow++;
				current += 3;
 
			} else if(m==0) {// ��ֵΪǰ��
				*targetrow++ = *current++;
				*targetrow++ = *current++;
				*targetrow++ = *current++;
				bgrow += 3;
			} else {
				b1 = *bgrow++;
				g1 = *bgrow++;
				r1 = *bgrow++;
 
				b2 = *current++;
				g2 = *current++;
				r2 = *current++;
 
				// Ȩ��
				wt = m / 255.0;
				
				// ���
				b = b1*wt + b2*(1.0 - wt);
				g = g1*wt + g2*(1.0 - wt);
				r = r1*wt + r2*(1.0 - wt);
 
				*targetrow++ = b;
				*targetrow++ = g;
				*targetrow++ = r;
			}
		}
	}
 
	return result;
}