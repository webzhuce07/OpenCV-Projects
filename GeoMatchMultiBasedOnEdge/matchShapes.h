#pragma once
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

vector<Point> ImageTemplateContours(Mat img_template)
{
	//�ҶȻ�
	Mat gray_img_template;
	cvtColor(img_template, gray_img_template, COLOR_BGR2GRAY);

	//��ֵ�ָ�
	Mat thresh_img_template;
	threshold(gray_img_template, thresh_img_template, 0, 255, THRESH_OTSU);
	//���ʹ���
	Mat ellipse = getStructuringElement(MORPH_ELLIPSE, Size(15, 15));
	Mat erode_img_template;
	//erode(thresh_img_template, erode_img_template, ellipse);
	morphologyEx(thresh_img_template, thresh_img_template, MORPH_OPEN, ellipse, Point(-1, -1), 1);

	//Ѱ�ұ߽�
	vector<vector<Point>> contours_template;
	vector<Vec4i> hierarchy;
	findContours(thresh_img_template, contours_template, hierarchy, RETR_LIST, CHAIN_APPROX_NONE, Point());

	//���Ʊ߽�
	drawContours(img_template, contours_template, 0, Scalar(0, 0, 255), 1, 8, hierarchy);


	return contours_template[0];
}


vector<Point2d> ShapeTemplateMatch(Mat image, vector<Point> imgTemplatecontours, double minMatchValue)
{
	vector<Point2d> image_coordinates;
	//�ҶȻ�
	Mat gray_img;
	cvtColor(image, gray_img, COLOR_BGR2GRAY);

	//��ֵ�ָ�
	Mat thresh_img;
	threshold(gray_img, thresh_img, 0, 255, THRESH_OTSU);

	//Ѱ�ұ߽�
	vector<vector<Point>> contours_img;
	vector<Vec4i> hierarchy;
	findContours(thresh_img, contours_img, hierarchy, RETR_LIST, CHAIN_APPROX_NONE, Point());
	//������״ģ�����ƥ��
	int min_pos = -1;
	double	min_value = minMatchValue;//ƥ���ֵ��С�ڸ�ֵ��ƥ��ɹ�
	for (int i = 0; i < contours_img.size(); i++)
	{
		//�������������ɸѡ��һЩû��Ҫ��С����
		if (contourArea(contours_img[i]) > 12)
		{
			//�õ�ƥ���ֵ 
			double value = matchShapes(contours_img[i], imgTemplatecontours, CONTOURS_MATCH_I3, 0.0);
			//��ƥ���ֵ���趨��ֵ���бȽ� 
			if (value < min_value)
			{
				min_pos = i;
				//����Ŀ��߽�
				//drawContours(image, contours_img, min_pos, Scalar(0, 0, 255), 1, 8, hierarchy, 0);

				//��ȡ���ĵ�
				Moments M;
				M = moments(contours_img[min_pos]);
				double cX = double(M.m10 / M.m00);
				double cY = double(M.m01 / M.m00);
				//��ʾĿ�����Ĳ���ȡ�����
				//circle(image, Point2d(cX, cY), 1, Scalar(0, 255, 0), 2, 8);
				//putText(image, "center", Point2d(cX - 20, cY - 20), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 1, 8);
							//��Ŀ����������궼���������� 
				image_coordinates.push_back(Point2d(cX, cY));//�������д�ŵ������
			}
		}
	}
	return image_coordinates;
}

