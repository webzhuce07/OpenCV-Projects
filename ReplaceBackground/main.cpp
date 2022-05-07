#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

Mat MatToSamples(Mat &image);
int main(int argc, char** argv)
{
	Mat src = imread("../1.jpg");
	if (src.empty()) {
		printf("could not load image...\n");
		return -1;
	}
	namedWindow("输入图像");
	imshow("输入图像", src);

	// 组装数据
	Mat points = MatToSamples(src);

	// 运行KMeans
	int numCluster = 4;
	Mat labels;
	Mat centers;
	TermCriteria criteria = TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 10, 0.1);
	kmeans(points, numCluster, labels, criteria, 3, KMEANS_PP_CENTERS, centers);



	// 显示图像分割结果
	Mat resultKMeans = Mat::zeros(src.size(), src.type());
	Scalar colorTab[] = {
		Scalar(0, 0, 255),
		Scalar(0, 255, 0),
		Scalar(255, 0, 0),
		Scalar(0, 255, 255),
		Scalar(255, 0, 255)
	};

	int index0 = 0;
	for (int row = 0; row < src.rows; row++) {
		for (int col = 0; col < src.cols; col++) {
			index0 = row * src.cols + col;
			int label = labels.at<int>(index0, 0);
			resultKMeans.at<Vec3b>(row, col)[0] = colorTab[label][0];
			resultKMeans.at<Vec3b>(row, col)[1] = colorTab[label][1];
			resultKMeans.at<Vec3b>(row, col)[2] = colorTab[label][2];
		}
	}
	imshow("kmeans", resultKMeans);

	for (int i = 0; i < centers.rows; i++) {
		int x = centers.at<float>(i, 0);
		int y = centers.at<float>(i, 1);
		printf("center %d = c.x : %d, c.y : %d\n", i, x, y);
	}


	// 去背景+遮罩生成
	Mat mask = Mat::zeros(src.size(), CV_8UC1);
	int index = src.rows * 2 + 2;
	int cindex = labels.at<int>(index, 0);
	int height = src.rows;
	int width = src.cols;
	//Mat dst;
	//src.copyTo(dst);
	for (int row = 0; row < height; row++) {
		for (int col = 0; col < width; col++) {
			index = row * width + col;
			int label = labels.at<int>(index, 0);
			if (label == cindex) { // 背景
				//dst.at<Vec3b>(row, col)[0] = 0;
				//dst.at<Vec3b>(row, col)[1] = 0;
				//dst.at<Vec3b>(row, col)[2] = 0;
				mask.at<uchar>(row, col) = 0;
			}
			else {
				mask.at<uchar>(row, col) = 255;
			}
		}
	}
	imshow("mask-遮罩", mask);

	// 腐蚀 + 高斯模糊
	Mat k = getStructuringElement(MORPH_RECT, Size(3, 3), Point(-1, -1));//用3*3像素进行腐蚀（减少3*3）
	erode(mask, mask, k);
	imshow("腐蚀-mask", mask);
	GaussianBlur(mask, mask, Size(3, 3), 0, 0);//用3*3像素进行高斯模糊（增加3*3）  所以边界不变
	imshow("高斯模糊-mask", mask);

	// 通道混合
	//RNG rng(12345);//随机数
	Vec3b color;
	color[0] = 0;//rng.uniform(0, 255);
	color[1] = 0;// rng.uniform(0, 255);
	color[2] = 255;// rng.uniform(0, 255);
	Mat result(src.size(), src.type());

	double w = 0.0;
	int b = 0, g = 0, r = 0;
	int b1 = 0, g1 = 0, r1 = 0;
	int b2 = 0, g2 = 0, r2 = 0;

	for (int row = 0; row < height; row++) {
		for (int col = 0; col < width; col++) {
			int m = mask.at<uchar>(row, col);
			if (m == 255) {
				result.at<Vec3b>(row, col) = src.at<Vec3b>(row, col); // 前景
			}
			else if (m == 0) {
				result.at<Vec3b>(row, col) = color; // 背景
			}
			else {
				w = m / 255.0;
				b1 = src.at<Vec3b>(row, col)[0];
				g1 = src.at<Vec3b>(row, col)[1];
				r1 = src.at<Vec3b>(row, col)[2];

				b2 = color[0];
				g2 = color[1];
				r2 = color[2];

				b = b1 * w + b2 * (1.0 - w);
				g = g1 * w + g2 * (1.0 - w);
				r = r1 * w + r2 * (1.0 - w);

				result.at<Vec3b>(row, col)[0] = b;
				result.at<Vec3b>(row, col)[1] = g;
				result.at<Vec3b>(row, col)[2] = r;
			}
		}
	}
	imshow("背景替换", result);

	waitKey(0);
	return 0;
}

Mat MatToSamples(Mat &image) {
	int w = image.cols;
	int h = image.rows;
	int samplecount = w * h;
	int dims = image.channels();
	Mat points(samplecount, dims, CV_32F, Scalar(10));

	int index = 0;
	for (int row = 0; row < h; row++) {
		for (int col = 0; col < w; col++) {
			index = row * w + col;
			Vec3b bgr = image.at<Vec3b>(row, col);
			points.at<float>(index, 0) = static_cast<int>(bgr[0]);
			points.at<float>(index, 1) = static_cast<int>(bgr[1]);
			points.at<float>(index, 2) = static_cast<int>(bgr[2]);
		}
	}
	return points;
}