#include<opencv2/opencv.hpp>
#include <vector>
using namespace cv;

int main(int argc, char *argv[])
{
	Mat image = cv::imread("1.jpg", 1);
	std::vector<cv::Mat> images;
	split(image, images);
	for (int i = 0; i < image.channels(); i++)
	{
		Mat& img = images[i];
		img.convertTo(img, CV_32F, 1, 0);
		cv::Mat  highPass;
		img.convertTo(highPass, CV_32F, 1, 0);

		cv::Mat EPFFilter;
		cv::bilateralFilter(highPass, EPFFilter, 15, 30, 60);
	    //cv:imwrite("bilateralFilter.jpg", EPFFilter);
		EPFFilter = EPFFilter - img;
		EPFFilter = EPFFilter + 128.0;
		//cv::imwrite("GaussianBlur0.jpg", EPFFilter);
		cv::GaussianBlur(EPFFilter, highPass, cv::Size(5, 5), 0, 0);
		//cv::imwrite("GaussianBlur.jpg", highPass);
		double opacity = 90.0;
		cv::Mat  dst = (img * (100.0 - opacity) + (img + 2.0 * highPass - 256.0) * opacity) / 100.0;
		//cv::imwrite("dst.jpg", dst);
		images[i] = dst;
	}

	Mat dst;
	merge(images, dst);

	cv::imwrite("dst.jpg", dst);

	waitKey(0);
	return EXIT_SUCCESS;
}