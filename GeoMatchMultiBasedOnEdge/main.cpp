#include "KcgMatch.h"
#include <time.h>

int main(int argc, char **argv) 
{
	// 实例化KcgMatch 
	// "demo/k"为存储模板的根目录 
	// "k"为模板的名字
	kcg_matching::KcgMatch kcg("../demo/k", "k");
	// 读取模板图像
	Mat model = imread("../demo/k/template.png");
	// 转灰度
	cvtColor(model, model, COLOR_BGR2GRAY);
	// 指定要制作的模板角度，尺度范围
	kcg_matching::AngleRange ar(-180.f, 180.f, 10.f);
	kcg_matching::ScaleRange sr(0.70f, 1.3f, 0.05f);
	// 开始制作模板
	kcg.MakingTemplates(model, ar, sr, 0, 30.f, 60.f);

	// 加载模板
	cout << "Loading model ......" << endl;
	kcg.LoadModel();
	cout << "Load succeed." << endl;

	// 读取搜索图像
	Mat source = imread("../demo/k/search.png");
	Mat draw_source;
	source.copyTo(draw_source);
	cvtColor(source, source, COLOR_BGR2GRAY);

	//Timer timer;
	// 开始匹配
	auto matches =
		kcg.Matching(source, 0.80f, 0.1f, 30.f, 0.9f,
			kcg_matching::PyramidLevel_2, 2, 12);
	//double t = timer.out("=== Match time ===");
	cout << "Final match size: " << matches.size() << endl << endl;

	// 画出匹配结果
	kcg.DrawMatches(draw_source, matches, Scalar(255, 0, 0));

	// 画出匹配时间
	rectangle(draw_source, Rect(Point(0, 0), Point(136, 20)), Scalar(255, 255, 255), -1);
	/*cv::putText(draw_source,
		"time: " + to_string(t) + "s",
		Point(0, 16), FONT_HERSHEY_PLAIN, 1.f, Scalar(0, 0, 0), 1);*/

	// 显示结果图像
	namedWindow("draw_source", 0);
	imshow("draw_source", draw_source);
	waitKey(0);
	system("pause");
}

//#include "matchShapes.h"
//#include <opencv2/opencv.hpp>
//
//using namespace cv;
//
//int main()
//{
//	Mat tempImage = imread("../template.jpg");
//	vector<Point> imgTemplatecontours = ImageTemplateContours(tempImage);
//
//	Mat image = imread("../test.jpg");
//	vector<Point2d> points =  ShapeTemplateMatch(image, imgTemplatecontours, 0.1);
//
//	for each (auto point in points)
//	{
//		Rect rect(point.x, point.y, 100, 100);
//		rectangle(image, rect, Scalar(255, 0, 255));
//
//	}
//	imshow("Result", image);
//	waitKey(0);
//	return EXIT_SUCCESS;
//}

//#include <opencv2/opencv.hpp>
//#include <ppl.h>
//#include <mutex>
//#include "Models.h"
//#include <time.h>
//
//using namespace cv;
//using namespace std;
//
//void main()
//{
//	//提取模板的梯度与坐标
//	int numLevels = 1;
//	Mat tempImage = imread("img/model3.bmp", IMREAD_ANYCOLOR);
//	cvtColor(tempImage, tempImage, COLOR_BGR2GRAY);
//	Models model;
//	createModels(tempImage, 0, 60, 10, 60, 150, 3, model);
//
//
//	Mat src = imread("img/model3_src3.bmp", IMREAD_ANYCOLOR);
//	cvtColor(src, src, COLOR_BGR2GRAY);
//
//	Mat gx, gy, g_rec;
//	calcGradientImage(src, 3, 60, 150, numLevels, gx, gy, g_rec);
//	float minScore = 0.9;
//	float norMinScore = minScore / model.pxCount;
//	float greediness = 0.7;
//	float normGreediness = ((1 - greediness * minScore) / (1 - greediness)) / model.pxCount; // precompute greedniness 
//
//
//	clock_t start = clock();
//	int parallel_num = gx.rows * gx.cols;
//	float resultScore = 0;
//	Point resultPoint(-1, -1);
//	mutex mt;
//	parallel_for(0, parallel_num, [&](int k)
//	{
//		int srcx = k % gx.cols;
//		int srcy = k / gx.cols;
//		float partialSum = 0;
//		int numCount = 0;
//		float partialScore = 0;
//		int _parallel_num = model.pxCount;
//		int sumCount;
//		mutex _mt;
//		parallel_for(0, _parallel_num, [&](int m)
//		{
//			int curx = srcx + model.pos[m].x - model.center.x;
//			int cury = srcy + model.pos[m].y - model.center.y;
//			if (curx > 0 && curx < gx.cols - 1 && cury > 0 && cury < gx.rows - 1)
//			{
//				float Gsx = gx.at<float>(cury, curx);
//				float Gsy = gy.at<float>(cury, curx);
//				float Gs_rec = g_rec.at<float>(cury, curx);
//				float Gtx = model.grads[m].x;
//				float Gty = model.grads[m].y;
//				float Gt_rec = model.grads[m].g_rec;
//				_mt.lock();
//				if ((Gtx != 0 || Gty != 0) && (Gsx != 0 || Gsy != 0))
//					partialSum = partialSum + (Gsx*Gtx + Gsy * Gty)*Gt_rec*Gs_rec;
//				numCount++;
//				partialScore = partialSum / numCount;
//				_mt.unlock();
//			}
//		});
//
//		if (partialScore > resultScore)
//		{
//			mt.lock();
//			resultScore = partialScore;
//			resultPoint.x = srcx;
//			resultPoint.y = srcy;
//			mt.unlock();
//		}
//	});
//
//	clock_t end = clock();
//	cout << "time cost = " << end - start << endl;
//
//	resultPoint.x = pow(2, numLevels)* resultPoint.x;
//	resultPoint.y = pow(2, numLevels)* resultPoint.y;
//
//	circle(src, resultPoint, 30, Scalar(255), 3);
//	imshow("src", src);
//	imwrite("result1.bmp", src);
//	waitKey(0);
//}