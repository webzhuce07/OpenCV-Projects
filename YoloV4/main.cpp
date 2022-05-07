#include<opencv2/opencv.hpp>
#include<opencv2/dnn.hpp>
#include <iostream>
#include <fstream>
#include<istream>
#include<string>

using namespace std;
using namespace cv;
using namespace dnn;

// ��ʼ������
float confThreshold = 0.5; // ���Ŷ���ֵ
float nmsThreshold = 0.4;  // �Ǽ���ֵ����(NMS)��ֵ
int inpWidth = 416;        // ��������ͼ����
int inpHeight = 416;       // ��������ͼ��߶�


// ������������ļ�
vector<string>classes;
// Load names of classes
string classesFile = "./model/coco.names";
ifstream ifs(classesFile.c_str());


// ����ģ�������ļ���Ȩ��
String config = "./model/yolov4.cfg";
String weights = "./model/yolov4.weights";

// ��������
Net net = readNetFromDarknet(config, weights);


// ��ȡ���������
vector<String> getOutputsNames(const Net& net)
{
	static vector<String> names;
	if (names.empty())
	{
		//Get the indices of the output layers, i.e. the layers with unconnected outputs
		vector<int> outLayers = net.getUnconnectedOutLayers();

		//get the names of all the layers in the network
		vector<String> layersNames = net.getLayerNames();

		// Get the names of the output layers in names
		names.resize(outLayers.size());
		for (size_t i = 0; i < outLayers.size(); ++i)
			names[i] = layersNames[outLayers[i] - 1];
	}
	return names;
}

// Draw the predicted bounding box
void drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame)
{
	//Draw a rectangle displaying the bounding box
	rectangle(frame, Point(left, top), Point(right, bottom), Scalar(0, 0, 255), 2);

	//Get the label for the class name and its confidence
	string label = format("%.2f", conf);
	if (!classes.empty())
	{
		CV_Assert(classId < (int)classes.size());
		label = classes[classId] + ":" + label;
	}

	//Display the label at the top of the bounding box
	int baseLine;
	Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.8, 1, &baseLine);
	top = max(top, labelSize.height);
	putText(frame, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 255, 0), 2);
}

// Remove the bounding boxes with low confidence using non-maxima suppression
void postprocess(Mat& frame, const vector<Mat>& outs)
{
	vector<int> classIds;
	vector<float> confidences;
	vector<Rect> boxes;

	for (size_t i = 0; i < outs.size(); ++i)
	{
		// Scan through all the bounding boxes output from the network and keep only the
		// ones with high confidence scores. Assign the box's class label as the class
		// with the highest score for the box.
		float* data = (float*)outs[i].data;
		for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols)
		{
			Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
			Point classIdPoint;
			double confidence;
			// Get the value and location of the maximum score
			minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
			if (confidence > confThreshold)
			{
				int centerX = (int)(data[0] * frame.cols);
				int centerY = (int)(data[1] * frame.rows);
				int width = (int)(data[2] * frame.cols);
				int height = (int)(data[3] * frame.rows);
				int left = centerX - width / 2;
				int top = centerY - height / 2;

				classIds.push_back(classIdPoint.x);
				confidences.push_back((float)confidence);
				boxes.push_back(Rect(left, top, width, height));
			}
		}
	}

	// Perform non maximum suppression to eliminate redundant overlapping boxes with
	// lower confidences
	vector<int> indices;
	NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);
	for (size_t i = 0; i < indices.size(); ++i)
	{
		int idx = indices[i];
		Rect box = boxes[idx];
		drawPred(classIds[idx], confidences[idx], box.x, box.y,
			box.x + box.width, box.y + box.height, frame);
	}
}

int main()
{
	Mat img = imread("../person.jpg");
	if (img.empty())
	{
		cout << "Image read error, please check again!" << endl;
	}
	string line;
	while (getline(ifs, line))
	{
		classes.push_back(line);
	}
	net.setPreferableBackend(DNN_BACKEND_OPENCV);
	net.setPreferableTarget(DNN_TARGET_CPU);
	// Create a 4D blob from a frame.
	Mat blob;
	blobFromImage(img, blob, 1 / 255.0, Size(inpWidth, inpHeight), Scalar(0, 0, 0), true, false);

	//Sets the input to the network
	net.setInput(blob);

	// Runs the forward pass to get output of the output layers
	vector<Mat> outs;
	net.forward(outs, getOutputsNames(net));

	// Remove the bounding boxes with low confidence
	postprocess(img, outs);

	// Put efficiency information. The function getPerfProfile returns the
	// overall time for inference(t) and the timings for each of the layers(in layersTimes)
	vector<double> layersTimes;
	double freq = getTickFrequency() / 1000;
	double t = net.getPerfProfile(layersTimes) / freq;
	string label = format("Inference time for a frame : %.2f ms", t);
	putText(img, label, Point(0, 20), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(255, 255, 0), 2);
	namedWindow("OpenCV_YoloV4_Demo", WINDOW_NORMAL);
	imshow("OpenCV_YoloV4_Demo", img);
	waitKey(0);

	return 0;
}