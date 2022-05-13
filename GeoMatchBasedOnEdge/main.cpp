//***********************************************************************
// Project		    : GeoMatch
// Author           : Shiju P K
// Email			: shijupk@gmail.com
// Created          : 10-01-2010
//
// File Name		: main.cpp
// Last Modified By : Shiju P K
// Last Modified On : 13-07-2010
// Description      : Defines the entry point for the console application.
//
// Copyright        : (c) . All rights reserved.
//***********************************************************************
//

#include <iostream>
#include <time.h>

#include "GeoMatch.h"
#include "CommandParser.h"

using namespace std;
int main(int argc, char** argv)
{
	void WrongUsage();

	CommandParser cp(argc, argv); // object to parse command line
	
	GeoMatch GM;				// object to implent geometric matching	
	int lowThreshold = 10;		//deafult value
	int highThreashold = 100;	//deafult value

	double minScore = 0.7;		//deafult value
	double greediness = 0.8;		//deafult value

	double total_time = 0;
	double score= 0;
	cv::Point result;

	//Load Template image 
	char *param;
	param = cp.GetParameter("-t");
	if(param==NULL)
	{
		cout<<"ERROR: Template image argument missing";
		WrongUsage();
		return -1;
	}

	cv::Mat templateImage = cv::imread(param, -1);
	if( templateImage.data == NULL )
	{
		cout<<"\nERROR: Could not load Template Image.\n"<<param;
		return 0;
	}
	
	param = cp.GetParameter("-s");
	if(param==NULL)
	{
		cout<<"ERROR: source image argument missing";
		WrongUsage();
		return -1;
	}
	//Load Search Image
	cv::Mat searchImage = cv::imread(param, -1 );
	if( searchImage.data == NULL )
	{
		cout<<"\nERROR: Could not load Search Image." <<param;
		return 0;
	}
	
	param = cp.GetParameter("-l"); //get Low threshold
	if(param != NULL )
		lowThreshold = atoi(param);
	
	param = cp.GetParameter("-h");
	if(param != NULL )
		highThreashold = atoi(param);//get high threshold
	
	param = cp.GetParameter("-m"); // get minimum score
	if(param != NULL )
		minScore = atof(param);

	param = cp.GetParameter("-g");//get greediness
	if(param != NULL )
		greediness = atof(param);
	
	cv::Size templateSize = cv::Size( templateImage.cols, templateImage.rows );
	cv::Mat grayTemplateImg;

	// Convert color image to gray image.
	if(templateImage.channels() == 3)
	{
		cvtColor(templateImage, grayTemplateImg, cv::COLOR_RGB2GRAY);
	}
	else
	{
		templateImage.copyTo(grayTemplateImg);
	}
	cout<< "\n Edge Based Template Matching Program\n";
	cout<< " ------------------------------------\n";
	
	if(!GM.CreateGeoMatchModel(grayTemplateImg,lowThreshold,highThreashold))
	{
		cout<<"ERROR: could not create model...";
		return 0;
	}
	GM.DrawContours(templateImage,CV_RGB( 255, 0, 0 ),1);
	cout<<" Shape model created.."<<"with  Low Threshold = "<<lowThreshold<<" High Threshold = "<<highThreashold<<endl;
	cv::Size searchSize = cv::Size( searchImage.cols, searchImage.rows );
	cv::Mat graySearchImg = cv::Mat::ones( searchSize, CV_8UC1);

	// Convert color image to gray image. 
	if(searchImage.channels() ==3)
		cvtColor(searchImage, graySearchImg, cv::COLOR_RGB2GRAY);
	else
	{
		searchImage.copyTo(graySearchImg);
	}
	cout<<" Finding Shape Model.."<<" Minumum Score = "<< minScore <<" Greediness = "<<greediness<<"\n\n";
	cout<< " ------------------------------------\n";
	clock_t start_time1 = clock();
	score = GM.FindGeoMatchModel(graySearchImg,minScore,greediness,&result);
	clock_t finish_time1 = clock();
	total_time = (double)(finish_time1-start_time1)/CLOCKS_PER_SEC;

	if(score>minScore) // if score is atleast 0.4
	{
		cout<<" Found at ["<<result.x<<", "<<result.y<<"]\n Score = "<<score<<"\n Searching Time = "<<total_time*1000<<"ms";
		GM.DrawContours(searchImage,result,CV_RGB( 0, 255, 0 ),1);
	}
	else
		cout<<" Object Not found";

	cout<< "\n ------------------------------------\n\n";
	cout<<"\n Press any key to exit!";

	//Display result
	cv::namedWindow("Template" );
	cv::imshow("Template",templateImage);
	cv::namedWindow("Search Image" );
	cv::imshow("Search Image",searchImage);
	// wait for both windows to be closed before releasing images
	cv::waitKey( 0 );
	return 1;
}


void WrongUsage()
{
	cout<< "\n Edge Based Template Matching Program\n" ;
	cout<< " ------------------------------------\n" ;
	cout<< "\nProgram arguments:\n\n";
	cout<< "     -t Template image name (image to be searched)\n\n" ;
	cout<< "     -h High Threshold (High threshold for creating template model)\n\n" ;
	cout<< "     -l Low Threshold (Low threshold for creating template model)\n\n" ;
	cout<< "     -s Search image name (image we are trying to find)\n\n" ;
	cout<< "     -m Minumum score (Minimum score required to proceed with search [0.0 to 1.0])\n\n" ;
	cout<< "     -g greediness (heuistic parameter to terminate search [0.0 to 1.0] )\n\n" ;

	cout<< "Example: GeoMatch -t Template.jpg -h 100 -l 10 -s Search1.jpg -m 0.7 -g 0.5 \n\n" ;
}

