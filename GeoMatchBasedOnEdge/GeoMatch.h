//***********************************************************************
// Project		    : GeoMatch
// Author           : Shiju P K
// Email			: shijupk@gmail.com
// Created          : 10-01-2010
//
// File Name		: GeoMatch.h
// Last Modified By : Shiju P K
// Last Modified On : 13-07-2010
// Description      : class to implement edge based template matching
//
// Copyright        : (c) . All rights reserved.
//***********************************************************************

#pragma once
#include <opencv2/opencv.hpp>
#include <math.h>

class GeoMatch
{
private:
	int				m_noOfCordinates;		//Number of elements in coordinate array
	cv::Point		*m_cordinates = nullptr;		//Coordinates array to store model points	
	int				m_modelHeight;		//Template height
	int				m_modelWidth;			//Template width
	double			*m_edgeMagnitude = nullptr;		//gradient magnitude
	double			*m_edgeDerivativeX = nullptr;	//gradient in X direction
	double			*m_edgeDerivativeY = nullptr;	//gradient in Y direction	
	cv::Point		m_centerOfGravity;	//Center of gravity of template 
	
	bool			modelDefined;
	
	void CreateDoubleMatrix(double **&matrix,cv::Size size);
	void ReleaseDoubleMatrix(double **&matrix,int size);
public:
	GeoMatch(void);
	GeoMatch(const cv::Mat& templateArr);
	~GeoMatch(void);

	int CreateGeoMatchModel(const cv::Mat& templateArr,double,double);
	double FindGeoMatchModel(const cv::Mat& srcarr,double minScore,double greediness, cv::Point *resultPoint);
	void DrawContours(cv::Mat pImage,cv::Point COG,cv::Scalar,int);
	void DrawContours(cv::Mat pImage,cv::Scalar,int);
};
