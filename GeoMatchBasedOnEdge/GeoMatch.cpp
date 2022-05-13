//***********************************************************************
// Project		    : GeoMatch
// Author           : Shiju P K
// Email			: shijupk@gmail.com
// Created          : 10-01-2010
//
// File Name		: GeoMatch.cpp
// Last Modified By : Shiju P K
// Last Modified On : 13-07-2010
// Description      : class to implement edge based template matching
//
// Copyright        : (c) . All rights reserved.
//***********************************************************************

#include "GeoMatch.h"

using namespace cv;

GeoMatch::GeoMatch(void)
{
	m_noOfCordinates = 0;  // Initialize  no of coordinates in model points
	modelDefined = false; 
}


int GeoMatch::CreateGeoMatchModel(const cv::Mat& templateArr,double maxContrast,double minContrast)
{
	// Convert IplImage to Matrix for integer operations
	Mat src;
	templateArr.convertTo(src, CV_16SC1);

	// set width and height
	cv::Size Ssize;
	Ssize.width =  src.cols;
	Ssize.height = src.rows;
	m_modelHeight = src.rows;		//Save Template height
	m_modelWidth = src.cols;			//Save Template width

	m_noOfCordinates=0;											//initialize	
	m_cordinates =  new cv::Point[ m_modelWidth * m_modelHeight];		//Allocate memory for coordinates of selected points in template image
	
	m_edgeMagnitude = new double[ m_modelWidth * m_modelHeight];		//Allocate memory for edge magnitude for selected points
	m_edgeDerivativeX = new double[m_modelWidth * m_modelHeight];			//Allocate memory for edge X derivative for selected points
	m_edgeDerivativeY = new double[m_modelWidth * m_modelHeight];			////Allocate memory for edge Y derivative for selected points


	// Calculate gradient of Template
	Mat gx = Mat::zeros(Ssize.height, Ssize.width, CV_16SC1);		//create Matrix to store X derivative
	Mat gy = Mat::zeros(Ssize.height, Ssize.width, CV_16SC1);		//create Matrix to store Y derivative
	Sobel( src, gx, CV_16SC1, 1,0, 3 );		//gradient in X direction			
	Sobel( src, gy, CV_16SC1, 0, 1, 3 );	//gradient in Y direction
	
	Mat nmsEdges = Mat::zeros(Ssize.height, Ssize.width, CV_32F);		//create Matrix to store Final nmsEdges
	const short* _sdx; 
	const short* _sdy; 
	double fdx,fdy;	
    double MagG, DirG;
	double MaxGradient = -99999.99;
	double direction;
	int *orients = new int[ Ssize.height *Ssize.width];
	int count = 0,i,j; // count variable;
	
	double **magMat;
	CreateDoubleMatrix(magMat ,Ssize);
	
	for( i = 1; i < Ssize.height-1; i++ )
    {
    	for( j = 1; j < Ssize.width-1; j++ )
        { 		 
				_sdx = (short*)(gx.data + gx.step*i);
				_sdy = (short*)(gy.data + gy.step*i);
				fdx = _sdx[j]; fdy = _sdy[j];        // read x, y derivatives

				MagG = sqrt((float)(fdx*fdx) + (float)(fdy*fdy)); //Magnitude = Sqrt(gx^2 +gy^2)
				direction = fastAtan2((float)fdy,(float)fdx);	 //Direction = inv tan (Gy / Gx)
				magMat[i][j] = MagG;
				
				if(MagG>MaxGradient)
					MaxGradient=MagG; // get maximum gradient value for normalizing.

				
					// get closest angle from 0, 45, 90, 135 set
                        if ( (direction>0 && direction < 22.5) || (direction >157.5 && direction < 202.5) || (direction>337.5 && direction<360)  )
                            direction = 0;
                        else if ( (direction>22.5 && direction < 67.5) || (direction >202.5 && direction <247.5)  )
                            direction = 45;
                        else if ( (direction >67.5 && direction < 112.5)||(direction>247.5 && direction<292.5) )
                            direction = 90;
                        else if ( (direction >112.5 && direction < 157.5)||(direction>292.5 && direction<337.5) )
                            direction = 135;
                        else 
							direction = 0;
				
			orients[count] = (int)direction;
			count++;
		}
	}
	
	count=0; // init count
	// non maximum suppression
	double leftPixel,rightPixel;
	
	for( i = 1; i < Ssize.height-1; i++ )
    {
		for( j = 1; j < Ssize.width-1; j++ )
        {
				switch ( orients[count] )
                {
                   case 0:
                        leftPixel  = magMat[i][j-1];
                        rightPixel = magMat[i][j+1];
                        break;
                    case 45:
                        leftPixel  = magMat[i-1][j+1];
						rightPixel = magMat[i+1][j-1];
                        break;
                    case 90:
                        leftPixel  = magMat[i-1][j];
                        rightPixel = magMat[i+1][j];
                        break;
                    case 135:
                        leftPixel  = magMat[i-1][j-1];
                        rightPixel = magMat[i+1][j+1];
                        break;
				 }
				// compare current pixels value with adjacent pixels
                if (( magMat[i][j] < leftPixel ) || (magMat[i][j] < rightPixel ) )
					(nmsEdges.data + nmsEdges.step*i)[j]=0;
                else
                    (nmsEdges.data + nmsEdges.step*i)[j]=(uchar)(magMat[i][j]/MaxGradient*255);
			
				count++;
			}
		}
	

	int RSum=0,CSum=0;
	int curX,curY;
	int flag=1;

	//Hysteresis  threshold
	for( i = 1; i < Ssize.height-1; i++ )
    {
		for( j = 1; j < Ssize.width; j++ )
        {
			_sdx = (short*)(gx.data + gx.step*i);
			_sdy = (short*)(gy.data + gy.step*i);
			fdx = _sdx[j]; fdy = _sdy[j];
				
			MagG = sqrt(fdx*fdx + fdy*fdy); //Magnitude = Sqrt(gx^2 +gy^2)
			DirG = fastAtan2((float)fdy,(float)fdx);	 //Direction = tan(y/x)
		
			////((uchar*)(imgGDir->imageData + imgGDir->widthStep*i))[j]= MagG;
			flag=1;
			if(((double)((nmsEdges.data + nmsEdges.step*i))[j]) < maxContrast)
			{
				if(((double)((nmsEdges.data + nmsEdges.step*i))[j])< minContrast)
				{
					
					(nmsEdges.data + nmsEdges.step*i)[j]=0;
					flag=0; // remove from edge
					////((uchar*)(imgGDir->imageData + imgGDir->widthStep*i))[j]=0;
				}
				else
				{   // if any of 8 neighboring pixel is not greater than max contrast remove from edge
					if( (((double)((nmsEdges.data + nmsEdges.step*(i-1)))[j-1]) < maxContrast)	&&
						(((double)((nmsEdges.data + nmsEdges.step*(i-1)))[j]) < maxContrast)		&&
						(((double)((nmsEdges.data + nmsEdges.step*(i-1)))[j+1]) < maxContrast)	&&
						(((double)((nmsEdges.data + nmsEdges.step*i))[j-1]) < maxContrast)		&&
						(((double)((nmsEdges.data + nmsEdges.step*i))[j+1]) < maxContrast)		&&
						(((double)((nmsEdges.data + nmsEdges.step*(i+1)))[j-1]) < maxContrast)	&&
						(((double)((nmsEdges.data + nmsEdges.step*(i+1)))[j]) < maxContrast)		&&
						(((double)((nmsEdges.data + nmsEdges.step*(i+1)))[j+1]) < maxContrast)	)
					{
						(nmsEdges.data + nmsEdges.step*i)[j]=0;
						flag=0;
						////((uchar*)(imgGDir->imageData + imgGDir->widthStep*i))[j]=0;
					}
				}
				
			}
			
			// save selected edge information
			curX=i;	curY=j;
			if(flag!=0)
			{
				if(fdx!=0 || fdy!=0)
				{		
					RSum=RSum+curX;	CSum=CSum+curY; // Row sum and column sum for center of gravity
					
					m_cordinates[m_noOfCordinates].x = curX;
					m_cordinates[m_noOfCordinates].y = curY;
					m_edgeDerivativeX[m_noOfCordinates] = fdx;
					m_edgeDerivativeY[m_noOfCordinates] = fdy;
					
					//handle divide by zero
					if(MagG!=0)
						m_edgeMagnitude[m_noOfCordinates] = 1/MagG;  // gradient magnitude 
					else
						m_edgeMagnitude[m_noOfCordinates] = 0;
															
					m_noOfCordinates++;
				}
			}
		}
	}

	m_centerOfGravity.x = RSum /m_noOfCordinates; // center of gravity
	m_centerOfGravity.y = CSum/m_noOfCordinates ;	// center of gravity
		
	// change coordinates to reflect center of gravity
	for(int m=0;m<m_noOfCordinates ;m++)
	{
		int temp;

		temp=m_cordinates[m].x;
		m_cordinates[m].x=temp-m_centerOfGravity.x;
		temp=m_cordinates[m].y;
		m_cordinates[m].y =temp-m_centerOfGravity.y;
	}
	
	////cvSaveImage("Edges.bmp",imgGDir);
	
	// free allocated memories
	delete[] orients;
	ReleaseDoubleMatrix(magMat ,Ssize.height);
	
	modelDefined=true;
	return 1;
}


double GeoMatch::FindGeoMatchModel(const cv::Mat& srcarr,double minScore,double greediness,cv::Point *resultPoint)
{
	double resultScore=0;
	double partialSum=0;
	double sumOfCoords=0;
	double partialScore;
	const short* _Sdx;
	const short* _Sdy;
	int i,j,m ;			// count variables
	double iTx,iTy,iSx,iSy;
	double gradMag;    
	int curX,curY;

	double **matGradMag;  //Gradient magnitude matrix
	
	Mat src;
	srcarr.convertTo(src, CV_16SC1);
	if( !modelDefined)
	{
		return 0;
	}

	// source image size
	cv::Size Ssize;
	Ssize.width =  src.cols;
	Ssize.height= src.rows;
	
	CreateDoubleMatrix(matGradMag ,Ssize); // create image to save gradient magnitude  values
		
	Mat Sdx = Mat::zeros( Ssize.height, Ssize.width, CV_16SC1 ); // X derivatives
	Mat Sdy = Mat::zeros( Ssize.height, Ssize.width, CV_16SC1 ); // y derivatives
	
	Sobel( src, Sdx, CV_16SC1, 1, 0, 3 );  // find X derivatives
	Sobel( src, Sdy, CV_16SC1, 0, 1, 3 ); // find Y derivatives
		
	// stopping criterias to search for model
	double normMinScore = minScore /m_noOfCordinates; // precompute minumum score 
	double normGreediness = ((1- greediness * minScore)/(1-greediness)) /m_noOfCordinates; // precompute greedniness 
		
	for( i = 0; i < Ssize.height; i++ )
    {
		 _Sdx = (short*)(Sdx.data + Sdx.step*(i));
		 _Sdy = (short*)(Sdy.data + Sdy.step*(i));
		
		 for( j = 0; j < Ssize.width; j++ )
		{
				iSx=_Sdx[j];  // X derivative of Source image
				iSy=_Sdy[j];  // Y derivative of Source image

				gradMag=sqrt((iSx*iSx)+(iSy*iSy)); //Magnitude = Sqrt(dx^2 +dy^2)
							
				if(gradMag!=0) // hande divide by zero
					matGradMag[i][j]=1/gradMag;   // 1/Sqrt(dx^2 +dy^2)
				else
					matGradMag[i][j]=0;
				
		}
	}
	for( i = 0; i < Ssize.height; i++ )
    {
			for( j = 0; j < Ssize.width; j++ )
             { 
				 partialSum = 0; // initialize partialSum measure
				 for(m=0;m<m_noOfCordinates;m++)
				 {
					 curX	= i + m_cordinates[m].x ;	// template X coordinate
					 curY	= j + m_cordinates[m].y ; // template Y coordinate
					 iTx	= m_edgeDerivativeX[m];	// template X derivative
					 iTy	= m_edgeDerivativeY[m];    // template Y derivative

					 if(curX<0 ||curY<0||curX>Ssize.height-1 ||curY>Ssize.width-1)
						 continue;
					 
					 _Sdx = (short*)(Sdx.data + Sdx.step*(curX));
					 _Sdy = (short*)(Sdy.data + Sdy.step*(curX));
						
					 iSx=_Sdx[curY]; // get curresponding  X derivative from source image
					 iSy=_Sdy[curY];// get curresponding  Y derivative from source image
						
					if((iSx!=0 || iSy!=0) && (iTx!=0 || iTy!=0))
					 {
						 //partial Sum  = Sum of(((Source X derivative* Template X drivative) + Source Y derivative * Template Y derivative)) / Edge magnitude of(Template)* edge magnitude of(Source))
						 partialSum = partialSum + ((iSx*iTx)+(iSy*iTy))*(m_edgeMagnitude[m] * matGradMag[curX][curY]);
									
					 }

					sumOfCoords = m + 1;
					partialScore = partialSum /sumOfCoords ;
					// check termination criteria
					// if partial score score is less than the score than needed to make the required score at that position
					// break serching at that coordinate.
					if( partialScore < (MIN((minScore -1) + normGreediness*sumOfCoords,normMinScore*  sumOfCoords)))
						break;
									
				}
				if(partialScore > resultScore)
				{
					resultScore = partialScore; //  Match score
					resultPoint->x = i;			// result coordinate X		
					resultPoint->y = j;			// result coordinate Y
				}
			} 
		}
	
	// free used resources and return score
	ReleaseDoubleMatrix(matGradMag ,Ssize.height);
	return resultScore;
}
// destructor
GeoMatch::~GeoMatch(void)
{
	if(m_cordinates)
		delete[] m_cordinates ;
	if(m_edgeMagnitude)
		delete[] m_edgeMagnitude;
	if(m_edgeDerivativeX)
		delete[] m_edgeDerivativeX;
	if(m_edgeDerivativeY)
		delete[] m_edgeDerivativeY;
}

//allocate memory for doubel matrix
void GeoMatch::CreateDoubleMatrix(double **&matrix,cv::Size size)
{
	matrix = new double*[size.height];
	for(int iInd = 0; iInd < size.height; iInd++)
		matrix[iInd] = new double[size.width];
}
// release memory
void GeoMatch::ReleaseDoubleMatrix(double **&matrix,int size)
{
	for(int iInd = 0; iInd < size; iInd++) 
        delete[] matrix[iInd]; 
}


// draw contours around result image
void GeoMatch::DrawContours(cv::Mat source,cv::Point COG,cv::Scalar color,int lineWidth)
{
	cv::Point point;
	point.y=COG.x;
	point.x=COG.y;
	for(int i=0; i<m_noOfCordinates; i++)
	{	
		point.y=m_cordinates[i].x + COG.x;
		point.x=m_cordinates[i].y + COG.y;
		line(source,point,point,color,lineWidth);
	}
}

// draw contour at template image
void GeoMatch::DrawContours(cv::Mat source,cv::Scalar color,int lineWidth)
{
	cv::Point point;
	for(int i=0; i<m_noOfCordinates; i++)
	{
		point.y=m_cordinates[i].x + m_centerOfGravity.x;
		point.x=m_cordinates[i].y + m_centerOfGravity.y;
		line(source,point,point,color,lineWidth);
	}
}

