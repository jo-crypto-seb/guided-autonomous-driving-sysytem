
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <gsl/gsl_fit.h>
#include <iostream>
#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <iomanip>

using namespace cv;
using namespace std;


float rho = 2; // distance resolution in pixels of the Hough grid
float theta = 1 * CV_PI / 180; // angular resolution in radians of the Hough grid
float hough_threshold = 15;	 // minimum number of votes(intersections in Hough grid cell)
float minLineLength = 10; //minimum number of pixels making up a line
float maxLineGap = 20;	//maximum gap in pixels between connectable line segments


//Region - of - interest vertices, 
//We want a trapezoid shape, with bottom edge at the bottom of the image
float trap_bottom_width = 0.85;  // width of bottom edge of trapezoid, expressed as percentage of image width
float trap_top_width = 0.07;     // ditto for top edge of trapezoid
float trap_height = 0.4;         // height of the trapezoid expressed as percentage of image height



Scalar lower_black = Scalar(0, 0, 0); // (RGB)
Scalar upper_black = Scalar(50, 50, 50);


Mat region_of_interest(Mat img_edges, Point *points)
{
	/*
	Applies an image mask.

	Only keeps the region of the image defined by the polygon
	formed from `vertices`. The rest of the image is set to black.
	*/

	Mat img_mask = Mat::zeros(img_edges.rows, img_edges.cols, CV_8UC1);


	Scalar ignore_mask_color = Scalar(255, 255, 255);
	const Point* ppt[1] = { points };
	int npt[] = { 4 };


	//filling pixels inside the polygon defined by "vertices" with the fill color
	fillPoly(img_mask, ppt, npt, 1, Scalar(255, 255, 255), LINE_8);


	//returning the image only where mask pixels are nonzero
	Mat img_masked;
	bitwise_and(img_edges, img_mask, img_masked);


	return img_masked;
}




void filter_colors(Mat _img_bgr, Mat &img_filtered)
{
	
	UMat img_bgr;
	_img_bgr.copyTo(img_bgr);
	UMat img_hsv, img_combine;
	UMat black_mask, black_image;


	//Filter black pixels
	inRange(img_bgr, lower_black, upper_black, black_mask);
	bitwise_and(img_bgr, img_bgr, black_image, black_mask);


	img_combine.copyTo(img_filtered);
}



void draw_line(Mat &img_line, vector<Vec4i> lines)
{
	if (lines.size() == 0) return;

	bool draw_right = true;
	bool draw_left = true;
	int width = img_line.cols;
	int height = img_line.rows;
	float slope_threshold = 0.5;
	vector<float> slopes;
	vector<Vec4i> new_lines;

	for (int i = 0; i < lines.size(); i++)
	{
		Vec4i line = lines[i];
		int x1 = line[0];
		int y1 = line[1];
		int x2 = line[2];
		int y2 = line[3];
		float slope;

		if (x2 - x1 == 0)
			slope = 999.0;
		else
			slope = (y2 - y1) / (float)(x2 - x1);

		if (abs(slope) > slope_threshold) {
			slopes.push_back(slope);
			new_lines.push_back(line);
		}
	}

	vector<Vec4i> right_lines;
	vector<Vec4i> left_lines;

	for (int i = 0; i < new_lines.size(); i++)
	{

		Vec4i line = new_lines[i];
		float slope = slopes[i];

		int x1 = line[0];
		int y1 = line[1];
		int x2 = line[2];
		int y2 = line[3];


		float cx = width * 0.5; //x coordinate of center of image

		if (slope > 0 && x1 > cx && x2 > cx)
			right_lines.push_back(line);
		else if (slope < 0 && x1 < cx && x2 < cx)
			left_lines.push_back(line);
	}

	double right_lines_x[1000];
	double right_lines_y[1000];
	float right_m, right_b;
	int right_index = 0;

	for (int i = 0; i < right_lines.size(); i++) {

		Vec4i line = right_lines[i];
		int x1 = line[0];
		int y1 = line[1];
		int x2 = line[2];
		int y2 = line[3];

		right_lines_x[right_index] = x1;
		right_lines_y[right_index] = y1;
		right_index++;
		right_lines_x[right_index] = x2;
		right_lines_y[right_index] = y2;
		right_index++;
	}


	if (right_index > 0) {

		double c0, c1, cov00, cov01, cov11, sumsq;
		gsl_fit_linear(right_lines_x, 1, right_lines_y, 1, right_index,
			&c0, &c1, &cov00, &cov01, &cov11, &sumsq);

		//printf("# best fit: Y = %g + %g X\n", c0, c1);

		right_m = c1;
		right_b = c0;
	}
	else {
		right_m = right_b = 1;

		draw_right = false;
	}


	
	double left_lines_x[1000];
	double left_lines_y[1000];
	float left_m, left_b;

	int left_index = 0;
	for (int i = 0; i < left_lines.size(); i++) {

		Vec4i line = left_lines[i];
		int x1 = line[0];
		int y1 = line[1];
		int x2 = line[2];
		int y2 = line[3];

		left_lines_x[left_index] = x1;
		left_lines_y[left_index] = y1;
		left_index++;
		left_lines_x[left_index] = x2;
		left_lines_y[left_index] = y2;
		left_index++;
	}


	if (left_index > 0) {
		double c0, c1, cov00, cov01, cov11, sumsq;
		gsl_fit_linear(left_lines_x, 1, left_lines_y, 1, left_index,
			&c0, &c1, &cov00, &cov01, &cov11, &sumsq);

		//printf("# best fit: Y = %g + %g X\n", c0, c1);

		left_m = c1;
		left_b = c0;
	}
	else {
		left_m = left_b = 1;

		draw_left = false;
	}




	//y = m*x + b--> x = (y - b) / m
	int y1 = height;
	int y2 = height * (1 - trap_height);
	float right_x1 = (y1 - right_b) / right_m;
	float right_x2 = (y2 - right_b) / right_m;
	float left_x1 = (y1 - left_b) / left_m;
	float left_x2 = (y2 - left_b) / left_m;

	y1 = int(y1);
	y2 = int(y2);
	right_x1 = int(right_x1);
	right_x2 = int(right_x2);
	left_x1 = int(left_x1);
	left_x2 = int(left_x2);

	if (draw_right)
		line(img_line, Point(right_x1, y1), Point(right_x2, y2), Scalar(255, 0, 0), 10);
	if (draw_left)
		line(img_line, Point(left_x1, y1), Point(left_x2, y2), Scalar(255, 0, 0), 10);
}

int main(int, char**)
{
	char buf[256];
	Mat img_bgr, img_gray, img_edges, img_hough, img_annotated;


	VideoCapture videoCapture(0);

	if (!videoCapture.isOpened())
	{
		cout << "Can't read video. \n" << endl;

		char a;
		cin >> a;

		return 1;
	}



	videoCapture.read(img_bgr);
	if (img_bgr.empty()) return -1;

	VideoWriter writer;
	int codec = VideoWriter::fourcc('X', 'V', 'I', 'D');  // select desired codec (must be available at runtime)
	double fps = 25.0;                          // framerate of the created video stream
	string filename = "./live_video.avi";             // name of the output video file
	writer.open(filename, codec, fps, img_bgr.size(), CV_8UC3);
	// check if we succeeded
	if (!writer.isOpened()) {
		cerr << "The output image is not available.\n";
		return -1;
	}


	videoCapture.read(img_bgr);
	int width = img_bgr.size().width;
	int height = img_bgr.size().height;

	int count = 0;

	while (1)
	{

		
		videoCapture.read(img_bgr);
		if (img_bgr.empty()) break;


		
		Mat img_filtered;
		filter_colors(img_bgr, img_filtered);

		
		cvtColor(img_filtered, img_gray, COLOR_BGR2GRAY);
		GaussianBlur(img_gray, img_gray, Size(3, 3), 0, 0);
		Canny(img_gray, img_edges, 50, 150);

		int width = img_filtered.cols;
		int height = img_filtered.rows;

		Point points[4];
		points[0] = Point((width * (1 - trap_bottom_width)) / 2, height);
		points[1] = Point((width * (1 - trap_top_width)) / 2, height - height * trap_height);
		points[2] = Point(width - (width * (1 - trap_top_width)) / 2, height - height * trap_height);
		points[3] = Point(width - (width * (1 - trap_bottom_width)) / 2, height);


		
		img_edges = region_of_interest(img_edges, points);


		UMat uImage_edges;
		img_edges.copyTo(uImage_edges);

		
		vector<Vec4i> lines;
		HoughLinesP(uImage_edges, lines, rho, theta, hough_threshold, minLineLength, maxLineGap);




		
		//(Linear Least-Squares Fitting)
		Mat img_line = Mat::zeros(img_bgr.rows, img_bgr.cols, CV_8UC3);
		draw_line(img_line, lines);




		
		addWeighted(img_bgr, 0.8, img_line, 1.0, 0.0, img_annotated);


		
		writer << img_annotated;

		count++;
		if (count == 10) imwrite("img_annota1ted.jpg", img_annotated);

		
		Mat img_result;
		resize(img_annotated, img_annotated, Size(width*0.7, height*0.7));
		resize(img_edges, img_edges, Size(width*0.7, height*0.7));
		cvtColor(img_edges, img_edges, COLOR_GRAY2BGR);
		hconcat(img_edges, img_annotated, img_result);
		imshow("Lane detection image ", img_result);

		if (waitKey(1) == 27) break; //ESC
	}


	return 0;
}
