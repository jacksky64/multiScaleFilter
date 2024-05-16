#pragma once

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>


namespace cp
{
	//convert like clone() method;
	 cv::Mat convert(cv::InputArray src, const int depth, const double alpha = 1.0, const double beta = 0.0);
	//a(x-b)+b
	 cv::Mat cenvertCentering(cv::InputArray src, int depth, double a = 1.0, double b = 127.5);

	//convert with gray color conversion like clone() method;
	 cv::Mat convertGray(cv::InputArray& src, const int depth, const double alpha = 1.0, const double beta = 0.0);

	//copyMakeBorder with normal parameters
	 cv::Mat border(cv::Mat& src, const int top, const int bottom, const int left, const int right, const int borderType = cv::BORDER_DEFAULT);
	//copyMakeBorder with one parameter
	 cv::Mat border(cv::Mat& src, const int r, const int borderType = cv::BORDER_DEFAULT);

	 void printMinMax(cv::InputArray src);
}