#include "domainTransformFilter.h"
#include "MultiScaleFilter.hpp"

#define CV_LIB_PREFIX comment(lib, "opencv_"

#define CV_LIB_VERSION CVAUX_STR(CV_MAJOR_VERSION)\
    CVAUX_STR(CV_MINOR_VERSION)\
    CVAUX_STR(CV_SUBMINOR_VERSION)

#ifdef _DEBUG
#define CV_LIB_SUFFIX CV_LIB_VERSION "d.lib")
#else
#define CV_LIB_SUFFIX CV_LIB_VERSION ".lib")
#endif

#define CV_LIBRARY(lib_name) CV_LIB_PREFIX CVAUX_STR(lib_name) CV_LIB_SUFFIX


#pragma CV_LIBRARY(core)
#pragma CV_LIBRARY(highgui)
#pragma CV_LIBRARY(imgcodecs)
#pragma CV_LIBRARY(imgproc)

#include <boost\property_tree\xml_parser.hpp>
#include <boost\program_options.hpp>
#include <boost\tokenizer.hpp>
#include <boost/algorithm/string/predicate.hpp>

#include <opencv2/ximgproc.hpp>
#include <opencv2/photo.hpp>
#include <opencv2/xphoto.hpp>

#include <filesystem>
#include <string>

namespace po = boost::program_options;

namespace fs = std::filesystem;

#ifndef VK_ESCAPE
#define VK_ESCAPE 0x1B
#endif // VK_ESCAPE

void smoothDemo(Mat& src)
{
	string wcname = "cws";
	namedWindow(wcname);
	string wname = "smooth";
	namedWindow(wname);

	int sc = 500;
	int ss = 30;
	int iteration = 2;

	createTrackbar("sigma_color", wcname, &sc, 2550);
	createTrackbar("sigma_space", wcname, &ss, 100);
	createTrackbar("iteration", wcname, &iteration, 10);
	int norm = 0;
	createTrackbar("normL1/L2", wcname, &norm, 1);
	int implimentation = 0;
	createTrackbar("impliment", wcname, &implimentation, 2);
	int sw = 0;
	createTrackbar("RF/NC/IC", wcname, &sw, 2);

	int key = 0;
	while (key != 'q' && key != VK_ESCAPE)
	{
		float scf = sc * 0.1f;
		Mat show;
		Mat input;

		input = src;

		int64 startTime = getTickCount();
		if (sw == 0)
		{
			domainTransformFilter(input, show, scf, ss, iteration, norm + 1, DTF_RF, implimentation);
		}
		else if (sw == 1)
		{
			domainTransformFilter(input, show, scf, ss, iteration, norm + 1, DTF_NC, implimentation);
		}
		else if (sw == 2)
		{
			domainTransformFilter(input, show, scf, ss, iteration, norm + 1, DTF_IC, implimentation);
		}

		double time = (getTickCount() - startTime) / (getTickFrequency());
		printf("domain transform filter: %f ms\n", time * 1000.0);

		show.convertTo(show, CV_8U);
		imshow(wname, show);
		imshow(wcname, Mat::zeros(cv::Size(800, 200), CV_8U));

		key = waitKey(1);
	}

	destroyWindow(wname);
	destroyWindow(wcname);

}

void jointSmoothDemo(Mat& src, Mat& guide)
{
	string wname = "smooth";
	namedWindow(wname);

	int sc = 500;
	int ss = 30;
	int iteration = 2;

	createTrackbar("sigma_color", wname, &sc, 2550);
	createTrackbar("sigma_space", wname, &ss, 100);
	createTrackbar("iteration", wname, &iteration, 10);
	int norm = 0;
	createTrackbar("normL1/L2", wname, &norm, 1);
	int implimentation = 0;
	createTrackbar("impliment", wname, &implimentation, 2);
	int sw = 0;
	createTrackbar("RF/NC/IC", wname, &sw, 5);

	int color = 0;
	createTrackbar("color", wname, &color, 1);

	int key = 0;
	while (key != 'q' && key != VK_ESCAPE)
	{
		float scf = sc * 0.1f;
		Mat show;
		Mat input;

		if (color == 0) cvtColor(src, input, COLOR_BGR2GRAY);
		else input = src;

		int64 startTime = getTickCount();
		if (sw == 0)
		{
			domainTransformFilter(input, show, scf, ss, iteration, norm + 1, DTF_RF, implimentation);
		}
		else if (sw == 2)
		{
			domainTransformFilter(input, show, scf, ss, iteration, norm + 1, DTF_NC, implimentation);
		}
		else if (sw == 4)
		{
			domainTransformFilter(input, show, scf, ss, iteration, norm + 1, DTF_IC, implimentation);
		}
		if (sw == 1)
		{
			domainTransformFilter(input, guide, show, scf, ss, iteration, norm + 1, DTF_RF, implimentation);
		}
		else if (sw == 3)
		{
			domainTransformFilter(input, guide, show, scf, ss, iteration, norm + 1, DTF_NC, implimentation);
		}
		else if (sw == 5)
		{
			domainTransformFilter(input, guide, show, scf, ss, iteration, norm + 1, DTF_IC, implimentation);
		}

		double time = (getTickCount() - startTime) / (getTickFrequency());
		printf("domain transform filter: %f ms\n", time * 1000.0);

		imshow(wname, show);
		key = waitKey(1);
	}

	destroyWindow(wname);
}

// Function to display the image with zoom level
void displayWithZoom(Mat& image, double zoomLevel, std::string name)
{
	Mat zoomedImage;
	resize(image, zoomedImage, Size(), zoomLevel, zoomLevel);
	imshow(name.c_str(), zoomedImage);
}

cv::Mat WL(cv::Mat& sc, int L, int H)
{
	cv::Mat lowerMsk;
	cv::inRange(sc, float(0), float(L), lowerMsk);

	cv::Mat higherMsk;
	cv::inRange(sc, float(H), float(1e7), higherMsk);
	cv::Mat r;
	r = (sc - float(L)) * (255. / float(H - L));
	r.setTo(0L, lowerMsk);
	r.setTo(255, higherMsk);

	r.convertTo(r, CV_8U);
	return r;
}


void drawText(std::string text, cv::Mat image)
{
	int fontFace = cv::FONT_HERSHEY_SIMPLEX;
	double fontScale = 1.0;
	int thickness = 2;
	cv::Scalar color(1);

	cv::Point textOrg(50, 50);
	cv::putText(image, text, textOrg, fontFace, fontScale, color, thickness);
}



cv::Mat removeEdge(cv::Mat& src, double minAnatomicLevel = 100., int erodeSize = 5)
{
	cv::Mat anatomicMask;
	cv::threshold(src, anatomicMask, minAnatomicLevel, 1., cv::ThresholdTypes::THRESH_BINARY_INV);
	anatomicMask.convertTo(anatomicMask, CV_8U);

	cv::Mat kernel = cv::Mat::ones(cv::Size(erodeSize, erodeSize), CV_32F);
	cv::Mat rc;
	cv::morphologyEx(anatomicMask, anatomicMask, cv::MorphTypes::MORPH_DILATE, kernel);
	src.copyTo(rc);
	rc.setTo(0, anatomicMask);
	return rc;
}


std::vector<cv::Mat>  multiFileDetailEnhancement(std::vector<cv::Mat>& src, float boostF, int sigma, int pyrLevels, int erodeSize, bool bUI)
{
	cp::MultiScaleBilateralFilter bf;
	bf.setPyramidComputeMethod(cp::MultiScaleFilter::OpenCV);
	cp::MultiScaleFilter::ScaleSpace scalespaceMethod = cp::MultiScaleFilter::ScaleSpace::Pyramid;

	int sigma_range = sigma;
	int ss = 0;
	int boost = int(boostF * 10.f + 0.5f);
	int level = pyrLevels;
	int erode = erodeSize;
	if (bUI)
	{
		int H = 320;
		int L = 80;

		string sname = "smoothed";
		string wname = "detail enhancement";
		namedWindow(wname);
		namedWindow(sname);

		string wccname = "cw";
		namedWindow(wccname);
		createTrackbar("sigma_range", wccname, &sigma_range, 2000);
		createTrackbar("level", wccname, &level, 9);
		createTrackbar("boost", wccname, &boost, 1000);
		createTrackbar("erode", wccname, &erode, 11);
		createTrackbar("L", wccname, &L, 1000);
		createTrackbar("H", wccname, &H, 2000);

		int key = 0;
		double zoomLevel = 1.0;

		Mat bfDst;
		int nPos = 0;

		while (key != 'q' && key != VK_ESCAPE)
		{
			switch (key)
			{
			case '+':
				zoomLevel += 0.1;
				break;

			case '-':
				zoomLevel -= 0.1;
				if (zoomLevel < 0.1) zoomLevel = 0.1;
				break;

			case 'n':
				nPos += 1;
				if (nPos >= src.size())
					nPos = 0;
				break;

			case 'm':
				nPos -= 1;
				if (nPos < 0)
					nPos = src.size() - 1;
				break;
			}

			cv::Mat img;
			src[nPos].copyTo(img);

			// erode edge
			double minAnatomicLevel = 100.;
			if (erode > 0)
				img = removeEdge(img, minAnatomicLevel, erode);

			int64 startTime = getTickCount();
			bf.filter(src[nPos], bfDst, sigma_range, ss, boost / 10, level, scalespaceMethod);
			double time = (getTickCount() - startTime) / (getTickFrequency());
			printf("domain transform filter with enhancement: %f ms\n", time * 1000.0);

			// saturate negative values
			cv::threshold(bfDst, bfDst, 0, 0, cv::THRESH_TOZERO);

			cv::Mat srcScaled;
			src[nPos].copyTo(srcScaled);
			srcScaled /= 50.f;
			rotate(srcScaled, srcScaled, cv::ROTATE_90_CLOCKWISE);
			flip(srcScaled, srcScaled, 1);
			displayWithZoom(WL(srcScaled, L, H), zoomLevel, wname);

			cv::Mat dstbfDst;
			bfDst.copyTo(dstbfDst);
			dstbfDst /= 50.f;
			rotate(dstbfDst, dstbfDst, cv::ROTATE_90_CLOCKWISE);
			flip(dstbfDst, dstbfDst, 1);
			drawText("Frame " + std::to_string(nPos) + "\\" + std::to_string(src.size()), dstbfDst);
			displayWithZoom(WL(dstbfDst, L, H), zoomLevel, sname);

			imshow(wccname, Mat::zeros(cv::Size(800, 200), CV_8U));
			key = waitKey(100);
		}
		cv::destroyWindow(wname);
		cv::destroyWindow(wccname);
	}


	std::vector<cv::Mat> d;
	for (auto & s : src)
	{
		double minAnatomicLevel = 100.;
		if (erode > 0)
			s = removeEdge(s, minAnatomicLevel, erode);

		cv::Mat r;
		bf.filter(s, r, sigma_range, ss, boost / 10, level, scalespaceMethod);

		// saturate negative values
		cv::threshold(r, r, 0, 0, cv::THRESH_TOZERO);

		d.push_back(r);
	}
	return d;
}



std::vector<std::string> processFolder(const std::string& folder_path)
{
	std::vector<std::string> allFiles;
	// Iterate over all files in the folder
	for (const auto& entry : fs::directory_iterator(folder_path))
	{
		// Get the file name from the path
		std::string file_name = entry.path().filename().string();
		std::string full_file_name = entry.path().string();
		allFiles.push_back(full_file_name);
	}
	return allFiles;
}



void enhFolder(const std::string& folder_path, const std::string& dest_path, float boost, int sigma_range, int pyrLevels, int erodeSize, bool bUI)
{
	std::vector<std::string> allFiles = processFolder(folder_path);
	std::vector<cv::Mat> src;
	for (const auto& v : allFiles)
	{
		Mat img = imread(v, IMREAD_ANYDEPTH | IMREAD_GRAYSCALE);

		img.convertTo(img, CV_32F);
		src.push_back(img);
	}

	auto r = multiFileDetailEnhancement(src, boost, sigma_range, pyrLevels, erodeSize, bUI);

	int nPos = 0;
	for (const auto& d : r)
	{
		std::filesystem::path p{ dest_path };
		std::filesystem::path fn{ allFiles[nPos] };
		p /= fn.filename();
		p.replace_extension("tif");
		cv::threshold(d, d, 0, 0, cv::THRESH_TOZERO);

		cv::imwrite(p.string(), d);
		nPos++;
	}
}

int main(int argc, char** argv)
{
	std::string inputImgOrFolder = ".\\inData";
	std::string outputImgOrFolder = ".\\outData";
	int sigma_range{ 300 };
	float boost{ 2.5f };
	int pyrLevels{ 6 };
	bool bUI{ false };
	int erodeSize = 5;

	std::string filterType;
	int halfWienerBlock{ 3 };
	float sigmaWiener{ 0.25f };

	// Declare the supported options.
	po::options_description desc("allowed options");
	desc.add_options()
		("help,h", "produce help message")
		("input,i", po::value<std::string>(&inputImgOrFolder), "input image folder")
		("out,o", po::value<std::string>(&outputImgOrFolder), "output image folder")
		("pyrLevels,p", po::value<int>(&pyrLevels), "pyramid levels")
		("erode,e", po::value<int>(&erodeSize), "erode size")
		("sigma,s", po::value<int>(&sigma_range), "sigma MSE gaussian")
		("boost,b", po::value<float>(&boost), "boost MSE gaussian")
		("ui,u", po::value<bool>(&bUI), "show UI");


	po::variables_map vm;
	try
	{
		po::store(po::parse_command_line(argc, argv, desc), vm);
		po::notify(vm);

		if (vm.count("help"))
		{
			std::cout << desc << "\n";
			return 1;
		}
	}
	catch (po::error& e)
	{
		std::cout << e.what() << "\n";
		return 0;
	}

	if (std::filesystem::exists(outputImgOrFolder) && !std::filesystem::is_directory(outputImgOrFolder))
	{
		std::cout << "output folder not valid\n";
		return -1;
	}

	if (!std::filesystem::exists(outputImgOrFolder))
		std::filesystem::create_directories(outputImgOrFolder);

	if (!std::filesystem::exists(inputImgOrFolder) || !std::filesystem::is_directory(inputImgOrFolder))
	{
		std::cout << "input folder not valid\n";
		return -1;
	}

	enhFolder(inputImgOrFolder, outputImgOrFolder, boost, sigma_range, pyrLevels, erodeSize, bUI);

	return 0;
}