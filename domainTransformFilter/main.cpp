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
	
	createTrackbar("sigma_color",wcname,&sc,2550);
	createTrackbar("sigma_space",wcname,&ss,100);
	createTrackbar("iteration",wcname,&iteration,10);
	int norm = 0;
	createTrackbar("normL1/L2",wcname,&norm,1);
	int implimentation=0;
	createTrackbar("impliment",wcname,&implimentation,2);
	int sw=0;
	createTrackbar("RF/NC/IC",wcname,&sw,2);

	int key = 0;
	while(key!='q' && key!=VK_ESCAPE)
	{
		float scf = sc*0.1f;
		Mat show;
		Mat input;
		
		input = src;
		
		int64 startTime = getTickCount();
		if(sw==0)
		{
			domainTransformFilter(input, show,scf,ss,iteration,norm+1,DTF_RF,implimentation);
		}
		else if(sw == 1)
		{
			domainTransformFilter(input, show,scf,ss,iteration,norm+1,DTF_NC,implimentation);
		}
		else if(sw == 2)
		{
			domainTransformFilter(input, show,scf,ss,iteration,norm+1,DTF_IC,implimentation);
		}

		double time = (getTickCount()-startTime)/(getTickFrequency());
		printf("domain transform filter: %f ms\n",time*1000.0);

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

	 createTrackbar("sigma_color",wname,&sc,2550);
	 createTrackbar("sigma_space",wname,&ss,100);
	 createTrackbar("iteration",wname,&iteration,10);
	 int norm = 0;
	 createTrackbar("normL1/L2",wname,&norm,1);
	 int implimentation=0;
	 createTrackbar("impliment",wname,&implimentation,2);
	 int sw=0;
	 createTrackbar("RF/NC/IC",wname,&sw,5);

	 int color = 0;
	 createTrackbar("color",wname,&color,1);

	 int key = 0;
	 while(key!='q' && key!=VK_ESCAPE)
	 {
		 float scf = sc*0.1f;
		 Mat show;
		 Mat input;

		 if(color==0) cvtColor(src,input,COLOR_BGR2GRAY);
		 else input = src;

		 int64 startTime = getTickCount();
		 if(sw==0)
		 {
			 domainTransformFilter(input,show,scf,ss,iteration,norm+1,DTF_RF,implimentation);
		 }
		 else if(sw == 2)
		 {
			 domainTransformFilter(input, show,scf,ss,iteration,norm+1,DTF_NC,implimentation);
		 }
		 else if(sw == 4)
		 {
			 domainTransformFilter(input, show,scf,ss,iteration,norm+1,DTF_IC,implimentation);
		 }
		 if(sw==1)
		 {
			 domainTransformFilter(input, guide,show,scf,ss,iteration,norm+1,DTF_RF,implimentation);
		 }
		 else if(sw == 3)
		 {
			 domainTransformFilter(input, guide, show,scf,ss,iteration,norm+1,DTF_NC,implimentation);
		 }
		 else if(sw == 5)
		 {
			 domainTransformFilter(input, guide, show,scf,ss,iteration,norm+1,DTF_IC,implimentation);
		 }

		 double time = (getTickCount()-startTime)/(getTickFrequency());
		 printf("domain transform filter: %f ms\n",time*1000.0);

		 imshow(wname,show);
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

cv::Mat WL(cv::Mat& sc,int L, int H)
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

cv::Mat detailEnhancement(Mat& src)
{
	cp::MultiScaleBilateralFilter bf;
	bf.setPyramidComputeMethod(cp::MultiScaleFilter::OpenCV);

	string sname = "smoothed";
	string wname = "detail enhancement";
	namedWindow(wname);
	namedWindow(sname);
	
	int sc = 25;
	int ss = 170;
	int iteration = 2;
	int boost = 430;
	int L = 300; 
	int H = 1300;
	
	string wccname = "cw";
	namedWindow(wccname);
	createTrackbar("s_color",wccname,&sc,1000);
	createTrackbar("s_space",wccname,&ss,1000);
	createTrackbar("iteration",wccname,&iteration,2);
	createTrackbar("boost",wccname,&boost,500);
	createTrackbar("L",wccname,&L,1000);
	createTrackbar("H",wccname,&H,2000);
	
	int key = 0;
	double zoomLevel = 1.0;

	Mat show;
	Mat smooth;
	Mat sub;
	Mat bfDst;

	while(key!='q' && key!=VK_ESCAPE)
	{
		if (key == '+') {
			zoomLevel += 0.1;
		}
		// Decrease zoom level when '-' key is pressed
		else if (key == '-') {
			zoomLevel -= 0.1;
			if (zoomLevel < 0.1) zoomLevel = 0.1;
		}

		int64 startTime = getTickCount();

		domainTransformFilter(src, smooth,ss,sc,iteration,DTF_L1,DTF_RF,DTF_BGRA_SSE_PARALLEL);
		//ximgproc::dtFilter(src, src, smooth, ss, sc, 0, iteration);

		//subtract(src,smooth,sub,noArray(),CV_32F);
		//sub*=(boost*0.1);
		//add(src,1*sub,sub,noArray(),CV_32F);
		
		cv::normalize(src, bfDst, 250, 0, NORM_MINMAX);

		bf.filter(src, bfDst, sc, ss, boost/100, 8);
		double time = (getTickCount()-startTime)/(getTickFrequency());
		printf("domain transform filter with enhancement: %f ms\n",time*1000.0);

		displayWithZoom(WL(src, L, H), zoomLevel, wname);
		displayWithZoom(WL(bfDst, L, H), zoomLevel, sname);
		
		//imshow(wname,show);
		imshow(wccname,Mat::zeros(cv::Size(800,200), CV_8U));
		key = waitKey(100);
	}

	cv::destroyWindow(wname);
	cv::destroyWindow(wccname);
	return bfDst;
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

std::vector<cv::Mat>  multiFileDetailEnhancement(std::vector<cv::Mat>& src)
{
	string sname = "smoothed";
	string wname = "detail enhancement";
	namedWindow(wname);
	namedWindow(sname);

	cp::MultiScaleBilateralFilter bf;
#if 1	
	bf.setPyramidComputeMethod(cp::MultiScaleFilter::OpenCV);
	int sc = 40;
	int ss = 170;
	int boost = 700;
	int L = 170;
	int H = 1500;
	int level = 6;
	cp::MultiScaleFilter::ScaleSpace scalespaceMethod = cp::MultiScaleFilter::ScaleSpace::Pyramid;
#else
	// bf.setPyramidComputeMethod(cp::MultiScaleFilter::OpenCV);
	int sc = 40;
	int ss = 1;
	int boost = 700;
	int L = 170;
	int H = 1500;
	int level = 4;
	cp::MultiScaleFilter::ScaleSpace scalespaceMethod = cp::MultiScaleFilter::ScaleSpace::DoG;
#endif

	string wccname = "cw";
	namedWindow(wccname);
	createTrackbar("s_color", wccname, &sc, 1000);
	createTrackbar("s_space", wccname, &ss, 1000);
	createTrackbar("boost", wccname, &boost, 1000);
	createTrackbar("L", wccname, &L, 1000);
	createTrackbar("H", wccname, &H, 2000);

	int key = 0;
	double zoomLevel = 1.0;

	Mat show;
	Mat smooth;
	Mat sub;
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
				nPos = src.size()-1;
			break;
		}


		int64 startTime = getTickCount();

		cv::normalize(src[nPos], bfDst, 250, 0, NORM_MINMAX);
		

		bf.filter(src[nPos], bfDst, sc, ss, boost / 100, level, scalespaceMethod);
		double time = (getTickCount() - startTime) / (getTickFrequency());
		printf("domain transform filter with enhancement: %f ms\n", time * 1000.0);

		
		drawText("Frame " + std::to_string(nPos) + "\\" + std::to_string(src.size()), bfDst);

		displayWithZoom(WL(src[nPos], L, H), zoomLevel, wname);
		displayWithZoom(WL(bfDst, L, H), zoomLevel, sname);


		//imshow(wname,show);
		imshow(wccname, Mat::zeros(cv::Size(800, 200), CV_8U));
		key = waitKey(100);
	}

	cv::destroyWindow(wname);
	cv::destroyWindow(wccname);
	std::vector<cv::Mat> d;
	for (auto const& s : src)
	{
		cv::Mat r;
		bf.filter(s, r, sc, ss, boost / 100, level, scalespaceMethod);
		d.push_back(r);
	}
	return d;
}


void bpdetailEnhancement(Mat& src)
{
	string wname = "detail enhancement";
	string wccname = "cws";
	namedWindow(wname);
	namedWindow(wccname);

	int sc = 20;
	int ss = 70;
	int sclp = 5;
	int sslp = 6;
	int iteration = 2;
	int boost = 30;
	createTrackbar("slp_color", wccname, &sclp, 255);
	createTrackbar("slp_space", wccname, &sslp, 255);
	createTrackbar("s_color", wccname, &sc, 255);
	createTrackbar("s_space", wccname, &ss, 255);
	createTrackbar("iteration", wccname, &iteration, 255);
	createTrackbar("boost", wccname, &boost, 500);
	int key = 0;

	Mat show;
	Mat smooth;
	Mat sub;
	Mat srcLp;
	while (key != 'q' && key != VK_ESCAPE)
	{
		int64 startTime = getTickCount();

		domainTransformFilter(src, srcLp, sslp, sclp, iteration, DTF_L1, DTF_RF, DTF_BGRA_SSE_PARALLEL);

		domainTransformFilter(srcLp, smooth, ss, sc, iteration, DTF_L1, DTF_RF, DTF_BGRA_SSE_PARALLEL);

		subtract(srcLp, smooth, sub, noArray(), CV_32F);
		sub *= (boost * 0.1);
		add(srcLp, 1 * sub, sub, noArray(), CV_32F);
		sub.convertTo(show, CV_8U);

		double time = (getTickCount() - startTime) / (getTickFrequency());
		printf("domain transform filter with enhancement: %f ms\n", time * 1000.0);

		imshow(wname, show);
		imshow(wccname, Mat::zeros(cv::Size(800, 200), CV_8U));
		key = waitKey(1);
	}

	destroyWindow(wname);
	destroyWindow(wccname);
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





cv::Mat removeEdge(cv::Mat& src, double minAnatomicLevel = 100.)
{
	cv::Mat anatomicMask;
	cv::threshold(src, anatomicMask, minAnatomicLevel, 1., cv::ThresholdTypes::THRESH_BINARY_INV);
	anatomicMask.convertTo(anatomicMask, CV_8U);

	cv::Mat kernel = cv::Mat::ones(cv::Size(5, 5), CV_32F);
	cv::Mat rc;
	cv::morphologyEx(anatomicMask, anatomicMask, cv::MorphTypes::MORPH_DILATE, kernel);
	src.copyTo(rc);
	rc.setTo(0, anatomicMask);
	return rc;
}


void enhFolder(const std::string& folder_path, const std::string& dest_path)
{
	std::vector<std::string> allFiles = processFolder(folder_path);
	std::vector<cv::Mat> src;
	for (const auto &v: allFiles )
	{ 
		Mat img = imread(v, IMREAD_ANYDEPTH | IMREAD_GRAYSCALE);
		double minAnatomicLevel = 100.;
		img = removeEdge(img, minAnatomicLevel);
	
		/*rotate(img, img, cv::ROTATE_90_CLOCKWISE);
		flip(img, img, 1);*/

		img.convertTo(img, CV_32F);
		cv::normalize(img, img, 1000, 0, NORM_MINMAX);
		src.push_back(img);
	}
	
	auto r = multiFileDetailEnhancement(src);

	int nPos = 0;
	for (const auto& d : r)
	{
		std::filesystem::path p{dest_path};
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
	std::string filterType;
	float fsigma{ 0.f }, fFiltAlpha{ 0.f };
	int halfPatch, halfSearchWindow, nTemporalFrames{ 1 }, nProcFrames{ 1 }, nSkipFrames{ 0 }, pyrLevels{ 0 };
	int halfWienerBlock{ 3 };
	int iteration{ 1 };
	float sigmaWiener{ 0.25f };
	bool useXCorr{ false };
	bool bGaussian{ false };
	bool bVST{ true };

	// Declare the supported options.
	po::options_description desc("allowed options");
	desc.add_options()
		("help,h", "produce help message")
		("input,i", po::value<std::string>(&inputImgOrFolder), "input image folder")
		("type,t", po::value<std::string>(&filterType), "filter type (dct, nlm, nlmMS, nlmMSW)")
		("sigma,s", po::value<float>(&fsigma), "noise std")
		("alpha,a", po::value<float>(&fFiltAlpha), "filter alpha")
		("patchSize,p", po::value<int>(&halfPatch), "half patch size")
		("iterate,j", po::value<int>(&iteration), "iteration counts")
		("searchSize,w", po::value<int>(&halfSearchWindow), "half search wnd size")
		("pyrLevels,y", po::value<int>(&pyrLevels), "pyramid levels")
		("gaussian,g", po::value<bool>(&bGaussian), "gaussian weights")
		("vst,v", po::value<bool>(&bVST), "variance stabilization")
		("tFrame,f", po::value<int>(&nTemporalFrames), "temporal frame number")
		("wienerHalfBlock,b", po::value<int>(&halfWienerBlock), "wiener half block size")
		("wienerSigma,c", po::value<float>(&sigmaWiener), "wiener noise sigma")
		("skipFrames,k", po::value<int>(&nSkipFrames), "skip frames")
		("xCorr,x", po::value<bool>(&useXCorr), "use correlation (fixed -> tbd)")
		("transientMode,u", "produce frames during settling time")
		("outFrames,n", po::value<int>(&nProcFrames), "processed output frames")
		("out,o", po::value<std::string>(&outputImgOrFolder), "output image folder");

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

	enhFolder(inputImgOrFolder, outputImgOrFolder);

	/*Mat img = imread("s_0024.png", IMREAD_ANYDEPTH | IMREAD_GRAYSCALE);
	
	cv::Mat anatomicMask;
	cv::threshold(img, anatomicMask, 100, 1, cv::ThresholdTypes::THRESH_BINARY_INV);
	anatomicMask.convertTo(anatomicMask, CV_8U);

	cv::Mat rc;
	cv::Mat kernel = cv::Mat::ones(cv::Size(15, 15),CV_32F);
	cv::morphologyEx(anatomicMask, anatomicMask, cv::MorphTypes::MORPH_DILATE, kernel );
	img.setTo(0, anatomicMask);

	rotate(img, rc, cv :: ROTATE_90_CLOCKWISE);
	flip(rc, rc, 1);
	
	rc.convertTo(rc, CV_32F);
	cv::normalize(rc, rc, 1000, 0, NORM_MINMAX);
	cv::Mat r = detailEnhancement(rc);
	cv::imwrite("s_0024Filt.tif", r);*/


	return 0;
}