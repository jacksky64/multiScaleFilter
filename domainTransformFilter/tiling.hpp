#pragma once
#pragma once

#include <opencv2/core.hpp>

#include <omp.h>
namespace cp
{
	//get online image size
	 cv::Size getTileAlignSize(const cv::Size src, const cv::Size div_size, const int r, const int align_x, const int align_y, const int left_multiple = 1, const int top_multiple = 1);
	 cv::Size getTileSize(const cv::Size src, const cv::Size div_size, const int r);

	//create a divided sub image
	 void cropTile(const cv::Mat& src, cv::Mat& dest, const cv::Size div_size, const cv::Point idx, const int topb, const int bottomb, const int leftb, const int rightb, const int borderType = cv::BORDER_DEFAULT);
	 void cropTile(const cv::Mat& src, cv::Mat& dest, const cv::Size div_size, const cv::Point idx, const int r, const int borderType = cv::BORDER_DEFAULT);
	 void cropTileAlign(const cv::Mat& src, cv::Mat& dest, const cv::Size div_size, const cv::Point idx, const int r, const int borderType = cv::BORDER_DEFAULT, const int align_x = 8, const int align_y = 1, const int leftmultiple = 1, const int topmultiple = 1);

	 void cropTile(const cv::Mat& src, cv::Mat& dest, const cv::Rect roi, const int topb, const int bottomb, const int leftb, const int rightb, const int borderType = cv::BORDER_DEFAULT);
	 void cropTile(const cv::Mat& src, cv::Mat& dest, const cv::Rect roi, const int r, const int borderType = cv::BORDER_DEFAULT);
	 void cropTileAlign(const cv::Mat& src, cv::Mat& dest, const cv::Rect roi, const int r, const int borderType = cv::BORDER_DEFAULT, const int align_x = 8, const int align_y = 1, const int leftmultiple = 1, const int topmultiple = 1);

	 void cropSplitTile(const cv::Mat& src, std::vector<cv::Mat>& dest, const cv::Size div_size, const cv::Point idx, const int topb, const int bottomb, const int leftb, const int rightb, const int borderType= cv::BORDER_DEFAULT);
	 void cropSplitTileAlign(const cv::Mat& src, std::vector<cv::Mat>& dest, const cv::Size div_size, const cv::Point idx, const int r, const int borderType = cv::BORDER_DEFAULT, const int align_x = 8, const int align_y = 1, const int leftmultiple = 1, const int topmultiple = 1);

	//set a divided sub image to a large image
	 void pasteTile(const cv::Mat& src, cv::Mat& dest,      const cv::Rect roi, const int top, const int left);
	 void pasteTile(const cv::Mat& src, cv::Mat& dest,      const cv::Rect roi, const int r);
	 void pasteTileAlign(const cv::Mat& src, cv::Mat& dest, const cv::Rect roi, const int r, const int left_multiple = 1, const int top_multiple = 1);

	 void pasteTile(const cv::Mat& src, cv::Mat& dest, const cv::Size div_size, const cv::Point idx, const int top, const int left);
	 void pasteTile(const cv::Mat& src, cv::Mat& dest, const cv::Size div_size, const cv::Point idx, const int r);
	 void pasteTileAlign(const cv::Mat& src, cv::Mat& dest, const cv::Size div_size, const cv::Point idx, const int r, const int left_multiple = 1, const int top_multiple = 1);

	 void pasteMergeTile(const std::vector <cv::Mat>& src, cv::Mat& dest, const cv::Size div_size, const cv::Point idx, const int top, const int left);
	 void pasteMergeTile(const std::vector <cv::Mat>& src, cv::Mat& dest, const cv::Size div_size, const cv::Point idx, const int r);
	 void pasteMergeTileAlign(const std::vector <cv::Mat>& src, cv::Mat& dest, const cv::Size div_size, const cv::Point idx, const int r, const int left_multiple = 1, const int top_multiple = 1);

	//split an image to sub images in std::vector 
	 void divideTiles(const cv::Mat& src, std::vector<cv::Mat>& dest, const cv::Size div_size, const int r, const int borderType = cv::BORDER_DEFAULT);
	 void divideTilesAlign(const cv::Mat& src, std::vector<cv::Mat>& dest, const cv::Size div_size, const int r, const int borderType = cv::BORDER_DEFAULT, const int align_x = 8, const int align_y = 1, const int left_multiple = 1, const int top_multiple = 1);

	//merge subimages in std::vector to an image
	 void conquerTiles(const std::vector<cv::Mat>& src, cv::Mat& dest, const cv::Size div_size, const int r);
	 void conquerTilesAlign(const std::vector<cv::Mat>& src, cv::Mat& dest, const cv::Size div_size, const int r, const int left_multiple = 1, const int top_multiple = 1);

	class  TileDivision
	{
		std::vector<cv::Point> pt;//left top point
		std::vector<cv::Size> tileSize;
		std::vector<int> threadnum;
		cv::Size div;
		cv::Size imgSize;
		int width_step = 0;
		int height_step = 0;

		void update_pt();
		bool isRecompute = true;
		bool preReturnFlag = false;
	public:

		//div.width * y + x;
		cv::Rect getROI(const int x, int y);
		cv::Rect getROI(const int index);

		TileDivision();
		void init(cv::Size imgSize, cv::Size div);
		TileDivision(cv::Size imgSize, cv::Size div);

		bool compute(const int width_step_, const int height_step_);

	};

	class  TileParallelBody
	{
		cp::TileDivision tdiv;
		cv::Size div;
		cv::Size tileSize;

		void init(const cv::Size div);
	protected:
		virtual void process(const cv::Mat& src, cv::Mat& dst, const int threadIndex, const int imageIndex) = 0;
		std::vector<cv::Mat> srcTile;
		std::vector<cv::Mat> dstTile;
		std::vector<cv::Mat> guideMaps;
		std::vector<std::vector<cv::Mat>> guideTile;
		int threadMax = omp_get_max_threads();
		bool isUseGuide = false;
		void initGuide(const cv::Size div, std::vector<cv::Mat>& guide);
	public:

		void invoker(const cv::Size div, const cv::Mat& src, cv::Mat& dst, const int tileBoundary, const int borderType = cv::BORDER_DEFAULT, const int depth = -1);
		void unsetUseGuide();
		cv::Size getTileSize();
		//void printParameter();
	};
}