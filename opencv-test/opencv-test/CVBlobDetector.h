#pragma once
#include <memory>
#include "opencv2/core.hpp"
#include <opencv2/features2d.hpp>

namespace cvblobdetector
{
	class CVBLobeDetector
	{
	private:
		cv::Ptr<cv::SimpleBlobDetector> _blobDetector;
	public:
		CVBLobeDetector(cv::SimpleBlobDetector::Params& param) : _blobDetector(cv::SimpleBlobDetector::create(param))
		{}

		void detectBLobs( cv::Mat image, std::vector<cv::KeyPoint>& keyPoints) const
		{
			//cv::SimpleBlobDetector::Params blobParams;
			//blobParams.filterByCircularity = true;
			//blobParams.minCircularity = 0.7;
			//blobParams.maxCircularity = 1.0;
			_blobDetector->detect(image, keyPoints);
		}
	};
}
