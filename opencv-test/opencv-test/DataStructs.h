#pragma once

#include "opencv2/core.hpp"

namespace structs
{
	struct RelationToNCamera
	{
		int Designator;
		cv::Mat HomogeneousTransformTo;
	};

	template<typename T>
	using EpiPair = std::pair<int, std::vector<cv::Vec<T, 3>>>;
}
