// opencv-test.cpp : Defines the entry point for the console application.
//
#include "stdafx.h"
#include <iostream>

#include "Camera.h"
#include "MathUtil.h"

//opencv inlcudes
#include "opencv2\core.hpp"
#include "opencv2\imgproc.hpp"
#include "opencv2\highgui.hpp"
#include "opencv2\features2d.hpp"
#include "opencv2\calib3d.hpp"
#include <boost/predef/other/endian.h>
#include <boost/filesystem.hpp>

// cv::Vec3i inputParams[0] = number of boards
// cv::Vec3i inputParams[1] = number of horizontal corners of checker board
// cv::Vec3i inputParams[2] = number of vertical corners of checker board

using math = mathutil::MathUtil;


void printStruct(std::string name, cv::Mat& in)
{
	auto cols = in.cols;
	auto rows = in.rows;
	//printf("%s \n", name);
	std::cout << name << std::endl;
	for (int i = 0; i < rows; ++i)
	{
		for (int j = 0; j < cols; ++j)
		{
			auto val = in.at<double>(i, j);
			printf("%f, ", static_cast<double>(val));
		}
		printf("\n");
	}
	printf("\n");
}

template <typename T1, typename T2>
static void drawEpipolarLines(const std::string& title, const cv::Matx<T1, 3, 3> F,
	const cv::Mat& img1, const cv::Mat& img2,
	const std::vector<cv::Point_<T2>> points1,
	const std::vector<cv::Point_<T2>> points2,
	const float inlierDistance = -1,
	const bool = true )
{
	CV_Assert(img1.size() == img2.size() && img1.type() == img2.type());
	cv::Mat outImg(img1.rows, img1.cols * 2, CV_8UC3);
	cv::Rect rect1(0, 0, img1.cols, img1.rows);
	cv::Rect rect2(img1.cols, 0, img1.cols, img1.rows);
	/*
	* Allow color drawing
	*/
	if (img1.type() == CV_8U)
	{
		cv::cvtColor(img1, outImg(rect1), CV_GRAY2BGR);
		cv::cvtColor(img2, outImg(rect2), CV_GRAY2BGR);
	}
	else
	{
		img1.copyTo(outImg(rect1));
		img2.copyTo(outImg(rect2));
	}
	std::vector<cv::Vec<T2, 3>> epilines1, epilines2;
	cv::computeCorrespondEpilines(points1, 1, F, epilines1); //Index starts with 1
	cv::computeCorrespondEpilines(points2, 2, F, epilines2);

	CV_Assert(points1.size() == points2.size() &&
		points2.size() == epilines1.size() &&
		epilines1.size() == epilines2.size());

	cv::RNG rng(0);
	for (size_t i = 0; i<points1.size(); i++)
	{
		if (inlierDistance > 0)
		{
			if (mathutil::MathUtil::distancePointLine(points1[i], epilines2[i]) > inlierDistance ||
				mathutil::MathUtil::distancePointLine(points2[i], epilines1[i]) > inlierDistance)
			{
				//The point match is no inlier
				continue;
			}
		}
		/*
		* Epipolar lines of the 1st point set are drawn in the 2nd image and vice-versa
		*/
		cv::Scalar color(rng(256), rng(256), rng(256));

		cv::line(outImg(rect2),
			cv::Point(0, -epilines1[i][2] / epilines1[i][1]),
			cv::Point(img1.cols, -(epilines1[i][2] + epilines1[i][0] * img1.cols) / epilines1[i][1]),
			color);
		cv::circle(outImg(rect1), points1[i], 3, color, -1, CV_AA);

		cv::line(outImg(rect1),
			cv::Point(0, -epilines2[i][2] / epilines2[i][1]),
			cv::Point(img2.cols, -(epilines2[i][2] + epilines2[i][0] * img2.cols) / epilines2[i][1]),
			color);
		cv::circle(outImg(rect2), points2[i], 3, color, -1, CV_AA);
	}
	cv::imshow(title, outImg);
	cv::waitKey(1);
}

static void help()
{
	std::cout << "press 0 to calibrate camera 0 \n";
	std::cout << "press 1 to calibrate camera 1 \n";
	std::cout << "press h to show help menu \n";
	std::cout << "press s to save settings \n";
	std::cout << "press shift + 0 to see camera 0 dynamics \n";
	std::cout << "press shift + 1 to see camera 1 dynamics \n";
}

//cv::Vec3i camCalibParam;
//// num of boardse
//camCalibParam[0] = 5;
//// num of horizontal corners
//camCalibParam[1] = 9;
//// num of vertical corners
//camCalibParam[2] = 6;
int main()
{
	auto showVideo = false;
	auto createVideo = false;


	std::vector<std::shared_ptr<Camera>> cameras;
	std::shared_ptr<Camera> camera1 = std::make_shared<Camera>(0);
	std::shared_ptr<Camera> camera2 = std::make_shared<Camera>(1);
	if (!boost::filesystem::is_directory("./CameraSettings"))
	{
		boost::filesystem::create_directory("./CameraSettings");
		camera1->setConfigFile("./CameraSettings/Camera1Config.xml");
		camera2->setConfigFile("./CameraSettings/Camera2Config.xml");
	}
	else
	{
		camera1->setConfigFile("./CameraSettings/Camera1Config.xml");
		camera2->setConfigFile("./CameraSettings/Camera2Config.xml");
	}


	cameras.push_back(camera1);
	cameras.push_back(camera2);

	std::future<void> epilineTask;
	auto index = 0;

	auto calibParams = math::ChArucoParams();
	calibParams.numXSquares = 5;
	calibParams.numYSquares = 7;
	//calibParams.squareLength = 200;
	//calibParams.markerLength = 120;
	calibParams.squareLength = 25;
	calibParams.markerLength = 15;
	
	help();
	while (true)
	{	
		char key;
		std::cin >> key;
		if (key == '0') // press 0 to calibrate camera 0
		{
			cameras[0]->haltThread();
			math::monoChAucoCalibrateCamera(calibParams, cameras[0]->getVideoCap(), cameras[0]->getIntrinsicParams()
				, cameras[0]->getDistCoeffs(), cameras[0]->getRotMat(), cameras[0]->getTransVec());
			cameras[0]->settingsSet();
			cameras[0]->resumeThread();
		}
		else if (key == '1') // press 1 to calibrate camera 1
		{
			cameras[1]->haltThread();
			math::monoChAucoCalibrateCamera(calibParams, cameras[1]->getVideoCap(), cameras[1]->getIntrinsicParams()
				, cameras[1]->getDistCoeffs(), cameras[1]->getRotMat(), cameras[1]->getTransVec());
			cameras[1]->settingsSet();
			cameras[1]->resumeThread();
		}
		else if (key == 's' || key == 'S') // press s to save settings
		{
			std::for_each(cameras.begin(), cameras.end(), [](std::shared_ptr<Camera> cam) {cam->saveConfig(); });
			std::cout << "all settings saved \n";
		}
		else if(key == 'h')
		{
			help();
		}
		else if (key == ')') // press shift + 0 to see camera 0 dynamics
		{
			const auto intrinsic = cameras[0]->getIntrinsicParams();
			const auto rot = cameras[0]->getRotMat();
			const auto trans = cameras[0]->getTransVec();
			std::cout << "=======================CAMERA 0 DYNAMICS=====================================\n";
			std::cout << "cameraMatrix = " << std::endl << intrinsic << std::endl << std::endl;
			std::cout << "mean rotation mat: " << std::endl << rot << std::endl << std::endl << std::endl;
			std::cout << "mean translation mat: " << std::endl << trans << std::endl << std::endl << std::endl;
		
		}
		else if(key == '!') // press shift + 1 to see camera 1 dynamics
		{
			const auto intrinsic = cameras[1]->getIntrinsicParams();
			const auto rot = cameras[1]->getRotMat();
			const auto trans = cameras[1]->getTransVec();
			std::cout << "=======================CAMERA 1 DYNAMICS=====================================\n";
			std::cout << "cameraMatrix = " << std::endl << intrinsic << std::endl << std::endl;
			std::cout << "mean rotation mat: " << std::endl << rot << std::endl << std::endl << std::endl;
			std::cout << "mean translation mat: " << std::endl << trans << std::endl << std::endl << std::endl;
		}
		else if(key == 'v') // create video 
		{
				auto epilineTask = std::async(std::launch::async,[&cameras]()
				{
					std::cout << "show videos!!!!\n";
					while (true)
					{
			
					}
				});
		}
		//else if (key == 'e') // press e to begin epiline calculation thread
		//{
		//	epilineTask = std::async(std::launch::async,[&cameras]()
		//	{
		//		std::cout << "epiline task started \n";
		//		while (true)
		//		{
		//			if (cameras[0]->ifSettingsSet() && cameras[1]->ifSettingsSet())
		//			{
		//				//iterate through the cameras
		//				for (auto out_it = cameras.begin(); out_it != cameras.end(); ++out_it)
		//				{
		//					// get all of the current cameras relationships
		//					std::vector<RelationToNCamera> camRel;
		//					(*out_it)->getCameraRelations(camRel);
		//					// iterate through the camera relations
		//					for (auto in_it = camRel.begin(); in_it != camRel.end(); ++in_it)
		//					{

		//						std::shared_ptr<Camera> pCam;
		//						auto camRelDes = (*in_it).Designator;
		//						// find and get a reference to the camera associated with the current relationship
		//						std::for_each(cameras.begin(), cameras.end(), [&pCam, &camRelDes](std::shared_ptr<Camera> cam)
		//						{
		//							if (cam->getCapID() == camRelDes)
		//							{
		//								pCam = cam;
		//								return;
		//							}
		//						});
		//						
		//						// vector that will hold our calculated epilines
		//						std::vector<cv::Vec3f> epilines;
		//						// get the keypoints from the camera
		//						std::vector<cv::Point2f> keyPoints;
		//						pCam->getKeyPoints(keyPoints);
		//						math::calcEpilines<float,float>((*in_it).FundamentalMatTo, keyPoints, epilines);
		//						(*out_it)->addEpilines(epilines);
		//					}
		//				}
		//			}
		//		}
		//	});
		//}

	}

	return 0;
}

