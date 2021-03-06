// opencv-test.cpp : Defines the entry point for the console application.
//
#include "stdafx.h"
#include <iostream>

#include "Camera.h"
#include "MathUtil.h"

//opencv inlcudes
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/calib3d.hpp"
#include <boost/filesystem.hpp>
#include <boost/date_time.hpp>

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
	std::cout << "press c to halt all camera captures and get record homogeneous transform to marker \n";
	std::cout << "press shift + 0 to see camera 0 dynamics \n";
	std::cout << "press shift + 1 to see camera 1 dynamics \n";
}

int main()
{
	auto createVideo = false;
	auto videoPath = "C:/Users/don_j/OneDrive/Videos";

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
		else if(key == 'c') // halts all of the camera captures
		{
			for (auto cam : cameras) { cam->haltThread(); };
			for (auto cam : cameras) { cam->captureMarkerDynamics(); };
			for (auto cam : cameras) { cam->resumeThread(); };
		}
		else if(key == 'r') // calculates all of the homogeneous coordinates to all other cameras
		{
			for (auto outCam : cameras)
			{
				for (auto inCam : cameras)
				{
					if (outCam->getCapID() != inCam->getCapID())
					{
						//get approximate homogeneous transform to scene for the outer Camera
						cv::Mat outCamTransfromToScene(4,4,CV_64F);
						for (auto transform : outCam->getMarkerTransfroms())
						{
							outCamTransfromToScene += transform;
						}
						outCamTransfromToScene = outCamTransfromToScene / outCam->getMarkerTransfroms().size();

						//get approximate homogeneous transform to scene for the inner camera
						cv::Mat inCamTransformToScene(4, 4, CV_64F);
						for(auto transform : inCam->getMarkerTransfroms())
						{
							inCamTransformToScene += transform;
						}
						inCamTransformToScene = inCamTransformToScene / inCam->getMarkerTransfroms().size();

						// pull rotation and translation out of inner camera homogeneous transfrom
						cv::Mat inRot = inCamTransformToScene(cv::Rect(0,0,3,3));
						cv::Mat inTrans = inCamTransformToScene.col(3);

						// flip inner camera rotation
						cv::Mat inFlippedRot;
						cv::transpose(inRot,inFlippedRot);
						// flip inner camera translation
						cv::Mat inFlippedTrans = -1 * (inFlippedRot * inTrans);

						// create new homogeneous transform for flipped inner camera transform
						cv::Mat inFlippedHomogTrans(4, 4, CV_64F);
						inFlippedRot.copyTo(inFlippedRot(cv::Rect(0, 0, 3, 3)));
						inFlippedTrans.copyTo(inFlippedHomogTrans(cv::Rect(3, 0, 1, 3)));

						// calculate transform from outer camera to inner camera
						cv::Mat transformOutToIn = outCamTransfromToScene * inFlippedHomogTrans;

						RelationToNCamera newRel;
						newRel.Designator = inCam->getCapID();
						transformOutToIn.copyTo(newRel.HomogeneousTransformTo);

						outCam->addRelation(newRel);
					}
				}
			}
		}
		else if(key == 'v') // create video 
		{
			if (!createVideo)
			{
				createVideo = true;

			}
			else
			{
				createVideo = false;
			}

			auto videoTask = std::async(std::launch::deferred | std::launch::async, [&cameras, &createVideo, &videoPath]()
			{
				std::cout << "creating video!!!!\n";
				cv::VideoWriter video_writer;
				cv::Mat frame1;
				cv::Mat frame2;
				int rows = 0;
				int cols = 0;
				cv::Mat3b res;
				auto time = boost::posix_time::second_clock::local_time().time_of_day();
				auto timeString = std::to_string(time.total_microseconds());

				while (createVideo)
				{
					cameras[0]->getCurrentFrames(frame1);
					cameras[1]->getCurrentFrames(frame2);
					if (!video_writer.isOpened())
					{
						std::stringstream ss;
						ss << videoPath << "/";
						ss << "Video" << timeString << "microseconds.avi";
						rows = cv::max(frame1.rows, frame2.rows);
						cols = frame1.cols + frame2.cols;
						if(!video_writer.open(ss.str(), cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 30, cv::Size(cols, rows)))
						{
							std::cout << "failed to create video at time: " << timeString << std::endl;
							std::cout << "at: " << ss.str() << std::endl;
							break;
						}
						res = cv::Mat3b(rows, cols, cv::Vec3b(0, 0, 0));
					}

					frame1.copyTo(res(cv::Rect(0, 0, frame1.cols, frame1.rows)));
					frame2.copyTo(res(cv::Rect(frame1.cols, 0, frame2.cols, frame2.rows)));
					cv::imshow(timeString,res);
					auto key = cv::waitKey(30);
					video_writer.write(res);
					if (key == 27)
					{
						break;
					}
				
				}
				createVideo = false;
				cv::destroyWindow(timeString);
				video_writer.release();
				return;
			});
		}

	}

	return 0;
}

