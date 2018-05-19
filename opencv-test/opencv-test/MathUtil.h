#pragma once
#include "stdafx.h"
#include <iostream>
#include <memory>

#include <opencv2/core/mat.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>
#include "Camera.h"

namespace mathutil
{
	class MathUtil
	{
	public :
		enum { DETECTION = 0, CAPTURING = 1, SETTING = 2, CALIBRATING = 3 };
		enum Pattern { CHESSBOARD, CIRCLES_GRID, ASYMMETRIC_CIRCLES_GRID };

		static double stereoCalibrateCamera(std::vector<int> &inputParams, std::shared_ptr<Camera> camera1, std::shared_ptr<Camera> camera2, cv::Mat& R, cv::Mat& T, cv::Mat& E, cv::Mat& F)
		{
			// inputParams[0] = number of boards
			// inputParams[1] = number of horizontal corners of checker board
			// inputParams[2] = number of vertical corners of checker board
			// inputParams[3] = square size

			auto numSquares = inputParams[1] * inputParams[2];
			auto board_sz = cv::Size(inputParams[1], inputParams[2]);

			std::vector<std::vector<cv::Point3f>> object_points;
			std::vector<std::vector<cv::Point2f>> imagePoints1, imagePoints2;
			std::vector<cv::Point2f> corners1, corners2;
			std::vector<std::vector<cv::Point2f>> left_img_points, right_img_points;

			cv::Mat img1;
			cv::Mat img2;
			cv::Mat img1_gray;
			cv::Mat img2_gray;

			auto found1 = false, found2 = false;

			for (;;)
			{
				if (!camera1->getVideoCap().read(img1))
				{
					std::cout << "Failed to read frame from Video Capture 1.";
					return -1;
				}

				if (!camera2->getVideoCap().read(img2))
				{
					std::cout << "Failed to read frame from Video Capture 2.";
					return -1;
				}


				found1 = cv::findChessboardCorners(img1, board_sz, corners1,
					CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_NORMALIZE_IMAGE);
				found2 = cv::findChessboardCorners(img2, board_sz, corners2,
					CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_NORMALIZE_IMAGE);

				cv::cvtColor(img1, img1_gray, CV_BGR2GRAY);
				cv::cvtColor(img2, img2_gray, CV_BGR2GRAY);

				if (found1)
				{
					cv::cornerSubPix(img1_gray, corners1, cv::Size(5, 5), cv::Size(-1, -1),
						cv::TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 30, 0.1));
					cv::drawChessboardCorners(img1_gray, board_sz, corners1, found1);
				}
				if (found2)
				{
					cv::cornerSubPix(img2_gray, corners2, cv::Size(5, 5), cv::Size(-1, -1),
						cv::TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 30, 0.1));
					cv::drawChessboardCorners(img2_gray, board_sz, corners2, found2);
				}

				cv::imshow("found1 corners", img1_gray);
				cv::imshow("found2 corners", img2_gray);
				const auto key = cv::waitKey(1);

				std::vector<cv::Point3f> obj;
				for (int j = 0; j<board_sz.height; j++)
				{
					for (int k = 0; k < board_sz.width;k++)
					{
						obj.push_back(cv::Point3f(static_cast<float>(k*inputParams[3]), static_cast<float>(j*inputParams[3]), 0.0f));
					}
				}
					

				if (found1 && found2 /*&& key == 32*/)
				{
					std::cout << "Found corners!" << std::endl;
					imagePoints1.push_back(corners1);
					imagePoints2.push_back(corners2);
					object_points.push_back(obj);
				}

				if (object_points.size() == inputParams[0])
				{
					break;
				}
			}
			// these matricies do nothing because these will have been calculated by monoCalibrateCamera
			cv::Mat camMat1;
			cv::Mat distCoeff1;
			cv::Mat camMat2;
			cv::Mat distCoeff2;
			const auto error = cv::stereoCalibrate(object_points, imagePoints1, imagePoints2, camMat1, distCoeff1, camMat2, distCoeff2, cv::Size(img1.cols, img1.rows), R, T, E, F
											, cv::CALIB_FIX_ASPECT_RATIO +
			                                 cv::CALIB_ZERO_TANGENT_DIST +
			                                 cv::CALIB_USE_INTRINSIC_GUESS +
			                                 cv::CALIB_SAME_FOCAL_LENGTH +
			                                 cv::CALIB_RATIONAL_MODEL +
			                                 cv::CALIB_FIX_K3 + cv::CALIB_FIX_K4 + cv::CALIB_FIX_K5,
			                                 cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 100, 1e-5));
			cv::destroyWindow("found1 corners");
			cv::destroyWindow("found2 corners");

			return error;
		}

		/**
		 * @param inputParams = the configuration parameters that set the number of boards, the number of horizontal and vertical corners
		 * @param videoCap = a reference to the video capture to the camera that needs to be calibrated
		 * @param intrinsicParams = return for the intrinsic paramaters
		 * @param  distCoeffs = return for the distortion coefficients
		 * @param rotationMat = return for the rotation matrix
		 * @param translationMat = return for the Translation vector as mat
		 */
		//static void monoCalibrateCamera(std::vector<int> &inputParams, cv::VideoCapture &videoCap, cv::Mat &intrinsicParams, cv::Mat &distCoeffs, cv::Mat &rotationMat, cv::Vec3d &translationVec)
		//{
		//	// cv::Vec3i inputParams[0] = number of boards
		//	// cv::Vec3i inputParams[1] = number of horizontal corners of checker board
		//	// cv::Vec3i inputParams[2] = number of vertical corners of checker board

		//	int count = 0;
		//	int numSquares = inputParams[1] * inputParams[2];
		//	cv::Size board_sz = cv::Size(inputParams[1], inputParams[2]);

		//	std::vector<std::vector<cv::Point3f>> object_points;
		//	std::vector<std::vector<cv::Point2f>> image_points;

		//	std::vector<cv::Point2f> corners;
		//	int successes = 0;

		//	cv::Mat image;
		//	cv::Mat gray_image;

		//	std::vector<cv::Point3f> obj;
		//	for (int j = 0; j<numSquares; j++)
		//		obj.push_back(cv::Point3f(j / inputParams[1], j%inputParams[1], 0.0f));

		//	videoCap.read(image);

		//	while (successes<inputParams[0])
		//	{

		//		cv::cvtColor(image, gray_image, CV_RGB2GRAY);

		//		bool found = cv::findChessboardCorners(image, board_sz, corners, CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FILTER_QUADS);

		//		if (found)
		//		{
		//			cv::cornerSubPix(gray_image, corners, cv::Size(11, 11), cv::Size(-1, -1), cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.1));
		//			cv::drawChessboardCorners(gray_image, board_sz, corners, found);
		//			successes++;
		//		}

		//		cv::imshow("win1", image);
		//		cv:: imshow("win2", gray_image);

		//		videoCap.read(image);
		//		int key = cv::waitKey(1);

		//		//if (key == 27)
		//		//	return 0;

		//		if (found != 0)
		//		{
		//			image_points.push_back(corners);
		//			object_points.push_back(obj);

		//			printf("Snap stored!\n");

		//			count++;
		//		}

		//		if (key == ' ' || count >= 40)
		//			break;
		//	}

		//	intrinsicParams.ptr<float>(0)[0] = 1;
		//	intrinsicParams.ptr<float>(1)[1] = 1;

		//	std::vector<cv::Mat> rotVecs;
		//	std::vector<cv::Mat> transVecs;

		//	cv::calibrateCamera(object_points, image_points, image.size(), intrinsicParams, distCoeffs, rotVecs, transVecs);

		//	translationVec[0] = transVecs.at(0).at<double>(0, 0);
		//	translationVec[1] = transVecs.at(0).at<double>(1, 0);
		//	translationVec[2] = transVecs.at(0).at<double>(2, 0);

		//	cv::Rodrigues(rotVecs.at(0), rotationMat);
		//	std::cout << "Mono-Configuration complete \n";
		//	cv::destroyWindow("win1");
		//	cv::destroyWindow("win2");
		//}

		static void monoCalibrateCamera(std::vector<int> &inputParams
			, cv::VideoCapture &videoCap
			, cv::Mat &intrinsicParams
			, cv::Mat &distCoeffs
			, cv::Mat &rotationMat
			, cv::Vec3d &translationVec
			, Pattern pattern = CHESSBOARD
			, int flags = 0)
		{
			// cv::Vec3i inputParams[0] = number of boards
			// cv::Vec3i inputParams[1] = number of horizontal corners of checker board
			// cv::Vec3i inputParams[2] = number of vertical corners of checker board
			// cv::Vec3i inputParams[3] = size of square side
			auto boardSize = cv::Size(inputParams[2], inputParams[1]);
			auto squareSize = inputParams[3];
			auto nBoards = inputParams[0];
			auto boardsFound = 0;
			auto aspectRatio = 0;
			cv::Mat cameraMatrix, distCoeff;
			clock_t prevTimestamp = 0;
			int mode = CAPTURING;
			std::vector<std::vector<cv::Point2f> > imagePoints;
			int delay = 1;

			bool undistortImage = false;

			for (auto i = 0;;i++)
			{
				cv::Mat image;
				cv::Mat grayImage;
				bool blink = false;

				videoCap >> image;

				if (aspectRatio == 0)
				{
					aspectRatio = static_cast<float> (image.cols/image.rows);
				}
				cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);
				std::vector<cv::Point2f> pointbuf;

				bool found = false;
				switch (pattern)
				{
				case CHESSBOARD:
					found = cv::findChessboardCorners(image, boardSize, pointbuf,
						cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_FAST_CHECK | cv::CALIB_CB_NORMALIZE_IMAGE);
					break;
				//case CIRCLES_GRID:
				//	found = cv::findCirclesGrid(view, boardSize, pointbuf);
				//	break;
				//case ASYMMETRIC_CIRCLES_GRID:
				//	found = cv::findCirclesGrid(view, boardSize, pointbuf, CALIB_CB_ASYMMETRIC_GRID);
				//	break;
				default:
					std::cout <<  "Unknown pattern type\n";
				}

				// improve the found corners' coordinate accuracy
				if (pattern == CHESSBOARD && found) cornerSubPix(grayImage, pointbuf, cv::Size(11, 11),
					cv::Size(-1, -1),cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.1));



				if (found)
				{
					drawChessboardCorners(image, boardSize, cv::Mat(pointbuf), found);
				}

				if (imagePoints.size() > nBoards)
				{
					mode = CALIBRATING;
				}
				
				cv::imshow("Chestboard", image);
				const auto key = cv::waitKey(1);
				if (mode == CAPTURING && found && key == 32) // 32 is the enter key
				{
					imagePoints.push_back(pointbuf);
					std::cout << "image points pushed \n";
				}

				if (mode == CALIBRATING)
				{
					cv::destroyWindow("Chestboard");
					std::vector<cv::Mat> rvecs;
					std::vector<cv::Mat> tvecs;
					std::vector<float> rProjErr;
					auto totalAvgError = 0.0;
					runCalibration(imagePoints
						, cv::Size(image.cols, image.rows)
						, boardSize
						, pattern
						, squareSize
						, aspectRatio
						, flags
						, intrinsicParams
						, distCoeffs
						, rvecs
						, tvecs
						, rProjErr
						,totalAvgError);
					std::cout << "Total Average Error: " << totalAvgError << "\n";
					translationVec[0] = tvecs.at(0).at<double>(0, 0);
					translationVec[1] = tvecs.at(0).at<double>(1, 0);
					translationVec[2] = tvecs.at(0).at<double>(2, 0);
					cv::Rodrigues(rvecs[0], rotationMat);
					return;
				}
			}
		}

		static void computeNormalVectorsFromPoints(std::vector<cv::Point2f>& imgpnts, cv::Mat& F, std::vector<cv::Point3d>& normVectors)
		{
			cv::Point3d tempPoint;
			if (imgpnts.size() > 0)
			{
				for(cv::Point2f kPnt : imgpnts)
				{
					computeNormalVectorFromPoint(kPnt, F, tempPoint);
					normVectors.push_back(tempPoint);
				}
			}
			else
			{
				//std::cout << "there are not key points in imgpnts \n";
			}
		}

		template <typename T>
		static float distancePointLine(const cv::Point_<T> point, const cv::Vec<T, 3>& line)
		{
			//Line is given as a*x + b*y + c = 0
			return std::fabsf(line(0)*point.x + line(1)*point.y + line(2))
				/ std::sqrt(line(0)*line(0) + line(1)*line(1));
		}

	
		template <typename T1, typename T2>
		static void calcEpilines(const cv::Matx<T1, 3, 3> F
							, std::vector<cv::Point2f> points
							, std::vector<cv::Vec<T2, 3>>& epilines)
		{
			if (points.size() == 0)
			{
				//std::cout << "there are no key points to calculate normal vectors to \n";
				return;
			}
			cv::computeCorrespondEpilines(points, 1, F,epilines);
		}
	
	
		private:
		static void computeNormalVectorFromPoint(cv::Point2f& imgpt, cv::Mat& F, cv::Point3d& dirout)
		{
			cv::Mat homogpt = (cv::Mat_<double>(3, 1) << imgpt.x, imgpt.y, 1.);
			homogpt = F.inv()*homogpt; // If F is not invertible, your calibration is messed up 
			dirout = cv::Point3f(homogpt);
			dirout = dirout / norm(dirout);
		}

		static bool runCalibration(std::vector<std::vector<cv::Point2f> > imagePoints,
			cv::Size imageSize, cv::Size boardSize, Pattern patternType,
			float squareSize, float aspectRatio,
			int flags, cv::Mat& cameraMatrix, cv::Mat& distCoeffs,
			std::vector<cv::Mat>& rvecs, std::vector<cv::Mat>& tvecs,
			std::vector<float>& reprojErrs,
			double& totalAvgErr)
		{
			cameraMatrix = cv::Mat::eye(3, 3, CV_64F);
			if (flags & cv::CALIB_FIX_ASPECT_RATIO)
				cameraMatrix.at<double>(0, 0) = aspectRatio;

			distCoeffs = cv::Mat::zeros(8, 1, CV_64F);

			std::vector<std::vector<cv::Point3f> > objectPoints(1);
			calcChessboardCorners(boardSize, squareSize, objectPoints[0], patternType);

			objectPoints.resize(imagePoints.size(), objectPoints[0]);

			double rms = calibrateCamera(objectPoints, imagePoints, imageSize, cameraMatrix,
				distCoeffs, rvecs, tvecs, flags | cv::CALIB_FIX_K4 | cv::CALIB_FIX_K5);
			///*|CALIB_FIX_K3*/|CALIB_FIX_K4|CALIB_FIX_K5);
			printf("RMS error reported by calibrateCamera: %g\n", rms);

			bool ok = checkRange(cameraMatrix) && checkRange(distCoeffs);

			totalAvgErr = computeReprojectionErrors(objectPoints, imagePoints,
				rvecs, tvecs, cameraMatrix, distCoeffs, reprojErrs);

			return ok;
		}

		static void calcChessboardCorners(cv::Size boardSize, float squareSize, std::vector<cv::Point3f>& corners, Pattern patternType = CHESSBOARD)
		{
			corners.resize(0);

			switch (patternType)
			{
			case CHESSBOARD:
			case CIRCLES_GRID:
				for (int i = 0; i < boardSize.height; i++)
					for (int j = 0; j < boardSize.width; j++)
						corners.push_back(cv::Point3f(float(j*squareSize),
							float(i*squareSize), 0));
				break;

			case ASYMMETRIC_CIRCLES_GRID:
				for (int i = 0; i < boardSize.height; i++)
					for (int j = 0; j < boardSize.width; j++)
						corners.push_back(cv::Point3f(float((2 * j + i % 2)*squareSize),
							float(i*squareSize), 0));
				break;

			default:
				CV_Error(cv::Error::StsBadArg, "Unknown pattern type\n");
			}
		}

		static double computeReprojectionErrors(
			const std::vector<std::vector<cv::Point3f> >& objectPoints,
			const std::vector<std::vector<cv::Point2f> >& imagePoints,
			const std::vector<cv::Mat>& rvecs, const std::vector<cv::Mat>& tvecs,
			const cv::Mat& cameraMatrix, const cv::Mat& distCoeffs,
			std::vector<float>& perViewErrors)
		{
			std::vector<cv::Point2f> imagePoints2;
			int i, totalPoints = 0;
			double totalErr = 0, err;
			perViewErrors.resize(objectPoints.size());

			for (i = 0; i < (int)objectPoints.size(); i++)
			{
				projectPoints(cv::Mat(objectPoints[i]), rvecs[i], tvecs[i],
					cameraMatrix, distCoeffs, imagePoints2);
				err = cv::norm(cv::Mat(imagePoints[i]), cv::Mat(imagePoints2), cv::NORM_L2);
				int n = (int)objectPoints[i].size();
				perViewErrors[i] = (float)std::sqrt(err*err / n);
				totalErr += err * err;
				totalPoints += n;
			}

			return std::sqrt(totalErr / totalPoints);
		}
	};
}
