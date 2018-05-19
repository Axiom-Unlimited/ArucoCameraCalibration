#pragma once
#include "stdafx.h"
#include <iostream>
#include <memory>

#include <opencv2/core/mat.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/aruco.hpp>
#include <opencv2/aruco/charuco.hpp>
#include "Camera.h"

namespace mathutil
{
	class MathUtil
	{
	public :

		struct ChArucoParams
		{
			int			numXSquares			= 0;
			int			numYSquares			= 0;
			double		squareLength		= 0;
			double		markerLength		= 0;
			int			dictionary			= cv::aruco::DICT_4X4_50;
			cv::Size	imageSize			= cv::Size(0,0);
			int			calibrationFlags	= 0;
			int			waitTime			= 1;
		};

		static double monoChAucoCalibrateCamera(ChArucoParams params
												, cv::VideoCapture &videoCap
												, cv::Mat &intrinsicParams
												, cv::Mat &distCoeffs
												, cv::Mat &rotationMat
												, cv::Mat &translationVec)
		{
			cv::Ptr<cv::aruco::DetectorParameters> detectorParams = cv::aruco::DetectorParameters::create();
			const auto dictionary = cv::aruco::getPredefinedDictionary(params.dictionary);

			// create charuco board object
			cv::Ptr<cv::aruco::CharucoBoard> charucoboard =
				cv::aruco::CharucoBoard::create(params.numXSquares, params.numYSquares, params.squareLength, params.markerLength, dictionary);

			// collect data from each frame
			std::vector< std::vector< std::vector<cv::Point2f > > > allCorners;
			std::vector< std::vector < int > > allIds;
			std::vector<cv::Mat > allImgs;
			cv::Size imgSize;

			while (videoCap.grab())
			{
				cv::Mat image, imageCopy;
				videoCap.retrieve(image);

				std::vector< int > ids;
				std::vector<std::vector<cv::Point2f > > corners, rejected;

				// detect markers
				detectMarkers(image, dictionary, corners, ids, detectorParams, rejected);

				// interpolate charuco corners
				cv::Mat currentCharucoCorners, currentCharucoIds;
				if (ids.size() > 0)
					interpolateCornersCharuco(corners, ids, image, charucoboard, currentCharucoCorners,
						currentCharucoIds);

				// draw results
				image.copyTo(imageCopy);
				if (ids.size() > 0) cv::aruco::drawDetectedMarkers(imageCopy, corners);

				if (currentCharucoCorners.total() > 0)
					cv::aruco::drawDetectedCornersCharuco(imageCopy, currentCharucoCorners, currentCharucoIds);

				putText(imageCopy, "Press 'c' to add current frame. 'ESC' to finish and calibrate",
					cv::Point(10, 20), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 0), 2);

				cv::imshow("Calibration Window", imageCopy);
				const auto key = static_cast<char>(cv::waitKey(params.waitTime));
				if (key == 27)// the esc key
				{
					cv::destroyWindow("Calibration Window");
					break;
				}
				if (key == 'c' && ids.size() > 0)
				{
					std::cout << "Frame captured" << std::endl;
					allCorners.push_back(corners);
					allIds.push_back(ids);
					allImgs.push_back(image);
					imgSize = image.size();
				}
			}

			// prepare data for charuco calibration
			auto nFrames = static_cast<int>(allCorners.size());
			std::vector<cv::Mat > allCharucoCorners;
			std::vector<cv::Mat > allCharucoIds;
			std::vector<cv::Mat > filteredImages;
			allCharucoCorners.reserve(nFrames);
			allCharucoIds.reserve(nFrames);

			cv::Mat cameraMatrix, distCo;
			std::vector<cv::Mat > rvecs, tvecs;

			for (int i = 0; i < nFrames; i++)
			{
				// interpolate using camera parameters
				cv::Mat currentCharucoCorners, currentCharucoIds;
				interpolateCornersCharuco(allCorners[i], allIds[i], allImgs[i], charucoboard,
					currentCharucoCorners, currentCharucoIds, cameraMatrix,
					distCo);

				allCharucoCorners.push_back(currentCharucoCorners);
				allCharucoIds.push_back(currentCharucoIds);
				filteredImages.push_back(allImgs[i]);
			}

			// calibrate camera using charuco
			const auto repError = calibrateCameraCharuco(allCharucoCorners, allCharucoIds, charucoboard, imgSize, cameraMatrix, distCo, rvecs, tvecs, params.calibrationFlags);
			std::cout << "cameraMatrix = " << std::endl << cameraMatrix << std::endl << std::endl;
			cameraMatrix.copyTo(intrinsicParams);
			distCoeffs.copyTo(distCoeffs);

			cv::Mat rsum = cv::Mat::zeros(3, 1, CV_64F);
			for (const auto r : rvecs)
			{
				rsum += r;
			}
			const cv::Mat rMean = rsum / rvecs.size();
			cv::Rodrigues(rMean, rotationMat);
			std::cout << "mean rotation mat: " << std::endl << rotationMat << std::endl << std::endl << std::endl;

			cv::Mat tsum = cv::Mat::zeros(3, 1, CV_64F);
			for (const auto t : tvecs)
			{
				tsum += t;
			}
			translationVec = tsum / tvecs.size();
			std::cout << "mean translation mat: " << std::endl << translationVec << std::endl << std::endl << std::endl;

			return repError;
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

	};


}
