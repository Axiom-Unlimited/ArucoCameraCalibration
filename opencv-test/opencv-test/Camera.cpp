#include "stdafx.h"
#include <memory>
#include <iostream>

#include "Camera.h"
#include "opencv2\core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/features2d.hpp"
#include "boost\filesystem.hpp"
#include "MathUtil.h"
#include "opencv2/aruco.hpp"

#define SHOW_BLOBS 1
#define SHOW_CAPTURE 1

using math = mathutil::MathUtil;

Camera::Camera(int captureId, cv::SimpleBlobDetector::Params blobParams) : _captureId(captureId),
								_videoCapture(cv::VideoCapture(captureId)),
								_intrinsicParams(cv::Mat(3, 3, CV_32FC1)),
								_blobDetector(blobParams),
								rot_to_AR(cv::Mat::zeros(0,0,CV_64F))
{
	_futureTask = std::async(std::launch::async | std::launch::deferred, &Camera::CameraRun, this);
}

cv::Mat& Camera::getIntrinsicParams()
{
	return _intrinsicParams;
}

cv::Mat& Camera::getDistCoeffs()
{
	return _distCoeffs;
}

cv::Mat& Camera::getRotMat()
{
	return _rotationMat;
}

cv::Vec3d& Camera::getTransVec()
{
	return _translationVecs;
}

void Camera::getKeyPoints(std::vector<cv::Point2f>& keypoints)
{
	mtx.lock();
	keypoints = _keyPoints;
	mtx.unlock();
}

cv::VideoCapture& Camera::getVideoCap()
{
	return  _videoCapture;
}

bool Camera::addBLobsToFrame(cv::Mat& imgOut, std::vector<cv::KeyPoint> blobPoints)
{
	_mutex.lock();
	if (_keyPoints.size() > 0 && imgOut.cols > 0 && imgOut.rows > 0)
	{
		cv::drawKeypoints(imgOut, blobPoints,imgOut);
		_mutex.unlock();
		return true;
	}
	_mutex.unlock();
	return false;
}

void Camera::drawEpilines(cv::Mat& input, std::vector<cv::Vec3f>& epilines)
{

	mtx.lock();
	if (epilines.size() > 0)
	{
		cv::RNG rng(0);
		for (auto i = 0; i < epilines.size(); i++)
		{
			cv::Scalar color(rng(256), rng(256), rng(256));

			cv::line(input,
				cv::Point(0, -epilines[i][2] / epilines[i][1]),
				cv::Point(input.cols, -(epilines[i][2] + epilines[i][0] * input.cols) / epilines[i][1]), color);
		}
	}
	mtx.unlock();
}

void Camera::settingsSet()
{
	SETTINGS_SET = true;
}

bool Camera::ifSettingsSet()
{
	return SETTINGS_SET;
}

void Camera::addRelation(RelationToNCamera relToCam)
{
	_mutex.lock();
	_cameraRelations.push_back(relToCam);
	_mutex.unlock();
}

int Camera::getCapID()
{
	return _captureId;
}

void Camera::setRotMat(cv::Mat& rot)
{
	_rotationMat = rot;
}

void Camera::setTranVec(cv::Vec3d& vec)
{
	_translationVecs = vec;
}

void Camera::setDistCoeff(cv::Mat& dist)
{
	_distCoeffs = dist;
}

void Camera::setIntrMat(cv::Mat& intr)
{
	_intrinsicParams = intr;
}

void Camera::getCameraRelations(std::vector<RelationToNCamera>& rel)
{
	rel = _cameraRelations;
}

void Camera::setConfigFile(std::string file) 
{
	_configFile = file;

	if (!boost::filesystem::exists(file))
	{
		std::cout << "File does not exist, so it was created. You must configure and save camera relations for camera " << std::to_string(_captureId) << ". \n";
		boost::filesystem::ofstream(file);
	}
	else
	{
		if(!_fileStorage.open(_configFile, cv::FileStorage::READ))
		{
			std::cout << "config file empty recalibrate and save configuration. \n";
			return;
		}

		// get the cameras relationship data for all other cameras in the system.
		cv::FileNode relNodes = _fileStorage["Relations"];
		for (auto it = relNodes.begin();it != relNodes.end();++it)
		{
			auto node = *it;
			std::cout << node.name() << "\n";
			if (node.isMap())
			{
				RelationToNCamera relToCam;
				node["Designator"] >> relToCam.Designator;
				node["RotationTo"] >> relToCam.RotationTo;
				node["TranslationTo"] >> relToCam.TranslationTo;
				node["FundamentalMatTo"] >> relToCam.FundamentalMatTo;
				node["EssentialMatTo"] >> relToCam.EssentialMatTo;
				_cameraRelations.push_back(relToCam);
			}
		}

		//get the cameras calibrated data
		cv::FileNode settings = _fileStorage["Settings"];
		settings["intrinsicParam"] >> _intrinsicParams;
		settings["distCoeffs"] >> _distCoeffs;
		settings["rotationMat"] >> _rotationMat;
		settings["translationVecs"] >> _translationVecs;
		_fileStorage.release();
		SETTINGS_SET = true;
	}
}

void Camera::saveConfig()
{
	_fileStorage.open(_configFile, cv::FileStorage::APPEND);
	cv::FileNode topParent = _fileStorage.getFirstTopLevelNode();
	_fileStorage << "Relations" << "{";
	for (auto it = _cameraRelations.begin();it != _cameraRelations.end();++it)
	{
		std::stringstream ss;
		ss << "CameraRelation" << std::to_string(it->Designator);
		_fileStorage << ss.str();
		_fileStorage << "{" << "Designator" << it->Designator << "RotationTo" << it->RotationTo;
		_fileStorage << "TranslationTo" << it->TranslationTo;
		_fileStorage << "FundamentalMatTo" << it->FundamentalMatTo;
		_fileStorage << "EssentialMatTo" << it->EssentialMatTo << "}";
	}
	_fileStorage << "}";
	_fileStorage << "Settings" << "{";
	_fileStorage << "intrinsicParam" << _intrinsicParams;
	_fileStorage << "distCoeffs" << _distCoeffs;
	_fileStorage << "rotationMat" << _rotationMat;
	_fileStorage << "translationVecs" << _translationVecs;
	_fileStorage << "}";
	_fileStorage.release();
	SETTINGS_SET = true;
}

void Camera::addEpilines(std::vector<cv::Vec3f>& epilines)
{
	_mutex.lock();
	_epilines = epilines;
	_mutex.unlock();
}

void Camera::haltThread()
{
	_futureTask.wait();
}

void Camera::reseumeThread()
{
	_futureTask.get();
}

// main thread for the Camera
void Camera::CameraRun()
{
	std::vector<cv::KeyPoint> localKeyPoint;
	std::vector<cv::Point2f> localPoints;
	while (_videoCapture.isOpened())
	{
		_videoCapture >> _currentFrame;

		mtx.lock();
		_currentFrame.copyTo(_blobFrame);
		_blobDetector.detectBLobs(_currentFrame, localKeyPoint);
		//undistort blob positions
		cv::KeyPoint::convert(localKeyPoint, _keyPoints);

		if (SETTINGS_SET)
		{
			_normVectors.clear();
			math::computeNormalVectorsFromPoints(_keyPoints,_intrinsicParams ,_normVectors);
		}
		mtx.unlock();

		if(SHOW_CAPTURE)
		{
			if (SHOW_BLOBS)
			{
				addBLobsToFrame(_blobFrame, localKeyPoint);
				//get the keypoints and normal vectors
				const auto numOfBLobs = _keyPoints.size();
				const auto numOFNormalVec = _normVectors.size();
				const auto numOfEpilines = _epilines.size();
				// burn keypoint # and normal vector# into frame
				cv::putText(_blobFrame, std::string("Blob#: " + std::to_string(numOfBLobs) 
					+ " NormVec: " + std::to_string(numOFNormalVec) 
					+ " Epilines: " + std::to_string(numOfEpilines)), cv::Point(30, 30), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0, 0, 255), 1.5);
				//draw epilines to frame
				drawEpilines(_blobFrame,_epilines);
				//combine blob frame with raw frame
				const auto rows = cv::max(_currentFrame.rows, _blobFrame.rows);
				const auto cols = _currentFrame.cols + _blobFrame.cols;
				cv::Mat3b res(rows, cols, cv::Vec3b(0, 0, 0));
				_currentFrame.copyTo(res(cv::Rect(0, 0, _currentFrame.cols, _currentFrame.rows)));
				_blobFrame.copyTo(res(cv::Rect(_currentFrame.cols, 0, _blobFrame.cols, _blobFrame.rows)));
				cv::imshow(std::string("Main Feed Camera: " + std::to_string(_captureId)), res);
			}
			else
			{
				cv::imshow(std::string("Main Feed Camera: " + std::to_string(_captureId)), _currentFrame);
			}
		}
		localKeyPoint.clear();
		localPoints.clear();
		cv::waitKey(1);
	}
}
















