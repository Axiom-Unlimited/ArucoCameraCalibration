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

Camera::Camera(int captureId, int markerSize, cv::Ptr<cv::aruco::Dictionary> dictionary, cv::Ptr<cv::aruco::DetectorParameters> detector_parameters) :
								  _aurcoDictionary(dictionary)
								, _captureId(captureId)	
								, _markerSize(markerSize)
								, _detectorParams(detector_parameters)
								, halt(false)
								,_videoCapture(cv::VideoCapture(captureId))
								,_intrinsicParams(cv::Mat(3, 3, CV_32FC1))
{
	_futureTask = std::async(std::launch::async | std::launch::deferred, &Camera::capture_frames, this);
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

cv::Mat& Camera::getTransVec()
{
	return _translationVecs;
}

cv::VideoCapture& Camera::getVideoCap()
{
	return  _videoCapture;
}


void Camera::settingsSet()
{
	SETTINGS_SET = true;
}

bool Camera::ifSettingsSet() const
{
	return SETTINGS_SET;
}

void Camera::addRelation(RelationToNCamera relToCam)
{
	_mutex.lock();
	_cameraRelations.push_back(relToCam);
	_mutex.unlock();
}

int Camera::getCapID() const
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

void Camera::getCameraRelations(std::vector<RelationToNCamera>& rel) const
{
	rel = _cameraRelations;
}

void Camera::setConfigFile(std::string file) 
{
	_configFile = file;

	if (!boost::filesystem::exists(file))
	{
		std::cout << "File does not exist, so it was created. You must configure and save camera relations for camera " << std::to_string(_captureId) << ". \n";
		boost::filesystem::ofstream(_configFile);
	}
	else
	{
		if(!_fileStorage.open(_configFile, cv::FileStorage::READ))
		{
			std::cout << "config file empty recalibrate and save configuration. \n";
			return;
		}

		std::cout << "config file found!!! \n";

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

void Camera::haltThread() 
{
	halt = true;
}

void Camera::resumeThread()
{
	halt = false;
}

// main thread for the Camera
void Camera::capture_frames()
{
	while (_videoCapture.isOpened())
	{
		if (halt){ continue;}

		_videoCapture >> _currentFrame;

		if (SETTINGS_SET)
		{
			std::vector<std::vector<cv::Point2f > > corners;
			std::vector< int > ids;
			detectMarkers(_currentFrame, _aurcoDictionary, corners, ids, _detectorParams);
			if (corners.size() > 0 && ids.size() > 0)
			{
				std::vector<cv::Vec3d > rvecs, tvecs;

				cv::aruco::estimatePoseSingleMarkers(corners, _markerSize, _intrinsicParams, _distCoeffs, rvecs, tvecs);
				cv::aruco::drawDetectedMarkers(_currentFrame, corners, ids);
				cv::Vec3d rsum;
				for (const auto r : rvecs)
				{
					rsum += r;
				}
				rsum = rsum / static_cast<double>(rvecs.size());

				cv::Vec3d tsum;
				for (const auto t : tvecs)
				{
					tsum += t;
				}
				tsum = tsum / static_cast<double>(tvecs.size());
				cv::aruco::drawAxis(_currentFrame, _intrinsicParams, _distCoeffs, rsum, tsum, _markerSize);
			}
		}

		if(SHOW_CAPTURE)
		{
			cv::imshow(std::string("Main Feed Camera: " + std::to_string(_captureId)), _currentFrame);
		}
		cv::waitKey(10);
	}
}
















