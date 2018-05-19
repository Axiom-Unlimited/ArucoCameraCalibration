#pragma once

#include "opencv2/core.hpp"
#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>
#include <memory>
#include "DataStructs.h";
#include <thread>
#include "CVBlobDetector.h"
#include <mutex>
#include <atomic>
#include <future>

using namespace structs;
using namespace cvblobdetector;

class Camera
{
private:
	// flags
	bool SETTINGS_SET = false;
	// mutexes
	std::mutex mtx;

	int _captureId;
	bool halt;
	cv::VideoCapture							_videoCapture;

	std::vector<std::shared_ptr<Camera>>		_pCameras{};

	cv::Mat										_intrinsicParams;
	cv::Mat										_distCoeffs;
	cv::Mat										_rotationMat;
	cv::Mat									_translationVecs;

	cv::Mat										_currentFrame;

	std::vector<RelationToNCamera>				_cameraRelations{};	

	cv::FileStorage								_fileStorage;

	std::string									_configFile;
	std::future<void>							_futureTask;
	std::mutex									_mutex;


	void capture_frames();

public:
	
	// getters for flags;
	void settingsSet();
	bool ifSettingsSet() const;
	// constructor and destructor
	Camera(int captureId);

	cv::Mat &getIntrinsicParams();
	cv::Mat &getDistCoeffs();
	cv::Mat &getRotMat();
	cv::Mat &getTransVec();

	bool getCorrectedFrames(cv::Mat &mat);

	cv::VideoCapture &getVideoCap();

	void addRelation(RelationToNCamera relToCam);

	int getCapID() const;

	void setRotMat(cv::Mat& rot);

	void setTranVec(cv::Vec3d& vec);

	void setDistCoeff(cv::Mat& dist);

	void setIntrMat(cv::Mat& intr);

	void getCameraRelations(std::vector<RelationToNCamera>& rel) const;

	void setConfigFile(std::string file);

	void saveConfig();

	void haltThread() ;

	void resumeThread();
};
