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
	cv::VideoCapture							_videoCapture;
	std::vector<cv::Point2f> 					_keyPoints;
	std::vector<cv::Point3d>					_normVectors;
	std::vector<std::shared_ptr<Camera>>		_pCameras;
	std::vector<cv::Vec3f>						_epilines;

	cv::Mat										_intrinsicParams;
	cv::Mat										_distCoeffs;
	cv::Mat										_rotationMat;
	cv::Vec3d									_translationVecs;

	cv::Mat										_currentFrame;
	cv::Mat										_blobFrame;

	std::vector<RelationToNCamera>				_cameraRelations;	
	std::vector<structs::EpiPair<cv::Vec3d>>	_camRelEpilines;

	cv::FileStorage								_fileStorage;

	std::string									_configFile;
	std::future<void>							_futureTask;
	std::mutex									_mutex;

	CVBLobeDetector								_blobDetector;


	void CameraRun();
	bool addBLobsToFrame(cv::Mat& imgOut,std::vector<cv::KeyPoint> blobPoints);
	void drawEpilines(cv::Mat& input, std::vector<cv::Vec3f>& epilines);

public:

	std::atomic<cv::Mat> rot_to_AR;

	// getters for flags;
	void settingsSet();
	bool ifSettingsSet();
	// constructor and destructor
	Camera(int captureId, cv::SimpleBlobDetector::Params blobParams);

	cv::Mat &getIntrinsicParams();
	cv::Mat &getDistCoeffs();
	cv::Mat &getRotMat();
	cv::Vec3d &getTransVec();

	void getKeyPoints(std::vector<cv::Point2f>& keypoints);

	bool getCorrectedFrames(cv::Mat &mat);

	cv::VideoCapture &getVideoCap();

	void addRelation(RelationToNCamera relToCam);

	int getCapID();

	void setRotMat(cv::Mat& rot);

	void setTranVec(cv::Vec3d& vec);

	void setDistCoeff(cv::Mat& dist);

	void setIntrMat(cv::Mat& intr);

	void getCameraRelations(std::vector<RelationToNCamera>& rel);

	void setConfigFile(std::string file);

	void saveConfig();

	void addEpilines(std::vector<cv::Vec3f>& epilines);

	void haltThread();

	void reseumeThread();
};
