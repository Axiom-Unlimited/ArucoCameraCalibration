#pragma once

#include <memory>

#include "opencv2/core.hpp"

#include "Camera.h"


class CameraManager
{
private:
	std::vector<std::shared_ptr<Camera>> _cameras;

	void cameraManagerRun();
public:
};
