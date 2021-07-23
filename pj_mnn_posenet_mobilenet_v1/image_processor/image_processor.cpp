/*** Include ***/
/* for general */
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>
#include <array>
#include <algorithm>
#include <chrono>
#include <fstream>
#include <memory>

/* for OpenCV */
#include <opencv2/opencv.hpp>

/* for My modules */
#include "CommonHelper.h"
#include "PoseEngine.h"
#include "ImageProcessor.h"

/*** Macro ***/
#define TAG "ImageProcessor"
#define PRINT(...)   COMMON_HELPER_PRINT(TAG, __VA_ARGS__)
#define PRINT_E(...) COMMON_HELPER_PRINT_E(TAG, __VA_ARGS__)

/*** Global variable ***/
std::unique_ptr<PoseEngine> s_engine;

/*** Function ***/
static cv::Scalar createCvColor(int32_t b, int32_t g, int32_t r) {
#ifdef CV_COLOR_IS_RGB
	return cv::Scalar(r, g, b);
#else
	return cv::Scalar(b, g, r);
#endif
}


int32_t ImageProcessor_initialize(const INPUT_PARAM* inputParam)
{
	if (s_engine) {
		PRINT_E("Already initialized\n");
		return -1;
	}

	s_engine.reset(new PoseEngine());
	if (s_engine->initialize(inputParam->workDir, inputParam->numThreads) != PoseEngine::RET_OK) {
		return -1;
	}
	return 0;
}

int32_t ImageProcessor_finalize(void)
{
	if (!s_engine) {
		PRINT_E("Not initialized\n");
		return -1;
	}

	if (s_engine->finalize() != PoseEngine::RET_OK) {
		return -1;
	}

	return 0;
}


int32_t ImageProcessor_command(int32_t cmd)
{
	if (!s_engine) {
		PRINT_E("Not initialized\n");
		return -1;
	}

	switch (cmd) {
	case 0:
	default:
		PRINT_E("command(%d) is not supported\n", cmd);
		return -1;
	}
}


int32_t ImageProcessor_process(cv::Mat* mat, OUTPUT_PARAM* outputParam)
{
	if (!s_engine) {
		PRINT_E("Not initialized\n");
		return -1;
	}

	cv::Mat& originalMat = *mat;
	PoseEngine::RESULT result;
	if (s_engine->invoke(originalMat, result) != PoseEngine::RET_OK) {
		return -1;
	}

	/* Draw the result */
	for (const auto& body : result.poseKeypointCoords) {
		for (const auto& part : body) {
			cv::circle(originalMat, cv::Point(static_cast<int32_t>(part.first), static_cast<int32_t>(part.second)), 5, cv::Scalar(0, 255, 0), -1);
		}
	}

	/* Return the results */
	outputParam->timePreProcess = result.timePreProcess;
	outputParam->timeInference = result.timeInference;
	outputParam->timePostProcess = result.timePostProcess;

	return 0;
}

