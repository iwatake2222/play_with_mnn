/*** Include ***/
/* for general */
#include <stdio.h>
#include <algorithm>
#include <fstream>
#include <functional>
#include <memory>
#include <sstream>
#include <vector>
#include <chrono>

/* for OpenCV */
#include <opencv2/opencv.hpp>

#include "ImageProcessor.h"

/*** Macro ***/
/* Model parameters */
#define MODEL_NAME   RESOURCE_DIR"/model/mobilenet_v2_1.0_224.mnn"
#define IMAGE_NAME   RESOURCE_DIR"/parrot.jpg"

/* Settings */
#define LOOP_NUM_FOR_TIME_MEASUREMENT 10

int main(int argc, const char* argv[])
{
	/*** Initialize ***/
	cv::Mat originalImage = cv::imread(IMAGE_NAME);

	/* Initialize image processor library */
	ImageProcessor_initialize(MODEL_NAME);

	/* Call image processor library */
	ImageProcessor_process(&originalImage);

	cv::imshow("originalImage", originalImage);

	/*** (Optional) Measure inference time ***/
	const auto& t0 = std::chrono::steady_clock::now();
	for (int i = 0; i < LOOP_NUM_FOR_TIME_MEASUREMENT; i++) {
		ImageProcessor_process(&originalImage);
	}
	const auto& t1 = std::chrono::steady_clock::now();
	std::chrono::duration<double> timeSpan = t1 - t0;
	printf("Inference time = %f [msec]\n", timeSpan.count() * 1000.0 / LOOP_NUM_FOR_TIME_MEASUREMENT);

	/* Fianlize image processor library */
	ImageProcessor_finalize();

	cv::waitKey(-1);

	return 0;
}
