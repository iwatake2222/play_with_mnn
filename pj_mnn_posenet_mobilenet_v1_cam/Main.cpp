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
#define MODEL_NAME   RESOURCE_DIR"/model/posenet-mobilenet_v1_075.mnn"

int main(int argc, const char* argv[])
{
	/*** Initialize ***/
	/* initialize camera */
	int originalImageWidth = 640;
	int originalImageHeight = 480;

	static cv::VideoCapture cap;
	cap = cv::VideoCapture(0);
	cap.set(cv::CAP_PROP_FRAME_WIDTH, originalImageWidth);
	cap.set(cv::CAP_PROP_FRAME_HEIGHT, originalImageHeight);
	// cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('B', 'G', 'R', '3'));
	cap.set(cv::CAP_PROP_BUFFERSIZE, 1);

	/* Initialize image processor library */
	ImageProcessor_initialize(MODEL_NAME);


	/***** Process for each frame *****/
	while (1) {
		const auto& timeAll0 = std::chrono::steady_clock::now();
		/*** Read image ***/
		const auto& timeCap0 = std::chrono::steady_clock::now();
		//cv::Mat originalImage = cv::imread(IMAGE_NAME);
		cv::Mat originalImage;
		cap.read(originalImage);
		cv::Mat outputImage;
		originalImage.copyTo(outputImage);		// need to copy because OpenCV may reuse or release captured mat
		const auto& timeCap1 = std::chrono::steady_clock::now();

		/* Call image processor library */
		const auto& timeProcess0 = std::chrono::steady_clock::now();
		ImageProcessor_process(&outputImage);
		const auto& timeProcess1 = std::chrono::steady_clock::now();

		cv::imshow("test", outputImage);
		if (cv::waitKey(1) == 'q') break;
		const auto& timePost1 = std::chrono::steady_clock::now();

		const auto& timeAll1 = std::chrono::steady_clock::now();
		printf("Total time = %.3lf [msec]\n", (timeAll1 - timeAll0).count() / 1000000.0);
		printf("Capture time = %.3lf [msec]\n", (timeCap1 - timeCap0).count() / 1000000.0);
		printf("Image processing time = %.3lf [msec]\n", (timeProcess1 - timeProcess0).count() / 1000000.0);
		printf("========\n");
	}

	/* Fianlize image processor library */
	ImageProcessor_finalize();

	return 0;
}
