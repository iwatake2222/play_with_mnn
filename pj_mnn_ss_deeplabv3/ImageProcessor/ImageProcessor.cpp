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

/* for MNN */
#include <MNN/ImageProcess.hpp>
#include <MNN/Interpreter.hpp>
#include <MNN/AutoTime.hpp>

/*** Macro ***/
#if defined(ANDROID) || defined(__ANDROID__)
#include <android/log.h>
#define TAG "MyApp_NDK"
#define PRINT(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)
#else
#define PRINT(...) printf(__VA_ARGS__)
#endif

/*** Global variables ***/
static MNN::Interpreter *net;
static MNN::Session* session;

/*** Functions ***/
int ImageProcessor_initialize(const char *modelFilename)
{
	/* Create interpreter */
	net = MNN::Interpreter::createFromFile(modelFilename);
	MNN::ScheduleConfig scheduleConfig;
	scheduleConfig.type  = MNN_FORWARD_AUTO;
	scheduleConfig.numThread = 4;
	// BackendConfig bnconfig;
	// bnconfig.precision = BackendConfig::Precision_Low;
	// config.backendConfig = &bnconfig;
	session = net->createSession(scheduleConfig);

	/* Get model information */
	auto input = net->getSessionInput(session, NULL);
	int modelChannel = input->channel();
	int modelHeight  = input->height();
	int modelWidth   = input->width();
	PRINT("model input size: widgh = %d , height = %d, channel = %d\n", modelWidth, modelHeight, modelChannel);

	return 0;
}

int ImageProcessor_process(cv::Mat *mat)
{
	/* Get size information */
	auto input = net->getSessionInput(session, NULL);
	int modelChannel = input->channel();
	int modelHeight = input->height();
	int modelWidth = input->width();
	int imageWidth = mat->size[1];
	int imageHeight = mat->size[0];

	/*** Pre process (resize, colorconversion, normalize) ***/
	MNN::CV::ImageProcess::Config imageProcessconfig;
	imageProcessconfig.filterType = MNN::CV::BILINEAR;
	float mean[3]     = {127.5f, 127.5f, 127.5f};
	float normals[3] = {0.00785f, 0.00785f, 0.00785f};
	std::memcpy(imageProcessconfig.mean, mean, sizeof(mean));
	std::memcpy(imageProcessconfig.normal, normals, sizeof(normals));
	imageProcessconfig.sourceFormat = MNN::CV::BGR;
	imageProcessconfig.destFormat   = MNN::CV::BGR;

	MNN::CV::Matrix trans;
	trans.setScale((float)imageWidth/modelWidth, (float)imageHeight/modelHeight);

	std::shared_ptr<MNN::CV::ImageProcess> pretreat(MNN::CV::ImageProcess::create(imageProcessconfig));
	pretreat->setMatrix(trans);
	pretreat->convert((uint8_t*)mat->data, imageWidth, imageHeight, 0, input);

	/*** Inference ***/
	net->runSession(session);

	/*** Post process ***/
	/* Retreive results */
	auto output = net->getSessionOutput(session, NULL);
	auto dimType = output->getDimensionType();
	dimType = MNN::Tensor::TENSORFLOW;

	std::shared_ptr<MNN::Tensor> outputUser(new MNN::Tensor(output, dimType));
	output->copyToHostTensor(outputUser.get());
	auto outputWidth = outputUser->shape()[2];
	auto outputHeight = outputUser->shape()[1];
	auto outputCannel = outputUser->shape()[3];
	printf("output size: width = %d, height = %d, channel = %d\n", outputWidth, outputHeight, outputCannel);

	auto values = outputUser->host<float>();
	cv::Mat outputImage = cv::Mat::zeros(outputHeight, outputWidth, CV_8UC3);
	for (int y = 0; y < outputHeight; y++) {
		for (int x = 0; x < outputWidth; x++) {
			int maxChannel = 0;
			float maxValue = 0;
			for (int c = 0; c < outputCannel; c++) {
				float value = values[y * (outputWidth * outputCannel) + x * outputCannel + c];
				if (value > maxValue) {
					maxValue = value;
					maxChannel = c;
				}
			}

			float colorRatioB = (maxChannel % 2 + 1) / 2.0;
			float colorRatioG = (maxChannel % 3 + 1) / 3.0;
			float colorRatioR = (maxChannel % 4 + 1) / 4.0;
			outputImage.data[(y * outputWidth + x) * 3 + 0] = (int)(255 * colorRatioB);
			outputImage.data[(y * outputWidth + x) * 3 + 1] = (int)(255 * colorRatioG);
			outputImage.data[(y * outputWidth + x) * 3 + 2] = (int)(255 * (1 - colorRatioR));

		}
	}

	cv::resize(outputImage, outputImage, mat->size());
	cv::add(*mat, outputImage, *mat);

	return 0;
}


int ImageProcessor_finalize(void)
{
	net->releaseSession(session);
	net->releaseModel();
	return 0;
}
