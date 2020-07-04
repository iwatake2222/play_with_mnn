/*** Include ***/
/* for general */
#include <stdio.h>
#include <stdlib.h>
#include <string>
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

#include "ImageProcessor.h"

/*** Macro ***/
#if defined(ANDROID) || defined(__ANDROID__)
#include <android/log.h>
#define TAG "MyApp_NDK"
#define PRINT(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)
#else
#define PRINT(...) printf(__VA_ARGS__)
#endif

#define CHECK(x)                              \
  if (!(x)) {                                                \
	PRINT("Error at %s:%d\n", __FILE__, __LINE__); \
	exit(1);                                                 \
  }

/*** Global variables ***/
static MNN::Interpreter *s_net;
static MNN::Session* s_session;
static std::vector<std::string> s_labels;

/*** Functions ***/
static void readLabel(const char* filename, std::vector<std::string> & labels)
{
	std::ifstream ifs(filename);
	if (ifs.fail()) {
		PRINT("failed to read %s\n", filename);
		return;
	}
	std::string str;
	while (getline(ifs, str)) {
		labels.push_back(str);
	}
}

int ImageProcessor_initialize(const char *modelFilename, INPUT_PARAM *inputParam)
{
	/* Create interpreter */
	s_net = MNN::Interpreter::createFromFile(modelFilename);
	CHECK(s_net != NULL);
	MNN::ScheduleConfig scheduleConfig;
	scheduleConfig.type  = MNN_FORWARD_AUTO;
	scheduleConfig.numThread = 4;
	// BackendConfig bnconfig;
	// bnconfig.precision = BackendConfig::Precision_Low;
	// config.backendConfig = &bnconfig;
	s_session = s_net->createSession(scheduleConfig);
	CHECK(s_session != NULL);

	/* Get model information */
	auto input = s_net->getSessionInput(s_session, NULL);
	int modelChannel = input->channel();
	int modelHeight  = input->height();
	int modelWidth   = input->width();
	PRINT("model input size: widgh = %d , height = %d, channel = %d\n", modelWidth, modelHeight, modelChannel);

	/* read label */
	readLabel(inputParam->labelFilename, s_labels);

	return 0;
}

int ImageProcessor_process(cv::Mat *mat, OUTPUT_PARAM *outputParam)
{
	/* Get size information */
	auto input = s_net->getSessionInput(s_session, NULL);
	CHECK(input != NULL);
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
	s_net->runSession(s_session);

	/*** Post process ***/
	/* Retreive results */
	auto output = s_net->getSessionOutput(s_session, NULL);
	CHECK(output != NULL);
	auto dimType = output->getDimensionType();

	std::shared_ptr<MNN::Tensor> outputUser(new MNN::Tensor(output, dimType));
	output->copyToHostTensor(outputUser.get());
	auto size = outputUser->elementSize();
	auto type = outputUser->getType();
	//PRINT("output size: size = %d\n", size);
	
	std::vector<std::pair<int, float>> tempValues(size);
	if (type.code == halide_type_float) {
		auto values = outputUser->host<float>();
		for (int i = 0; i < size; ++i) {
			tempValues[i] = std::make_pair(i, values[i]);
		}
	} else if (type.code == halide_type_uint && type.bytes() == 1) {
		auto values = outputUser->host<uint8_t>();
		for (int i = 0; i < size; ++i) {
			tempValues[i] = std::make_pair(i, values[i]);
		}
	} else {
		PRINT("should not reach here\n");
	}

	/* Find the highest score */
	std::sort(tempValues.begin(), tempValues.end(), [](std::pair<int, float> a, std::pair<int, float> b) { return a.second > b.second; });
	int length = size > 10 ? 10 : size;
	PRINT("==========\n");
	for (int i = 0; i < length; ++i) {
		PRINT("%s(%d): %f\n", s_labels[tempValues[i].first].c_str(), tempValues[i].first, tempValues[i].second);
	}

	/* Draw the result */
	std::string resultStr;
	resultStr = "Result:" + s_labels[tempValues[0].first] + "(" + std::to_string(tempValues[0].first) + ") (score=" + std::to_string(tempValues[0].second) + ")";
	cv::putText(*mat, resultStr, cv::Point(100, 100), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0, 0, 0), 3);
	cv::putText(*mat, resultStr, cv::Point(100, 100), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0, 255, 0), 1);
	
	/* Contain the results */
	outputParam->classId = tempValues[0].first;
	snprintf(outputParam->label, sizeof(outputParam->label), s_labels[tempValues[0].first].c_str());
	outputParam->score = tempValues[0].second;

	return 0;
}


int ImageProcessor_finalize(void)
{
	s_net->releaseSession(s_session);
	s_net->releaseModel();
	delete s_net;
	return 0;
}
