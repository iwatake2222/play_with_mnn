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
/* Model parameters */
#define MODEL_NAME   RESOURCE_DIR"/model/deeplabv3_257_mv_gpu.mnn"
#define IMAGE_NAME   RESOURCE_DIR"/cat.jpg"

/* Settings */
#define LOOP_NUM_FOR_TIME_MEASUREMENT 100


int main(int argc, const char* argv[])
{
	/*** Initialize ***/
	/* Create interpreter */
	std::shared_ptr<MNN::Interpreter> net(MNN::Interpreter::createFromFile(MODEL_NAME));
	MNN::ScheduleConfig scheduleConfig;
	scheduleConfig.type  = MNN_FORWARD_AUTO;
	scheduleConfig.numThread = 4;
	// BackendConfig bnconfig;
	// bnconfig.precision = BackendConfig::Precision_Low;
	// config.backendConfig = &bnconfig;
	auto session = net->createSession(scheduleConfig);

	/* Get model information */
	auto input = net->getSessionInput(session, NULL);
	int modelChannel = input->channel();
	int modelHeight  = input->height();
	int modelWidth   = input->width();
	printf("model input size: widgh = %d , height = %d, channel = %d\n", modelWidth, modelHeight, modelChannel);

	/***** Process for each frame *****/
	/*** Read image ***/
	cv::Mat originalImage = cv::imread(IMAGE_NAME);
	int imageWidth = originalImage.size[1];
	int imageHeight = originalImage.size[0];
	printf("image size: width = %d, height = %d\n", imageWidth, imageHeight);
	
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
	pretreat->convert((uint8_t*)originalImage.data, imageWidth, imageHeight, 0, input);

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

			float colorRatio = (float)maxChannel / outputCannel;	// 0 ~ 1.0
			outputImage.data[(y * outputWidth + x) * 3 + 0] = 0xFF * colorRatio;
			outputImage.data[(y * outputWidth + x) * 3 + 1] = 0xFF * (0.5 + colorRatio/2);
			outputImage.data[(y * outputWidth + x) * 3 + 2] = 0xFF * (1 - colorRatio);

		}
	}
	cv::imshow("originalImage", originalImage); cv::waitKey(1);
	cv::imshow("outputImage", outputImage); cv::waitKey(1);
	cv::waitKey(-1);


	/*** (Optional) Measure inference time ***/
	const auto& t0 = std::chrono::steady_clock::now();
	for (int i = 0; i < LOOP_NUM_FOR_TIME_MEASUREMENT; i++) {
		net->runSession(session);
	}
	const auto& t1 = std::chrono::steady_clock::now();
	std::chrono::duration<double> timeSpan = t1 - t0;
	printf("Inference time = %f [msec]\n", timeSpan.count() * 1000.0 / LOOP_NUM_FOR_TIME_MEASUREMENT);

	/*** Finalize ***/
	net->releaseSession(session);

	return 0;
}
