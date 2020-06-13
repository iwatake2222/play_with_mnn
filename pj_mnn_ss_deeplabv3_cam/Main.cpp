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

	/* initialize camera */
	static cv::VideoCapture cap;
	cap = cv::VideoCapture(0);
	cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
	cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
	// cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('B', 'G', 'R', '3'));
	cap.set(cv::CAP_PROP_BUFFERSIZE, 1);

	while (1) {
		const auto& timeAll0 = std::chrono::steady_clock::now();
		/*** Read image ***/
		const auto& timeCap0 = std::chrono::steady_clock::now();
		//cv::Mat originalImage = cv::imread(IMAGE_NAME);
		cv::Mat originalImage;
		cap.read(originalImage);
		
		int imageWidth = originalImage.size[1];
		int imageHeight = originalImage.size[0];
		//printf("image size: width = %d, height = %d\n", imageWidth, imageHeight);
		const auto& timeCap1 = std::chrono::steady_clock::now();

		/*** Pre process (resize, colorconversion, normalize) ***/
		const auto& timePre0 = std::chrono::steady_clock::now();
		MNN::CV::ImageProcess::Config imageProcessconfig;
		imageProcessconfig.filterType = MNN::CV::BILINEAR;
		float mean[3] = { 127.5f, 127.5f, 127.5f };
		float normals[3] = { 0.00785f, 0.00785f, 0.00785f };
		std::memcpy(imageProcessconfig.mean, mean, sizeof(mean));
		std::memcpy(imageProcessconfig.normal, normals, sizeof(normals));
		imageProcessconfig.sourceFormat = MNN::CV::BGR;
		imageProcessconfig.destFormat = MNN::CV::BGR;

		MNN::CV::Matrix trans;
		trans.setScale((float)imageWidth / modelWidth, (float)imageHeight / modelHeight);

		std::shared_ptr<MNN::CV::ImageProcess> pretreat(MNN::CV::ImageProcess::create(imageProcessconfig));
		pretreat->setMatrix(trans);
		pretreat->convert((uint8_t*)originalImage.data, imageWidth, imageHeight, 0, input);
		const auto& timePre1 = std::chrono::steady_clock::now();

		/*** Inference ***/
		const auto& timeInference0 = std::chrono::steady_clock::now();
		net->runSession(session);
		const auto& timeInference1 = std::chrono::steady_clock::now();

		/*** Post process ***/
		const auto& timePost0 = std::chrono::steady_clock::now();
		/* Retreive results */
		auto output = net->getSessionOutput(session, NULL);
		auto dimType = output->getDimensionType();
		dimType = MNN::Tensor::TENSORFLOW;

		std::shared_ptr<MNN::Tensor> outputUser(new MNN::Tensor(output, dimType));
		output->copyToHostTensor(outputUser.get());
		auto outputWidth = outputUser->shape()[2];
		auto outputHeight = outputUser->shape()[1];
		auto outputCannel = outputUser->shape()[3];
		//printf("output size: width = %d, height = %d, channel = %d\n", outputWidth, outputHeight, outputCannel);

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
		cv::imshow("originalImage", originalImage);
		cv::imshow("outputImage", outputImage);
		if (cv::waitKey(1) == 'q') break;
		const auto& timePost1 = std::chrono::steady_clock::now();

		const auto& timeAll1 = std::chrono::steady_clock::now();
		printf("Total time = %.3lf [msec]\n", (timeAll1 - timeAll0).count() / 1000000.0);
		printf("Capture time = %.3lf [msec]\n", (timeCap1 - timeCap0).count() / 1000000.0);
		printf("Inference time = %.3lf [msec]\n", (timeInference1 - timeInference0).count() / 1000000.0);
		printf("PreProcess time = %.3lf [msec]\n", (timePre1 - timePre0).count() / 1000000.0);
		printf("PostProcess time = %.3lf [msec]\n", (timePost1 - timePost0).count() / 1000000.0);
		printf("========\n");
	}


	/*** Finalize ***/
	net->releaseSession(session);

	return 0;
}
