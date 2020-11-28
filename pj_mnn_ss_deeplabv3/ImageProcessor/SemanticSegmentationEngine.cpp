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

/* for OpenCV */
#include <opencv2/opencv.hpp>

/* for My modules */
#include "CommonHelper.h"
#include "InferenceHelper.h"
#include "SemanticSegmentationEngine.h"

/*** Macro ***/
#define TAG "SemanticSegmentationEngine"
#define PRINT(...)   COMMON_HELPER_PRINT(TAG, __VA_ARGS__)
#define PRINT_E(...) COMMON_HELPER_PRINT_E(TAG, __VA_ARGS__)

/* Model parameters */
#define MODEL_NAME   "deeplabv3_257_mv_gpu.mnn"


/*** Function ***/
int32_t SemanticSegmentationEngine::initialize(const std::string& workDir, const int32_t numThreads)
{
	/* Set model information */
	std::string modelFilename = workDir + "/model/" + MODEL_NAME;

	/* Set input tensor info */
	m_inputTensorList.clear();
	InputTensorInfo inputTensorInfo;
	inputTensorInfo.name = "sub_7";
	inputTensorInfo.tensorType = TensorInfo::TENSOR_TYPE_FP32;
	inputTensorInfo.tensorDims.batch = 1;
	inputTensorInfo.tensorDims.width = 257;
	inputTensorInfo.tensorDims.height = 257;
	inputTensorInfo.tensorDims.channel = 3;
	inputTensorInfo.data = nullptr;
	inputTensorInfo.dataType = InputTensorInfo::DATA_TYPE_IMAGE;
	inputTensorInfo.imageInfo.width = -1;
	inputTensorInfo.imageInfo.height = -1;
	inputTensorInfo.imageInfo.channel = -1;
	inputTensorInfo.imageInfo.cropX = -1;
	inputTensorInfo.imageInfo.cropY = -1;
	inputTensorInfo.imageInfo.cropWidth = -1;
	inputTensorInfo.imageInfo.cropHeight = -1;
	inputTensorInfo.imageInfo.isBGR = true;
	inputTensorInfo.imageInfo.swapColor = false;
	inputTensorInfo.normalize.mean[0] = 0.5f;   	/* https://github.com/tensorflow/examples/blob/master/lite/examples/image_segmentation/android/lib_interpreter/src/main/java/org/tensorflow/lite/examples/imagesegmentation/tflite/ImageSegmentationModelExecutor.kt#L236 */
	inputTensorInfo.normalize.mean[1] = 0.5f;
	inputTensorInfo.normalize.mean[2] = 0.5f;
	inputTensorInfo.normalize.norm[0] = 0.5f;
	inputTensorInfo.normalize.norm[1] = 0.5f;
	inputTensorInfo.normalize.norm[2] = 0.5f;
#if 0
	/* Convert to speeden up normalization:  ((src / 255) - mean) / norm  = src * 1 / (255 * norm) - (mean / norm) */
	for (int32_t i = 0; i < 3; i++) {
		inputTensorInfo.normalize.mean[i] /= inputTensorInfo.normalize.norm[i];
		inputTensorInfo.normalize.norm[i] *= 255.0f;
		inputTensorInfo.normalize.norm[i] = 1.0f / inputTensorInfo.normalize.norm[i];
	}
#endif
#if 1
	/* Convert to speeden up normalization:  ((src / 255) - mean) / norm = (src  - (mean * 255))  * (1 / (255 * norm)) */
	for (int32_t i = 0; i < 3; i++) {
		inputTensorInfo.normalize.mean[i] *= 255.0f;
		inputTensorInfo.normalize.norm[i] *= 255.0f;
		inputTensorInfo.normalize.norm[i] = 1.0f / inputTensorInfo.normalize.norm[i];
	}
#endif

	m_inputTensorList.push_back(inputTensorInfo);

	/* Set output tensor info */
	m_outputTensorList.clear();
	OutputTensorInfo outputTensorInfo;
	outputTensorInfo.name = "ResizeBilinear_3";
	outputTensorInfo.tensorType = TensorInfo::TENSOR_TYPE_FP32;
	m_outputTensorList.push_back(outputTensorInfo);

	/* Create and Initialize Inference Helper */
	//m_inferenceHelper.reset(InferenceHelper::create(InferenceHelper::OPEN_CV));
	//m_inferenceHelper.reset(InferenceHelper::create(InferenceHelper::TENSOR_RT));
	//m_inferenceHelper.reset(InferenceHelper::create(InferenceHelper::NCNN));
	m_inferenceHelper.reset(InferenceHelper::create(InferenceHelper::MNN));

	if (!m_inferenceHelper) {
		return RET_ERR;
	}
	if (m_inferenceHelper->setNumThread(numThreads) != InferenceHelper::RET_OK) {
		m_inferenceHelper.reset();
		return RET_ERR;
	}
	if (m_inferenceHelper->initialize(modelFilename, m_inputTensorList, m_outputTensorList) != InferenceHelper::RET_OK) {
		m_inferenceHelper.reset();
		return RET_ERR;
	}


	return RET_OK;
}

int32_t SemanticSegmentationEngine::finalize()
{
	if (!m_inferenceHelper) {
		PRINT_E("Inference helper is not created\n");
		return RET_ERR;
	}
	m_inferenceHelper->finalize();
	return RET_OK;
}


int32_t SemanticSegmentationEngine::invoke(const cv::Mat& originalMat, RESULT& result)
{
	if (!m_inferenceHelper) {
		PRINT_E("Inference helper is not created\n");
		return RET_ERR;
	}
	/*** PreProcess ***/
	const auto& tPreProcess0 = std::chrono::steady_clock::now();
	InputTensorInfo& inputTensorInfo = m_inputTensorList[0];
#if 1
	/* do resize and color conversion here because some inference engine doesn't support these operations */
	cv::Mat imgSrc;
	cv::resize(originalMat, imgSrc, cv::Size(inputTensorInfo.tensorDims.width, inputTensorInfo.tensorDims.height));
	if (inputTensorInfo.imageInfo.channel == 3 && inputTensorInfo.imageInfo.swapColor) {
		cv::cvtColor(imgSrc, imgSrc, cv::COLOR_BGR2RGB);
	}
	inputTensorInfo.data = imgSrc.data;
	inputTensorInfo.dataType = InputTensorInfo::DATA_TYPE_IMAGE;
	inputTensorInfo.imageInfo.width = imgSrc.cols;
	inputTensorInfo.imageInfo.height = imgSrc.rows;
	inputTensorInfo.imageInfo.channel = imgSrc.channels();
	inputTensorInfo.imageInfo.cropX = 0;
	inputTensorInfo.imageInfo.cropY = 0;
	inputTensorInfo.imageInfo.cropWidth = imgSrc.cols;
	inputTensorInfo.imageInfo.cropHeight = imgSrc.rows;
	inputTensorInfo.imageInfo.isBGR = true;
	inputTensorInfo.imageInfo.swapColor = false;
#else
	/* Test other input format */
	cv::Mat imgSrc;
	inputTensorInfo.data = originalMat.data;
	inputTensorInfo.dataType = InputTensorInfo::DATA_TYPE_IMAGE;
	inputTensorInfo.imageInfo.width = originalMat.cols;
	inputTensorInfo.imageInfo.height = originalMat.rows;
	inputTensorInfo.imageInfo.channel = originalMat.channels();
	inputTensorInfo.imageInfo.cropX = 0;
	inputTensorInfo.imageInfo.cropY = 0;
	inputTensorInfo.imageInfo.cropWidth = originalMat.cols;
	inputTensorInfo.imageInfo.cropHeight = originalMat.rows;
	inputTensorInfo.imageInfo.isBGR = true;
	inputTensorInfo.imageInfo.swapColor = true;
	//InferenceHelper::preProcessByOpenCV(inputTensorInfo, false, imgSrc);
	InferenceHelper::preProcessByOpenCV(inputTensorInfo, true, imgSrc);
	inputTensorInfo.data = imgSrc.data;
	//inputTensorInfo.dataType = InputTensorInfo::DATA_TYPE_BLOB_NHWC;
	inputTensorInfo.dataType = InputTensorInfo::DATA_TYPE_BLOB_NCHW;
#endif
	if (m_inferenceHelper->preProcess(m_inputTensorList) != InferenceHelper::RET_OK) {
		return RET_ERR;
	}
	const auto& tPreProcess1 = std::chrono::steady_clock::now();

	/*** Inference ***/
	const auto& tInference0 = std::chrono::steady_clock::now();
	if (m_inferenceHelper->invoke(m_outputTensorList) != InferenceHelper::RET_OK) {
		return RET_ERR;
	}
	const auto& tInference1 = std::chrono::steady_clock::now();

	/*** PostProcess ***/
	const auto& tPostProcess0 = std::chrono::steady_clock::now();
	/* Create mask image */
	int32_t outputWidth = m_outputTensorList[0].tensorDims.width;
	int32_t outputHeight = m_outputTensorList[0].tensorDims.height;
	int32_t outputCannel = m_outputTensorList[0].tensorDims.channel;
	float_t* values = static_cast<float_t*>(m_outputTensorList[0].data);
	cv::Mat maskImage = cv::Mat::zeros(outputHeight, outputWidth, CV_8UC3);
	for (int32_t y = 0; y < outputHeight; y++) {
		for (int32_t x = 0; x < outputWidth; x++) {
			int32_t maxChannel = 0;
			float_t maxValue = 0;
			for (int32_t c = 0; c < outputCannel; c++) {
				//float_t value = values[y * (outputWidth * outputCannel) + x * outputCannel + c];	// NHWC
				float_t value = values[c * (outputWidth * outputHeight) + y * outputWidth + x];	// NCHW
				if (value > maxValue) {
					maxValue = value;
					maxChannel = c;
				}
			}

			float_t colorRatioB = (maxChannel % 2 + 1) / 2.0f;
			float_t colorRatioG = (maxChannel % 3 + 1) / 3.0f;
			float_t colorRatioR = (maxChannel % 4 + 1) / 4.0f;
			maskImage.data[(y * outputWidth + x) * 3 + 0] = (int)(255 * colorRatioB);
			maskImage.data[(y * outputWidth + x) * 3 + 1] = (int)(255 * colorRatioG);
			maskImage.data[(y * outputWidth + x) * 3 + 2] = (int)(255 * (1 - colorRatioR));

		}
	}
	const auto& tPostProcess1 = std::chrono::steady_clock::now();

	/* Return the results */
	result.maskImage = maskImage;
	result.timePreProcess = static_cast<std::chrono::duration<double_t>>(tPreProcess1 - tPreProcess0).count() * 1000.0;
	result.timeInference = static_cast<std::chrono::duration<double_t>>(tInference1 - tInference0).count() * 1000.0;
	result.timePostProcess = static_cast<std::chrono::duration<double_t>>(tPostProcess1 - tPostProcess0).count() * 1000.0;;

	return RET_OK;
}
