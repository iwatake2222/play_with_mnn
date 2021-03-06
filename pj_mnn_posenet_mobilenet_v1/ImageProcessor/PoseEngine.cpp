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
#include "PoseEngine.h"

/*** Macro ***/
#define TAG "PoseEngine"
#define PRINT(...)   COMMON_HELPER_PRINT(TAG, __VA_ARGS__)
#define PRINT_E(...) COMMON_HELPER_PRINT_E(TAG, __VA_ARGS__)

/* Model parameters */
#define MODEL_NAME   "posenet-mobilenet_v1_075.mnn"
/* better to use the same aspect as the input image */
static const int32_t MODEL_WIDTH = 225;
static const int32_t MODEL_HEIGHT = 225;


/*** [Start] Retrieved from demo source code in MNN (https://github.com/alibaba/MNN/blob/master/demo/exec/multiPose.cpp) ***/
/***         modified by iwatake2222 **/
/* Settings */
#define OUTPUT_STRIDE 16

#define MAX_POSE_DETECTIONS 10
#define NUM_KEYPOINTS 17
#define SCORE_THRESHOLD 0.5
#define MIN_POSE_SCORE 0.25
#define NMS_RADIUS 20
#define LOCAL_MAXIMUM_RADIUS 1
#define OFFSET_NODE_NAME "offset_2"
#define DISPLACE_FWD_NODE_NAME "displacement_fwd_2"
#define DISPLACE_BWD_NODE_NAME "displacement_bwd_2"
#define HEATMAPS "heatmap"

const std::vector<std::string> PoseNames{ "nose",         "leftEye",       "rightEye",  "leftEar",    "rightEar",
										 "leftShoulder", "rightShoulder", "leftElbow", "rightElbow", "leftWrist",
										 "rightWrist",   "leftHip",       "rightHip",  "leftKnee",   "rightKnee",
										 "leftAnkle",    "rightAnkle" };

const std::vector<std::pair<std::string, std::string>> PoseChain{
	{"nose", "leftEye"},          {"leftEye", "leftEar"},        {"nose", "rightEye"},
	{"rightEye", "rightEar"},     {"nose", "leftShoulder"},      {"leftShoulder", "leftElbow"},
	{"leftElbow", "leftWrist"},   {"leftShoulder", "leftHip"},   {"leftHip", "leftKnee"},
	{"leftKnee", "leftAnkle"},    {"nose", "rightShoulder"},     {"rightShoulder", "rightElbow"},
	{"rightElbow", "rightWrist"}, {"rightShoulder", "rightHip"}, {"rightHip", "rightKnee"},
	{"rightKnee", "rightAnkle"} };


inline float clip(float value, float min, float max) {
	if (value < 0) {
		return 0;
	} else if (value > max) {
		return max;
	} else {
		return value;
	}
}

static std::pair<float,float> getCoordsFromTensor(const OutputTensorInfo& dataTensor, int id, int x, int y, bool getCoord = true) {
	// dataTensor must be [1,c,h,w]
	auto dataPtr = (float*)dataTensor.data;
	const int xOffset = dataTensor.tensorDims.channel / 2;
	const int indexPlane = y * dataTensor.tensorDims.width + x;
	const int indexY = id * dataTensor.tensorDims.width * dataTensor.tensorDims.height + indexPlane;
	const int indexX = (id + xOffset) * dataTensor.tensorDims.width * dataTensor.tensorDims.height + indexPlane;
	std::pair<float,float> point;
	if (getCoord) {
		point = std::pair<float, float>(dataPtr[indexX], dataPtr[indexY]);
	} else {
		point = std::pair<float, float>(0.0f, dataPtr[indexY]);
	}
	return point;
};

// decode pose and posenet model reference from https://github.com/rwightman/posenet-python
static int decodePoseImpl(float curScore, int curId, const std::pair<float,float>& originalOnImageCoords, const OutputTensorInfo& heatmaps,
	const OutputTensorInfo& offsets, const OutputTensorInfo& displacementFwd, const OutputTensorInfo& displacementBwd,
	std::vector<float>& instanceKeypointScores, std::vector<std::pair<float,float>>& instanceKeypointCoords) {
	instanceKeypointScores[curId] = curScore;
	instanceKeypointCoords[curId] = originalOnImageCoords;
	const int height = heatmaps.tensorDims.height;
	const int width = heatmaps.tensorDims.width;
	std::map<std::string, int> poseNamesID;
	for (int i = 0; i < PoseNames.size(); ++i) {
		poseNamesID[PoseNames[i]] = i;
	}

	auto traverseToTargetKeypoint = [=](int edgeId, const std::pair<float,float>& sourcekeypointCoord, int targetKeypointId,
		const OutputTensorInfo& displacement) {
		int sourceKeypointIndicesX =
			static_cast<int>(clip(round(sourcekeypointCoord.first / (float)OUTPUT_STRIDE), 0, (float)(width - 1)));
		int sourceKeypointIndicesY =
			static_cast<int>(clip(round(sourcekeypointCoord.second / (float)OUTPUT_STRIDE), 0, (float)(height - 1)));

		auto displacementCoord =
			getCoordsFromTensor(displacement, edgeId, sourceKeypointIndicesX, sourceKeypointIndicesY);
		float displacedPointX = sourcekeypointCoord.first + displacementCoord.first;
		float displacedPointY = sourcekeypointCoord.second + displacementCoord.second;

		int displacedPointIndicesX =
			static_cast<int>(clip(round(displacedPointX / OUTPUT_STRIDE), 0, (float)(width - 1)));
		int displacedPointIndicesY =
			static_cast<int>(clip(round(displacedPointY / OUTPUT_STRIDE), 0, (float)(height - 1)));

		float score =
			getCoordsFromTensor(heatmaps, targetKeypointId, displacedPointIndicesX, displacedPointIndicesY, false).second;
		auto offset = getCoordsFromTensor(offsets, targetKeypointId, displacedPointIndicesX, displacedPointIndicesY);

		std::pair<float,float> imageCoord;
		imageCoord.first = displacedPointIndicesX * OUTPUT_STRIDE + offset.first;
		imageCoord.second = displacedPointIndicesY * OUTPUT_STRIDE + offset.second;

		return std::make_pair(score, imageCoord);
	};
	
	for (int edge = (int)PoseChain.size() - 1; edge >= 0; --edge) {
		const int targetKeypointID = poseNamesID[PoseChain[edge].first];
		const int sourceKeypointID = poseNamesID[PoseChain[edge].second];
		if (instanceKeypointScores[sourceKeypointID] > 0.0 && instanceKeypointScores[targetKeypointID] == 0.0) {
			auto curInstance = traverseToTargetKeypoint(edge, instanceKeypointCoords[sourceKeypointID],
				targetKeypointID, displacementBwd);
			instanceKeypointScores[targetKeypointID] = curInstance.first;
			instanceKeypointCoords[targetKeypointID] = curInstance.second;
		}
	}

	for (int edge = 0; edge < PoseChain.size(); ++edge) {
		const int sourceKeypointID = poseNamesID[PoseChain[edge].first];
		const int targetKeypointID = poseNamesID[PoseChain[edge].second];
		if (instanceKeypointScores[sourceKeypointID] > 0.0 && instanceKeypointScores[targetKeypointID] == 0.0) {
			auto curInstance = traverseToTargetKeypoint(edge, instanceKeypointCoords[sourceKeypointID],
				targetKeypointID, displacementFwd);
			instanceKeypointScores[targetKeypointID] = curInstance.first;
			instanceKeypointCoords[targetKeypointID] = curInstance.second;
		}
	}

	return 0;
}


static int decodeMultiPose(const OutputTensorInfo& offsets, const OutputTensorInfo& displacementFwd, const OutputTensorInfo& displacementBwd, const OutputTensorInfo& heatmaps, 
	std::vector<float>& poseScores,
	std::vector<std::vector<float>>& poseKeypointScores,
	std::vector<std::vector<std::pair<float,float>>>& poseKeypointCoords) {
	// keypoint_id, score, coord((x,y))
	typedef std::pair<int, std::pair<float, std::pair<float,float>>> partsType;
	std::vector<partsType> parts;

	const int channel = heatmaps.tensorDims.channel;
	const int height = heatmaps.tensorDims.height;
	const int width = heatmaps.tensorDims.width;
	auto maximumFilter = [&parts, width, height](const int id, const float* startPtr) {
		for (int y = 0; y < height; ++y) {
			for (int x = 0; x < width; ++x) {
				// check whether (y,x) is the max value around the neighborhood
				bool isMaxVaule = true;
				float maxValue = startPtr[y * width + x];
				{
					for (int i = -LOCAL_MAXIMUM_RADIUS; i < (LOCAL_MAXIMUM_RADIUS + 1); ++i) {
						for (int j = -LOCAL_MAXIMUM_RADIUS; j < (LOCAL_MAXIMUM_RADIUS + 1); ++j) {
							float value = 0.0f;
							int yCoord = y + i;
							int xCoord = x + j;
							if (yCoord >= 0 && yCoord < height && xCoord >= 0 && xCoord < width) {
								value = startPtr[yCoord * width + xCoord];
							}
							if (maxValue < value) {
								isMaxVaule = false;
								break;
							}
						}
					}
				}

				if (isMaxVaule && maxValue >= SCORE_THRESHOLD) {
					std::pair<float,float> coord((float)x, (float)y);
					parts.push_back(std::make_pair(id, std::make_pair(maxValue, coord)));
				}
			}
		}
	};

	float* scoresPtr = static_cast< float*>(heatmaps.data);

	for (int id = 0; id < channel; ++id) {
		auto idScoresPtr = scoresPtr + id * width * height;
		maximumFilter(id, idScoresPtr);
	}

	// sort the parts according to score
	std::sort(parts.begin(), parts.end(),
		[](const partsType& a, const partsType& b) { return a.second.first > b.second.first; });

	const int squareNMSRadius = NMS_RADIUS * NMS_RADIUS;

	auto withinNMSRadius = [=, &poseKeypointCoords](const std::pair<float,float>& point, const int id) {
		bool withinThisPointRadius = false;
		for (int i = 0; i < poseKeypointCoords.size(); ++i) {
			const auto& curPoint = poseKeypointCoords[i][id];
			const auto sum = powf((curPoint.first - point.first), 2) + powf((curPoint.second - point.second), 2);
			if (sum <= squareNMSRadius) {
				withinThisPointRadius = true;
				break;
			}
		}
		return withinThisPointRadius;
	};

	std::vector<float> instanceKeypointScores(NUM_KEYPOINTS);
	std::vector<std::pair<float,float>> instanceKeypointCoords(NUM_KEYPOINTS);

	auto getInstanceScore = [&]() {
		float notOverlappedScores = 0.0f;
		const int poseNums = (const int)poseKeypointCoords.size();
		if (poseNums == 0) {
			for (int i = 0; i < NUM_KEYPOINTS; ++i) {
				notOverlappedScores += instanceKeypointScores[i];
			}
		} else {
			for (int id = 0; id < NUM_KEYPOINTS; ++id) {
				if (!withinNMSRadius(instanceKeypointCoords[id], id)) {
					notOverlappedScores += instanceKeypointScores[id];
				}
			}
		}

		return notOverlappedScores / NUM_KEYPOINTS;
	};

	int poseCount = 0;
	for (const auto& part : parts) {
		if (poseCount >= MAX_POSE_DETECTIONS) {
			break;
		}
		const auto curScore = part.second.first;
		const auto curId = part.first;
		const auto& curPoint = part.second.second;

		const auto offsetXY = getCoordsFromTensor(offsets, curId, (int)curPoint.first, (int)curPoint.second);
		std::pair<float,float> originalOnImageCoords;
		originalOnImageCoords.first = curPoint.first * OUTPUT_STRIDE + offsetXY.first;
		originalOnImageCoords.second = curPoint.second * OUTPUT_STRIDE + offsetXY.second;

		if (withinNMSRadius(originalOnImageCoords, curId)) {
			continue;
		}
		::memset(instanceKeypointScores.data(), 0, sizeof(float) * NUM_KEYPOINTS);
		::memset(instanceKeypointCoords.data(), 0, sizeof(std::pair<float,float>) * NUM_KEYPOINTS);
		decodePoseImpl(curScore, curId, originalOnImageCoords, heatmaps, offsets, displacementFwd, displacementBwd,
			instanceKeypointScores, instanceKeypointCoords);

		float poseScore = getInstanceScore();
		if (poseScore > MIN_POSE_SCORE) {
			poseScores.push_back(poseScore);
			poseKeypointScores.push_back(instanceKeypointScores);
			poseKeypointCoords.push_back(instanceKeypointCoords);
			poseCount++;
		}
	}


	return 0;
}




/*** Function ***/
int32_t PoseEngine::initialize(const std::string& workDir, const int32_t numThreads)
{
	/* Set model information */
	std::string modelFilename = workDir + "/model/" + MODEL_NAME;

	/* Set input tensor info */
	m_inputTensorList.clear();
	InputTensorInfo inputTensorInfo;
	inputTensorInfo.name = "image";
	inputTensorInfo.tensorType = TensorInfo::TENSOR_TYPE_FP32;
	inputTensorInfo.tensorDims.batch = 1;
	inputTensorInfo.tensorDims.width = static_cast<int32_t>((float)MODEL_WIDTH / (float)OUTPUT_STRIDE) * OUTPUT_STRIDE + 1;
	inputTensorInfo.tensorDims.height = static_cast<int32_t>((float)MODEL_HEIGHT / (float)OUTPUT_STRIDE) * OUTPUT_STRIDE + 1;
	inputTensorInfo.tensorDims.channel = 3;
	inputTensorInfo.dataType = InputTensorInfo::DATA_TYPE_IMAGE;
	inputTensorInfo.normalize.mean[0] = 0.5f;   	/* https://github.com/alibaba/MNN/blob/master/demo/exec/multiPose.cpp#L343 */
	inputTensorInfo.normalize.mean[1] = 0.5f;
	inputTensorInfo.normalize.mean[2] = 0.5f;
	inputTensorInfo.normalize.norm[0] = 0.5f;
	inputTensorInfo.normalize.norm[1] = 0.5f;
	inputTensorInfo.normalize.norm[2] = 0.5f;
	m_inputTensorList.push_back(inputTensorInfo);

	/* Set output tensor info */
	m_outputTensorList.clear();
	OutputTensorInfo outputTensorInfo;
	outputTensorInfo.tensorType = TensorInfo::TENSOR_TYPE_FP32;
	outputTensorInfo.name = OFFSET_NODE_NAME;
	m_outputTensorList.push_back(outputTensorInfo);
	outputTensorInfo.name = DISPLACE_FWD_NODE_NAME;
	m_outputTensorList.push_back(outputTensorInfo);
	outputTensorInfo.name = DISPLACE_BWD_NODE_NAME;
	m_outputTensorList.push_back(outputTensorInfo);
	outputTensorInfo.name = HEATMAPS;
	m_outputTensorList.push_back(outputTensorInfo);

	/* Create and Initialize Inference Helper */
	//m_inferenceHelper.reset(InferenceHelper::create(InferenceHelper::OPEN_CV));
	//m_inferenceHelper.reset(InferenceHelper::create(InferenceHelper::TENSOR_RT));
	//m_inferenceHelper.reset(InferenceHelper::create(InferenceHelper::NCNN));
	m_inferenceHelper.reset(InferenceHelper::create(InferenceHelper::MNN));
	//m_inferenceHelper.reset(InferenceHelper::create(InferenceHelper::TENSORFLOW_LITE));
	//m_inferenceHelper.reset(InferenceHelper::create(InferenceHelper::TENSORFLOW_LITE_EDGETPU));
	//m_inferenceHelper.reset(InferenceHelper::create(InferenceHelper::TENSORFLOW_LITE_GPU));
	//m_inferenceHelper.reset(InferenceHelper::create(InferenceHelper::TENSORFLOW_LITE_XNNPACK));

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
	
	/* Check if input tensor info is set */
	for (const auto& inputTensorInfo : m_inputTensorList) {
		if ((inputTensorInfo.tensorDims.width <= 0) || (inputTensorInfo.tensorDims.height <= 0) || inputTensorInfo.tensorType == TensorInfo::TENSOR_TYPE_NONE) {
			PRINT_E("Invalid tensor size\n");
			m_inferenceHelper.reset();
			return RET_ERR;
		}
	}

	return RET_OK;
}

int32_t PoseEngine::finalize()
{
	if (!m_inferenceHelper) {
		PRINT_E("Inference helper is not created\n");
		return RET_ERR;
	}
	m_inferenceHelper->finalize();
	return RET_OK;
}


int32_t PoseEngine::invoke(const cv::Mat& originalMat, RESULT& result)
{
	if (!m_inferenceHelper) {
		PRINT_E("Inference helper is not created\n");
		return RET_ERR;
	}
	/*** PreProcess ***/
	const auto& tPreProcess0 = std::chrono::steady_clock::now();
	InputTensorInfo& inputTensorInfo = m_inputTensorList[0];
	/* do resize and color conversion here because some inference engine doesn't support these operations */
	cv::Mat imgSrc;
	cv::resize(originalMat, imgSrc, cv::Size(inputTensorInfo.tensorDims.width, inputTensorInfo.tensorDims.height));
#ifndef CV_COLOR_IS_RGB
	cv::cvtColor(imgSrc, imgSrc, cv::COLOR_BGR2RGB);
#endif
	inputTensorInfo.data = imgSrc.data;
	inputTensorInfo.dataType = InputTensorInfo::DATA_TYPE_IMAGE;
	inputTensorInfo.imageInfo.width = imgSrc.cols;
	inputTensorInfo.imageInfo.height = imgSrc.rows;
	inputTensorInfo.imageInfo.channel = imgSrc.channels();
	inputTensorInfo.imageInfo.cropX = 0;
	inputTensorInfo.imageInfo.cropY = 0;
	inputTensorInfo.imageInfo.cropWidth = imgSrc.cols;
	inputTensorInfo.imageInfo.cropHeight = imgSrc.rows;
	inputTensorInfo.imageInfo.isBGR = false;
	inputTensorInfo.imageInfo.swapColor = false;
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
	std::vector<float> poseScores;
	std::vector<std::vector<float>> poseKeypointScores;
	std::vector<std::vector<std::pair<float,float>>> poseKeypointCoords;	// x, y
	decodeMultiPose(m_outputTensorList[0], m_outputTensorList[1], m_outputTensorList[2], m_outputTensorList[3], poseScores, poseKeypointScores, poseKeypointCoords);

	float scaleX = static_cast<float>(originalMat.cols) / inputTensorInfo.tensorDims.width;
	float scaleY = static_cast<float>(originalMat.rows) / inputTensorInfo.tensorDims.height;
	for (int32_t i = 0; i < poseScores.size(); ++i) {
		for (int32_t id = 0; id < NUM_KEYPOINTS; ++id) {
			poseKeypointCoords[i][id].first *= scaleX;
			poseKeypointCoords[i][id].second *= scaleY;
		}
	}
	const auto& tPostProcess1 = std::chrono::steady_clock::now();

	/* Return the results */
	result.poseScores = poseScores;
	result.poseKeypointScores = poseKeypointScores;
	result.poseKeypointCoords = poseKeypointCoords;
	result.timePreProcess = static_cast<std::chrono::duration<double>>(tPreProcess1 - tPreProcess0).count() * 1000.0;
	result.timeInference = static_cast<std::chrono::duration<double>>(tInference1 - tInference0).count() * 1000.0;
	result.timePostProcess = static_cast<std::chrono::duration<double>>(tPostProcess1 - tPostProcess0).count() * 1000.0;;

	return RET_OK;
}
