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
/*** Macro ***/
#if defined(ANDROID) || defined(__ANDROID__)
#include <android/log.h>
#define TAG "MyApp_NDK"
#define PRINT(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)
#else
#define PRINT(...) printf(__VA_ARGS__)
#endif


/*** [Start] Retrieved from demo source code in MNN (multiPose.cpp/.h) ***/
using namespace MNN;
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

static CV::Point getCoordsFromTensor(const Tensor* dataTensor, int id, int x, int y, bool getCoord = true) {
	// dataTensor must be [1,c,h,w]
	auto dataPtr = dataTensor->host<float>();
	const int xOffset = dataTensor->channel() / 2;
	const int indexPlane = y * dataTensor->stride(2) + x;
	const int indexY = id * dataTensor->stride(1) + indexPlane;
	const int indexX = (id + xOffset) * dataTensor->stride(1) + indexPlane;
	CV::Point point;
	if (getCoord) {
		point.set(dataPtr[indexX], dataPtr[indexY]);
	} else {
		point.set(0.0, dataPtr[indexY]);
	}
	return point;
};

// decode pose and posenet model reference from https://github.com/rwightman/posenet-python
static int decodePoseImpl(float curScore, int curId, const CV::Point& originalOnImageCoords, const Tensor* heatmaps,
	const Tensor* offsets, const Tensor* displacementFwd, const Tensor* displacementBwd,
	std::vector<float>& instanceKeypointScores, std::vector<CV::Point>& instanceKeypointCoords) {
	instanceKeypointScores[curId] = curScore;
	instanceKeypointCoords[curId] = originalOnImageCoords;
	const int height = heatmaps->height();
	const int width = heatmaps->width();
	std::map<std::string, int> poseNamesID;
	for (int i = 0; i < PoseNames.size(); ++i) {
		poseNamesID[PoseNames[i]] = i;
	}

	auto traverseToTargetKeypoint = [=](int edgeId, const CV::Point& sourcekeypointCoord, int targetKeypointId,
		const Tensor* displacement) {
		int sourceKeypointIndicesX =
			static_cast<int>(clip(round(sourcekeypointCoord.fX / (float)OUTPUT_STRIDE), 0, (float)(width - 1)));
		int sourceKeypointIndicesY =
			static_cast<int>(clip(round(sourcekeypointCoord.fY / (float)OUTPUT_STRIDE), 0, (float)(height - 1)));

		auto displacementCoord =
			getCoordsFromTensor(displacement, edgeId, sourceKeypointIndicesX, sourceKeypointIndicesY);
		float displacedPointX = sourcekeypointCoord.fX + displacementCoord.fX;
		float displacedPointY = sourcekeypointCoord.fY + displacementCoord.fY;

		int displacedPointIndicesX =
			static_cast<int>(clip(round(displacedPointX / OUTPUT_STRIDE), 0, (float)(width - 1)));
		int displacedPointIndicesY =
			static_cast<int>(clip(round(displacedPointY / OUTPUT_STRIDE), 0, (float)(height - 1)));

		float score =
			getCoordsFromTensor(heatmaps, targetKeypointId, displacedPointIndicesX, displacedPointIndicesY, false).fY;
		auto offset = getCoordsFromTensor(offsets, targetKeypointId, displacedPointIndicesX, displacedPointIndicesY);

		CV::Point imageCoord;
		imageCoord.fX = displacedPointIndicesX * OUTPUT_STRIDE + offset.fX;
		imageCoord.fY = displacedPointIndicesY * OUTPUT_STRIDE + offset.fY;

		return std::make_pair(score, imageCoord);
	};

	MNN_ASSERT((NUM_KEYPOINTS - 1) == PoseChain.size());

	for (int edge = PoseChain.size() - 1; edge >= 0; --edge) {
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


static int decodeMultiPose(const Tensor* offsets, const Tensor* displacementFwd, const Tensor* displacementBwd,
	const Tensor* heatmaps, std::vector<float>& poseScores,
	std::vector<std::vector<float>>& poseKeypointScores,
	std::vector<std::vector<CV::Point>>& poseKeypointCoords, CV::Point& scale) {
	// keypoint_id, score, coord((x,y))
	typedef std::pair<int, std::pair<float, CV::Point>> partsType;
	std::vector<partsType> parts;

	const int channel = heatmaps->channel();
	const int height = heatmaps->height();
	const int width = heatmaps->width();
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
					CV::Point coord;
					coord.set(x, y);
					parts.push_back(std::make_pair(id, std::make_pair(maxValue, coord)));
				}
			}
		}
	};

	auto scoresPtr = heatmaps->host<float>();

	for (int id = 0; id < channel; ++id) {
		auto idScoresPtr = scoresPtr + id * width * height;
		maximumFilter(id, idScoresPtr);
	}

	// sort the parts according to score
	std::sort(parts.begin(), parts.end(),
		[](const partsType& a, const partsType& b) { return a.second.first > b.second.first; });

	const int squareNMSRadius = NMS_RADIUS * NMS_RADIUS;

	auto withinNMSRadius = [=, &poseKeypointCoords](const CV::Point& point, const int id) {
		bool withinThisPointRadius = false;
		for (int i = 0; i < poseKeypointCoords.size(); ++i) {
			const auto& curPoint = poseKeypointCoords[i][id];
			const auto sum = powf((curPoint.fX - point.fX), 2) + powf((curPoint.fY - point.fY), 2);
			if (sum <= squareNMSRadius) {
				withinThisPointRadius = true;
				break;
			}
		}
		return withinThisPointRadius;
	};

	std::vector<float> instanceKeypointScores(NUM_KEYPOINTS);
	std::vector<CV::Point> instanceKeypointCoords(NUM_KEYPOINTS);

	auto getInstanceScore = [&]() {
		float notOverlappedScores = 0.0f;
		const int poseNums = poseKeypointCoords.size();
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

		const auto offsetXY = getCoordsFromTensor(offsets, curId, (int)curPoint.fX, (int)curPoint.fY);
		CV::Point originalOnImageCoords;
		originalOnImageCoords.fX = curPoint.fX * OUTPUT_STRIDE + offsetXY.fX;
		originalOnImageCoords.fY = curPoint.fY * OUTPUT_STRIDE + offsetXY.fY;

		if (withinNMSRadius(originalOnImageCoords, curId)) {
			continue;
		}
		::memset(instanceKeypointScores.data(), 0, sizeof(float) * NUM_KEYPOINTS);
		::memset(instanceKeypointCoords.data(), 0, sizeof(CV::Point) * NUM_KEYPOINTS);
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

	// scale the pose keypoint coords
	for (int i = 0; i < poseCount; ++i) {
		for (int id = 0; id < NUM_KEYPOINTS; ++id) {
			poseKeypointCoords[i][id].fX *= scale.fX;
			poseKeypointCoords[i][id].fY *= scale.fY;
		}
	}

	return 0;
}

/*** [End] Retrieved from demo source code in MNN (multiPose.cpp/.h) ***/
static MNN::Interpreter *net;
static Session* session;
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


	return 0;
}

int ImageProcessor_process(cv::Mat *mat)
{
	cv::Mat originalImage = *mat;

	/* Fix model input size */
	int imageWidth = 225;
	int imageHeight = 225;
	auto input = net->getSessionInput(session, NULL);
	static int targetWidth = static_cast<int>((float)imageWidth / (float)OUTPUT_STRIDE) * OUTPUT_STRIDE + 1;
	static int targetHeight = static_cast<int>((float)imageHeight / (float)OUTPUT_STRIDE) * OUTPUT_STRIDE + 1;
	net->resizeTensor(input, { 1, 3, targetHeight, targetWidth });
	net->resizeSession(session);
	int modelChannel = input->channel();
	int modelHeight = input->height();
	int modelWidth = input->width();
	//PRINT("model input size: widgh = %d , height = %d, channel = %d\n", modelWidth, modelHeight, modelChannel);

	int originalImageWidth = originalImage.cols;
	int originalImageHeight = originalImage.cols;
	MNN::CV::Point scale;
	scale.fX = (float)originalImageWidth / (float)targetWidth;
	scale.fY = (float)originalImageHeight / (float)targetHeight;

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
	trans.postScale(1.0 / targetWidth, 1.0 / targetHeight);
	trans.postScale(originalImageWidth, originalImageHeight);

	std::shared_ptr<MNN::CV::ImageProcess> pretreat(MNN::CV::ImageProcess::create(imageProcessconfig));
	pretreat->setMatrix(trans);
	pretreat->convert((uint8_t*)originalImage.data, originalImageWidth, originalImageHeight, 0, input);
	const auto& timePre1 = std::chrono::steady_clock::now();

	/*** Inference ***/
	const auto& timeInference0 = std::chrono::steady_clock::now();
	net->runSession(session);
	const auto& timeInference1 = std::chrono::steady_clock::now();

	/*** Post process ***/
	/* Retreive results */
	const auto& timePost0 = std::chrono::steady_clock::now();
	auto offsets = net->getSessionOutput(session, OFFSET_NODE_NAME);
	auto displacementFwd = net->getSessionOutput(session, DISPLACE_FWD_NODE_NAME);
	auto displacementBwd = net->getSessionOutput(session, DISPLACE_BWD_NODE_NAME);
	auto heatmaps = net->getSessionOutput(session, HEATMAPS);

	MNN::Tensor offsetsHost(offsets, MNN::Tensor::CAFFE);
	MNN::Tensor displacementFwdHost(displacementFwd, MNN::Tensor::CAFFE);
	MNN::Tensor displacementBwdHost(displacementBwd, MNN::Tensor::CAFFE);
	MNN::Tensor heatmapsHost(heatmaps, MNN::Tensor::CAFFE);

	offsets->copyToHostTensor(&offsetsHost);
	displacementFwd->copyToHostTensor(&displacementFwdHost);
	displacementBwd->copyToHostTensor(&displacementBwdHost);
	heatmaps->copyToHostTensor(&heatmapsHost);

	std::vector<float> poseScores;
	std::vector<std::vector<float>> poseKeypointScores;
	std::vector<std::vector<MNN::CV::Point>> poseKeypointCoords;

	decodeMultiPose(&offsetsHost, &displacementFwdHost, &displacementBwdHost, &heatmapsHost, poseScores,
		poseKeypointScores, poseKeypointCoords, scale);

	const int poseCount = poseScores.size();
	for (int i = 0; i < poseCount; ++i) {
		if (poseScores[i] > MIN_POSE_SCORE) {
			for (int id = 0; id < NUM_KEYPOINTS; ++id) {
				if (poseKeypointScores[i][id] > SCORE_THRESHOLD) {
					CV::Point point = poseKeypointCoords[i][id];
					cv::circle(originalImage, cv::Point(point.fX, point.fY), 10, cv::Scalar(0, 255, 0), -1);
				}
			}
		}
	}
	
	return 0;
}


int ImageProcessor_finalize(void)
{
	net->releaseSession(session);
	net->releaseModel();
	return 0;
}
