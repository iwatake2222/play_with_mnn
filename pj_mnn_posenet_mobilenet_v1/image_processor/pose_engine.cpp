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
#include "common_helper.h"
#include "inference_helper.h"
#include "pose_engine.h"

/*** Macro ***/
#define TAG "PoseEngine"
#define PRINT(...)   COMMON_HELPER_PRINT(TAG, __VA_ARGS__)
#define PRINT_E(...) COMMON_HELPER_PRINT_E(TAG, __VA_ARGS__)

/* Model parameters */
#define MODEL_NAME   "posenet-mobilenet_v1_075.mnn"
#define INPUT_NAME   "image"
#define OUTPUT_NAME  "MobilenetV2/Predictions/Reshape_1"
#define TENSORTYPE    TensorInfo::kTensorTypeFp32
#define IS_NCHW       true
/* better to use the same aspect as the input image */
static const int32_t MODEL_WIDTH = 225;
static const int32_t MODEL_HEIGHT = 225;
#define INPUT_DIMS    { 1, 3, static_cast<int32_t>((float)MODEL_HEIGHT / (float)OUTPUT_STRIDE) * OUTPUT_STRIDE + 1, static_cast<int32_t>((float)MODEL_WIDTH / (float)OUTPUT_STRIDE) * OUTPUT_STRIDE + 1 }

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
    auto dataPtr = (float*)dataTensor.data;
    const int xOffset = dataTensor.tensor_dims[1] / 2;
    const int indexPlane = y * dataTensor.tensor_dims[3] + x;
    const int indexY = id * dataTensor.tensor_dims[3] * dataTensor.tensor_dims[2] + indexPlane;
    const int indexX = (id + xOffset) * dataTensor.tensor_dims[3] * dataTensor.tensor_dims[2] + indexPlane;
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
    const int height = heatmaps.tensor_dims[2];
    const int width = heatmaps.tensor_dims[3];
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
    std::vector<float>& pose_scores,
    std::vector<std::vector<float>>& pose_keypoint_scores,
    std::vector<std::vector<std::pair<float,float>>>& pose_eypoint_coords) {
    // keypoint_id, score, coord((x,y))
    typedef std::pair<int, std::pair<float, std::pair<float,float>>> partsType;
    std::vector<partsType> parts;

    const int channel = heatmaps.tensor_dims[1];
    const int height = heatmaps.tensor_dims[2];
    const int width = heatmaps.tensor_dims[3];
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

    auto withinNMSRadius = [=, &pose_eypoint_coords](const std::pair<float,float>& point, const int id) {
        bool withinThisPointRadius = false;
        for (int i = 0; i < pose_eypoint_coords.size(); ++i) {
            const auto& curPoint = pose_eypoint_coords[i][id];
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
        const int poseNums = (const int)pose_eypoint_coords.size();
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
            pose_scores.push_back(poseScore);
            pose_keypoint_scores.push_back(instanceKeypointScores);
            pose_eypoint_coords.push_back(instanceKeypointCoords);
            poseCount++;
        }
    }


    return 0;
}

/*** Function ***/
int32_t PoseEngine::Initialize(const std::string& work_dir, const int32_t num_threads)
{
    /* Set model information */
    std::string model_filename = work_dir + "/model/" + MODEL_NAME;

    /* Set input tensor info */
    input_tensor_info_list_.clear();
    InputTensorInfo input_tensor_info(INPUT_NAME, TENSORTYPE, IS_NCHW);
    input_tensor_info.tensor_dims = INPUT_DIMS;
    input_tensor_info.data_type = InputTensorInfo::kDataTypeImage;
    input_tensor_info.normalize.mean[0] = 0.5f;   	/* https://github.com/alibaba/MNN/blob/master/demo/exec/multiPose.cpp#L343 */
    input_tensor_info.normalize.mean[1] = 0.5f;
    input_tensor_info.normalize.mean[2] = 0.5f;
    input_tensor_info.normalize.norm[0] = 0.5f;
    input_tensor_info.normalize.norm[1] = 0.5f;
    input_tensor_info.normalize.norm[2] = 0.5f;
    input_tensor_info_list_.push_back(input_tensor_info);

    /* Set output tensor info */
    output_tensor_info_list_.clear();
    output_tensor_info_list_.push_back(OutputTensorInfo(OFFSET_NODE_NAME, TENSORTYPE));
    output_tensor_info_list_.push_back(OutputTensorInfo(DISPLACE_FWD_NODE_NAME, TENSORTYPE));
    output_tensor_info_list_.push_back(OutputTensorInfo(DISPLACE_BWD_NODE_NAME, TENSORTYPE));
    output_tensor_info_list_.push_back(OutputTensorInfo(HEATMAPS, TENSORTYPE));

    /* Create and Initialize Inference Helper */
    inference_helper_.reset(InferenceHelper::Create(InferenceHelper::kMnn));

    if (!inference_helper_) {
        return kRetErr;
    }
    if (inference_helper_->SetNumThreads(num_threads) != InferenceHelper::kRetOk) {
        inference_helper_.reset();
        return kRetErr;
    }
    if (inference_helper_->Initialize(model_filename, input_tensor_info_list_, output_tensor_info_list_) != InferenceHelper::kRetOk) {
        inference_helper_.reset();
        return kRetErr;
    }

    return kRetOk;
}

int32_t PoseEngine::Finalize()
{
    if (!inference_helper_) {
        PRINT_E("Inference helper is not created\n");
        return kRetErr;
    }
    inference_helper_->Finalize();
    return kRetOk;
}


int32_t PoseEngine::Process(const cv::Mat& original_mat, Result& result)
{
    if (!inference_helper_) {
        PRINT_E("Inference helper is not created\n");
        return kRetErr;
    }
    /*** PreProcess ***/
    const auto& t_pre_process0 = std::chrono::steady_clock::now();
    InputTensorInfo& input_tensor_info = input_tensor_info_list_[0];
    /* do resize and color conversion here because some inference engine doesn't support these operations */
    cv::Mat img_src;
    cv::resize(original_mat, img_src, cv::Size(input_tensor_info.GetWidth(), input_tensor_info.GetHeight()));
#ifndef CV_COLOR_IS_RGB
    cv::cvtColor(img_src, img_src, cv::COLOR_BGR2RGB);
#endif
    input_tensor_info.data = img_src.data;
    input_tensor_info.data_type = InputTensorInfo::kDataTypeImage;
    input_tensor_info.image_info.width = img_src.cols;
    input_tensor_info.image_info.height = img_src.rows;
    input_tensor_info.image_info.channel = img_src.channels();
    input_tensor_info.image_info.crop_x = 0;
    input_tensor_info.image_info.crop_y = 0;
    input_tensor_info.image_info.crop_width = img_src.cols;
    input_tensor_info.image_info.crop_height = img_src.rows;
    input_tensor_info.image_info.is_bgr = false;
    input_tensor_info.image_info.swap_color = false;
    if (inference_helper_->PreProcess(input_tensor_info_list_) != InferenceHelper::kRetOk) {
        return kRetErr;
    }
    const auto& t_pre_process1 = std::chrono::steady_clock::now();

    /*** Inference ***/
    const auto& t_inference0 = std::chrono::steady_clock::now();
    if (inference_helper_->Process(output_tensor_info_list_) != InferenceHelper::kRetOk) {
        return kRetErr;
    }
    const auto& t_inference1 = std::chrono::steady_clock::now();

    /*** PostProcess ***/
    const auto& t_post_process0 = std::chrono::steady_clock::now();
    std::vector<float> pose_scores;
    std::vector<std::vector<float>> pose_keypoint_scores;
    std::vector<std::vector<std::pair<float,float>>> pose_eypoint_coords;	// x, y
    decodeMultiPose(output_tensor_info_list_[0], output_tensor_info_list_[1], output_tensor_info_list_[2], output_tensor_info_list_[3], pose_scores, pose_keypoint_scores, pose_eypoint_coords);

    float scaleX = static_cast<float>(original_mat.cols) / input_tensor_info.GetWidth();
    float scaleY = static_cast<float>(original_mat.rows) / input_tensor_info.GetHeight();
    for (int32_t i = 0; i < pose_scores.size(); ++i) {
        for (int32_t id = 0; id < NUM_KEYPOINTS; ++id) {
            pose_eypoint_coords[i][id].first *= scaleX;
            pose_eypoint_coords[i][id].second *= scaleY;
        }
    }
    const auto& t_post_process1 = std::chrono::steady_clock::now();

    /* Return the results */
    result.pose_scores = pose_scores;
    result.pose_keypoint_scores = pose_keypoint_scores;
    result.pose_eypoint_coords = pose_eypoint_coords;
    result.time_pre_process = static_cast<std::chrono::duration<double>>(t_pre_process1 - t_pre_process0).count() * 1000.0;
    result.time_inference = static_cast<std::chrono::duration<double>>(t_inference1 - t_inference0).count() * 1000.0;
    result.time_post_process = static_cast<std::chrono::duration<double>>(t_post_process1 - t_post_process0).count() * 1000.0;;

    return kRetOk;
}
