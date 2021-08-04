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
#include <memory>

/* for OpenCV */
#include <opencv2/opencv.hpp>

/* for My modules */
#include "common_helper.h"
#include "pose_engine.h"
#include "image_processor.h"

/*** Macro ***/
#define TAG "ImageProcessor"
#define PRINT(...)   COMMON_HELPER_PRINT(TAG, __VA_ARGS__)
#define PRINT_E(...) COMMON_HELPER_PRINT_E(TAG, __VA_ARGS__)

/*** Global variable ***/
std::unique_ptr<PoseEngine> s_engine;

/*** Function ***/
static cv::Scalar CreateCvColor(int32_t b, int32_t g, int32_t r) {
#ifdef CV_COLOR_IS_RGB
    return cv::Scalar(r, g, b);
#else
    return cv::Scalar(b, g, r);
#endif
}


int32_t ImageProcessor::Initialize(const InputParam& input_param)
{
    if (s_engine) {
        PRINT_E("Already initialized\n");
        return -1;
    }

    s_engine.reset(new PoseEngine());
    if (s_engine->Initialize(input_param.work_dir, input_param.num_threads) != PoseEngine::kRetOk) {
        return -1;
    }
    return 0;
}

int32_t ImageProcessor::Finalize(void)
{
    if (!s_engine) {
        PRINT_E("Not initialized\n");
        return -1;
    }

    if (s_engine->Finalize() != PoseEngine::kRetOk) {
        return -1;
    }

    return 0;
}


int32_t ImageProcessor::Command(int32_t cmd)
{
    if (!s_engine) {
        PRINT_E("Not initialized\n");
        return -1;
    }

    switch (cmd) {
    case 0:
    default:
        PRINT_E("command(%d) is not supported\n", cmd);
        return -1;
    }
}


int32_t ImageProcessor::Process(cv::Mat& mat, Result& result)
{
    if (!s_engine) {
        PRINT_E("Not initialized\n");
        return -1;
    }

    PoseEngine::Result pose_result;
    if (s_engine->Process(mat, pose_result) != PoseEngine::kRetOk) {
        return -1;
    }

    /* Draw the result */
    for (const auto& body : pose_result.pose_eypoint_coords) {
        for (const auto& part : body) {
            cv::circle(mat, cv::Point(static_cast<int32_t>(part.first), static_cast<int32_t>(part.second)), 5, cv::Scalar(0, 255, 0), -1);
        }
    }

    /* Return the results */
    result.time_pre_process = pose_result.time_pre_process;
    result.time_inference = pose_result.time_inference;
    result.time_post_process = pose_result.time_post_process;

    return 0;
}

