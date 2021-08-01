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
#include "style_transfer_engine.h"

/*** Macro ***/
#define TAG "StyleTransferEngine"
#define PRINT(...)   COMMON_HELPER_PRINT(TAG, __VA_ARGS__)
#define PRINT_E(...) COMMON_HELPER_PRINT_E(TAG, __VA_ARGS__)

/* Model parameters */
#define MODEL_NAME   "017_Artistic-Style-Transfer_transfer.mnn"
#define INPUT_IMAGE_NAME   "inputs"
#define INPUT_IMAGE_DIMS    { 1, 3, 384, 384 }
#define INPUT_STYLE_NAME   "inputs_1"
#define INPUT_STYLE_DIMS    { 1, 100, 1, 1 }
#define OUTPUT_NAME  "model/tf.math.sigmoid/Sigmoid"
#define TENSORTYPE    TensorInfo::kTensorTypeFp32
#define IS_NCHW       true


/*** Function ***/
int32_t StyleTransferEngine::Initialize(const std::string& work_dir, const int32_t num_threads)
{
	/* Set model information */
	std::string model_filename = work_dir + "/model/" + MODEL_NAME;

	/* Set input tensor info */
	input_tensor_info_list_.clear();
	InputTensorInfo input_tensor_info(INPUT_IMAGE_NAME, TENSORTYPE, IS_NCHW);
	input_tensor_info.tensor_dims = INPUT_IMAGE_DIMS;
	input_tensor_info.data_type = InputTensorInfo::kDataTypeImage;
	input_tensor_info.normalize.mean[0] = 0.0f;
	input_tensor_info.normalize.mean[1] = 0.0f;
	input_tensor_info.normalize.mean[2] = 0.0f;
	input_tensor_info.normalize.norm[0] = 1.0f;
	input_tensor_info.normalize.norm[1] = 1.0f;
	input_tensor_info.normalize.norm[2] = 1.0f;
	input_tensor_info_list_.push_back(input_tensor_info);
	InputTensorInfo input_tensor_info_style(INPUT_STYLE_NAME, TENSORTYPE, IS_NCHW);
	input_tensor_info_style.tensor_dims = INPUT_STYLE_DIMS;
	input_tensor_info_style.data_type = InputTensorInfo::kDataTypeBlobNhwc;
	input_tensor_info_style.normalize.mean[0] = 0.0f;
	input_tensor_info_style.normalize.mean[1] = 0.0f;
	input_tensor_info_style.normalize.mean[2] = 0.0f;
	input_tensor_info_style.normalize.norm[0] = 1.0f;
	input_tensor_info_style.normalize.norm[1] = 1.0f;
	input_tensor_info_style.normalize.norm[2] = 1.0f;
	input_tensor_info_list_.push_back(input_tensor_info_style);

	/* Set output tensor info */
	output_tensor_info_list_.clear();
	output_tensor_info_list_.push_back(OutputTensorInfo(OUTPUT_NAME, TENSORTYPE));

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

int32_t StyleTransferEngine::Finalize()
{
	if (!inference_helper_) {
		PRINT_E("Inference helper is not created\n");
		return kRetErr;
	}
	inference_helper_->Finalize();
	return kRetOk;
}


int32_t StyleTransferEngine::Process(const cv::Mat& original_mat, const float styleBottleneck[], const int32_t lengthStyleBottleneck, Result& result)
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

	InputTensorInfo& inputTensorInfoBottleneck = input_tensor_info_list_[1];
	inputTensorInfoBottleneck.data = const_cast<float*>(styleBottleneck);
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
	cv::Mat out_mat_fp(cv::Size(output_tensor_info_list_[0].tensor_dims[2], output_tensor_info_list_[0].tensor_dims[1]), CV_32FC3, const_cast<float*>(output_tensor_info_list_[0].GetDataAsFloat()));
	cv::Mat out_mat;
	out_mat_fp.convertTo(out_mat, CV_8UC3, 255);
	const auto& t_post_process1 = std::chrono::steady_clock::now();

	/* Return the results */
	result.image = out_mat;
	result.time_pre_process = static_cast<std::chrono::duration<double>>(t_pre_process1 - t_pre_process0).count() * 1000.0;
	result.time_inference = static_cast<std::chrono::duration<double>>(t_inference1 - t_inference0).count() * 1000.0;
	result.time_post_process = static_cast<std::chrono::duration<double>>(t_post_process1 - t_post_process0).count() * 1000.0;;

	return kRetOk;
}

