#include <jni.h>
#include <string>

#include <opencv2/opencv.hpp>
#include "ImageProcessor.h"

#define MODEL_FILENAME "/sdcard/models/posenet-mobilenet_v1_075.mnn"

extern "C" JNIEXPORT jint JNICALL
Java_com_iwatake_viewandroid_MainActivity_ImageProcessorInitialize(
        JNIEnv* env,
        jobject /* this */) {

    int ret = 0;
    ret = ImageProcessor_initialize(MODEL_FILENAME);
    return ret;
}

extern "C" JNIEXPORT jint JNICALL
Java_com_iwatake_viewandroid_MainActivity_ImageProcessorProcess(
        JNIEnv* env,
        jobject, /* this */
        jlong   objMat) {

    int ret = 0;
    cv::Mat* mat = (cv::Mat*) objMat;
    ret = ImageProcessor_process(mat);
    return ret;
}

extern "C" JNIEXPORT jint JNICALL
Java_com_iwatake_viewandroid_MainActivity_ImageProcessorFinalize(
        JNIEnv* env,
        jobject /* this */) {

    int ret = 0;
    ret = ImageProcessor_finalize();
    return ret;
}
