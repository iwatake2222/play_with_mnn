
#ifndef IMAGE_PROCESSOR_H_
#define IMAGE_PROCESSOR_H_

namespace cv {
	class Mat;
};

int ImageProcessor_initialize(const char *modelFilename);
int ImageProcessor_process(cv::Mat *mat);
int ImageProcessor_finalize(void);

#endif
