// cuda_wavelet.h
#ifndef CUDA_WAVELET_H
#define CUDA_WAVELET_H

#include <opencv2/opencv.hpp>

// Function to initialize CUDA device
void initializeCUDA();

// Function to compress image using CUDA
cv::Mat compressImageCUDA(const cv::Mat& input, double threshold = 0.3, int decompositionLevels = 3);

#endif // CUDA_WAVELET_H

