#include "cuda_wavelet.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <vector>
#include <cmath>

// Error checking macro for CUDA calls
#define CUDA_CHECK(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error in " << __FILE__ << " at line " << __LINE__ << ": " \
                  << cudaGetErrorString(err) << std::endl; \
        exit(EXIT_FAILURE); \
    } \
} while (0)

// Host constants for temporary use
const double h_H0 = 1.0 / std::sqrt(2.0);
const double h_H1 = 1.0 / std::sqrt(2.0);
const double h_G0 = 1.0 / std::sqrt(2.0);
const double h_G1 = -1.0 / std::sqrt(2.0);

// Device constants
__constant__ double H0, H1, G0, G1;

// Function to find next power of 2
int nextPowerOf2cuda(int n) {
    int power = 1;
    while (power < n) {
        power *= 2;
    }
    return power;
}

// Function to pad image to power of 2
cv::Mat padToPowerOf2cuda(const cv::Mat& input) {
    int targetRows = nextPowerOf2cuda(input.rows);
    int targetCols = nextPowerOf2cuda(input.cols);

    if (targetRows == input.rows && targetCols == input.cols) {
        return input.clone();
    }

    cv::Mat padded;
    cv::copyMakeBorder(input, padded, 0, targetRows - input.rows, 0, targetCols - input.cols,
        cv::BORDER_REPLICATE);
    return padded;
}

// New function to pad image to even dimensions
cv::Mat padToEvenCUDA(const cv::Mat& input) {
    int targetRows = input.rows % 2 == 0 ? input.rows : input.rows + 1;
    int targetCols = input.cols % 2 == 0 ? input.cols : input.cols + 1;

    if (targetRows == input.rows && targetCols == input.cols) {
        return input.clone();
    }

    cv::Mat padded;
    cv::copyMakeBorder(input, padded, 0, targetRows - input.rows, 0, targetCols - input.cols,
        cv::BORDER_REPLICATE);
    return padded;
}

// Function to crop back to original size
cv::Mat cropToOriginalSizecuda(const cv::Mat& input, int originalRows, int originalCols) {
    return input(cv::Rect(0, 0, originalCols, originalRows));
}

// CUDA kernel for row transform
__global__ void transformRowsKernel(double* d_data, double* d_temp, int currentRows, int currentCols, int stride) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < currentRows && i < currentCols / 2) {
        d_temp[row * stride + i] = H0 * d_data[row * stride + 2 * i] + H1 * d_data[row * stride + 2 * i + 1];
        d_temp[row * stride + i + currentCols / 2] = G0 * d_data[row * stride + 2 * i] + G1 * d_data[row * stride + 2 * i + 1];
    }
}

// CUDA kernel for column transform
__global__ void transformColsKernel(double* d_data, double* d_temp, int currentRows, int currentCols, int stride) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < currentCols && i < currentRows / 2) {
        d_temp[i * stride + col] = H0 * d_data[(2 * i) * stride + col] + H1 * d_data[(2 * i + 1) * stride + col];
        d_temp[(i + currentRows / 2) * stride + col] = G0 * d_data[(2 * i) * stride + col] + G1 * d_data[(2 * i + 1) * stride + col];
    }
}

// CUDA kernel for inverse row transform
__global__ void inverseTransformRowsKernel(double* d_data, double* d_temp, int currentRows, int currentCols, int stride) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < currentRows && i < currentCols / 2) {
        d_temp[row * stride + 2 * i] = H0 * d_data[row * stride + i] + G0 * d_data[row * stride + i + currentCols / 2];
        d_temp[row * stride + 2 * i + 1] = H1 * d_data[row * stride + i] + G1 * d_data[row * stride + i + currentCols / 2];
    }
}

// CUDA kernel for inverse column transform
__global__ void inverseTransformColsKernel(double* d_data, double* d_temp, int currentRows, int currentCols, int stride) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < currentCols && i < currentRows / 2) {
        d_temp[(2 * i) * stride + col] = H0 * d_data[i * stride + col] + G0 * d_data[(i + currentRows / 2) * stride + col];
        d_temp[(2 * i + 1) * stride + col] = H1 * d_data[i * stride + col] + G1 * d_data[(i + currentRows / 2) * stride + col];
    }
}

// CUDA kernel for copying data from temp buffer back to data buffer
__global__ void copyBufferKernel(double* d_data, double* d_temp, int rows, int cols, int stride) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols) {
        d_data[row * stride + col] = d_temp[row * stride + col];
    }
}

// CUDA kernel for thresholding
__global__ void thresholdKernel(double* d_data, int rows, int cols, int stride, double threshold) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols) {
        if (fabs(d_data[row * stride + col]) < threshold) {
            d_data[row * stride + col] = 0.0;
        }
    }
}

// Function to transform rows using CUDA
void transformRowsCUDA(double* d_data, double* d_temp, int currentRows, int currentCols, int stride) {
    dim3 blockSize(16, 16);
    dim3 gridSize((currentCols / 2 + blockSize.x - 1) / blockSize.x,
        (currentRows + blockSize.y - 1) / blockSize.y);

    transformRowsKernel << <gridSize, blockSize >> > (d_data, d_temp, currentRows, currentCols, stride);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    gridSize = dim3((currentCols + blockSize.x - 1) / blockSize.x,
        (currentRows + blockSize.y - 1) / blockSize.y);
    copyBufferKernel << <gridSize, blockSize >> > (d_data, d_temp, currentRows, currentCols, stride);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

// Function to transform columns using CUDA
void transformColsCUDA(double* d_data, double* d_temp, int currentRows, int currentCols, int stride) {
    dim3 blockSize(16, 16);
    dim3 gridSize((currentCols + blockSize.x - 1) / blockSize.x,
        (currentRows / 2 + blockSize.y - 1) / blockSize.y);

    transformColsKernel << <gridSize, blockSize >> > (d_data, d_temp, currentRows, currentCols, stride);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    gridSize = dim3((currentCols + blockSize.x - 1) / blockSize.x,
        (currentRows + blockSize.y - 1) / blockSize.y);
    copyBufferKernel << <gridSize, blockSize >> > (d_data, d_temp, currentRows, currentCols, stride);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

// Function for inverse transform of rows using CUDA
void inverseTransformRowsCUDA(double* d_data, double* d_temp, int currentRows, int currentCols, int stride) {
    dim3 blockSize(16, 16);
    dim3 gridSize((currentCols / 2 + blockSize.x - 1) / blockSize.x,
        (currentRows + blockSize.y - 1) / blockSize.y);

    inverseTransformRowsKernel << <gridSize, blockSize >> > (d_data, d_temp, currentRows, currentCols, stride);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    gridSize = dim3((currentCols + blockSize.x - 1) / blockSize.x,
        (currentRows + blockSize.y - 1) / blockSize.y);
    copyBufferKernel << <gridSize, blockSize >> > (d_data, d_temp, currentRows, currentCols, stride);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

// Function for inverse transform of columns using CUDA
void inverseTransformColsCUDA(double* d_data, double* d_temp, int currentRows, int currentCols, int stride) {
    dim3 blockSize(16, 16);
    dim3 gridSize((currentCols + blockSize.x - 1) / blockSize.x,
        (currentRows / 2 + blockSize.y - 1) / blockSize.y);

    inverseTransformColsKernel << <gridSize, blockSize >> > (d_data, d_temp, currentRows, currentCols, stride);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    gridSize = dim3((currentCols + blockSize.x - 1) / blockSize.x,
        (currentRows + blockSize.y - 1) / blockSize.y);
    copyBufferKernel << <gridSize, blockSize >> > (d_data, d_temp, currentRows, currentCols, stride);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

// Function to apply threshold using CUDA
void thresholdDataCUDA(double* d_data, int rows, int cols, int stride, double threshold) {
    dim3 blockSize(16, 16);
    dim3 gridSize((cols + blockSize.x - 1) / blockSize.x,
        (rows + blockSize.y - 1) / blockSize.y);

    thresholdKernel << <gridSize, blockSize >> > (d_data, rows, cols, stride, threshold);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

// 2D transform with CUDA
void transform2DCUDA(double* d_data, double* d_temp, int rows, int cols, int stride, int levels) {
    for (int level = 0; level < levels; level++) {
        int currentRows = rows >> level;
        int currentCols = cols >> level;

        if (currentRows < 2 || currentCols < 2) break;

        transformRowsCUDA(d_data, d_temp, currentRows, currentCols, stride);
        transformColsCUDA(d_data, d_temp, currentRows, currentCols, stride);
    }
}

// 2D inverse transform with CUDA
void inverseTransform2DCUDA(double* d_data, double* d_temp, int rows, int cols, int stride, int levels) {
    for (int level = levels - 1; level >= 0; level--) {
        int currentRows = rows >> level;
        int currentCols = cols >> level;

        if (currentRows < 2 || currentCols < 2) break;

        inverseTransformColsCUDA(d_data, d_temp, currentRows, currentCols, stride);
        inverseTransformRowsCUDA(d_data, d_temp, currentRows, currentCols, stride);
    }
}

// Initialize CUDA device and set constants
void initializeCUDA() {
    int deviceCount;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0) {
        std::cerr << "Error: No CUDA-capable devices found!" << std::endl;
        exit(EXIT_FAILURE);
    }

    CUDA_CHECK(cudaSetDevice(0));

    cudaDeviceProp deviceProp;
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, 0));
    std::cout << "Using CUDA device: " << deviceProp.name << std::endl;
    std::cout << "Compute capability: " << deviceProp.major << "." << deviceProp.minor << std::endl;
    std::cout << "Total global memory: " << deviceProp.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;

    CUDA_CHECK(cudaMemcpyToSymbol(H0, &h_H0, sizeof(double)));
    CUDA_CHECK(cudaMemcpyToSymbol(H1, &h_H1, sizeof(double)));
    CUDA_CHECK(cudaMemcpyToSymbol(G0, &h_G0, sizeof(double)));
    CUDA_CHECK(cudaMemcpyToSymbol(G1, &h_G1, sizeof(double)));
}

// Main compression function using CUDA
cv::Mat compressImageCUDA(const cv::Mat& input, double threshold, int decompositionLevels) {
    int originalRows = input.rows;
    int originalCols = input.cols;
    bool isColor = input.channels() == 3;

    if (isColor) {
        std::vector<cv::Mat> channels;
        cv::split(input, channels);
        std::vector<cv::Mat> compressedChannels(channels.size());

        for (int i = 0; i < channels.size(); i++) {
            // Use padToEvenCUDA for levels = 1, otherwise padToPowerOf2CUDA
            cv::Mat padded = (decompositionLevels == 1) ? padToEvenCUDA(channels[i]) : padToPowerOf2cuda(channels[i]);
            cv::Mat normalized;
            padded.convertTo(normalized, CV_64F, 1.0 / 255.0);

            int rows = normalized.rows;
            int cols = normalized.cols;
            int stride = cols;

            double* d_data;
            double* d_temp;
            CUDA_CHECK(cudaMalloc(&d_data, rows * cols * sizeof(double)));
            CUDA_CHECK(cudaMalloc(&d_temp, rows * cols * sizeof(double)));

            CUDA_CHECK(cudaMemcpy(d_data, normalized.ptr<double>(0), rows * cols * sizeof(double), cudaMemcpyHostToDevice));

            transform2DCUDA(d_data, d_temp, rows, cols, stride, decompositionLevels);
            thresholdDataCUDA(d_data, rows, cols, stride, threshold);
            inverseTransform2DCUDA(d_data, d_temp, rows, cols, stride, decompositionLevels);

            CUDA_CHECK(cudaMemcpy(normalized.ptr<double>(0), d_data, rows * cols * sizeof(double), cudaMemcpyDeviceToHost));

            CUDA_CHECK(cudaFree(d_data));
            CUDA_CHECK(cudaFree(d_temp));

            cv::Mat compressed;
            normalized.convertTo(compressed, CV_8U, 255.0);
            compressedChannels[i] = cropToOriginalSizecuda(compressed, originalRows, originalCols);
        }

        cv::Mat compressed;
        cv::merge(compressedChannels, compressed);
        return compressed;
    }
    else {
        // Use padToEvenCUDA for levels = 1, otherwise padToPowerOf2CUDA
        cv::Mat padded = (decompositionLevels == 1) ? padToEvenCUDA(input) : padToPowerOf2cuda(input);
        cv::Mat normalized;
        padded.convertTo(normalized, CV_64F, 1.0 / 255.0);

        int rows = normalized.rows;
        int cols = normalized.cols;
        int stride = cols;

        double* d_data;
        double* d_temp;
        CUDA_CHECK(cudaMalloc(&d_data, rows * cols * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_temp, rows * cols * sizeof(double)));

        CUDA_CHECK(cudaMemcpy(d_data, normalized.ptr<double>(0), rows * cols * sizeof(double), cudaMemcpyHostToDevice));

        transform2DCUDA(d_data, d_temp, rows, cols, stride, decompositionLevels);
        thresholdDataCUDA(d_data, rows, cols, stride, threshold);
        inverseTransform2DCUDA(d_data, d_temp, rows, cols, stride, decompositionLevels);

        CUDA_CHECK(cudaMemcpy(normalized.ptr<double>(0), d_data, rows * cols * sizeof(double), cudaMemcpyDeviceToHost));

        CUDA_CHECK(cudaFree(d_data));
        CUDA_CHECK(cudaFree(d_temp));

        cv::Mat compressed;
        normalized.convertTo(compressed, CV_8U, 255.0);
        return cropToOriginalSizecuda(compressed, originalRows, originalCols);
    }
}