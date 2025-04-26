#include <iostream>
#include <vector>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <omp.h>
#include "cuda_wavelet.h" // Include CUDA header
#include <filesystem>
#include <algorithm>
#include <fstream>

// Haar wavelet coefficients
const double H0 = 1.0 / std::sqrt(2.0);
const double H1 = 1.0 / std::sqrt(2.0);
const double G0 = 1.0 / std::sqrt(2.0);
const double G1 = -1.0 / std::sqrt(2.0);

// Function to find next power of 2
int nextPowerOf2(int n) {
    int power = 1;
    while (power < n) {
        power *= 2;
    }
    return power;
}

// Function to pad image to power of 2
cv::Mat padToPowerOf2(const cv::Mat& input) {
    int targetRows = nextPowerOf2(input.rows);
    int targetCols = nextPowerOf2(input.cols);

    if (targetRows == input.rows && targetCols == input.cols) {
        return input.clone();
    }

    cv::Mat padded;
    cv::copyMakeBorder(input, padded, 0, targetRows - input.rows, 0, targetCols - input.cols,
        cv::BORDER_REPLICATE);
    return padded;
}

// New function to pad image to even dimensions
cv::Mat padToEven(const cv::Mat& input) {
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
cv::Mat cropToOriginalSize(const cv::Mat& input, int originalRows, int originalCols) {
    return input(cv::Rect(0, 0, originalCols, originalRows));
}

// Serial version functions
void transformRowsSerial(std::vector<std::vector<double>>& data, int currentRows, int currentCols) {
    std::vector<std::vector<double>> temp = data;
    for (int row = 0; row < currentRows; row++) {
        for (int i = 0; i < currentCols / 2; i++) {
            temp[row][i] = H0 * data[row][2 * i] + H1 * data[row][2 * i + 1];
            temp[row][i + currentCols / 2] = G0 * data[row][2 * i] + G1 * data[row][2 * i + 1];
        }
    }
    for (int row = 0; row < currentRows; row++) {
        for (int col = 0; col < currentCols; col++) {
            data[row][col] = temp[row][col];
        }
    }
}

void transformColsSerial(std::vector<std::vector<double>>& data, int currentRows, int currentCols) {
    std::vector<std::vector<double>> temp = data;
    for (int col = 0; col < currentCols; col++) {
        for (int i = 0; i < currentRows / 2; i++) {
            temp[i][col] = H0 * data[2 * i][col] + H1 * data[2 * i + 1][col];
            temp[i + currentRows / 2][col] = G0 * data[2 * i][col] + G1 * data[2 * i + 1][col];
        }
    }
    for (int row = 0; row < currentRows; row++) {
        for (int col = 0; col < currentCols; col++) {
            data[row][col] = temp[row][col];
        }
    }
}

void inverseTransformRowsSerial(std::vector<std::vector<double>>& data, int currentRows, int currentCols) {
    std::vector<std::vector<double>> temp = data;
    for (int row = 0; row < currentRows; row++) {
        for (int i = 0; i < currentCols / 2; i++) {
            temp[row][2 * i] = H0 * data[row][i] + G0 * data[row][i + currentCols / 2];
            temp[row][2 * i + 1] = H1 * data[row][i] + G1 * data[row][i + currentCols / 2];
        }
    }
    for (int row = 0; row < currentRows; row++) {
        for (int col = 0; col < currentCols; col++) {
            data[row][col] = temp[row][col];
        }
    }
}

void inverseTransformColsSerial(std::vector<std::vector<double>>& data, int currentRows, int currentCols) {
    std::vector<std::vector<double>> temp = data;
    for (int col = 0; col < currentCols; col++) {
        for (int i = 0; i < currentRows / 2; i++) {
            temp[2 * i][col] = H0 * data[i][col] + G0 * data[i + currentRows / 2][col];
            temp[2 * i + 1][col] = H1 * data[i][col] + G1 * data[i + currentRows / 2][col];
        }
    }
    for (int row = 0; row < currentRows; row++) {
        for (int col = 0; col < currentCols; col++) {
            data[row][col] = temp[row][col];
        }
    }
}

void transform2DSerial(std::vector<std::vector<double>>& image, int levels) {
    int rows = image.size();
    int cols = image[0].size();
    for (int level = 0; level < levels; level++) {
        int currentRows = rows >> level;
        int currentCols = cols >> level;
        if (currentRows < 2 || currentCols < 2) break;
        transformRowsSerial(image, currentRows, currentCols);
        transformColsSerial(image, currentRows, currentCols);
    }
}

void inverseTransform2DSerial(std::vector<std::vector<double>>& image, int levels) {
    int rows = image.size();
    int cols = image[0].size();
    for (int level = levels - 1; level >= 0; level--) {
        int currentRows = rows >> level;
        int currentCols = cols >> level;
        if (currentRows < 2 || currentCols < 2) break;
        inverseTransformColsSerial(image, currentRows, currentCols);
        inverseTransformRowsSerial(image, currentRows, currentCols);
    }
}

void thresholdDataSerial(std::vector<std::vector<double>>& image, double threshold) {
    for (int i = 0; i < image.size(); i++) {
        for (int j = 0; j < image[i].size(); j++) {
            if (fabs(image[i][j]) < threshold) {
                image[i][j] = 0.0;
            }
        }
    }
}

cv::Mat compressImageSerial(const cv::Mat& input, double threshold, int decompositionLevels) {
    int originalRows = input.rows;
    int originalCols = input.cols;
    bool isColor = input.channels() == 3;

    if (isColor) {
        std::vector<cv::Mat> channels;
        cv::split(input, channels);
        for (int i = 0; i < channels.size(); i++) {
            // Use padToEven for levels = 1, otherwise padToPowerOf2
            cv::Mat padded = (decompositionLevels == 1) ? padToEven(channels[i]) : padToPowerOf2(channels[i]);
            cv::Mat normalized;
            padded.convertTo(normalized, CV_64F, 1.0 / 255.0);
            std::vector<std::vector<double>> channelData(normalized.rows, std::vector<double>(normalized.cols));
            for (int r = 0; r < normalized.rows; r++) {
                for (int c = 0; c < normalized.cols; c++) {
                    channelData[r][c] = normalized.at<double>(r, c);
                }
            }
            transform2DSerial(channelData, decompositionLevels);
            thresholdDataSerial(channelData, threshold);
            inverseTransform2DSerial(channelData, decompositionLevels);
            for (int r = 0; r < normalized.rows; r++) {
                for (int c = 0; c < normalized.cols; c++) {
                    normalized.at<double>(r, c) = channelData[r][c];
                }
            }
            cv::Mat compressed;
            normalized.convertTo(compressed, CV_8U, 255.0);
            channels[i] = cropToOriginalSize(compressed, originalRows, originalCols);
        }
        cv::Mat compressed;
        cv::merge(channels, compressed);
        return compressed;
    }
    else {
        // Use padToEven for levels = 1, otherwise padToPowerOf2
        cv::Mat padded = (decompositionLevels == 1) ? padToEven(input) : padToPowerOf2(input);
        cv::Mat normalized;
        padded.convertTo(normalized, CV_64F, 1.0 / 255.0);
        std::vector<std::vector<double>> imageData(normalized.rows, std::vector<double>(normalized.cols));
        for (int i = 0; i < normalized.rows; i++) {
            for (int j = 0; j < normalized.cols; j++) {
                imageData[i][j] = normalized.at<double>(i, j);
            }
        }
        transform2DSerial(imageData, decompositionLevels);
        thresholdDataSerial(imageData, threshold);
        inverseTransform2DSerial(imageData, decompositionLevels);
        for (int i = 0; i < normalized.rows; i++) {
            for (int j = 0; j < normalized.cols; j++) {
                normalized.at<double>(i, j) = imageData[i][j];
            }
        }
        cv::Mat compressed;
        normalized.convertTo(compressed, CV_8U, 255.0);
        return cropToOriginalSize(compressed, originalRows, originalCols);
    }
}

// OpenMP version functions
void transformRowsOpenMP(std::vector<std::vector<double>>& data, int currentRows, int currentCols) {
    std::vector<std::vector<double>> temp = data;
#pragma omp parallel for
    for (int row = 0; row < currentRows; row++) {
        for (int i = 0; i < currentCols / 2; i++) {
            temp[row][i] = H0 * data[row][2 * i] + H1 * data[row][2 * i + 1];
            temp[row][i + currentCols / 2] = G0 * data[row][2 * i] + G1 * data[row][2 * i + 1];
        }
    }
#pragma omp parallel for
    for (int row = 0; row < currentRows; row++) {
        for (int col = 0; col < currentCols; col++) {
            data[row][col] = temp[row][col];
        }
    }
}

void transformColsOpenMP(std::vector<std::vector<double>>& data, int currentRows, int currentCols) {
    std::vector<std::vector<double>> temp = data;
#pragma omp parallel for
    for (int col = 0; col < currentCols; col++) {
        for (int i = 0; i < currentRows / 2; i++) {
            temp[i][col] = H0 * data[2 * i][col] + H1 * data[2 * i + 1][col];
            temp[i + currentRows / 2][col] = G0 * data[2 * i][col] + G1 * data[2 * i + 1][col];
        }
    }
#pragma omp parallel for
    for (int row = 0; row < currentRows; row++) {
        for (int col = 0; col < currentCols; col++) {
            data[row][col] = temp[row][col];
        }
    }
}

void inverseTransformRowsOpenMP(std::vector<std::vector<double>>& data, int currentRows, int currentCols) {
    std::vector<std::vector<double>> temp = data;
#pragma omp parallel for
    for (int row = 0; row < currentRows; row++) {
        for (int i = 0; i < currentCols / 2; i++) {
            temp[row][2 * i] = H0 * data[row][i] + G0 * data[row][i + currentCols / 2];
            temp[row][2 * i + 1] = H1 * data[row][i] + G1 * data[row][i + currentCols / 2];
        }
    }
#pragma omp parallel for
    for (int row = 0; row < currentRows; row++) {
        for (int col = 0; col < currentCols; col++) {
            data[row][col] = temp[row][col];
        }
    }
}

void inverseTransformColsOpenMP(std::vector<std::vector<double>>& data, int currentRows, int currentCols) {
    std::vector<std::vector<double>> temp = data;
#pragma omp parallel for
    for (int col = 0; col < currentCols; col++) {
        for (int i = 0; i < currentRows / 2; i++) {
            temp[2 * i][col] = H0 * data[i][col] + G0 * data[i + currentRows / 2][col];
            temp[2 * i + 1][col] = H1 * data[i][col] + G1 * data[i + currentRows / 2][col];
        }
    }
#pragma omp parallel for
    for (int row = 0; row < currentRows; row++) {
        for (int col = 0; col < currentCols; col++) {
            data[row][col] = temp[row][col];
        }
    }
}

void transform2DOpenMP(std::vector<std::vector<double>>& image, int levels) {
    int rows = image.size();
    int cols = image[0].size();
    for (int level = 0; level < levels; level++) {
        int currentRows = rows >> level;
        int currentCols = cols >> level;
        if (currentRows < 2 || currentCols < 2) break;
        transformRowsOpenMP(image, currentRows, currentCols);
        transformColsOpenMP(image, currentRows, currentCols);
    }
}

void inverseTransform2DOpenMP(std::vector<std::vector<double>>& image, int levels) {
    int rows = image.size();
    int cols = image[0].size();
    for (int level = levels - 1; level >= 0; level--) {
        int currentRows = rows >> level;
        int currentCols = cols >> level;
        if (currentRows < 2 || currentCols < 2) break;
        inverseTransformColsOpenMP(image, currentRows, currentCols);
        inverseTransformRowsOpenMP(image, currentRows, currentCols);
    }
}

void thresholdDataOpenMP(std::vector<std::vector<double>>& image, double threshold) {
    int rows = image.size();
    int cols = image[0].size();
#pragma omp parallel for collapse(2)
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if (fabs(image[i][j]) < threshold) {
                image[i][j] = 0.0;
            }
        }
    }
}

cv::Mat compressImageOpenMP(const cv::Mat& input, double threshold, int decompositionLevels) {
    int originalRows = input.rows;
    int originalCols = input.cols;
    bool isColor = input.channels() == 3;

    if (isColor) {
        std::vector<cv::Mat> channels;
        cv::split(input, channels);
#pragma omp parallel sections
        {
#pragma omp section
            {
                // Use padToEven for levels = 1, otherwise padToPowerOf2
                cv::Mat padded = (decompositionLevels == 1) ? padToEven(channels[0]) : padToPowerOf2(channels[0]);
                cv::Mat normalized;
                padded.convertTo(normalized, CV_64F, 1.0 / 255.0);
                std::vector<std::vector<double>> channelData(normalized.rows, std::vector<double>(normalized.cols));
                for (int r = 0; r < normalized.rows; r++) {
                    for (int c = 0; c < normalized.cols; c++) {
                        channelData[r][c] = normalized.at<double>(r, c);
                    }
                }
                transform2DOpenMP(channelData, decompositionLevels);
                thresholdDataOpenMP(channelData, threshold);
                inverseTransform2DOpenMP(channelData, decompositionLevels);
                for (int r = 0; r < normalized.rows; r++) {
                    for (int c = 0; c < normalized.cols; c++) {
                        normalized.at<double>(r, c) = channelData[r][c];
                    }
                }
                cv::Mat compressed;
                normalized.convertTo(compressed, CV_8U, 255.0);
                channels[0] = cropToOriginalSize(compressed, originalRows, originalCols);
            }
#pragma omp section
            {
                // Use padToEven for levels = 1, otherwise padToPowerOf2
                cv::Mat padded = (decompositionLevels == 1) ? padToEven(channels[1]) : padToPowerOf2(channels[1]);
                cv::Mat normalized;
                padded.convertTo(normalized, CV_64F, 1.0 / 255.0);
                std::vector<std::vector<double>> channelData(normalized.rows, std::vector<double>(normalized.cols));
                for (int r = 0; r < normalized.rows; r++) {
                    for (int c = 0; c < normalized.cols; c++) {
                        channelData[r][c] = normalized.at<double>(r, c);
                    }
                }
                transform2DOpenMP(channelData, decompositionLevels);
                thresholdDataOpenMP(channelData, threshold);
                inverseTransform2DOpenMP(channelData, decompositionLevels);
                for (int r = 0; r < normalized.rows; r++) {
                    for (int c = 0; c < normalized.cols; c++) {
                        normalized.at<double>(r, c) = channelData[r][c];
                    }
                }
                cv::Mat compressed;
                normalized.convertTo(compressed, CV_8U, 255.0);
                channels[1] = cropToOriginalSize(compressed, originalRows, originalCols);
            }
#pragma omp section
            {
                // Use padToEven for levels = 1, otherwise padToPowerOf2
                cv::Mat padded = (decompositionLevels == 1) ? padToEven(channels[2]) : padToPowerOf2(channels[2]);
                cv::Mat normalized;
                padded.convertTo(normalized, CV_64F, 1.0 / 255.0);
                std::vector<std::vector<double>> channelData(normalized.rows, std::vector<double>(normalized.cols));
                for (int r = 0; r < normalized.rows; r++) {
                    for (int c = 0; c < normalized.cols; c++) {
                        channelData[r][c] = normalized.at<double>(r, c);
                    }
                }
                transform2DOpenMP(channelData, decompositionLevels);
                thresholdDataOpenMP(channelData, threshold);
                inverseTransform2DOpenMP(channelData, decompositionLevels);
                for (int r = 0; r < normalized.rows; r++) {
                    for (int c = 0; c < normalized.cols; c++) {
                        normalized.at<double>(r, c) = channelData[r][c];
                    }
                }
                cv::Mat compressed;
                normalized.convertTo(compressed, CV_8U, 255.0);
                channels[2] = cropToOriginalSize(compressed, originalRows, originalCols);
            }
        }
        cv::Mat compressed;
        cv::merge(channels, compressed);
        return compressed;
    }
    else {
        // Use padToEven for levels = 1, otherwise padToPowerOf2
        cv::Mat padded = (decompositionLevels == 1) ? padToEven(input) : padToPowerOf2(input);
        cv::Mat normalized;
        padded.convertTo(normalized, CV_64F, 1.0 / 255.0);
        std::vector<std::vector<double>> imageData(normalized.rows, std::vector<double>(normalized.cols));
#pragma omp parallel for collapse(2)
        for (int i = 0; i < normalized.rows; i++) {
            for (int j = 0; j < normalized.cols; j++) {
                imageData[i][j] = normalized.at<double>(i, j);
            }
        }
        transform2DOpenMP(imageData, decompositionLevels);
        thresholdDataOpenMP(imageData, threshold);
        inverseTransform2DOpenMP(imageData, decompositionLevels);
#pragma omp parallel for collapse(2)
        for (int i = 0; i < normalized.rows; i++) {
            for (int j = 0; j < normalized.cols; j++) {
                normalized.at<double>(i, j) = imageData[i][j];
            }
        }
        cv::Mat compressed;
        normalized.convertTo(compressed, CV_8U, 255.0);
        return cropToOriginalSize(compressed, originalRows, originalCols);
    }
}

// Modified savePerformanceToCSV function
void savePerformanceToCSV(const std::string& filename,
    const std::vector<std::string>& imageNames,
    const std::vector<double>& serialTimes,
    const std::vector<double>& openMPTimes,
    const std::vector<double>& cudaTimes,
    const std::vector<std::pair<int, int>>& imageDimensions,
    int numThreads,
    double threshold,
    int levels) {

    // Determine which algorithms were run based on the vectors having data
    bool hasSerialData = !serialTimes.empty();
    bool hasOpenMPData = !openMPTimes.empty();
    bool hasCUDAData = !cudaTimes.empty();

    std::ofstream csvFile(filename);
    if (!csvFile.is_open()) {
        std::cerr << "Error: Could not open CSV file for writing: " << filename << std::endl;
        return;
    }

    // Create header based on which algorithms were run
    csvFile << "Image,Dimensions(WxH)";
    if (hasSerialData) csvFile << ",SerialTime(s)";
    if (hasOpenMPData) csvFile << ",OpenMPTime(s)";
    if (hasCUDAData) csvFile << ",CUDATime(s)";

    // Add headers for speedup and efficiency calculations
    if (hasSerialData && hasOpenMPData) csvFile << ",OpenMPSpeedup,OpenMPEfficiency(%)";
    if (hasSerialData && hasCUDAData) csvFile << ",CUDASpeedup";
    if (hasOpenMPData && hasCUDAData && !hasSerialData) csvFile << ",CUDAvsOpenMPSpeedup";

    csvFile << "\n";

    // Write data for each image
    for (size_t i = 0; i < imageNames.size(); i++) {
        csvFile << imageNames[i] << ","
            << imageDimensions[i].first << "x" << imageDimensions[i].second;

        // Write timing data
        if (hasSerialData) {
            if (i < serialTimes.size()) {
                csvFile << "," << serialTimes[i];
            }
            else {
                csvFile << ",N/A";
            }
        }

        if (hasOpenMPData) {
            if (i < openMPTimes.size()) {
                csvFile << "," << openMPTimes[i];
            }
            else {
                csvFile << ",N/A";
            }
        }

        if (hasCUDAData) {
            if (i < cudaTimes.size()) {
                csvFile << "," << cudaTimes[i];
            }
            else {
                csvFile << ",N/A";
            }
        }

        // Calculate and write speedups and efficiency
        if (hasSerialData && hasOpenMPData) {
            if (i < serialTimes.size() && i < openMPTimes.size() && openMPTimes[i] > 0) {
                double openMPSpeedup = serialTimes[i] / openMPTimes[i];
                double openMPEfficiency = (openMPSpeedup / numThreads) * 100;
                csvFile << "," << openMPSpeedup << "," << openMPEfficiency;
            }
            else {
                csvFile << ",N/A,N/A";
            }
        }

        if (hasSerialData && hasCUDAData) {
            if (i < serialTimes.size() && i < cudaTimes.size() && cudaTimes[i] > 0) {
                double cudaSpeedup = serialTimes[i] / cudaTimes[i];
                csvFile << "," << cudaSpeedup;
            }
            else {
                csvFile << ",N/A";
            }
        }

        if (hasOpenMPData && hasCUDAData && !hasSerialData) {
            if (i < openMPTimes.size() && i < cudaTimes.size() && cudaTimes[i] > 0) {
                double cudaVsOpenMPSpeedup = openMPTimes[i] / cudaTimes[i];
                csvFile << "," << cudaVsOpenMPSpeedup;
            }
            else {
                csvFile << ",N/A";
            }
        }

        csvFile << "\n";
    }

    // Calculate and write averages row
    double totalWidth = 0.0, totalHeight = 0.0;
    for (const auto& dim : imageDimensions) {
        totalWidth += dim.first;
        totalHeight += dim.second;
    }

    double avgWidth = imageDimensions.empty() ? 0 : totalWidth / imageDimensions.size();
    double avgHeight = imageDimensions.empty() ? 0 : totalHeight / imageDimensions.size();

    // Calculate average times
    double totalSerialTime = 0.0, totalOpenMPTime = 0.0, totalCUDATime = 0.0;
    for (const auto& time : serialTimes) totalSerialTime += time;
    for (const auto& time : openMPTimes) totalOpenMPTime += time;
    for (const auto& time : cudaTimes) totalCUDATime += time;

    double avgSerialTime = serialTimes.empty() ? 0 : totalSerialTime / serialTimes.size();
    double avgOpenMPTime = openMPTimes.empty() ? 0 : totalOpenMPTime / openMPTimes.size();
    double avgCUDATime = cudaTimes.empty() ? 0 : totalCUDATime / cudaTimes.size();

    // Write average row
    csvFile << "Average,"
        << static_cast<int>(avgWidth) << "x" << static_cast<int>(avgHeight);

    if (hasSerialData) csvFile << "," << avgSerialTime;
    if (hasOpenMPData) csvFile << "," << avgOpenMPTime;
    if (hasCUDAData) csvFile << "," << avgCUDATime;

    // Calculate and write average speedups and efficiency
    if (hasSerialData && hasOpenMPData && avgOpenMPTime > 0) {
        double avgOpenMPSpeedup = avgSerialTime / avgOpenMPTime;
        double avgOpenMPEfficiency = (avgOpenMPSpeedup / numThreads) * 100;
        csvFile << "," << avgOpenMPSpeedup << "," << avgOpenMPEfficiency;
    }
    else if (hasSerialData && hasOpenMPData) {
        csvFile << ",N/A,N/A";
    }

    if (hasSerialData && hasCUDAData && avgCUDATime > 0) {
        double avgCUDASpeedup = avgSerialTime / avgCUDATime;
        csvFile << "," << avgCUDASpeedup;
    }
    else if (hasSerialData && hasCUDAData) {
        csvFile << ",N/A";
    }

    if (hasOpenMPData && hasCUDAData && !hasSerialData && avgCUDATime > 0) {
        double avgCUDAvsOpenMPSpeedup = avgOpenMPTime / avgCUDATime;
        csvFile << "," << avgCUDAvsOpenMPSpeedup;
    }
    else if (hasOpenMPData && hasCUDAData && !hasSerialData) {
        csvFile << ",N/A";
    }

    csvFile << "\n";

    // Add additional information about the test settings
    //csvFile << "\nTest Settings:\n";
    //csvFile << "Threshold," << threshold << "\n";
    //csvFile << "Decomposition Levels," << levels << "\n";
    //if (hasOpenMPData) csvFile << "OpenMP Threads," << numThreads << "\n";

    csvFile.close();
    std::cout << "Performance data saved to: " << filename << std::endl;
}

// MAIN FUNCTION
int main(int argc, char** argv) {
    try {
        std::string folderPath = (argc > 1) ? argv[1] : "./images/run";
        std::string savePath = (argc > 1) ? argv[1] : "./images";
        std::cout << "Processing images from folder: " << folderPath << std::endl;
        std::vector<std::pair<int, int>> imageDimensions;

        // Set up OpenMP threads
        omp_set_num_threads(16);
        int numThreads = omp_get_max_threads();
        std::cout << "Number of available OpenMP threads: " << numThreads << std::endl;

        // User selection variables
        bool runSerial = false;
        bool runOpenMP = false;
        bool runCUDA = false;

        // Present menu to user
        int choice;
        std::cout << "\n=== IMAGE COMPRESSION ALGORITHM SELECTION ===\n";
        std::cout << "1. Run Serial version only\n";
        std::cout << "2. Run OpenMP version only\n";
        std::cout << "3. Run CUDA version only\n";
        std::cout << "4. Run all versions (Serial, OpenMP, and CUDA)\n";
        std::cout << "Please enter your choice (1-4): ";
        std::cin >> choice;

        // Process user choice
        switch (choice) {
        case 1:
            runSerial = true;
            std::cout << "Selected: Serial version only\n";
            break;
        case 2:
            runOpenMP = true;
            std::cout << "Selected: OpenMP version only\n";
            break;
        case 3:
            runCUDA = true;
            // Initialize CUDA only if needed
            initializeCUDA();
            std::cout << "Selected: CUDA version only\n";
            break;
        case 4:
            runSerial = true;
            runOpenMP = true;
            runCUDA = true;
            initializeCUDA();
            std::cout << "Selected: All versions (Serial, OpenMP, and CUDA)\n";
            break;
        default:
            std::cout << "Invalid choice. Running all versions by default.\n";
            runSerial = true;
            runOpenMP = true;
            runCUDA = true;
            initializeCUDA();
            break;
        }

        double threshold = 0.3;
        int levels = 1;

        std::vector<std::string> imageFiles;
        try {
            for (const auto& entry : std::filesystem::directory_iterator(folderPath)) {
                if (entry.is_regular_file()) {
                    std::string extension = entry.path().extension().string();
                    std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);
                    if (extension == ".jpg" || extension == ".jpeg" || extension == ".png" ||
                        extension == ".bmp" || extension == ".tiff" || extension == ".tif") {
                        imageFiles.push_back(entry.path().string());
                    }
                }
            }
        }
        catch (const std::filesystem::filesystem_error& e) {
            std::cerr << "Filesystem error: " << e.what() << std::endl;
            return -1;
        }

        if (imageFiles.empty()) {
            std::cerr << "No image files found in " << folderPath << std::endl;
            return -1;
        }

        std::cout << "Found " << imageFiles.size() << " image files to process." << std::endl;

        // Only create directories for selected algorithms
        std::string outputFolderSerial, outputFolderOpenMP, outputFolderCUDA;

        if (runSerial) {
            outputFolderSerial = savePath + "/compressed_serial";
            try {
                std::filesystem::create_directory(outputFolderSerial);
            }
            catch (const std::filesystem::filesystem_error& e) {
                std::cerr << "Error creating Serial output directory: " << e.what() << std::endl;
            }
        }

        if (runOpenMP) {
            outputFolderOpenMP = savePath + "/compressed_openmp";
            try {
                std::filesystem::create_directory(outputFolderOpenMP);
            }
            catch (const std::filesystem::filesystem_error& e) {
                std::cerr << "Error creating OpenMP output directory: " << e.what() << std::endl;
            }
        }

        if (runCUDA) {
            outputFolderCUDA = savePath + "/compressed_cuda";
            try {
                std::filesystem::create_directory(outputFolderCUDA);
            }
            catch (const std::filesystem::filesystem_error& e) {
                std::cerr << "Error creating CUDA output directory: " << e.what() << std::endl;
            }
        }

        double totalSerialTime = 0.0;
        double totalOpenMPTime = 0.0;
        double totalCUDATime = 0.0;
        std::vector<std::string> imageNames;
        std::vector<double> serialTimes;
        std::vector<double> openMPTimes;
        std::vector<double> cudaTimes;

        for (const auto& imagePath : imageFiles) {
            std::cout << "\n=== Processing image: " << imagePath << " ===" << std::endl;

            cv::Mat input = cv::imread(imagePath);
            if (input.empty()) {
                std::cerr << "Error: Could not read the image at path: " << imagePath << std::endl;
                continue;
            }

            std::cout << "Successfully loaded image, size: " << input.cols << "x" << input.rows << " pixels" << std::endl;

            std::string baseFilename = std::filesystem::path(imagePath).stem().string();
            imageNames.push_back(baseFilename);

            imageDimensions.push_back(std::make_pair(input.cols, input.rows));
            std::cout << "Image dimensions: " << input.cols << "x" << input.rows << " pixels" << std::endl;

            std::vector<int> compression_params;
            compression_params.push_back(cv::IMWRITE_PNG_COMPRESSION);
            compression_params.push_back(9);

            // Run Serial version if selected
            double serialTime = 0.0;
            if (runSerial) {
                std::cout << "\n--- Running Serial Version ---" << std::endl;
                auto startTimeSerial = std::chrono::high_resolution_clock::now();
                cv::Mat compressedSerial = compressImageSerial(input, threshold, levels);
                auto endTimeSerial = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> durationSerial = endTimeSerial - startTimeSerial;
                serialTime = durationSerial.count();
                std::cout << "Serial compression time: " << serialTime << " seconds" << std::endl;
                serialTimes.push_back(serialTime);

                std::string outputPathSerial = outputFolderSerial + "/" + baseFilename + "_compressed.png";
                cv::imwrite(outputPathSerial, compressedSerial, compression_params);
                std::cout << "Serial compressed image saved as: " << outputPathSerial << std::endl;

                totalSerialTime += serialTime;
            }

            // Run OpenMP version if selected
            double openMPTime = 0.0;
            if (runOpenMP) {
                std::cout << "\n--- Running OpenMP Version ---" << std::endl;
                auto startTimeOpenMP = std::chrono::high_resolution_clock::now();
                cv::Mat compressedOpenMP = compressImageOpenMP(input, threshold, levels);
                auto endTimeOpenMP = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> durationOpenMP = endTimeOpenMP - startTimeOpenMP;
                openMPTime = durationOpenMP.count();
                std::cout << "OpenMP compression time: " << openMPTime << " seconds" << std::endl;
                openMPTimes.push_back(openMPTime);

                std::string outputPathOpenMP = outputFolderOpenMP + "/" + baseFilename + "_compressed.png";
                cv::imwrite(outputPathOpenMP, compressedOpenMP, compression_params);
                std::cout << "OpenMP compressed image saved as: " << outputPathOpenMP << std::endl;

                totalOpenMPTime += openMPTime;
            }

            // Run CUDA version if selected
            double cudaTime = 0.0;
            if (runCUDA) {
                std::cout << "\n--- Running CUDA Version ---" << std::endl;
                auto startTimeCUDA = std::chrono::high_resolution_clock::now();
                cv::Mat compressedCUDA = compressImageCUDA(input, threshold, levels);
                auto endTimeCUDA = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> durationCUDA = endTimeCUDA - startTimeCUDA;
                cudaTime = durationCUDA.count();
                std::cout << "CUDA compression time: " << cudaTime << " seconds" << std::endl;
                cudaTimes.push_back(cudaTime);

                std::string outputPathCUDA = outputFolderCUDA + "/" + baseFilename + "_compressed.png";
                cv::imwrite(outputPathCUDA, compressedCUDA, compression_params);
                std::cout << "CUDA compressed image saved as: " << outputPathCUDA << std::endl;

                totalCUDATime += cudaTime;
            }

            // Print performance metrics for this image (only if multiple versions are run)
            if ((runSerial && runOpenMP) || (runSerial && runCUDA) || (runOpenMP && runCUDA)) {
                std::cout << "\n--- Performance Metrics for " << baseFilename << " ---" << std::endl;

                if (runSerial && runOpenMP) {
                    double openMPSpeedup = serialTime / openMPTime;
                    double openMPEfficiency = (openMPSpeedup / numThreads) * 100;
                    std::cout << "OpenMP Speedup: " << openMPSpeedup << "x" << std::endl;
                    std::cout << "OpenMP Efficiency: " << openMPEfficiency << "%" << std::endl;
                }

                if (runSerial && runCUDA) {
                    double cudaSpeedup = serialTime / cudaTime;
                    std::cout << "CUDA Speedup: " << cudaSpeedup << "x" << std::endl;
                }

                if (runOpenMP && runCUDA && !runSerial) {
                    double cudaVsOpenMPSpeedup = openMPTime / cudaTime;
                    std::cout << "CUDA vs OpenMP Speedup: " << cudaVsOpenMPSpeedup << "x" << std::endl;
                }
            }
        }

        // Print overall performance summary
        std::cout << "\n=== OVERALL PERFORMANCE SUMMARY ===\n" << std::endl;
        std::cout << "Total images processed: " << imageFiles.size() << std::endl;

        if (runSerial) {
            std::cout << "Total Serial execution time: " << totalSerialTime << " seconds" << std::endl;
            double avgSerialTime = totalSerialTime / imageFiles.size();
            std::cout << "Average Serial time per image: " << avgSerialTime << " seconds" << std::endl;
        }

        if (runOpenMP) {
            std::cout << "Total OpenMP execution time: " << totalOpenMPTime << " seconds" << std::endl;
            double avgOpenMPTime = totalOpenMPTime / imageFiles.size();
            std::cout << "Average OpenMP time per image: " << avgOpenMPTime << " seconds" << std::endl;
        }

        if (runCUDA) {
            std::cout << "Total CUDA execution time: " << totalCUDATime << " seconds" << std::endl;
            double avgCUDATime = totalCUDATime / imageFiles.size();
            std::cout << "Average CUDA time per image: " << avgCUDATime << " seconds" << std::endl;
        }

        // Calculate and display average speedups if multiple versions were run
        if (runSerial && runOpenMP) {
            double avgSpeedupOpenMP = totalSerialTime / totalOpenMPTime;
            std::cout << "Average OpenMP Speedup: " << avgSpeedupOpenMP << "x" << std::endl;
            std::cout << "OpenMP Efficiency: " << (avgSpeedupOpenMP / numThreads) * 100 << "%" << std::endl;
        }

        if (runSerial && runCUDA) {
            double avgSpeedupCUDA = totalSerialTime / totalCUDATime;
            std::cout << "Average CUDA Speedup: " << avgSpeedupCUDA << "x" << std::endl;
        }

        if (runOpenMP && runCUDA && !runSerial) {
            double avgSpeedupCUDAvsOpenMP = totalOpenMPTime / totalCUDATime;
            std::cout << "Average CUDA vs OpenMP Speedup: " << avgSpeedupCUDAvsOpenMP << "x" << std::endl;
        }

        std::cout << "\nCompression settings for all versions:" << std::endl;
        std::cout << "- Threshold: " << threshold << std::endl;
        std::cout << "- Decomposition levels: " << levels << std::endl;

        // Save results to CSV only if at least one algorithm was run
        if (runSerial || runOpenMP || runCUDA) {
            std::string csvFilename = folderPath + "/performance_results.csv";
            savePerformanceToCSV(csvFilename, imageNames, serialTimes, openMPTimes, cudaTimes,
                imageDimensions, numThreads, threshold, levels);
            std::cout << "Performance results saved to: " << csvFilename << std::endl;
        }

        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }
}