#include <opencv2/opencv.hpp>
#include <iostream>
#include <cuda_runtime.h>
#include <chrono>

using namespace cv;
using namespace std;
using namespace chrono;


__global__ void negativeTransformation(unsigned char* input, unsigned char* output, int width, int height, int step) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int index = y * step + x;
    output[index] = 255 - input[index];
}

void negativeTransformationCPU(const Mat& input, Mat& output) {
    // Prosta implementacja negatywu
    for (int y = 0; y < input.rows; ++y) {
        for (int x = 0; x < input.cols; ++x) {
            output.at<uchar>(y, x) = 255 - input.at<uchar>(y, x);
        }
    }
}

__global__ void sobelFilter(unsigned char* input, unsigned char* output, int width, int height, int step) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    float Gx = 0;
    float Gy = 0;

    if (x > 0 && x < width - 1 && y > 0 && y < height - 1) {
        Gx = (input[(y - 1) * step + (x + 1)] - input[(y - 1) * step + (x - 1)])
           + 2 * (input[y * step + (x + 1)] - input[y * step + (x - 1)])
           + (input[(y + 1) * step + (x + 1)] - input[(y + 1) * step + (x - 1)]);

        Gy = (input[(y - 1) * step + (x - 1)] - input[(y + 1) * step + (x - 1)])
           + 2 * (input[(y - 1) * step + x] - input[(y + 1) * step + x])
           + (input[(y - 1) * step + (x + 1)] - input[(y + 1) * step + (x + 1)]);

        output[y * step + x] = min(255, int(sqrtf(Gx * Gx + Gy * Gy)));
    }
}

void sobelFilterCPU(const Mat& input, Mat& output) {
    int Gx[3][3] = {
        {-1, 0, 1},
        {-2, 0, 2},
        {-1, 0, 1}
    };

    int Gy[3][3] = {
        {-1, -2, -1},
        { 0,  0,  0},
        { 1,  2,  1}
    };

    for (int y = 1; y < input.rows - 1; ++y) {
        for (int x = 1; x < input.cols - 1; ++x) {
            float sumX = 0.0, sumY = 0.0;

            for (int k = -1; k <= 1; k++) {
                for (int j = -1; j <= 1; j++) {
                    sumX += input.at<uchar>(y + k, x + j) * Gx[k + 1][j + 1];
                    sumY += input.at<uchar>(y + k, x + j) * Gy[k + 1][j + 1];
                }
            }

            int val = static_cast<int>(sqrt(sumX * sumX + sumY * sumY));
            output.at<uchar>(y, x) = min(max(val, 0), 255);
        }
    }
}

__global__ void blurFilter(unsigned char* input, unsigned char* output, int width, int height, int step) {
    int blurSize = 1;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int pixelCount = 0;
    int totalR = 0, totalG = 0, totalB = 0;

    // Uśrednianie wartości pikseli w oknie blurSize x blurSize
    for(int blurRow = -blurSize; blurRow <= blurSize; blurRow++) {
        for(int blurCol = -blurSize; blurCol <= blurSize; blurCol++) {
            int curX = x + blurCol;
            int curY = y + blurRow;

            // Sprawdzenie granic
            if(curX >= 0 && curX < width && curY >= 0 && curY < height) {
                int index = (curY * step) + (curX * 3);
                totalB += input[index];
                totalG += input[index + 1];
                totalR += input[index + 2];
                pixelCount++;
            }
        }
    }

    int index = (y * step) + (x * 3);
    output[index] = totalB / pixelCount;
    output[index + 1] = totalG / pixelCount;
    output[index + 2] = totalR / pixelCount;
}

void blurFilterCPU(const Mat& input, Mat& output) {
    int blurSize = 1;
    for (int y = 0; y < input.rows; ++y) {
        for (int x = 0; x < input.cols; ++x) {
            int total = 0;
            int count = 0;

            for (int dy = -blurSize; dy <= blurSize; ++dy) {
                for (int dx = -blurSize; dx <= blurSize; ++dx) {
                    int nx = x + dx;
                    int ny = y + dy;

                    if (nx >= 0 && ny >= 0 && nx < input.cols && ny < input.rows) {
                        total += input.at<uchar>(ny, nx);
                        count++;
                    }
                }
            }

            output.at<uchar>(y, x) = total / count;
        }
    }
}


int main() {
    int VAR = 1;
    string video_path = "org.avi"; // Zaktualizuj ścieżkę do pliku wideo
    VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        cout << "Nie mozna zaladowac wideo" << endl;
        return -1;
    }

    int frame_width = cap.get(CAP_PROP_FRAME_WIDTH);
    int frame_height = cap.get(CAP_PROP_FRAME_HEIGHT);
    VideoWriter video("output.avi", VideoWriter::fourcc('M', 'J', 'P', 'G'), 10, Size(frame_width, frame_height), false);

    Mat frame, frameCPU;

    auto startGPU = high_resolution_clock::now();

    while (true) {
        cap >> frame;
        if (frame.empty()) {
            break;
        }

        frameCPU = frame.clone();
        cvtColor(frame, frame, COLOR_BGR2GRAY);

        unsigned char *input, *output;
        cudaMallocManaged(&input, frame.rows * frame.cols);
        cudaMallocManaged(&output, frame.rows * frame.cols);
        memcpy(input, frame.data, frame.rows * frame.cols);

        dim3 blockSize(16, 16);
        dim3 gridSize((frame.cols + blockSize.x - 1) / blockSize.x, (frame.rows + blockSize.y - 1) / blockSize.y);

        switch(VAR) {
          case 1:
            negativeTransformation<<<gridSize, blockSize>>>(input, output, frame.cols, frame.rows, frame.step);
            cudaDeviceSynchronize();
            break;
          case 2:
            blurFilter<<<gridSize, blockSize>>>(input, output, frame.cols, frame.rows, frame.step);
            cudaDeviceSynchronize();
            break;
          case 3:
            sobelFilter<<<gridSize, blockSize>>>(input, output, frame.cols, frame.rows, frame.step);
            cudaDeviceSynchronize();
            break;
        }

        Mat output_frame(frame.rows, frame.cols, CV_8UC1, output);
        video.write(output_frame);

        cudaFree(input);
        cudaFree(output);
    }

    auto endGPU = high_resolution_clock::now();

    cap.open(video_path);

    auto startCPU = high_resolution_clock::now();

    while (true) {
        cap >> frameCPU;
        if (frameCPU.empty()) {
            break;
        }

        cvtColor(frameCPU, frameCPU, COLOR_BGR2GRAY);

        switch(VAR) {
          case 1:
            negativeTransformationCPU(frameCPU, frameCPU);
            break;
          case 2:
            blurFilterCPU(frameCPU, frameCPU);
            break;
          case 3:
            sobelFilterCPU(frameCPU, frameCPU);
            break;
        }
    }

    auto endCPU = high_resolution_clock::now();

    auto durationGPU = duration_cast<microseconds>(endGPU - startGPU);
    auto durationCPU = duration_cast<microseconds>(endCPU - startCPU);
    cout << "Całkowity czas na GPU: " << durationGPU.count() << " microseconds" << endl;
    cout << "Całkowity czas na CPU: " << durationCPU.count() << " microseconds" << endl;

    cap.release();
    video.release();

    return 0;
}
