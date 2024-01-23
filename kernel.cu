#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>

cv::Mat applySobelFilter(const cv::Mat& frame) {
    cv::Mat gray, grad_x, grad_y;
    cv::Mat abs_grad_x, abs_grad_y, grad;
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    cv::Sobel(gray, grad_x, CV_16S, 1, 0, 3, 1, 0, cv::BORDER_DEFAULT);
    cv::convertScaleAbs(grad_x, abs_grad_x);
    cv::Sobel(gray, grad_y, CV_16S, 0, 1, 3, 1, 0, cv::BORDER_DEFAULT);
    cv::convertScaleAbs(grad_y, abs_grad_y);
    cv::addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad);

    return grad;
}

cv::Mat applyCannyEdgeDetection(const cv::Mat& frame) {
    cv::Mat gray, edges;
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    cv::Canny(gray, edges, 50, 150);

    return edges;
}

cv::Mat applyLaplacianEdgeDetection(const cv::Mat& frame) {
    cv::Mat gray, laplacian;
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    cv::Laplacian(gray, laplacian, CV_16S, 3);
    cv::convertScaleAbs(laplacian, laplacian);

    return laplacian;
}


int main() {
    cv::VideoCapture cap("nagranie.avi");
    if (!cap.isOpened()) {
        std::cerr << "Błąd podczas otwierania pliku wideo." << std::endl;
        return -1;
    }
    int frame_width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int frame_height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    cv::VideoWriter video("output.avi", cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
        10, cv::Size(frame_width, frame_height), false);


    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        cv::Mat sobelFrame = applySobelFilter(frame);
        cv::Mat cannyFrame = applyCannyEdgeDetection(frame);
        cv::Mat laplacianFrame = applyLaplacianEdgeDetection(frame);
        video.write(laplacianFrame);
    }


    cap.release();
    video.release();

    return 0;
}
