#include <opencv2/opencv.hpp>
#include <iostream>

const int channel_num = 3;

__global__
void to_gray_kenerl(uchar* output, uchar* input, int width, int height) {
    // the x and y is depend on the Cartesian coordinate system 
    // which is opposite to the C multidimensional array indexing convention
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= height || col >= width) {
        return;
    }

    int gray_index = row * width + col;

    if (gray_index >= width * height) {
        return;
    }

    int color_index = gray_index * channel_num;
    uchar r = input[color_index];
    uchar g = input[color_index + 1];
    uchar b = input[color_index + 2];
    output[gray_index] = 0.21f*r + 0.71f*g + 0.07f*b;
}

void to_gray(uchar* output_h, uchar* input_h, int width, int height) {
    uchar *input_d, *output_d;
    cudaMalloc((void**)&input_d, width * height * channel_num);
    cudaMalloc((void**)&output_d, width * height);
    cudaMemcpy(input_d, input_h, sizeof(uchar)*width*height*channel_num, cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16);
    dim3 gridSize(width*1.0/blockSize.x, height*1.0/blockSize.y);

    to_gray_kenerl<<<gridSize, blockSize>>>(output_d, input_d, width, height);

    cudaMemcpy(output_h, output_d, sizeof(uchar)*width*height, cudaMemcpyDeviceToHost);
    cudaFree(input_d);
    cudaFree(output_d);
}

uchar input_pixel_array[5000 * 5000 * channel_num];
uchar output_pixel_array[5000 * 5000];

// nvcc -lopencv_core -lopencv_imgcodecs -lopencv_highgui -I /usr/local/include/opencv4 ./color_image_to_gray.cu

int main() {
    
    cv::Mat image = cv::imread("./image/desktop.jpg");

   if (image.empty()) {
        std::cerr << "cant't read image file." << std::endl;
        return -1;
    }

    int height = image.rows;
    int width = image.cols;

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            cv::Vec3b color = image.at<cv::Vec3b>(i, j);
            for (int k = 0; k < 3; k++) {
                input_pixel_array[(i * width + j) * channel_num + k] = color[k];
            }
        }
    }

    std::cout << "image's height: " << height << std::endl;
    std::cout << "image's weight: " << width << std::endl; 
    std::cout << "the color(R, G, B) of first pixel : ";

    for (int k = 0; k < 3; k++) {
        std::cout << int(input_pixel_array[k]) << " ";
    }
    std::cout << std::endl;

    to_gray(output_pixel_array, input_pixel_array, width, height);

    cv::Mat gray_image(height, width, CV_8U, output_pixel_array);
    bool saved = cv::imwrite("./image/gray_desktop.jpg", gray_image);

    if (saved) {
        std::cout << "gray image has been saved as ./image/gray_desktop.jpg" << std::endl;
    } else {
        std::cerr << "can't save the image" << std::endl;
    }

    return 0;
}

