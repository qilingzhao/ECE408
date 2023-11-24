#include <iostream>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <iomanip>  // For std::setw to format the output
#include <chrono>

const size_t TiledSize = 16;

__global__
void multi_kernel(float* M, float* N, float* P, size_t mat_len) {
    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    int gx = bx * TiledSize + tx;
    int gy = by * TiledSize + ty;

    __shared__ float sd_M[TiledSize][TiledSize];
    __shared__ float sd_N[TiledSize][TiledSize];

    float p = 0;
    for (int ph = 0; ph < (mat_len + TiledSize - 1) / TiledSize; ph++) {
        // load from global memery to shared_memory
        int Mx = bx * TiledSize + tx;
        int My = ph * TiledSize + ty;
        int M_idx = Mx * mat_len + My;
        // Attention: the Mx, My boundary check!
        sd_M[tx][ty] = (Mx >= mat_len || My >= mat_len) ? 0 : M[M_idx];

        int Nx = ph * TiledSize + tx;
        int Ny = by * TiledSize + ty;
        int N_idx = Nx * mat_len + Ny;
        // Attention: the Nx, Ny boundary check!
        sd_N[tx][ty] = (Nx >= mat_len || Ny >= mat_len) ? 0 : N[N_idx];

        __syncthreads();

        // printf("sd_M[%d][%d]: %f, sd_N[%d][%d]: %f\n", tx, ty, sd_M[tx][ty], tx, ty, sd_N[tx][ty]);

        // calculate the value in the TILED
        for (int i = 0; i < TiledSize; i++) {
            p += sd_M[tx][i] * sd_N[i][ty];
        }
        __syncthreads();
    }
    int P_idx = gx * mat_len + gy;
    // Attention: the gx, gy boundary check!
    if (gx < mat_len && gy < mat_len) {
        P[P_idx] = p;
    }
}

// consider two matrixs are the same size square.
__host__
void martix_multi(const float* mat_a_h, const float* mat_b_h, float* mat_c_h, size_t mat_len) {
    float *mat_a_d, *mat_b_d, *mat_c_d;
    size_t byte_size = mat_len * mat_len * sizeof(float);
    cudaMalloc((void**)&mat_a_d, byte_size);
    cudaMalloc((void**)&mat_b_d, byte_size);
    cudaMalloc((void**)&mat_c_d, byte_size);

    cudaMemcpy(mat_a_d, mat_a_h, byte_size, cudaMemcpyHostToDevice);
    cudaMemcpy(mat_b_d, mat_b_h, byte_size, cudaMemcpyHostToDevice);

    dim3 blockSize(TiledSize, TiledSize);
    dim3 gridSize((mat_len + TiledSize - 1) / TiledSize, (mat_len + TiledSize - 1) / TiledSize);
    multi_kernel<<<gridSize, blockSize>>>(mat_a_d, mat_b_d, mat_c_d, mat_len);

    cudaMemcpy(mat_c_h, mat_c_d, byte_size, cudaMemcpyDeviceToHost);
    cudaFree(mat_a_d);
    cudaFree(mat_b_d);
    cudaFree(mat_c_d);
}

// Function to create the matrix and initialize it with random float values
float* createMatrix(int rows, int cols) {
    float* matrix = new float[rows * cols];

    // Initialize the matrix with random float values
    for (int i = 0; i < rows * cols; i++) {
        // Generate random float between 0 and 1
        // You can modify the range as per your requirement
        matrix[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    }

    return matrix;
}

float* multiplyMatricesCPU(const float* matrixA, const float* matrixB, const int size) {
    float* resultMatrix = new float[size * size];

    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            resultMatrix[i * size + j] = 0.0f;
            for (int k = 0; k < size; k++) {
                resultMatrix[i * size + j] += matrixA[i * size + k] * matrixB[k * size + j];
            }
        }
    }

    return resultMatrix;
}

bool areEqual(float* array1, float* array2, int size, float epsilon = 1e-3) {
    for (int i = 0; i < size; i++) {
        if (std::fabs(array1[i] - array2[i]) > epsilon) {
            std::cout << "first not equal at " << array1[i] << " " << array2[i] << " " << i << std::endl;
            return false;
        }
    }
    return true;
}

// Function to print a 1D array as a 2D matrix
void printMatrix(float* array, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            std::cout << std::setw(10) << array[i * cols + j] << " ";
        }
        std::cout << std::endl;
    }
}

const int MaxDisplayMatrixLen = 10;
int main() {
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    std::cout << "sharedMemPerBlock: " << deviceProp.sharedMemPerBlock << " Bytes" << "\t(" << deviceProp.sharedMemPerBlock/1024 << "KB)" << std::endl;

    const int mat_len = :1000;
    // Seed for random number generation
    std::srand(std::time(nullptr));
    float* M = createMatrix(mat_len, mat_len);
    float* N = createMatrix(mat_len, mat_len);
    float* P = new float[mat_len * mat_len];
    memset(P, 0, mat_len * mat_len * sizeof(float));

    auto gpu_start = std::chrono::high_resolution_clock::now();
    martix_multi(M, N, P, mat_len);
    auto gpu_stop = std::chrono::high_resolution_clock::now();
    auto gpu_duration = std::chrono::duration_cast<std::chrono::milliseconds>(gpu_stop - gpu_start);
    std::cout << "finish GPU: " << gpu_duration.count() << " ms" << std::endl;

    auto cpu_start = std::chrono::high_resolution_clock::now();
    float* P_cpu = multiplyMatricesCPU(M, N, mat_len);
    auto cpu_stop = std::chrono::high_resolution_clock::now();
    auto cpu_duration = std::chrono::duration_cast<std::chrono::milliseconds>(cpu_stop - cpu_start);
    std::cout << "finish CPU: " << cpu_duration.count() << " ms" << std::endl;

    bool eq = areEqual(P, P_cpu, mat_len*mat_len);
    std::cout << (eq ? "equals!" : "not equal.") << std::endl;

    if (mat_len <= MaxDisplayMatrixLen) {
        std::cout << " ------ M ------" << std::endl;
        printMatrix(M, mat_len, mat_len);
        std::cout << " ------ N ------" << std::endl;
        printMatrix(N, mat_len, mat_len);
        std::cout << " ------ P ------" << std::endl;
        printMatrix(P, mat_len, mat_len);
        std::cout << " ------ P_cpu ------" << std::endl;
        printMatrix(P_cpu, mat_len, mat_len);
    }
    delete[] M;
    delete[] N;
    delete[] P;
    delete[] P_cpu;
    return 0;
}
