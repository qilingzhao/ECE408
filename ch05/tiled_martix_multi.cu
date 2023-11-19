#include <iostream>
#include <cmath>
#include <cstdlib>
#include <ctime>

__constant__ size_t TiledSize = 16;

__global__
void multi_kernel(float* M, float* N, float* P, size_t mat_len) {
    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    int gx = bx * TiledSize + tx;
    int gy = by * TiledSize + ty;

    __shared__ float sd_M[TiledSize][TiledSize];
    __shared__ float sd_N[TiledSize][TiledSize];

    for (int ph = 0; ph < mat_len/TiledSize; ph++) {
        // load from global memery to shared_memory
        int Mx = bx * TiledSize + tx;
        int My = ph * TiledSize + ty;
        int M_idx = Mx * mat_len + My;
        sd_M[tx][ty] = (M_idx >= (mat_len * mat_len)) ? 0 : M[M_idx];

        int Nx = ph * TiledSize + tx;
        int Ny = by * TiledSize + ty;
        int N_idx = Nx * mat_len + Ny;
        sd_N[tx][ty] = (N_idx >= (mat_len * mat_len)) ? 0 : N[N_idx];

        __syncthreads();

        // calculate the value in the TILED
        float p = 0;
        for (int i = 0; i < TiledSize; i++) {
            p += sd_M[tx][i] * sd_N[i][ty];
        }
        int P_idx = gx * mat_len + gy;
        P[P_idx] = (P_idx >= mat_len * mat_len) ? 0 : p;
        __syncthreads();
    }
}

// consider two matrixs are the same size square.
__host__
void martix_multi(float* mat_a_h, float* mat_b_h, float* mat_c_h, size_t mat_len) {
    float *mat_a_d, *mat_b_d, *mat_c_d;
    size_t byte_size = mat_len * mat_len * sizeof(float);
    cudaMalloc((void**)&mat_a_d, byte_size);
    cudaMalloc((void**)&mat_b_d, byte_size);
    cudaMalloc((void**)&mat_c_d, byte_size);

    cudaMemcpy(mat_a_d, mat_a_h, byte_size, cudaMemcpyHostToDevice);
    cudaMemcpy(mat_b_d, mat_b_h, byte_size, cudaMemcpyHostToHost);

    dim3 blockSize(TiledSize, TiledSize);
    dim3 gridSize((mat_len + TiledSize - 1) / TiledSize, (mat_len + TiledSize - 1) / TiledSize);
    multi_kernel<<<gridSize, blockSize>>>(mat_a_d, mat_b_d, mat_c_d, mat_len);

    cudaMemcpy(mat_c_h, mat_c_d, byte_size, cudaMemcpyHostToHost);
    cudaFree(mat_a_d);
    cudaFree(mat_b_d);
    cudaFree(mat_c_d);
}

// Function to create the matrix and initialize it with random float values
float* createMatrix(int rows, int cols) {
    float* matrix = new float[rows * cols];

    // Seed for random number generation
    std::srand(std::time(nullptr));

    // Initialize the matrix with random float values
    for (int i = 0; i < rows * cols; i++) {
        // Generate random float between 0 and 1
        // You can modify the range as per your requirement
        matrix[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    }

    return matrix;
}

float* multiplyMatricesCPU(float* matrixA, float* matrixB, int size) {
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

bool areEqual(float* array1, float* array2, int size, float epsilon = 1e-6) {
    for (int i = 0; i < size; i++) {
        if (std::fabs(array1[i] - array2[i]) > epsilon) {
            return false;
        }
    }
    return true;
}

int main() {
    cudaDeviceProp deviceProp;
    std::cout << "sharedMemPerBlock: " << deviceProp.sharedMemPerBlock << " Bytes" << std::endl;
    const int mat_len = 5000;
    float* M = createMatrix(mat_len, mat_len);
    float* N = createMatrix(mat_len, mat_len);
    float* P = nullptr;
    martix_multi(M, N, P, mat_len);
    float* P_cpu = multiplyMatricesCPU(M, N, mat_len);
    
    bool eq = areEqual(P, P_cpu, mat_len*mat_len);
    std::cout << (eq ? "equals!" : "not equal.") << std::endl;

    delete[] M;
    delete[] N;
    delete[] P;
    delete[] P_cpu;
    return 0;
}