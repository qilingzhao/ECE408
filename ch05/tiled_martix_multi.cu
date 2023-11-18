#include <iostream>
#include <cmath>

__global__
void multi_kernel(float* mat_a_d, float* mat_b_d, float* mat_c_d, size_t mat_size) {
    int c_x = blockDim.x * blockIdx.x + threadIdx.x;
    int c_y = blockDim.y * blockIdx.y + threadIdx.y;

    if (c_x >= mat_size || c_y >= mat_size) {
        return;
    }

    // load memeory
    __shared__ float shared_a_block[blockDim.x * blockDim.y];
    __shared__ float shared_b_block[blockDim.x * blockDim.y];

    int c_idx = c_x * mat_size + c_y;
    int shared_idx = threadIdx.x * blockDim.y + threadIdx.y;
    shared_a_block[shared_idx] = mat_a_d[c_idx];
    shared_b_block[shared_idx] = mat_b_d[c_idx];
    __syncthreads();

    // calculate
    __shared__ float shared_c_block[blockDim.x * blockDim.y];
    cudaMemset(shared_c_block, 0, blockDim.x * blockDim.y * sizeof(float));
    float sum = 0;
    for (int i = 0; i < blockDim.y; i++) {
        sum += shared_a_block[threadIdx.x * blockDim.y + i] * 
                shared_b_block[i * blockDim.y + threadIdx.y];
    }
    shared_c_block[threadIdx.x * blockDim.y + threadIdx.y] += sum;
    __syncthreads();

    for (int i = 0; i < blockDim.x * blockDim.y; i++) {
        int x_in_block = i / blockDim.x;
        int y_in_block = i % blockDim.x;
        int x_in_grid = x_in_block + blockDim.x * blockIdx.x;
        int y_in_grid = y_in_block + blockDim.y * blockIdx.y;
        mat_c_d[x_in_grid * mat_size + y_in_grid] = shared_c_block[i];
    }
}

// consider two matrixs are the same size square.
__host__
void martix_multi(float* mat_a_h, float* mat_b_h, float* mat_c_h, size_t mat_size) {
    float *mat_a_d, *mat_b_d, *mat_c_d;
    size_t byte_size = mat_size * mat_size * sizeof(float);
    cudaMalloc((void**)&mat_a_d, byte_size);
    cudaMalloc((void**)&mat_b_d, byte_size);
    cudaMalloc((void**)&mat_c_d, byte_size);

    cudaMemcpy(mat_a_d, mat_a_h, byte_size, cudaMemcpyHostToDevice);
    cudaMemcpy(mat_b_d, mat_b_h, byte_size, cudaMemcpyHostToHost);
    cudaMemset(mat_c_d, 0, byte_size);

    int block_dim_len = 16;
    dim3 blockSize(block_dim_len, block_dim_len);
    dim3 gridSize(ceil(mat_size*1.0/block_dim_len), ceil(mat_size*1.0/block_dim_len));
    multi_kernel<<<gridSize, blockSize>>>(mat_a_d, mat_b_d, mat_c_d, mat_size);

    cudaMemcpy(mat_c_h, mat_c_d, byte_size, cudaMemcpyHostToHost);
    cudaFree(mat_a_d);
    cudaFree(mat_b_d);
    cudaFree(mat_c_d);
}
int main() {
    cudaDeviceProp deviceProp;
    std::cout << "sharedMemPerBlock: " << deviceProp.sharedMemPerBlock << std::endl;
    
    return 0;
}