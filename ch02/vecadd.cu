#include <cstdio>

__global__
void vec_add_kernel(float* A, float* B, float* C, int n) {
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    
    if (index < n) {
        C[index] = A[index] + B[index];
    }
}

void vec_add(float* A_h, float* B_h, float* C_h, int N) {

    float *A_d, *B_d, *C_d;
    cudaMalloc((void**)&A_d, sizeof(float)*N);
    cudaMalloc((void**)&B_d, sizeof(float)*N);
    cudaMalloc((void**)&C_d, sizeof(float)*N);
    cudaMemcpy(A_d, A_h, sizeof(float)*N, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, sizeof(float)*N, cudaMemcpyHostToDevice);

    int threadNumPerBlock = 256;
    // dim3 blockNum(ceil(1.0*N/threadNumPerBlock));
    dim3 blockNum((N+threadNumPerBlock-1)/threadNumPerBlock);
    vec_add_kernel<<<blockNum, threadNumPerBlock>>>(A_d, B_d, C_d, N);

    cudaMemcpy(C_h, C_d, sizeof(float)*N, cudaMemcpyDeviceToHost);
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}

int main() {

    const int LEN = 10;
    float arr1[LEN] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    float arr2[LEN] = {9, 8, 7, 6, 5, 4, 3, 2, 1, 70}; 
    float ans[LEN] = {123};

    vec_add(arr1, arr2, ans, LEN);

    for (int i = 0; i < LEN; i++) {
        printf("ans[%d]: %f\n", i, ans[i]);
    }

    return 0;
}