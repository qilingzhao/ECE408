#include <cstdio>
#include <iostream>
using namespace std;

struct Matrix
{
    float* p = nullptr;
    size_t height = 0;
    size_t weight = 0;
};

__global__
void matrix_multi_kernel(const Matrix& M, const Matrix& N, Matrix& P, const size_t TileLen) {
    // TODO: check matrix length

    size_t tx = threadIdx.x; size_t ty = threadIdx.y;
    size_t bx = blockIdx.x; size_t by = blockIdx.y;
    size_t gx = bx * TileLen + tx;
    size_t gy = by * TileLen + ty;

    extern float shm[];

    size_t max_len = max(max(M.height, M.weight), N.weight);
    for (size_t ph = 0; ph < (max_len + TileLen - 1) / TileLen; ph++) {
        // load matrix data from global memory to shm;
        size_t M_x = gx;
        size_t M_y = ph * TileLen + ty;
        size_t M_idx = M_x * M.weight + M_y;
        size_t shm_m_idx = tx * TileLen + ty;
        shm[shm_m_idx] = (M_x < M.height && M_y < M.weight) ? M.p[M_idx] : 0.0;

        size_t N_x = ph * TileLen + tx;
        size_t N_y = gy;
        size_t N_idx = N_x * N.weight + N_y;
        size_t shm_n_idx = TileLen * TileLen + tx * TileLen + ty;
        shm[shm_n_idx] = (N_x < N.height && N_y < N.weight) ? N.p[N_idx] : 0.0;
        __syncthreads();

        // calculate value at P[tx][ty]
        double sum = 0;
        for (int i = 0; i < TileLen; i++) {
            size_t m_idx = tx * TileLen + i;
            size_t n_idx = TileLen * TileLen + i * TileLen + ty;
            sum += shm[m_idx] * shm[n_idx];
        }
        P.p[gx * P.weight + gy] += (gx < P.height && gy < P.weight) ? sum : 0.0;
        __syncthreads();
    }
}

__host__ 
size_t calc_tiled_len() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    return sqrt(prop.sharedMemPerBlock / sizeof(float) / 2);
}
__host__
void martix_multi(const Matrix& M_h, const Matrix& N_h, Matrix& P_h) {
    Matrix M_d, N_d, P_d;
    M_d = M_h; N_d = N_h; P_d = P_h;
    cudaMalloc((void**)(&M_d.p), sizeof(float)*M_h.height*M_h.weight);
    cudaMalloc((void**)(&N_d.p), sizeof(float)*N_h.height*N_h.weight);
    cudaMalloc((void**)(&P_d.p), sizeof(float)*P_h.height*P_h.weight);
    cudaMemset(P_d.p, 0, sizeof(float)*P_h.height*P_h.weight);
    cudaMemcpy(M_d.p, M_h.p, sizeof(float)*M_h.height*M_h.weight, cudaMemcpyHostToDevice);
    cudaMemcpy(N_d.p, N_h.p, sizeof(float)*N_h.height*N_h.weight, cudaMemcpyHostToDevice);

    size_t tiled_len = calc_tiled_len();
    std::cout << "tiled_len is " << tiled_len << std::endl;
    dim3 blockSize(tiled_len, tiled_len);
    size_t blockNum = (max(P_h.height, P_h.weight) + tiled_len - 1) / tiled_len;
    std::cout << "blockNum is " << blockNum << std::endl;
    dim3 gridSize(blockNum, blockNum);
    cudaMemcpy(N_d.p, N_h.p, sizeof(float)*N_h.height*N_h.weight, cudaMemcpyDeviceToHost);
    cudaFree(M_d.p);
    cudaFree(N_d.p);
    cudaFree(P_d.p);

}

int main() {
    return 0;
}