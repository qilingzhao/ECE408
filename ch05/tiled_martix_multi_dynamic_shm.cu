#include <cstdio>
using namespace std;

struct Matrix
{
    float* p = nullptr;
    size_t height = 0;
    size_t weight = 0;
};

__global__
void matrix_multi_kernel(const Matrix& M, const Matrix& N, Matrix& P, const size_t TileLen) {
    // TODO: check matrix boundary

    size_t tx = threadIdx.x; size_t ty = threadIdx.y;
    size_t bx = blockIdx.x; size_t by = blockIdx.y;
    size_t gx = bx * TileLen + tx;
    size_t gy = by * TileLen + ty;

    extern float shm[];

    size_t max_len = max(P.height, P.weight);
    for (size_t ph = 0; ph < (max_len + TileLen - 1) / TileLen; ph++) {
        // load matrix data from global memory to shm;
        size_t M_x = ph * TileLen + tx;
        size_t M_y = ph * TileLen + ty;
        size_t shm_m_idx = 0;

    }
}

int main() {
    return 0;
}