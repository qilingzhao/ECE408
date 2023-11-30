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

    int tx = threadIdx.x; int ty = threadIdx.y;
    int bx = blockIdx.x; int by = blockIdx.y;
    int gx = bx * TileLen + tx;
    int gy = by * TileLen + ty;

    extern float shm[];

    size_t max_len = max(P.height, P.weight);
    for (int ph = 0; ph < (max_len + TileLen - 1) / TileLen; ph++) {
        // load matrix data from global memory to shm;
        
    }
}

int main() {
    return 0;
}