#include <iostream>

int main() {
    
    int devCount;
    cudaGetDeviceCount(&devCount);
    std::cout << "device count is " << devCount << std::endl;

    cudaDeviceProp devProp;
    for (uint i = 0; i < devCount; i++) {
        cudaGetDeviceProperties(&devProp, i);
        std::cout << "devProp.maxThreadsPerBlock: " << devProp.maxThreadsPerBlock << std::endl;
        std::cout << "the number of SMs(devProp.multiProcessorCount): " << devProp.multiProcessorCount << std::endl;
        std::cout << "clockRate: " << devProp.clockRate << std::endl;
        std::cout << "Maximum number of threads in a block(maxThreadsDim): x: " << devProp.maxThreadsDim[0] <<
                 ", y: " << devProp.maxThreadsDim[1] << 
                 ", z: " << devProp.maxThreadsDim[2] << std::endl;
        std::cout << "Maxium number of blocks in a grid(maxGridSize): x: " << devProp.maxGridSize[0] <<
                ", y: " << devProp.maxGridSize[1] << 
                ", z: " << devProp.maxGridSize[2] << std::endl;
        std::cout << "devProp.regsPerBlock/Grid: " << devProp.regsPerBlock << std::endl;
        std::cout << "devProp.warpSize: " << devProp.warpSize << std::endl;
    }

    return 0;
}
// The output of Tesla T4
// device count is 1
// devProp.maxThreadsPerBlock: 1024
// the number of SMs(devProp.multiProcessorCount): 40
// clockRate: 1590000
// Maximum number of threads in a block(maxThreadsDim): x: 1024, y: 1024, z: 64
// Maxium number of blocks in a grid(maxGridSize): x: 2147483647, y: 65535, z: 65535
// devProp.regsPerBlock/Grid: 65536
// devProp.warpSize: 32
