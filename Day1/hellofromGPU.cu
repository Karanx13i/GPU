#include <cstdio>

__global__ void helloFromGPU() {
    printf("Hello from GPU thread!\n");
}

int main() {
    helloFromGPU<<<1, 1>>>();
    cudaDeviceSynchronize();
    return 0;
}

