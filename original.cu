#include <iostream>
#include <cuda.h>

template<typename T>
__global__ void axpy(T a, T *x, T *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  y[i] += a * x[i];
}


int main(int argc, char* argv[]) {
  const int kDataLen = 4;

  float a = 2.0f;
  float host_x[kDataLen] = {1.0f, 2.0f, 3.0f, 4.0f};
  float host_y[kDataLen] = {1.0f, 2.0f, 3.0f, 4.0f};

  float* device_x;
  float* device_y;

  cudaMalloc(&device_x, kDataLen * sizeof(float));

  cudaMalloc(&device_y, kDataLen * sizeof(float));

  cudaMalloc(&device_y, kDataLen * sizeof(double));

  cudaMemcpy(device_x, host_x, kDataLen * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(device_y, host_y, kDataLen * sizeof(float), cudaMemcpyHostToDevice);

  axpy<<<1, kDataLen>>>(a, device_x, device_y);

  cudaDeviceSynchronize();

  cudaMemcpy(host_y, device_y, kDataLen * sizeof(float), cudaMemcpyDeviceToHost);

  // Print the results.
  for (int i = 0; i < kDataLen; ++i) {
    std::cout << "y[" << i << "] = " << host_y[i] << "\n";
  }

  cudaDeviceReset();
  return 0;
}
