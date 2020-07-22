#include <iostream>
#include <cuda.h>

template<typename T>
__global__ void axpy(T a, T *x, T *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  y[i] += a * x[i];
}
// cyclic distribution ⇒ coalesced memory access. In OpenMP,  If you use schedule(static:chunk:1), it forces cyclic distribution. (dynamic:chunk:1) is not cyclic, but could end up to be cyclic
__global__ void axpy_cudakernel_cyclic(float *y, float *x, float a, int kDataLen) {
   // int i;
    int thread_num = threadIdx.x + blockIdx.x * blockDim.x;
    int total_threads = gridDim.x * blockDim.x;
    //int block_size = kDataLen / total_threads; //dividable, TODO handle non-dividiable later
    if (thread_num < kDataLen) {
        int start_index = thread_num;
        for (int i=start_index; i<kDataLen; i+=total_threads) 
            y[i] += a*x[i];
    }
}



int main(int argc, char* argv[]) {
  const int kDataLen = 10240;

  float a = 2.0f;
  float host_x[kDataLen];
  float host_y[kDataLen];
  for(int i=0; i<10240; i++) host_x[i] = i;//rand();
  for(int i=0; i<10240; i++) host_y[i] = i;//rand();
  //float host_x[kDataLen] = {1.0f, 2.0f, 3.0f, 4.0f};
 /// float host_y[kDataLen] = {1.0f, 2.0f, 3.0f, 4.0f};

  float* device_x;
  float* device_y;

  cudaMalloc(&device_x, kDataLen * sizeof(float));

  cudaMalloc(&device_y, kDataLen * sizeof(float));

  cudaMalloc(&device_y, kDataLen * sizeof(double));

  cudaMemcpy(device_x, host_x, kDataLen * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(device_y, host_y, kDataLen * sizeof(float), cudaMemcpyHostToDevice);

  //dim3 block_size(16, 16);
  //dim3 grid_size(2, 2);
  for(int i = 0; i<100; i++){
    axpy_cudakernel_cyclic<<<2, 256>>>( device_y, device_x, a, kDataLen);
  }

  cudaDeviceSynchronize();

  cudaMemcpy(host_y, device_y, kDataLen * sizeof(float), cudaMemcpyDeviceToHost);

  // Print the results.
  /*for (int i = 0; i < kDataLen; ++i) {
    std::cout << "y[" << i << "] = " << host_y[i] << "\n";
  }*/

  cudaDeviceReset();
  return 0;
}
