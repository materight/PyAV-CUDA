#include <cuda_runtime_api.h>
#include <cuda.h>

__global__ void cuda_nv12_to_rgb(uint8_t *in_y, uint8_t *in_uv, uint8_t *out_rgb, size_t height, size_t width) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    out_rgb[tid] = 0;
}


void nv12_to_rgb(uint8_t *in_y, uint8_t *in_uv, uint8_t *out_rgb, size_t height, size_t width) {
	const dim3 blockDim(32, 16, 1);
	const dim3 gridDim((width+(2*blockDim.x-1))/(2*blockDim.x), (height+(blockDim.y-1))/blockDim.y, 1);

    cuda_nv12_to_rgb<<<gridDim, blockDim>>>(in_y, in_uv, out_rgb, height, width);

    cudaGetLastError();
    cudaDeviceSynchronize();
}
