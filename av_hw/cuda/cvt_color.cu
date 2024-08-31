#include <stdint.h>
#include <cuda_runtime.h>


__global__ void cuda_nv12_to_rgb(uint8_t *in_y, uint8_t *in_uv, uint8_t *out_rgb, int height, int width, int pitch, int full_color_range) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int y_index = y * pitch + x;
    int uv_index = (y / 2) * pitch + (x / 2) * 2;

    uint8_t Y = in_y[y_index];
    uint8_t U = in_uv[uv_index];
    uint8_t V = in_uv[uv_index + 1];

    int C = Y - 16;
    int D = U - 128;
    int E = V - 128;

    uint8_t R = min(max((298 * C + 409 * E + 128) >> 8, 0), 255);
    uint8_t G = min(max((298 * C - 100 * D - 208 * E + 128) >> 8, 0), 255);
    uint8_t B = min(max((298 * C + 516 * D + 128) >> 8, 0), 255);

    int rgb_index = (y * width + x) * 3;
    out_rgb[rgb_index] = R;
    out_rgb[rgb_index + 1] = G;
    out_rgb[rgb_index + 2] = B;
}


// Host function to launch the CUDA kernel
extern "C" {
    cudaError_t nv12_to_rgb(uint8_t *in_y, uint8_t *in_uv, uint8_t *out_rgb, int height, int width, int pitch, int full_color_range) {
        dim3 block(16, 16);
        dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

        cuda_nv12_to_rgb<<<grid, block>>>(in_y, in_uv, out_rgb, height, width, pitch, full_color_range);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) return err;
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) return err;
        return cudaSuccess;
    }
}
