#include <stdint.h>
#include <cuda_runtime.h>


template<class T>
__device__ static T clamp(T x, T lower, T upper) {
    return x < lower ? lower : (x > upper ? upper : x);
}

template<bool FullColorRange>
__global__ void cudaNV12ToRGB(
    uint8_t *in_y,
    uint8_t *in_uv,
    uint8_t *out_rgb,
    int height,
    int width,
    int pitch
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int y_index = y * pitch + x;
    int uv_index = (y / 2) * pitch + (x / 2) * 2;

    uint8_t Y = in_y[y_index];
    uint8_t U = in_uv[uv_index];
    uint8_t V = in_uv[uv_index + 1];

    float fY = (int)Y - 0;
    float fU = (int)U - 128;
    float fV = (int)V - 128;

    uint8_t R, G, B;
    if constexpr (FullColorRange) {
        R = clamp(1.000f * fY +             + 1.402f * fV, 0.0f, 255.0f);
        G = clamp(1.000f * fY - 0.344f * fU - 0.714f * fV, 0.0f, 255.0f);
        B = clamp(1.000f * fY + 1.772f * fU              , 0.0f, 255.0f);
    } else {
        fY -= 16;
        R = clamp(1.164f * fY +             + 1.596f * fV, 0.0f, 255.0f);
        G = clamp(1.164f * fY - 0.392f * fU - 0.813f * fV, 0.0f, 255.0f);
        B = clamp(1.164f * fY + 2.017f * fU              , 0.0f, 255.0f);
    }

    int rgb_index = (y * width + x) * 3;
    out_rgb[rgb_index] = R;
    out_rgb[rgb_index + 1] = G;
    out_rgb[rgb_index + 2] = B;
}


// Host function to launch the CUDA kernel
extern "C" {
    cudaError_t NV12ToRGB(uint8_t *in_y, uint8_t *in_uv, uint8_t *out_rgb, int height, int width, int pitch, bool full_color_range) {
        dim3 block(16, 16);
        dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

        if (full_color_range) {
            cudaNV12ToRGB<true><<<grid, block>>>(in_y, in_uv, out_rgb, height, width, pitch);
        } else {
            cudaNV12ToRGB<false><<<grid, block>>>(in_y, in_uv, out_rgb, height, width, pitch);
        }

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) return err;
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) return err;
        return cudaSuccess;
    }
}
