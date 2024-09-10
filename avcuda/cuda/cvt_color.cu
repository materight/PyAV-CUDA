#include <stdint.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 32

template<class T>
__device__ static T clamp(T x, T lower, T upper) {
    return x < lower ? lower : (x > upper ? upper : x);
}


template<bool FullColorRange>
__global__ void NV12ToRGB_kernel(
    uint8_t *inY,
    uint8_t *inUV,
    uint8_t *outRGB,
    int height,
    int width,
    int pitch
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int yIdx = y * pitch + x;
    int uvIdx = (y / 2) * pitch + (x / 2) * 2;

    uint8_t Y = inY[yIdx];
    uint8_t U = inUV[uvIdx];
    uint8_t V = inUV[uvIdx + 1];

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

    int rgbIdx = (y * width + x) * 3;
    outRGB[rgbIdx] = R;
    outRGB[rgbIdx + 1] = G;
    outRGB[rgbIdx + 2] = B;
}


__global__ void RGBToNV12_kernel(
    uint8_t *inRGB,
    uint8_t *outY,
    uint8_t *outUV,
    int height,
    int width,
    int pitch
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int rgbIdx = (y * width + x) * 3;
    uint8_t R = inRGB[rgbIdx];
    uint8_t G = inRGB[rgbIdx + 1];
    uint8_t B = inRGB[rgbIdx + 2];

    uint8_t Y = clamp(0.257f * R + 0.504f * G + 0.098f * B + 16.0f, 0.0f, 255.0f);

    int yIdx = y * pitch + x;
    outY[yIdx] = Y;

    if ((x % 2 == 0) && (y % 2 == 0)) {
        uint8_t U = clamp(-0.148f * R - 0.291f * G + 0.439f * B + 128.0f, 0.0f, 255.0f);
        uint8_t V = clamp( 0.439f * R - 0.368f * G - 0.071f * B + 128.0f, 0.0f, 255.0f);

        int uvIdx = (y / 2) * pitch + (x / 2) * 2;
        outUV[uvIdx] = U;
        outUV[uvIdx + 1] = V;
    }
}


inline int divCeil(int num, int den) {
    return (num + (den - 1)) / den;
}

cudaError_t checkCudaErrorAndSync() {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return err;
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) return err;
    return cudaSuccess;
}


extern "C" {
    cudaError_t NV12ToRGB(uint8_t *inY, uint8_t *inUV, uint8_t *outRGB, int height, int width, int pitch, bool fullColorRange) {
        dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
        dim3 gridSize(divCeil(width, blockSize.x), divCeil(height, blockSize.y));

        if (fullColorRange) {
            NV12ToRGB_kernel<true><<<gridSize, blockSize>>>(inY, inUV, outRGB, height, width, pitch);
        } else {
            NV12ToRGB_kernel<false><<<gridSize, blockSize>>>(inY, inUV, outRGB, height, width, pitch);
        }

        return checkCudaErrorAndSync();
    }

    cudaError_t RGBToNV12(uint8_t *inRGB, uint8_t *outY, uint8_t *outUV, int height, int width, int pitch) {
        dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE / 4);
        dim3 gridSize(divCeil(width, blockSize.x), divCeil(height, blockSize.y));

        RGBToNV12_kernel<<<gridSize, blockSize>>>(inRGB, outY, outUV, height, width, pitch);

        return checkCudaErrorAndSync();
    }
}
