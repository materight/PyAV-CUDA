#include <stdint.h>
#include <cuda_runtime.h>

cudaError_t NV12ToRGB(uint8_t *inY, uint8_t *inUV, uint8_t *outRGB, int height, int width, int pitchY, int pitchUV, int fullColorRange);
cudaError_t RGBToNV12(uint8_t *inRGB, uint8_t *outY, uint8_t *outUV, int height, int width, int pitchY, int pitchUV);
