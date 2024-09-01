#include <stdint.h>
#include <cuda_runtime.h>

cudaError_t NV12ToRGB(uint8_t *in_y, uint8_t *in_uv, uint8_t *out_rgb, int height, int width, int pitch, int full_color_range);
