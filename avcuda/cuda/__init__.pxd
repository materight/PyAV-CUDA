from libc.stdint cimport uint8_t

cdef extern from "cuda.h" nogil:
    ctypedef unsigned long long CUdeviceptr_v2
    ctypedef CUdeviceptr_v2 CUdeviceptr


cdef extern from "driver_types.h" nogil:
    cdef enum cudaError:
        cudaSuccess = 0

    ctypedef cudaError cudaError_t

    cdef const char* cudaGetErrorString(cudaError_t error)

    ctypedef void* cudaStream_t

    cdef enum cudaMemcpyKind:
        cudaMemcpyHostToHost = 0
        cudaMemcpyHostToDevice = 1
        cudaMemcpyDeviceToHost = 2
        cudaMemcpyDeviceToDevice = 3
        cudaMemcpyDefault = 4


cdef extern from "cuda_runtime.h" nogil:
    cdef cudaError_t cudaMemcpy2D(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind)
    cdef cudaError_t cudaFree(void* devPtr)


# Custom CUDA kernels
cdef extern from "cuda/cvt_color.h" nogil:
    cudaError_t NV12ToRGB(uint8_t *inY, uint8_t *inUV, uint8_t *outRGB, int height, int width, int pitchY, int pitchUV, int fullColorRange)
    cudaError_t RGBToNV12(uint8_t *inRGB, uint8_t *outY, uint8_t *outU, uint8_t *outV, int height, int width, int pitchY, int pitchUV)
