from libc.stdint cimport uint8_t

cdef extern from "cuda.h" nogil:
    ctypedef unsigned long long CUdeviceptr_v2
    ctypedef CUdeviceptr_v2 CUdeviceptr


cdef extern from "driver_types.h" nogil:
    cdef enum cudaError:
        cudaSuccess = 0

    ctypedef cudaError cudaError_t

    cdef const char* cudaGetErrorString(cudaError_t error)

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
    cudaError_t nv12_to_rgb(uint8_t *in_y, uint8_t *in_uv, uint8_t *out_rgb, int height, int width, int pitch, int full_color_range)
