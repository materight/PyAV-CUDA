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


cdef extern from "npp.h" nogil:
    ctypedef enum NppStatus:
        NPP_NO_ERROR = 0

    ctypedef struct NppiSize:
        int width
        int height

    ctypedef unsigned char Npp8u

    cdef NppStatus nppiNV12ToRGB_8u_P2C3R(const Npp8u* const pSrc[2], int rSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI)
    cdef NppStatus nppiNV12ToRGB_709HDTV_8u_P2C3R(const Npp8u* const pSrc[2], int rSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI)
    cdef NppStatus nppiNV12ToBGR_8u_P2C3R(const Npp8u* const pSrc[2], int rSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI)
    cdef NppStatus nppiNV12ToBGR_709HDTV_8u_P2C3R(const Npp8u* const pSrc[2], int rSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI)

    cdef NppStatus nppiRGBToYCbCr420_8u_C3P3R(const Npp8u* pSrc, int nSrcStep, Npp8u* pDst[3], int rDstStep[3], NppiSize oSizeROI)
    cdef NppStatus nppiRGBToYCbCr420_JPEG_8u_C3P3R(const Npp8u* pSrc, int nSrcStep, Npp8u* pDst[3], int aDstStep[3], NppiSize oSizeROI)
    cdef NppStatus nppiBGRToYCbCr420_8u_C3P3R(const Npp8u* pSrc, int nSrcStep, Npp8u* pDst[3], int rDstStep[3], NppiSize oSizeROI)
    cdef NppStatus nppiBGRToYCbCr420_JPEG_8u_C3P3R(const Npp8u* pSrc, int nSrcStep, Npp8u* pDst[3], int aDstStep[3], NppiSize oSizeROI)
