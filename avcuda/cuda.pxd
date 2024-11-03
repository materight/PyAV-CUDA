from libc.stdint cimport uint8_t


ctypedef unsigned long long CUdeviceptr

cdef extern from "npp.h" nogil:
    ctypedef enum NppStatus:
        NPP_NO_ERROR = 0
        NPP_ERROR = -2

    ctypedef struct NppiSize:
        int width
        int height

    ctypedef unsigned char Npp8u

    cdef NppStatus nppiNV12ToRGB_709CSC_8u_P2C3R(const Npp8u* const pSrc[2], int rSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI)
    cdef NppStatus nppiNV12ToRGB_709HDTV_8u_P2C3R(const Npp8u* const pSrc[2], int rSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI)
    cdef NppStatus nppiNV12ToBGR_709CSC_8u_P2C3R(const Npp8u* const pSrc[2], int rSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI)
    cdef NppStatus nppiNV12ToBGR_709HDTV_8u_P2C3R(const Npp8u* const pSrc[2], int rSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI)

    cdef NppStatus nppiRGBToYCbCr420_8u_C3P3R(const Npp8u* pSrc, int nSrcStep, Npp8u* pDst[3], int rDstStep[3], NppiSize oSizeROI)
    cdef NppStatus nppiRGBToYCbCr420_JPEG_8u_C3P3R(const Npp8u* pSrc, int nSrcStep, Npp8u* pDst[3], int aDstStep[3], NppiSize oSizeROI)
    cdef NppStatus nppiBGRToYCbCr420_8u_C3P3R(const Npp8u* pSrc, int nSrcStep, Npp8u* pDst[3], int rDstStep[3], NppiSize oSizeROI)
    cdef NppStatus nppiBGRToYCbCr420_JPEG_8u_C3P3R(const Npp8u* pSrc, int nSrcStep, Npp8u* pDst[3], int aDstStep[3], NppiSize oSizeROI)
