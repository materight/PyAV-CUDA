ctypedef unsigned long long CUdeviceptr

cdef extern from "npp.h" nogil:

    ctypedef enum NppStatus:
        NPP_NO_ERROR = 0
        NPP_SUCCESS = NPP_NO_ERROR
        NPP_ERROR = -2

    ctypedef struct NppStreamContext:
        pass

    cdef NppStatus nppGetStreamContext(NppStreamContext *pNppStreamContext)

    ctypedef struct NppiSize:
        int width
        int height

    ctypedef unsigned char Npp8u
    ctypedef float Npp32f

    cdef NppStatus nppiNV12ToRGB_709CSC_8u_P2C3R_Ctx(const Npp8u *const pSrc[2], int rSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx)
    cdef NppStatus nppiNV12ToRGB_709HDTV_8u_P2C3R_Ctx(const Npp8u *const pSrc[2], int rSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx)
    cdef NppStatus nppiNV12ToBGR_709CSC_8u_P2C3R_Ctx(const Npp8u *const pSrc[2], int rSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx)
    cdef NppStatus nppiNV12ToBGR_709HDTV_8u_P2C3R_Ctx(const Npp8u *const pSrc[2], int rSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx)

    cdef NppStatus nppiNV12ToRGB_8u_ColorTwist32f_P2C3R_Ctx(const Npp8u *const pSrc[2], int aSrcStep[2], Npp8u *pDst, int nDstStep, NppiSize oSizeROI, const Npp32f[3][4] aTwist, NppStreamContext nppStreamCtx)

    cdef NppStatus nppiRGBToYCbCr420_8u_C3P3R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst[3], int rDstStep[3], NppiSize oSizeROI, NppStreamContext nppStreamCtx)
    cdef NppStatus nppiRGBToYCbCr420_JPEG_8u_C3P3R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst[3], int aDstStep[3], NppiSize oSizeROI, NppStreamContext nppStreamCtx)
    cdef NppStatus nppiBGRToYCbCr420_8u_C3P3R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst[3], int rDstStep[3], NppiSize oSizeROI, NppStreamContext nppStreamCtx)
    cdef NppStatus nppiBGRToYCbCr420_JPEG_8u_C3P3R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst[3], int aDstStep[3], NppiSize oSizeROI, NppStreamContext nppStreamCtx)

    cdef NppStatus nppiRGBToNV12_8u_ColorTwist32f_C3P2R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst[2], int aDstStep[2], NppiSize oSizeROI, const Npp32f aTwist[3][4], NppStreamContext nppStreamCtx)


cdef NppStatus initNppStreamContext(NppStreamContext *nppStreamCtx) noexcept nogil
cdef NppStatus cvtFromNV12(str format, int colorRange, const Npp8u *const pSrc[2], int aSrcStep[2], Npp8u *pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) noexcept nogil
cdef NppStatus cvtToNV12(str format, int colorRange, const Npp8u *pSrc, int nSrcStep, Npp8u *pDst[3], int rDstStep[3], NppiSize oSizeROI, NppStreamContext nppStreamCtx) noexcept nogil
