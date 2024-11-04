cimport libav


cdef NppStatus initNppStreamContext(NppStreamContext *nppStreamCtx) noexcept nogil:
    return nppGetStreamContext(nppStreamCtx)


cdef NppStatus cvtFromNV12(str format, int colorRange, const Npp8u *const pSrc[2], int aSrcStep[2], Npp8u *pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) noexcept nogil:
    if format == "rgb24" and colorRange == libav.AVCOL_RANGE_MPEG:
        return nppiNV12ToRGB_709CSC_8u_P2C3R_Ctx(pSrc, aSrcStep[0], pDst, nDstStep, oSizeROI, nppStreamCtx)
    elif format == "rgb24" and colorRange == libav.AVCOL_RANGE_JPEG:
        return nppiNV12ToRGB_709HDTV_8u_P2C3R_Ctx(pSrc, aSrcStep[0], pDst, nDstStep, oSizeROI, nppStreamCtx)
    elif format == "bgr24" and colorRange == libav.AVCOL_RANGE_MPEG:
        return nppiNV12ToBGR_709CSC_8u_P2C3R_Ctx(pSrc, aSrcStep[0], pDst, nDstStep, oSizeROI, nppStreamCtx)
    elif format == "bgr24" and colorRange == libav.AVCOL_RANGE_JPEG:
        return nppiNV12ToBGR_709HDTV_8u_P2C3R_Ctx(pSrc, aSrcStep[0], pDst, nDstStep, oSizeROI, nppStreamCtx)
    else:
        return NPP_ERROR


cdef NppStatus cvtToNV12(str format, int colorRange, const Npp8u *pSrc, int nSrcStep, Npp8u *pDst[3], int rDstStep[3], NppiSize oSizeROI, NppStreamContext nppStreamCtx) noexcept nogil:
    if format == "rgb24" and colorRange == libav.AVCOL_RANGE_MPEG:
        return nppiRGBToYCbCr420_8u_C3P3R_Ctx(pSrc, nSrcStep, pDst, rDstStep, oSizeROI, nppStreamCtx)
    elif format == "rgb24" and colorRange == libav.AVCOL_RANGE_JPEG:
        return nppiRGBToYCbCr420_JPEG_8u_C3P3R_Ctx(pSrc, nSrcStep, pDst, rDstStep, oSizeROI, nppStreamCtx)
    elif format == "bgr24" and colorRange == libav.AVCOL_RANGE_MPEG:
        return nppiBGRToYCbCr420_8u_C3P3R_Ctx(pSrc, nSrcStep, pDst, rDstStep, oSizeROI, nppStreamCtx)
    elif format == "bgr24" and colorRange == libav.AVCOL_RANGE_JPEG:
        return nppiBGRToYCbCr420_JPEG_8u_C3P3R_Ctx(pSrc, nSrcStep, pDst, rDstStep, oSizeROI, nppStreamCtx)
    else:
        return NPP_ERROR


# From NV12:
#   src[0]' = src[0] + M[0][3]
#   src[1]' = src[1] + M[1][3]
#   src[2]' = src[2] + M[2][3]
#
#   dst[0] = M[0][0] * src[0]' + M[0][1] * src[1]' + M[0][2] * src[2]'
#   dst[1] = M[1][0] * src[0]' + M[1][1] * src[1]' + M[1][2] * src[2]'
#   dst[2] = M[2][0] * src[0]' + M[2][1] * src[1]' + M[2][2] * src[2]'

cdef Npp32f[3][4] NV12_TO_RGB_MPEG = [
    [1.164,  0.000,  1.596,  -16.000],
    [1.164, -0.392, -0.813, -128.000],
    [1.164,  2.017,  0.000, -128.000],
]

cdef Npp32f[3][4] NV12_TO_RGB_JPEG = [
    [1.000,  0.000,  1.402,    0.000],
    [1.000, -0.344, -0.714, -128.000],
    [1.000,  1.772,  0.000, -128.000],
]

cdef Npp32f[3][4] NV12_TO_BGR_MPEG = [
    [NV12_TO_RGB_MPEG[2][0], NV12_TO_RGB_MPEG[2][1], NV12_TO_RGB_MPEG[2][2], NV12_TO_RGB_MPEG[2][3]],
    [NV12_TO_RGB_MPEG[1][0], NV12_TO_RGB_MPEG[1][1], NV12_TO_RGB_MPEG[1][2], NV12_TO_RGB_MPEG[1][3]],
    [NV12_TO_RGB_MPEG[0][0], NV12_TO_RGB_MPEG[0][1], NV12_TO_RGB_MPEG[0][2], NV12_TO_RGB_MPEG[0][3]],
]

cdef Npp32f[3][4] NV12_TO_BGR_JPEG = [
    [NV12_TO_RGB_JPEG[2][0], NV12_TO_RGB_JPEG[2][1], NV12_TO_RGB_JPEG[2][2], NV12_TO_RGB_JPEG[2][3]],
    [NV12_TO_RGB_JPEG[1][0], NV12_TO_RGB_JPEG[1][1], NV12_TO_RGB_JPEG[1][2], NV12_TO_RGB_JPEG[1][3]],
    [NV12_TO_RGB_JPEG[0][0], NV12_TO_RGB_JPEG[0][1], NV12_TO_RGB_JPEG[0][2], NV12_TO_RGB_JPEG[0][3]],
]


# To NV12:
#   dst[0] = M[0][0] * src[0] + M[0][1] * src[1] + M[0][2] * src[2] + M[0][3]
#   dst[1] = M[1][0] * src[0] + M[1][1] * src[1] + M[1][2] * src[2] + M[1][3]
#   dst[2] = M[2][0] * src[0] + M[2][1] * src[1] + M[2][2] * src[2] + M[2][3]

