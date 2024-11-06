cimport libav


cdef NppStatus initNppStreamContext(NppStreamContext *nppStreamCtx) noexcept nogil:
    return nppGetStreamContext(nppStreamCtx)


cdef NppStatus cvtFromNV12(str format, int colorRange, const Npp8u *const pSrc[2], int aSrcStep[2], Npp8u *pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) noexcept nogil:
    if format == "rgb24" and colorRange == libav.AVCOL_RANGE_MPEG:
        return nppiNV12ToRGB_8u_ColorTwist32f_P2C3R_Ctx(pSrc, aSrcStep, pDst, nDstStep, oSizeROI, NV12_TO_RGB_MPEG, nppStreamCtx)
    elif format == "rgb24" and colorRange == libav.AVCOL_RANGE_JPEG:
        return nppiNV12ToRGB_8u_ColorTwist32f_P2C3R_Ctx(pSrc, aSrcStep, pDst, nDstStep, oSizeROI, NV12_TO_RGB_JPEG, nppStreamCtx)
    elif format == "bgr24" and colorRange == libav.AVCOL_RANGE_MPEG:
        return nppiNV12ToRGB_8u_ColorTwist32f_P2C3R_Ctx(pSrc, aSrcStep, pDst, nDstStep, oSizeROI, NV12_TO_BGR_MPEG, nppStreamCtx)
    elif format == "bgr24" and colorRange == libav.AVCOL_RANGE_JPEG:
        return nppiNV12ToRGB_8u_ColorTwist32f_P2C3R_Ctx(pSrc, aSrcStep, pDst, nDstStep, oSizeROI, NV12_TO_BGR_JPEG, nppStreamCtx)
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


# The conversion matrices are applied as follows:
#   dst[0] = M[0][0] * src[0] + M[0][1] * src[1] + M[0][2] * src[2] + M[0][3]
#   dst[1] = M[1][0] * src[0] + M[1][1] * src[1] + M[1][2] * src[2] + M[1][3]
#   dst[2] = M[2][0] * src[0] + M[2][1] * src[1] + M[2][2] * src[2] + M[2][3]


# From NV12
cdef Npp32f[3][4] NV12_TO_RGB_MPEG = [
    [1.164,  0.000,  1.596, 1.164*-16 +  0.000*-128 +  1.596*-128],
    [1.164, -0.392, -0.813, 1.164*-16 + -0.392*-128 + -0.813*-128],
    [1.164,  2.017,  0.000, 1.164*-16 +  2.017*-128 +  0.000*-128],
]

cdef Npp32f[3][4] NV12_TO_RGB_JPEG = [
    [1.000,  0.000,  1.402, 1.000*-0 +  0.000*-128 +  1.402*-128],
    [1.000, -0.344, -0.714, 1.000*-0 + -0.344*-128 + -0.714*-128],
    [1.000,  1.772,  0.000, 1.000*-0 +  1.772*-128 +  0.000*-128],
]

cdef Npp32f[3][4] NV12_TO_BGR_MPEG = [
    NV12_TO_RGB_MPEG[2],
    NV12_TO_RGB_MPEG[1],
    NV12_TO_RGB_MPEG[0],
]

cdef Npp32f[3][4] NV12_TO_BGR_JPEG = [
    NV12_TO_RGB_JPEG[2],
    NV12_TO_RGB_JPEG[1],
    NV12_TO_RGB_JPEG[0],
]
