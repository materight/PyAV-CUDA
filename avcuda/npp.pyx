cimport libav


cdef NppStatus initNppStreamContext(NppStreamContext *nppStreamCtx) noexcept nogil:
    return nppGetStreamContext(nppStreamCtx)


cdef NppStatus cvtFromNV12(str format, int colorRange, const Npp8u *const pSrc[2], int aSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) noexcept nogil:
    if format == "rgb24" and colorRange == libav.AVCOL_RANGE_MPEG:
        return nppiNV12ToRGB_709CSC_8u_P2C3R_Ctx(pSrc, aSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx)
    elif format == "rgb24" and colorRange == libav.AVCOL_RANGE_JPEG:
        return nppiNV12ToRGB_709HDTV_8u_P2C3R_Ctx(pSrc, aSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx)
    elif format == "bgr24" and colorRange == libav.AVCOL_RANGE_MPEG:
        return nppiNV12ToBGR_709CSC_8u_P2C3R_Ctx(pSrc, aSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx)
    elif format == "bgr24" and colorRange == libav.AVCOL_RANGE_JPEG:
        return nppiNV12ToBGR_709HDTV_8u_P2C3R_Ctx(pSrc, aSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx)
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
