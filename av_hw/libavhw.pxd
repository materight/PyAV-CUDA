cimport libav # Import headers from av

cdef extern from "libavutil/buffer.h" nogil:
    cdef struct AVBuffer
    cdef struct AVBufferRef

    cdef AVBufferRef* av_buffer_ref(AVBufferRef *buf)
    cdef void av_buffer_unref(AVBufferRef **buf)


cdef extern from "libavutil/hwcontext.h" nogil:

    enum AVHWDeviceType:
        AV_HWDEVICE_TYPE_NONE
        AV_HWDEVICE_TYPE_VDPAU
        AV_HWDEVICE_TYPE_CUDA
        AV_HWDEVICE_TYPE_VAAPI
        AV_HWDEVICE_TYPE_DXVA2
        AV_HWDEVICE_TYPE_QSV
        AV_HWDEVICE_TYPE_VIDEOTOOLBOX
        AV_HWDEVICE_TYPE_D3D11VA
        AV_HWDEVICE_TYPE_DRM
        AV_HWDEVICE_TYPE_OPENCL
        AV_HWDEVICE_TYPE_MEDIACODEC

    cdef int av_hwdevice_ctx_create(AVBufferRef **device_ctx, AVHWDeviceType type, const char *device, libav.AVDictionary *opts, int flags)


cdef extern from "libavcodec/avcodec.h" nogil:
        
    cdef struct AVCodecContext:   
        AVBufferRef *hw_device_ctx
