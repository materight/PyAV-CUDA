from libc.stdint cimport uint8_t
cimport libav


cdef extern from "libavutil/buffer.h" nogil:

    cdef struct AVBuffer:
        uint8_t* data
        int size

    cdef struct AVBufferRef:
        AVBuffer* buffer
        uint8_t* data
        int size

    cdef AVBufferRef* av_buffer_ref(AVBufferRef *buf)
    cdef void av_buffer_unref(AVBufferRef **buf)


cdef extern from "libavutil/avutil.h" nogil:
    enum AVPixelFormat:
        AV_PIX_FMT_CUDA


cdef extern from "libavutil/hwcontext.h" nogil:

    enum AVHWDeviceType:
        AV_HWDEVICE_TYPE_NONE
        AV_HWDEVICE_TYPE_CUDA

    cdef struct AVHWFramesContext:
        AVPixelFormat format
        AVPixelFormat sw_format
        int width
        int height
        int initial_pool_size

    cdef int av_hwdevice_ctx_create(AVBufferRef **device_ctx, AVHWDeviceType type, const char *device, libav.AVDictionary *opts, int flags)
    cdef AVBufferRef* av_hwframe_ctx_alloc(AVBufferRef *device_ctx)
    cdef int av_hwframe_ctx_init(AVBufferRef *ref)
    cdef int av_hwframe_get_buffer(AVBufferRef* hwframe_ctx, libav.AVFrame *frame, int flags)


cdef extern from "libavcodec/avcodec.h" nogil:
        
    cdef struct AVCodecContext:
        AVBufferRef* hw_device_ctx
        AVBufferRef* hw_frames_ctx
        AVPixelFormat sw_pix_fmt
        AVPixelFormat pix_fmt
        int width
        int height
