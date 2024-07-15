cimport libav
from av.codec.context cimport CodecContext

from av_hw cimport libavhw


def hwdevice_ctx_create(codec_context: CodecContext):
    cdef libavhw.AVHWDeviceType hw_device_type = libavhw.AVHWDeviceType.AV_HWDEVICE_TYPE_CUDA
    cdef libavhw.AVBufferRef *hw_device_ctx = NULL
    cdef err = libavhw.av_hwdevice_ctx_create(&hw_device_ctx, hw_device_type, NULL, NULL, 0)
    if err < 0:
        raise RuntimeError(f"Failed to create specified HW device. {libav.av_err2str(err).decode('utf-8')}.")
    (<libavhw.AVCodecContext*> codec_context.ptr).hw_device_ctx = libavhw.av_buffer_ref(hw_device_ctx)
