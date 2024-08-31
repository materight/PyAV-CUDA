cimport libav
from libc.stdint cimport uint8_t
from av.video.frame cimport VideoFrame
from av.codec.context cimport CodecContext
import torch

from avhardware cimport libavhw, cuda
from avhardware.libavhw cimport AVBufferRef, AVHWDeviceType, AVCodecContext


cdef class HWDeviceContext:

    cdef AVBufferRef* ptr
    cdef int device

    def __cinit__(self, int device):
        self.ptr = NULL
        self.device = device

        cdef err = libavhw.av_hwdevice_ctx_create(
            &self.ptr,
            libavhw.AV_HWDEVICE_TYPE_CUDA,
            str(self.device).encode(),
            NULL,
            0
        )
        if err < 0:
            raise RuntimeError(f"Failed to create specified HW device. {libav.av_err2str(err).decode('utf-8')}.")

    def close(self):
        if self.ptr:
            libavhw.av_buffer_unref(&self.ptr)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def attach(self, CodecContext codec_context):
        (<AVCodecContext*> codec_context.ptr).hw_device_ctx = libavhw.av_buffer_ref(self.ptr)

    def to_tensor(self, frame: VideoFrame) -> torch.Tensor:
        tensor = torch.empty((frame.ptr.height, frame.ptr.width, 3), dtype=torch.uint8, device=torch.device('cuda', self.device))
        cdef cuda.CUdeviceptr tensor_ptr = tensor.data_ptr()
        with nogil:
            err =  cuda.nv12_to_rgb(
                <uint8_t*> frame.ptr.data[0],
                <uint8_t*> frame.ptr.data[1],
                <uint8_t*> tensor_ptr,
                frame.ptr.height,
                frame.ptr.width,
                frame.ptr.linesize[0],
                (frame.ptr.color_range == libav.AVCOL_RANGE_JPEG), # Use full color range for yuvj420p format
            )
            if err != cuda.cudaSuccess:
                raise RuntimeError(f"Failed to decode CUDA frame: {cuda.cudaGetErrorString(err).decode('utf-8')}.")
        return tensor
