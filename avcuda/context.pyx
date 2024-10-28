cimport libav
from libc.stdint cimport uint8_t, uintptr_t
from av.video.frame cimport VideoFrame
from av.codec.context cimport CodecContext
from av.video.format cimport get_video_format
import torch

from avcuda cimport libavhw, cuda
from avcuda.libavhw cimport AVBufferRef, AVHWDeviceType, AVCodecContext, AVHWFramesContext


cdef class HWDeviceContext:
    cdef AVBufferRef* ptr

    def __cinit__(self, int device):
        self.ptr = NULL

        # Since we are re-using the pytorch context, we need to ensure that the CUDA context is initialized
        torch.cuda.init()
        torch.cuda.synchronize()

        err = libavhw.av_hwdevice_ctx_create(
            &self.ptr,
            libavhw.AV_HWDEVICE_TYPE_CUDA,
            str(device).encode(),
            NULL,
            libavhw.AV_CUDA_USE_CURRENT_CONTEXT,
        )
        if err < 0:
            raise RuntimeError(f"Failed to create specified HW device. {libav.av_err2str(err).decode('utf-8')}.")


cdef dict[int, HWDeviceContext] device_contexts = {}

cdef HWDeviceContext get_device_context(int device):
    if device not in device_contexts:
        device_contexts[device] = HWDeviceContext(device)
    return device_contexts[device]


def init_hwcontext(CodecContext codec_context, int device):
    cdef AVBufferRef* hw_device_ctx = get_device_context(device).ptr

    cdef AVCodecContext* ctx = <AVCodecContext*> codec_context.ptr
    ctx.hw_device_ctx = libavhw.av_buffer_ref(hw_device_ctx)

    cdef AVHWFramesContext* frames_ctx
    if codec_context.is_encoder:
        ctx.sw_pix_fmt = ctx.pix_fmt
        ctx.pix_fmt = libavhw.AV_PIX_FMT_CUDA

        ctx.hw_frames_ctx = libavhw.av_hwframe_ctx_alloc(hw_device_ctx)
        if not ctx.hw_frames_ctx:
            raise RuntimeError("Failed to allocate CUDA frame context.")

        frames_ctx = <AVHWFramesContext*> ctx.hw_frames_ctx.data
        frames_ctx.format = ctx.pix_fmt
        frames_ctx.sw_format = ctx.sw_pix_fmt
        frames_ctx.width = ctx.width
        frames_ctx.height = ctx.height
        frames_ctx.initial_pool_size = 5

        err = libavhw.av_hwframe_ctx_init(ctx.hw_frames_ctx)
        if err < 0:
            raise RuntimeError(f"Failed to initialize CUDA frame context. {libav.av_err2str(err).decode('utf-8')}.")
        codec_context.pix_fmt = "cuda"


def to_tensor(frame: VideoFrame, device: int) -> torch.Tensor:
    tensor = torch.empty((frame.ptr.height, frame.ptr.width, 3), dtype=torch.uint8, device=torch.device('cuda', device))
    cdef cuda.CUdeviceptr tensor_ptr = tensor.data_ptr()
    with nogil:
        err = cuda.NV12ToRGB(
            <uint8_t*> frame.ptr.data[0],
            <uint8_t*> frame.ptr.data[1],
            <uint8_t*> tensor_ptr,
            frame.ptr.height,
            frame.ptr.width,
            frame.ptr.linesize[0],
            frame.ptr.linesize[1],
            (frame.ptr.color_range == libav.AVCOL_RANGE_JPEG), # Use full color range for yuvj420p format
        )
        if err != cuda.cudaSuccess:
            raise RuntimeError(f"Failed to decode CUDA frame: {cuda.cudaGetErrorString(err).decode('utf-8')}.")
    return tensor


def from_tensor(CodecContext codec_context, tensor: torch.Tensor) -> VideoFrame:
    cdef cuda.CUdeviceptr tensor_ptr = tensor.data_ptr()
    cdef int height = tensor.shape[0]
    cdef int width = tensor.shape[1]
    frame = VideoFrame(0, 0, format="cuda") # Allocate an empty frame with the final format
    with nogil:
        frame.ptr = libav.av_frame_alloc()
        frame.ptr.height = height
        frame.ptr.width = width
        frame.ptr.format = libavhw.AV_PIX_FMT_CUDA
        err = libavhw.av_hwframe_get_buffer((<AVCodecContext*> codec_context.ptr).hw_frames_ctx, frame.ptr, 0)
        if err < 0:
            raise RuntimeError(f"Failed to allocate CUDA frame: {libav.av_err2str(err).decode('utf-8')}.")

        err_cuda = cuda.RGBToNV12(
            <uint8_t*> tensor_ptr,
            <uint8_t*> frame.ptr.data[0],
            <uint8_t*> frame.ptr.data[1],
            <uint8_t*> frame.ptr.data[2],
            frame.ptr.height,
            frame.ptr.width,
            frame.ptr.linesize[0],
            frame.ptr.linesize[1],
        )
        if err != cuda.cudaSuccess:
            raise RuntimeError(f"Failed to encode CUDA frame: {cuda.cudaGetErrorString(err_cuda).decode('utf-8')}.")
    frame._init_user_attributes() # Update frame's internal attributes
    return frame
