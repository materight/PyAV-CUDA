cimport libav
from libc.stdint cimport uint8_t, uintptr_t
from av.video.frame cimport VideoFrame
from av.codec.context cimport CodecContext
from av.video.format cimport get_video_format
import torch

from avcuda cimport libavhw, npp
from avcuda.libavhw cimport AVBufferRef, AVHWDeviceType, AVCodecContext, AVHWFramesContext
from avcuda.npp cimport CUdeviceptr, NppStreamContext, initNppStreamContext, NppStatus, Npp8u, Npp32f, NppiSize


cdef int AV_CUDA_USE_CURRENT_CONTEXT = 2  # From libavutil/hwcontext_cuda.h

cdef class HWDeviceContext:
    cdef AVBufferRef* hwdevice_ctx
    cdef NppStreamContext npp_ctx

    def __cinit__(self, int device):
        self.hwdevice_ctx = NULL

        # Since we are re-using the pytorch context, we need to ensure that the CUDA context is initialized
        torch.cuda.init()
        torch.cuda.synchronize()

        err = libavhw.av_hwdevice_ctx_create(
            &self.hwdevice_ctx,
            libavhw.AV_HWDEVICE_TYPE_CUDA,
            str(device).encode(),
            NULL,
            AV_CUDA_USE_CURRENT_CONTEXT,
        )
        if err < 0:
            raise RuntimeError(f"Failed to create specified HW device. {libav.av_err2str(err).decode('utf-8')}.")

        err = initNppStreamContext(&self.npp_ctx)
        if err != npp.NPP_SUCCESS:
            raise RuntimeError(f"Failed to get NPP stream context: {err}.")


cdef dict[int, HWDeviceContext] device_contexts = {}

cdef HWDeviceContext get_device_context(int device):
    if device not in device_contexts:
        device_contexts[device] = HWDeviceContext(device)
    return device_contexts[device]


def init_hwcontext(codec_context: CodecContext, device: int = 0):
    cdef AVBufferRef* hw_device_ctx = get_device_context(device).hwdevice_ctx

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
        frames_ctx.initial_pool_size = 3

        err = libavhw.av_hwframe_ctx_init(ctx.hw_frames_ctx)
        if err < 0:
            raise RuntimeError(f"Failed to initialize CUDA frame context. {libav.av_err2str(err).decode('utf-8')}.")
        codec_context.pix_fmt = "cuda"


def to_tensor(frame: VideoFrame, device: int = 0, format: str = "rgb24") -> torch.Tensor:
    if frame.format.name != "cuda":
        raise ValueError(f"Input frame must be in CUDA format, got {frame.format.name}.")

    cdef const Npp8u* src[2]
    src[0] = <Npp8u*> frame.ptr.data[0]
    src[1] = <Npp8u*> frame.ptr.data[1]
    cdef int[2] src_pitch = [frame.ptr.linesize[0], frame.ptr.linesize[1]]

    tensor = torch.empty((frame.ptr.height, frame.ptr.width, 3), dtype=torch.uint8, device=torch.device('cuda', device))
    cdef CUdeviceptr tensor_data_ptr = tensor.data_ptr()
    cdef Npp8u* dst = <Npp8u*> tensor_data_ptr
    cdef int dst_pitch = tensor.stride(0)

    cdef NppiSize roi = NppiSize(frame.ptr.width, frame.ptr.height)

    cdef NppStreamContext npp_ctx = get_device_context(device).npp_ctx

    cdef NppStatus status
    with nogil:
        status = npp.cvtFromNV12(format, frame.ptr.color_range, src, src_pitch, dst, dst_pitch, roi, npp_ctx)
    if status != npp.NPP_SUCCESS:
        raise RuntimeError(f"Failed to convert frame to tensor: {status}.")

    return tensor


def from_tensor(tensor: torch.Tensor, codec_context: CodecContext, format: str = "rgb24") -> VideoFrame:
    cdef CUdeviceptr tensor_data_ptr = tensor.data_ptr()
    cdef const Npp8u* src = <Npp8u*> tensor_data_ptr
    cdef int src_pitch = tensor.stride(0)

    # Allocate an empty frame with the final format
    cdef VideoFrame frame = VideoFrame(0, 0, format="cuda")
    frame.ptr = libav.av_frame_alloc()
    frame.ptr.height = tensor.shape[0]
    frame.ptr.width = tensor.shape[1]
    frame.ptr.format = libavhw.AV_PIX_FMT_CUDA
    frame.ptr.color_range = libav.AVCOL_RANGE_MPEG
    err = libavhw.av_hwframe_get_buffer((<AVCodecContext*> codec_context.ptr).hw_frames_ctx, frame.ptr, 0)
    if err < 0:
        raise RuntimeError(f"Failed to allocate CUDA frame: {libav.av_err2str(err).decode('utf-8')}.")

    cdef Npp8u* dst[3]
    dst[0] = <Npp8u*> frame.ptr.data[0]
    dst[1] = <Npp8u*> frame.ptr.data[1]
    dst[2] = <Npp8u*> frame.ptr.data[2]
    cdef int[3] dst_pitch = [frame.ptr.linesize[0], frame.ptr.linesize[1], frame.ptr.linesize[2]]

    cdef NppiSize roi = NppiSize(frame.ptr.width, frame.ptr.height)

    cdef NppStreamContext npp_ctx = get_device_context(tensor.device.index).npp_ctx

    cdef NppStatus status
    with nogil:
        status = npp.cvtToNV12(format, frame.ptr.color_range, src, src_pitch, dst, dst_pitch, roi, npp_ctx)
    if status != npp.NPP_NO_ERROR:
        raise RuntimeError(f"Failed to convert tensor to frame: {status}.")

    frame._init_user_attributes() # Update frame's internal attributes
    return frame
