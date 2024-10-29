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
        frames_ctx.initial_pool_size = 3

        err = libavhw.av_hwframe_ctx_init(ctx.hw_frames_ctx)
        if err < 0:
            raise RuntimeError(f"Failed to initialize CUDA frame context. {libav.av_err2str(err).decode('utf-8')}.")
        codec_context.pix_fmt = "cuda"


def to_tensor(frame: VideoFrame, device: int, pix_fmt: str = "rgb24") -> torch.Tensor:
    if frame.format.name != "cuda":
        raise ValueError(f"Input frame must be in CUDA format, got {frame.format.name}.")

    cdef const cuda.Npp8u* src[2]
    src[0] = <cuda.Npp8u *> frame.ptr.data[0]
    src[1] = <cuda.Npp8u *> frame.ptr.data[1]
    cdef int src_pitch = frame.ptr.linesize[0]

    tensor = torch.empty((frame.ptr.height, frame.ptr.width, 3), dtype=torch.uint8, device=torch.device('cuda', device))
    cdef cuda.CUdeviceptr tensor_data_ptr = tensor.data_ptr()
    cdef cuda.Npp8u* dst = <cuda.Npp8u*> tensor_data_ptr
    cdef int dst_pitch = tensor.stride(0)

    cdef cuda.NppiSize roi = cuda.NppiSize(frame.ptr.width, frame.ptr.height)

    with nogil:
        if pix_fmt == "rgb24":
            if frame.ptr.color_range == libav.AVCOL_RANGE_JPEG:
                cuda.nppiNV12ToRGB_709HDTV_8u_P2C3R(src, src_pitch, dst, dst_pitch, roi)
            else:
                cuda.nppiNV12ToRGB_8u_P2C3R(src, src_pitch, dst, dst_pitch, roi)
        
        elif pix_fmt == "bgr24":
            if frame.ptr.color_range == libav.AVCOL_RANGE_JPEG:
                cuda.nppiNV12ToBGR_709HDTV_8u_P2C3R(src, src_pitch, dst, dst_pitch, roi)
            else:
                cuda.nppiNV12ToBGR_8u_P2C3R(src, src_pitch, dst, dst_pitch, roi)

        else:
            raise ValueError(f"Unsupported pixel format: {pix_fmt}.")
    return tensor


def from_tensor(tensor: torch.Tensor, codec_context: CodecContext, pix_fmt: str = "rgb24") -> VideoFrame:
    cdef cuda.CUdeviceptr tensor_data_ptr = tensor.data_ptr()
    cdef const cuda.Npp8u* src = <cuda.Npp8u*> tensor_data_ptr
    cdef int src_pitch = tensor.stride(0)

    # Allocate an empty frame with the final format
    cdef VideoFrame frame = VideoFrame(0, 0, format="cuda")
    frame.ptr = libav.av_frame_alloc()
    frame.ptr.height = tensor.shape[0]
    frame.ptr.width = tensor.shape[1]
    frame.ptr.format = libavhw.AV_PIX_FMT_CUDA
    err = libavhw.av_hwframe_get_buffer((<AVCodecContext*> codec_context.ptr).hw_frames_ctx, frame.ptr, 0)
    if err < 0:
        raise RuntimeError(f"Failed to allocate CUDA frame: {libav.av_err2str(err).decode('utf-8')}.")

    cdef cuda.Npp8u* dst[3]
    dst[0] = <cuda.Npp8u *> frame.ptr.data[0]
    dst[1] = <cuda.Npp8u *> frame.ptr.data[1]
    dst[2] = <cuda.Npp8u *> frame.ptr.data[2]
    cdef int[3] dst_pitch = [frame.ptr.linesize[0], frame.ptr.linesize[1], frame.ptr.linesize[2]]

    cdef cuda.NppiSize roi = cuda.NppiSize(frame.ptr.width, frame.ptr.height)

    with nogil:
        if pix_fmt == "rgb24":
            if frame.ptr.color_range == libav.AVCOL_RANGE_JPEG:
                cuda.nppiRGBToYCbCr420_JPEG_8u_C3P3R(src, src_pitch, dst, dst_pitch, roi)
            else:
                cuda.nppiRGBToYCbCr420_8u_C3P3R(src, src_pitch, dst, dst_pitch, roi)
        
        elif pix_fmt == "bgr24":
            if frame.ptr.color_range == libav.AVCOL_RANGE_JPEG:
                cuda.nppiBGRToYCbCr420_JPEG_8u_C3P3R(src, src_pitch, dst, dst_pitch, roi)
            else:
                cuda.nppiBGRToYCbCr420_8u_C3P3R(src, src_pitch, dst, dst_pitch, roi)

        else:
            raise ValueError(f"Unsupported pixel format: {pix_fmt}.")
    frame._init_user_attributes() # Update frame's internal attributes
    return frame
