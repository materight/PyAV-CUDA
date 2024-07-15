import torch
from av.video.frame cimport VideoFrame

from av_hw cimport libavhw

def video_frame_to_tensor(frame: VideoFrame) -> torch.Tensor:
    avframe = frame.ptr
    cdef size_t height = avframe.height
    cdef size_t width = avframe.width
    cdef size_t uv_offset = width * height
    nv12 = torch.empty((height + height // 2, width), dtype=torch.uint8, device="cuda:0")
    cdef libavhw.CUdeviceptr nv12_ptr = nv12.data_ptr()
    with nogil:
        ret = libavhw.cudaMemcpy2D(
            <void*> (<libavhw.CUdeviceptr> nv12_ptr),
            width,
            <const void*> (<libavhw.CUdeviceptr> avframe.data[0]),
            avframe.linesize[0],
            width,
            height,
            libavhw.cudaMemcpyKind.cudaMemcpyDeviceToDevice
        )
        ret = libavhw.cudaMemcpy2D(
            <void*> (<libavhw.CUdeviceptr> nv12_ptr + uv_offset),
            width,
            <const void*> (<libavhw.CUdeviceptr> avframe.data[1]),
            avframe.linesize[1],
            width,
            height // 2,
            libavhw.cudaMemcpyKind.cudaMemcpyDeviceToDevice
        )
    return nv12
