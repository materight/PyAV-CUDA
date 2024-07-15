from libc.stdint cimport uint8_t

from av.video.frame cimport VideoFrame
import torch

from av_hw cimport libavhw, cuda

def video_frame_to_tensor(frame: VideoFrame) -> torch.Tensor:
    avframe = frame.ptr
    cdef int height = avframe.height
    cdef int width = avframe.width
    tensor = torch.empty((height, width, 3), dtype=torch.uint8, device="cuda:0").contiguous()
    cdef cuda.CUdeviceptr tensor_ptr = tensor.data_ptr()
    with nogil:
        # ret = libavhw.cudaMemcpy2D(
        #     <void*> (<libavhw.CUdeviceptr> nv12_ptr),
        #     width,
        #     <const void*> (<libavhw.CUdeviceptr> avframe.data[0]),
        #     avframe.linesize[0],
        #     width,
        #     height,
        #     libavhw.cudaMemcpyKind.cudaMemcpyDeviceToDevice
        # )
        # ret = libavhw.cudaMemcpy2D(
        #     <void*> (<libavhw.CUdeviceptr> nv12_ptr + uv_offset),
        #     width,
        #     <const void*> (<libavhw.CUdeviceptr> avframe.data[1]),
        #     avframe.linesize[1],
        #     width,
        #     height // 2,
        #     libavhw.cudaMemcpyKind.cudaMemcpyDeviceToDevice
        # )
        err =  cuda.nv12_to_rgb(
            <uint8_t*> avframe.data[0],
            <uint8_t*>avframe.data[1],
            <uint8_t*>tensor_ptr,
            height,
            width,
            avframe.linesize[0],
        )
        if err != cuda.cudaSuccess:
            raise RuntimeError(f"Failed to decode CUDA frame: {cuda.cudaGetErrorString(err).decode('utf-8')}.")
    return tensor
