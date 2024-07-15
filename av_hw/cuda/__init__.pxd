from libc.stdint cimport uint8_t

cdef extern from "cuda/cvt_color.h" nogil:
    void nv12_to_rgb(uint8_t *in_y, uint8_t *in_uv, uint8_t *out_rgb, int height, int width)
