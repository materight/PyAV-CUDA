# PyAV-Hardware
Extension of [PyAV](https://github.com/PyAV-Org/PyAV) with hardware decoding support. Can be used to accelerate video decoding with FFmpeg on NVIDIA GPUs. Supports PyTorch and implements CUDA-accelerated kernels for color space conversion.

## Installation

- Build and install FFmpeg with [hardware acceleration support](https://pytorch.org/audio/stable/build.ffmpeg.html).

- Re-install `PyAV` from sources to enable hardware acceleration. For instance, if FFmpeg was installed in `/opt/ffmpeg`:
    ```sh
    pip uninstall av
    PKG_CONFIG_LIBDIR="/opt/ffmpeg/lib/pkgconfig" pip install av --no-binary av --no-cache
    ```
    If the installation was successful, `h264_cuvid` should appear in the available codecs:
    ```python
    import av
    print(av.codecs_available)
    ```

- Install PyAV-HW:
    ```sh
    PKG_CONFIG_LIBDIR="/opt/ffmpeg/lib/pkgconfig" pip install avhw
    ```

- To test the installation, run `python examples/benchmark.py`. The output should be something like:
    ```
    CPU decoding took 31.91s
    GPU decoding took 7.07s
    ```


## Usage

To use hardware decoding, instantiate an `HWDeviceContext` and attach it to a `VideoStream`. Note that an `HWDeviceContext` can only be shared by multiple `VideoStream` instances to save memory.

```python
import av
import avhardware

CUDA_DEVICE = 0

with (
    av.open("video.mp4") as container,
    avhardware.HWDeviceContext(CUDA_DEVICE) as hwdevice_ctx,
):
        stream = container.streams.video[0]
        hwdevice_ctx.attach(stream.codec_context)

        # Converts frame into an RGB PyTorch tensor on the same device
        for frame in container.decode(stream):
            frame_tensor = hwdevice_ctx.to_tensor(frame)
```