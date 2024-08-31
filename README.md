# PyAV-Hardware

**PyAV-Hardware** is an extension of [PyAV](https://github.com/PyAV-Org/PyAV) that adds support for hardware-accelerated video decoding using Nvidia GPUs. It integrates with FFmpeg and PyTorch, providing CUDA-accelerated kernels for efficient color space conversion.

## Installation

1. Build and install FFmpeg with [hardware acceleration support](https://pytorch.org/audio/stable/build.ffmpeg.html).

2. To enable hardware acceleration in PyAV, it needs to be reinstalled from source. Assuming FFmpeg is installed in `/opt/ffmpeg`, run:
    ```bash
    pip uninstall av
    PKG_CONFIG_LIBDIR="/opt/ffmpeg/lib/pkgconfig" pip install av --no-binary av --no-cache
    ```
    If the installation was successful, `h264_cuvid` should appear between the available codecs:
    ```python
    import av
    print(av.codecs_available)
    ```

3. Install PyAV-Hardware:
    ```bash
    PKG_CONFIG_LIBDIR="/opt/ffmpeg/lib/pkgconfig" CUDA_HOME="/usr/local/cuda" pip install avhwardware
    ```

4. Test the installation by running `python examples/benchmark.py`. The output should show something like:
    ```
    Running CPU decoding... took 34.99s
    Running GPU decoding... took 8.30s
    ```


## Usage

To use hardware decoding, instantiate an `HWDeviceContext` and attach it to a `VideoStream`. Note that an `HWDeviceContext` can be shared by multiple `VideoStream` instances to save memory.

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

        # Convert frames into RGB PyTorch tensors on the same device
        for frame in container.decode(stream):
            frame_tensor = hwdevice_ctx.to_tensor(frame)
```