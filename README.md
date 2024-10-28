# PyAV-CUDA
[![PyPI version](https://img.shields.io/pypi/v/avcuda)](https://pypi.org/project/avcuda/)

**PyAV-CUDA** is an extension of [PyAV](https://github.com/PyAV-Org/PyAV) that adds support for hardware-accelerated video decoding using Nvidia GPUs. It integrates with FFmpeg and PyTorch, providing CUDA-accelerated kernels for efficient color space conversion.

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

3. Install PyAV-CUDA:
    ```bash
    PKG_CONFIG_LIBDIR="/opt/ffmpeg/lib/pkgconfig" CUDA_HOME="/usr/local/cuda" pip install avcuda
    ```

4. Test the installation by running `python examples/benchmark_decode.py`. The output should show something like:
    ```
    Running CPU decoding... took 34.99s
    Running GPU decoding... took 8.30s
    ```


## Usage

### Decoding

```python
import av
import avcuda

CUDA_DEVICE = 0

with av.open("video.mp4") as container:
    stream = container.streams.video[0]
    avcuda.init_hwcontext(stream.codec_context, CUDA_DEVICE)

    for avframe in container.decode(stream):
        frame_tensor = avcuda.to_tensor(avframe, CUDA_DEVICE)
```

### Encoding

```python
import av
import avcuda

CUDA_DEVICE = 0

NUM_FRAMES = 100
FPS = 30
WIDTH = 640
HEIGHT = 480

with av.open("video.mp4", "w") as container:
    stream = container.add_stream("h264_nvenc", rate=FPS)
    stream.pix_fmt, stream.width, stream.height = "yuv420p", WIDTH, HEIGHT

    avcuda.init_hwcontext(stream.codec_context, CUDA_DEVICE)

    for _ in range(NUM_FRAMES):
        frame_tensor = torch.randint(0, 255, (HEIGHT, WIDTH, 3), dtype=torch.uint8, device=CUDA_DEVICE)
        avframe = avcuda.from_tensor(frame_tensor, stream.codec_context) 

        for packet in stream.encode(avframe):
            container.mux(packet)
```
