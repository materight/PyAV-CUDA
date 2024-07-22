# PyAV-hw
Extension of [PyAV](https://github.com/PyAV-Org/PyAV) with hardware decoding support.

## Installation

- Build and install FFmpeg with hardware acceleration support. You can follow [this guide](https://pytorch.org/audio/stable/build.ffmpeg.html) from the PyTorch docs.

- Re-install `PyAV` from sources to enable hardware acceleration. For instance, if FFmpeg was installed in `/opt/ffmpeg`:
    ```sh
    pip uninstall av
    PKG_CONFIG_LIBDIR="/opt/ffmpeg/lib/pkgconfig" pip install av --no-binary av --no-cache
    ```

- If the installation was successful, `h264_cuvid` should appear between the available codecs:
    ```python
    import av
    print(av.codecs_available)
    ```

- Install PyAV-HW:
    ```sh
    PKG_CONFIG_LIBDIR="/opt/ffmpeg/lib/pkgconfig" pip install pyav_hw
    ```

- To test the installation, run `python examples/benchmark.py`.