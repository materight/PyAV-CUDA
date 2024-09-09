import time
from pathlib import Path

import av
import av.datasets
import avcuda
import cv2
import torch
from av.video.frame import VideoFrame

INPUT = av.datasets.curated("pexels/time-lapse-video-of-sunset-by-the-sea-854400.mp4")
OUT_DIR = Path(__file__).parent / "out" / "encode"
DEVICE = torch.device("cuda", index=0)
N_RUNS = 1
FPS = 30


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True, parents=True)

    with av.open(INPUT) as container:
        stream = container.streams.video[0]
        frames_cpu = [frame.to_ndarray(format="rgb24") for frame in container.decode(stream)]
        frames_gpu = [torch.from_numpy(frame).to(DEVICE) for frame in frames_cpu]

    # Test CPU encoding
    print("Running CPU encoding...", end=" ", flush=True)
    cpu_start_time = time.perf_counter()

    # for _ in range(N_RUNS):
    #     with av.open(OUT_DIR / "cpu.mp4", "w") as container:
    #         stream = container.add_stream("libx264", rate=FPS)
    #         stream.pix_fmt = "yuv420p"
    #         stream.width = frames_cpu[0].shape[1]
    #         stream.height = frames_cpu[0].shape[0]

    #         for frame in frames_cpu:
    #             avframe = VideoFrame.from_ndarray(frame, format="rgb24")
    #             for packet in stream.encode(avframe):
    #                 container.mux(packet)
    #         stream.close()

    cpu_elapsed = time.perf_counter() - cpu_start_time
    print(f"took {cpu_elapsed:.2f}s")

    # Test GPU decoding with no copy
    print("Running GPU encoding...", end=" ", flush=True)
    gpu_start_time = time.perf_counter()

    with avcuda.HWDeviceContext(DEVICE.index) as hwdevice_ctx:
        for _ in range(N_RUNS):
            with av.open(OUT_DIR / "cpu.mp4", "w") as container:
                stream = container.add_stream("h264_nvenc", rate=FPS)
                stream.pix_fmt = "yuv420p"
                stream.width = frames_cpu[0].shape[1]
                stream.height = frames_cpu[0].shape[0]
                hwdevice_ctx.attach_encoder(stream.codec_context)

                for frame in frames_gpu:
                    avframe = hwdevice_ctx.from_tensor(stream.codec_context, frame)
                    for packet in stream.encode(avframe):
                        container.mux(packet)
                stream.close()

    gpu_elapsed = time.perf_counter() - gpu_start_time
    print(f"took {gpu_elapsed:.2f}s")

    # Test difference in decoded images
    cpu_frame = cv2.imread(str(OUT_DIR / "cpu.png"))
    gpu_frame = cv2.imread(str(OUT_DIR / "gpu.png"))
    diff = cv2.absdiff(cpu_frame, gpu_frame)
    print(f"Max diff in px values between CPU and GPU decoded frames: {diff.max()}")


if __name__ == "__main__":
    main()
