import time
from pathlib import Path

import av
import av.datasets
import av_hw
import cv2
import torch

VIDEO_FILE = av.datasets.curated("pexels/time-lapse-video-of-sunset-by-the-sea-854400.mp4")
OUT_DIR = Path(__file__).parent / "out"
N_RUNS = 5


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True, parents=True)
    frame: av.VideoFrame

    # Test CPU decoding + copy to GPU
    cpu_start_time = time.perf_counter()
    for i in range(N_RUNS):
        with av.open(VIDEO_FILE) as container:
            stream = container.streams.video[0]
            for frame in container.decode(stream):
                frame_ndarray = frame.to_ndarray(format="rgb24")
                frame_tensor = torch.from_numpy(frame_ndarray).to("cuda:0", non_blocking=True)
                if i == 0 and frame.time == 0:
                    img = cv2.cvtColor(frame_ndarray, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(str(OUT_DIR / "cpu.png"), img)
            torch.cuda.synchronize()
    cpu_elapsed = time.perf_counter() - cpu_start_time
    print(f"CPU decoding took {cpu_elapsed:.2f}s")

    # Test GPU decoding
    gpu_start_time = time.perf_counter()
    for i in range(N_RUNS):
        with av.open(VIDEO_FILE) as container:
            stream = container.streams.video[0]
            av_hw.hwdevice_ctx_create(stream.codec_context)
            for frame in container.decode(stream):
                frame_tensor = av_hw.video_frame_to_tensor(frame)
                if i == 0 and frame.time == 0:
                    img = cv2.cvtColor(frame_tensor.cpu().numpy(), cv2.COLOR_RGB2BGR)
                    cv2.imwrite(str(OUT_DIR / "gpu.png"), img)
            torch.cuda.synchronize()
    gpu_elapsed = time.perf_counter() - gpu_start_time
    print(f"GPU decoding took {gpu_elapsed:.2f}s")

    # Test difference in decoded images
    cpu_frame = cv2.imread(str(OUT_DIR / "cpu.png"))
    gpu_frame = cv2.imread(str(OUT_DIR / "gpu.png"))
    diff = cv2.absdiff(cpu_frame, gpu_frame)
    print(f"Max diff in px values between CPU and GPU decoding: {diff.max()}")


if __name__ == "__main__":
    main()
