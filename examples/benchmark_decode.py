import time
from pathlib import Path

import av
import av.datasets
import avcuda
import cv2
import torch

INPUT = av.datasets.curated("pexels/time-lapse-video-of-sunset-by-the-sea-854400.mp4")
OUT_DIR = Path(__file__).parent / "out" / "decode"
DEVICE = torch.device("cuda", index=0)
N_RUNS = 10


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True, parents=True)

    options = {}
    if INPUT.startswith("rtsp://"):
        options["flags"] = "+low_delay"
        options["rtsp_transport"] = "tcp"

    # Test GPU decoding with no copy
    print("Running GPU decoding...", end=" ", flush=True)
    gpu_start_time = time.perf_counter()

    with avcuda.HWDeviceContext(DEVICE.index) as hwdevice_ctx:
        for i in range(N_RUNS):
            with av.open(INPUT, options=options) as container:
                stream = container.streams.video[0]

                hwdevice_ctx.attach(stream.codec_context)

                for frame_idx, frame in enumerate(container.decode(stream)):
                    frame_tensor = hwdevice_ctx.to_tensor(frame)

                    if i == 0 and frame_idx == 0:
                        img = cv2.cvtColor(frame_tensor.cpu().numpy(), cv2.COLOR_RGB2BGR)
                        cv2.imwrite(str(OUT_DIR / "gpu.png"), img)
                stream.close()

    gpu_elapsed = time.perf_counter() - gpu_start_time
    print(f"took {gpu_elapsed:.2f}s")

    # Test CPU decoding with copy to GPU
    print("Running CPU decoding...", end=" ", flush=True)
    cpu_start_time = time.perf_counter()

    for i in range(N_RUNS):
        with av.open(INPUT, options=options) as container:
            stream = container.streams.video[0]

            for frame_idx, frame in enumerate(container.decode(stream)):
                frame_ndarray = frame.to_ndarray(format="rgb24")
                frame_tensor = torch.from_numpy(frame_ndarray).to(DEVICE)

                if i == 0 and frame_idx == 0:
                    img = cv2.cvtColor(frame_ndarray, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(str(OUT_DIR / "cpu.png"), img)
            stream.close()

    cpu_elapsed = time.perf_counter() - cpu_start_time
    print(f"took {cpu_elapsed:.2f}s")

    # Test difference in decoded images
    cpu_frame = cv2.imread(str(OUT_DIR / "cpu.png"))
    gpu_frame = cv2.imread(str(OUT_DIR / "gpu.png"))
    diff = cv2.absdiff(cpu_frame, gpu_frame)
    print(f"Max diff in px values between CPU and GPU decoded frames: {diff.max()}")


if __name__ == "__main__":
    main()
