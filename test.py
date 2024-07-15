import av
import av_hw
import cv2

VIDEO_FILE = "/home/cerrion/videos/sample_cavolle.mp4"


def main() -> None:
    # ctx = av.CodecContext.create("h264_cuvid", "r")
    # av_hw.hwdevice_ctx_create(ctx)
    # ctx.options["hwaccel_output_format"] = "cuda"
    with av.open(VIDEO_FILE) as container:
        stream = container.streams.video[0]
        # ctx.extradata = stream.codec_context.extradata
        av_hw.hwdevice_ctx_create(stream.codec_context)
        for packet in container.demux(stream):
            for frame in stream.decode(packet):
                frame: av.VideoFrame
                if frame.time == 0:
                    print("Time:", frame.time)
                    print("Format:", frame.format)
                    frame_tensor = av_hw.video_frame_to_tensor(frame)
                    print("Frame shape:", frame_tensor.shape, frame_tensor.device)
                    im = cv2.cvtColor(frame_tensor.cpu().numpy(), cv2.COLOR_RGB2BGR)
                    print("Img shape:", im.shape)
                    cv2.imwrite("/home/cerrion/test.png", im)
                else:
                    frame_tensor = av_hw.video_frame_to_tensor(frame)
    print("Done")


if __name__ == "__main__":
    main()
