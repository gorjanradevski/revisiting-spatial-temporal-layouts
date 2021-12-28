import numpy as np
import ffmpeg


def load_video(in_filepath: str):
    """Loads a video from a filepath."""
    probe = ffmpeg.probe(in_filepath)
    video_stream = next(
        (stream for stream in probe["streams"] if stream["codec_type"] == "video"),
        None,
    )
    width = int(video_stream["width"])
    height = int(video_stream["height"])
    out, _ = (
        ffmpeg.input(in_filepath)
        .output("pipe:", format="rawvideo", pix_fmt="rgb24")
        # https://github.com/kkroening/ffmpeg-python/issues/68#issuecomment-443752014
        .global_args("-loglevel", "error")
        .run(capture_stdout=True)
    )
    video = np.frombuffer(out, np.uint8).reshape([-1, height, width, 3])

    return video
