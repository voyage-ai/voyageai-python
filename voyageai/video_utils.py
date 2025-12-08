from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from os import PathLike
from typing import IO, Any, Dict, Optional, Tuple, Union

from voyageai.error import VideoProcessingError

try:
    import ffmpeg  # type: ignore[import]
except ImportError:  # pragma: no cover - handled lazily in functions
    ffmpeg = None  # type: ignore[assignment]


class Video:
    """
    Represents a video payload, optionally pre-optimized, ready to be passed
    into Client.multimodal_embed in list-of-list format.

    Internally, this class stores the video bytes and minimal metadata.
    It does NOT decode frames in Python.
    """

    def __init__(
        self,
        data: bytes,
        *,
        model: str,
        optimized: bool = False,
        mime_type: Optional[str] = None,
        num_pixels: Optional[int] = None,
        num_frames: Optional[int] = None,
        estimated_num_tokens: Optional[int] = None,
    ) -> None:
        self._data = data
        # The multimodal model this video is intended for (used to pick
        # client_config and video token accounting parameters).
        self.model = model
        self.optimized = optimized
        self.mime_type = mime_type
        # Aggregate video usage characteristics. These are optional and, when
        # present, should reflect the state of the currently stored bytes.
        self.num_pixels: Optional[int] = num_pixels
        self.num_frames: Optional[int] = num_frames
        self.estimated_num_tokens: Optional[int] = estimated_num_tokens

    @classmethod
    def from_path(
        cls,
        path: Union[str, PathLike[str]],
        *,
        model: str,
        optimize: bool = True,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ) -> "Video":
        """
        Create a Video object from a local filesystem path.

        If optimize=True, call voyageai.video_utils.optimize_video(...)
        with this path and optimizer_kwargs, and return the optimized Video.

        If optimize=False, ignore optimizer_kwargs and just load bytes from path.
        """
        if optimize:
            # Delegate to the optimizer, normalizing kwargs to an empty dict if needed.
            return optimize_video(
                path,
                model=model,
                **(optimizer_kwargs or {}),
            )

        # optimize is False: read bytes directly.
        with open(path, "rb") as f:
            data = f.read()

        # Best-effort attempt to populate usage metadata. If probing fails
        # (e.g. ffmpeg missing or invalid file or client_config not available),
        # we silently fall back to None.
        num_pixels, num_frames, estimated_tokens = _compute_basic_usage_for_path(path, model=model)

        return cls(
            data=data,
            model=model,
            optimized=False,
            num_pixels=num_pixels,
            num_frames=num_frames,
            estimated_num_tokens=estimated_tokens,
        )

    @classmethod
    def from_file(
        cls,
        file_obj: IO[bytes],
        *,
        model: str,
        optimize: bool = True,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ) -> "Video":
        """
        Create a Video object from a file-like object (IO[bytes]).

        If optimize=True, call voyageai.video_utils.optimize_video(...)
        with the raw bytes and optimizer_kwargs, and return the optimized Video.

        If optimize=False, ignore optimizer_kwargs and wrap the bytes directly.
        """
        data = file_obj.read()

        if optimize:
            return optimize_video(
                data,
                model=model,
                **(optimizer_kwargs or {}),
            )

        # optimize is False: wrap the bytes as-is, but try to compute metadata.
        num_pixels: Optional[int] = None
        num_frames: Optional[int] = None
        estimated_tokens: Optional[int] = None

        temp_file: Optional[tempfile.NamedTemporaryFile] = None
        try:
            temp_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
            temp_file.write(data)
            temp_file.flush()
            num_pixels, num_frames, estimated_tokens = _compute_basic_usage_for_path(
                temp_file.name, model=model
            )
        finally:
            if temp_file is not None:
                temp_name = temp_file.name
                try:
                    temp_file.close()
                except Exception:
                    pass
                if temp_name and os.path.exists(temp_name):
                    try:
                        os.unlink(temp_name)
                    except OSError:
                        pass

        return cls(
            data=data,
            model=model,
            optimized=False,
            num_pixels=num_pixels,
            num_frames=num_frames,
            estimated_num_tokens=estimated_tokens,
        )

    def to_bytes(self) -> bytes:
        """
        Return the encoded video as raw bytes.
        """
        return self._data

    def to_file(self, path: Union[str, PathLike[str]]) -> None:
        """
        Save the encoded video bytes to the given path.
        """
        with open(path, "wb") as f:
            f.write(self._data)


def _load_video_bytes(video: Union[str, PathLike[str], bytes, Video]) -> bytes:
    """
    Helper to normalize the supported video input types into raw bytes.
    """
    if isinstance(video, Video):
        return video.to_bytes()

    if isinstance(video, (str, os.PathLike)):
        with open(video, "rb") as f:
            return f.read()

    if isinstance(video, (bytes, bytearray)):
        return bytes(video)

    raise TypeError(
        f"Unsupported video type {type(video)!r}. Expected str, PathLike, bytes, or Video."
    )


def _ensure_ffmpeg_available() -> None:
    if ffmpeg is None:
        raise ImportError(
            "ffmpeg-python is required for video optimization. "
            "Install `ffmpeg-python` and ensure `ffmpeg` is available on PATH."
        )
    if shutil.which("ffmpeg") is None:
        raise EnvironmentError(
            "The `ffmpeg` executable was not found on PATH. "
            "Please install ffmpeg and make sure it is accessible in your environment."
        )
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
    except Exception as exc:
        raise EnvironmentError(
            "Failed to execute `ffmpeg`. Please verify your ffmpeg installation."
        ) from exc


def _probe_video(path: Union[str, PathLike[str]]) -> Dict[str, Any]:
    """
    Probe video metadata using ffmpeg.probe and return key properties.
    """
    _ensure_ffmpeg_available()

    probe = ffmpeg.probe(str(path))  # type: ignore[union-attr]
    video_stream = next(
        (s for s in probe["streams"] if s.get("codec_type") == "video"),
        None,
    )
    if video_stream is None:
        raise ValueError("No video stream found in input video")

    format_info = probe.get("format", {})
    duration_str = video_stream.get("duration", format_info.get("duration", 0.0))

    return {
        "width": int(video_stream["width"]),
        "height": int(video_stream["height"]),
        "r_frame_rate": video_stream.get("r_frame_rate", "0/0"),
        "duration": float(duration_str),
    }


def _compute_basic_usage_for_path(
    path: Union[str, PathLike[str]],
    *,
    model: str,
) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    """
    Compute simple usage statistics (num_pixels, num_frames, estimated_tokens)
    for a video at the given path.

    This uses the native resolution and frame rate of the source video and,
    when available, the model's configured pixel-to-token ratio for multimodal
    video. If the client config cannot be loaded, num_pixels and num_frames
    are still computed, but estimated_tokens is returned as None.

    If probing fails (e.g. ffmpeg is missing or the file is not a valid video),
    this function returns (None, None, None) instead of raising.
    """
    try:
        meta = _probe_video(path)
    except Exception:
        return None, None, None

    width = meta["width"]
    height = meta["height"]
    duration = meta["duration"]
    fps = _parse_fps(meta["r_frame_rate"])

    if fps <= 0 or duration <= 0:
        return None, None, None

    frames = int(fps * duration)
    if frames % 2 == 1:
        frames -= 1
    if frames <= 0:
        return None, None, None

    video_config = _get_video_token_config(model)
    pixels_per_frame_raw = width * height
    if pixels_per_frame_raw <= 0:
        return None, None, None

    if video_config is None:
        num_pixels = pixels_per_frame_raw * frames
        estimated_tokens = None
        return num_pixels, frames, estimated_tokens

    min_video_pixels, max_video_pixels, video_pixel_to_token_ratio = video_config
    pixels_per_frame = max(
        min_video_pixels,
        min(max_video_pixels, pixels_per_frame_raw),
    )

    num_pixels = pixels_per_frame * frames
    estimated_tokens = max(1, (pixels_per_frame * frames) // max(video_pixel_to_token_ratio, 1))

    return num_pixels, frames, estimated_tokens


def _get_video_token_config(
    model: str,
) -> Optional[Tuple[int, int, int]]:
    """
    Load the multimodal video token accounting parameters for the given model.

    Returns (min_video_pixels, max_video_pixels, video_pixel_to_token_ratio) or
    None if the client config could not be loaded.
    """
    try:
        from voyageai._base import _get_client_config  # type: ignore
    except Exception:
        return None

    try:
        client_config = _get_client_config(model)
    except Exception:
        return None

    try:
        min_video_pixels = int(client_config["multimodal_video_pixels_min"])
        max_video_pixels = int(client_config["multimodal_video_pixels_max"])
        video_pixel_to_token_ratio = int(client_config["multimodal_video_to_tokens_ratio"])
    except (KeyError, TypeError, ValueError):
        return None

    return min_video_pixels, max_video_pixels, video_pixel_to_token_ratio


def _parse_fps(r_frame_rate: str) -> float:
    num, _, den = r_frame_rate.partition("/")
    try:
        num_f = float(num)
        den_f = float(den) if den else 1.0
        return num_f / den_f if den_f != 0 else 0.0
    except ValueError:
        return 0.0


def _round_to_multiple(value: int, multiple: int) -> int:
    """
    Round `value` to the nearest multiple of `multiple`, at least `multiple`.
    """
    if multiple <= 0:
        return value
    rounded = int(round(value / multiple) * multiple)
    return max(multiple, rounded)


def _compute_target_fps(
    original_fps: float,
    duration_sec: float,
    max_video_tokens: int,
    tokens_per_frame: int = 16,
) -> float:
    """
    Approximate a target fps given max_video_tokens.

    Assumes roughly `tokens_per_frame` tokens per frame. This is a heuristic
    and may be tuned later or aligned more closely with server-side
    accounting.

    The computed fps is additionally constrained so that the resulting frame
    count is at least two frames and strictly positive.

    Raises:
        ValueError: If `max_video_tokens` is less than or equal to 0.
        VideoProcessingError: If the input video has invalid duration or
            frame rate, or if the token budget is too small to sample at
            least two frames.
    """
    if max_video_tokens <= 0:
        raise ValueError("'max_video_tokens' must be greater than 0")

    if original_fps <= 0 or duration_sec <= 0:
        raise VideoProcessingError(
            "Invalid video duration or frame rate. Please provide a valid video duration and frame rate."
        )

    max_frames = max_video_tokens // max(tokens_per_frame, 1)
    approx_fps_limit = max_frames / duration_sec

    if approx_fps_limit <= 0 or max_frames < 2:
        raise VideoProcessingError(
            "The provided video cannot be downsampled to fit within the specified 'max_video_tokens'. "
            "Please increase 'max_video_tokens' or resize your video before proceeding."
        )

    target_fps = min(original_fps, approx_fps_limit)

    # Encourage an even frame count by slightly adjusting fps so that
    # floor(target_fps * duration) is even, approximating "drop last frame if odd".
    frames = int(target_fps * duration_sec)
    if frames > 1 and frames % 2 == 1:
        frames -= 1
        target_fps = frames / duration_sec

    return target_fps


def optimize_video(
    video: Union[str, PathLike[str], bytes, Video],
    *,
    model: str,
    resize: bool = True,
    resize_multiple: int = 28,
    downsample_fps: bool = True,
    max_video_tokens: int = 32000,
) -> Video:
    """
    Optimize video using ffmpeg-python.

    - If `video` is str or PathLike: treat it as a local path.
    - If `video` is bytes: treat it as raw encoded video bytes.
    - If `video` is a Video: re-optimize that video.

    resize:
        If True, resize video dimensions to the nearest multiple of
        `resize_multiple` to improve backend preprocessor performance.

    resize_multiple:
        The integer multiple for width/height rounding (e.g., 28).

    downsample_fps:
        If True, downsample the frame rate to keep within max_video_tokens.

    max_video_tokens:
        Approximate token budget for video frames. This function uses a simple
        heuristic tokens-per-frame estimate to select a target fps.

    This function requires `ffmpeg-python` and the `ffmpeg` binary to be
    installed and available on PATH.

    Returns:
        A Video instance containing the optimized MP4 bytes.

    Raises:
        TypeError: If the provided `video` input type is unsupported.
        VideoProcessingError: If optimization fails (for example, due to
            ffmpeg errors, invalid metadata, or a `max_video_tokens` value
            that is too small to sample at least two frames).
    """
    _ensure_ffmpeg_available()

    temp_file: Optional[tempfile.NamedTemporaryFile] = None
    input_path: Union[str, PathLike[str]]

    # 1. Normalize input to a file on disk for ffmpeg to read.
    if isinstance(video, Video):
        temp_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        temp_file.write(video.to_bytes())
        temp_file.flush()
        input_path = temp_file.name
    elif isinstance(video, (str, os.PathLike)):
        input_path = str(video)
    elif isinstance(video, (bytes, bytearray)):
        temp_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        temp_file.write(bytes(video))
        temp_file.flush()
        input_path = temp_file.name
    else:
        raise TypeError(f"Unsupported video input type: {type(video)!r}")

    try:
        # 2. Probe metadata and compute target dimensions / fps.
        meta = _probe_video(input_path)
        width = meta["width"]
        height = meta["height"]
        duration = meta["duration"]
        original_fps = _parse_fps(meta["r_frame_rate"])

        # Load model-specific video token accounting parameters, if available.
        video_config = _get_video_token_config(model)

        if resize:
            target_width = _round_to_multiple(width, resize_multiple)
            target_height = _round_to_multiple(height, resize_multiple)
        else:
            target_width = width
            target_height = height

        if downsample_fps and max_video_tokens is not None and video_config is not None:
            # Estimate tokens-per-frame based on the current spatial resolution,
            # using the same pixel-to-token ratio and pixel bounds as the
            # multimodal video client configuration.
            min_video_pixels, max_video_pixels, video_pixel_to_token_ratio = video_config

            pixels_per_frame_raw = target_width * target_height
            pixels_per_frame = max(
                min_video_pixels,
                min(max_video_pixels, pixels_per_frame_raw),
            )
            tokens_per_frame = max(
                1,
                pixels_per_frame // max(video_pixel_to_token_ratio, 1),
            )
            target_fps = _compute_target_fps(
                original_fps=original_fps,
                duration_sec=duration,
                max_video_tokens=max_video_tokens,
                tokens_per_frame=tokens_per_frame,
            )
        else:
            target_fps = original_fps

        # Approximate frame count after FPS adjustment, and ensure an even
        # number of frames by dropping the last one if needed.
        frames = 0
        if target_fps > 0 and duration > 0:
            frames = int(target_fps * duration)
            if frames % 2 == 1:
                frames -= 1
            if frames < 0:
                frames = 0

        # Aggregate usage estimates based on the current resolution.
        pixels_per_frame_raw = target_width * target_height
        num_pixels: Optional[int] = None
        estimated_tokens: Optional[int] = None
        if frames > 0 and pixels_per_frame_raw > 0:
            if video_config is not None:
                min_video_pixels, max_video_pixels, video_pixel_to_token_ratio = video_config
                pixels_per_frame = max(
                    min_video_pixels,
                    min(max_video_pixels, pixels_per_frame_raw),
                )
                num_pixels = pixels_per_frame * frames
                estimated_tokens = max(
                    1, (pixels_per_frame * frames) // max(video_pixel_to_token_ratio, 1)
                )
            else:
                # Fallback: we can still expose num_pixels if desired, but we
                # cannot provide a model-aligned token estimate.
                num_pixels = pixels_per_frame_raw * frames
                estimated_tokens = None

        # 3. Build the ffmpeg filter graph.
        stream = ffmpeg.input(str(input_path))  # type: ignore[union-attr]

        if target_fps > 0:
            stream = stream.filter("fps", fps=target_fps)

        if resize:
            stream = stream.filter(
                "scale",
                target_width,
                target_height,
                flags="bicubic",
            )

        # 4. Output settings: short GOP, no audio, H.264, yuv420p, rate control.
        x264_params = (
            "bframes=0:"
            "ref=1:"
            "cabac=0:"
            "weightp=0:"
            "deblock=0,0:"
            "scenecut=0:"
            "keyint=60:"
            "min-keyint=60"
        )

        output_kwargs = {
            "format": "mp4",
            "vcodec": "libx264",
            "pix_fmt": "yuv420p",
            "preset": "veryfast",
            "crf": 20,
            "maxrate": "6M",
            "bufsize": "12M",
            "an": None,  # drop audio
            "r": target_fps if target_fps > 0 else None,
            "x264-params": x264_params,
        }

        # Remove None values (ffmpeg-python does not accept them as kwargs).
        output_kwargs = {k: v for k, v in output_kwargs.items() if v is not None}

        # Write to a temporary file to avoid MP4-on-pipe limitations on some
        # ffmpeg builds (e.g. "muxer does not support non seekable output").
        out_fd, out_path = tempfile.mkstemp(suffix=".mp4")
        os.close(out_fd)

        stream = ffmpeg.output(stream, str(out_path), **output_kwargs)  # type: ignore[union-attr]
        stream = stream.overwrite_output()  # type: ignore[union-attr]

        try:
            out, err = ffmpeg.run(
                stream,
                capture_stdout=True,
                capture_stderr=True,
            )  # type: ignore[union-attr]
        except Exception as e:
            # If this is an ffmpeg.Error, surface stderr for easier debugging.
            if hasattr(e, "stderr"):
                stderr = getattr(e, "stderr")
                decoded = (
                    stderr.decode("utf-8", errors="ignore")
                    if isinstance(stderr, (bytes, bytearray))
                    else str(stderr)
                )
                raise VideoProcessingError(f"ffmpeg optimization failed: {decoded}") from e
            raise

        # Ensure that ffmpeg actually wrote a non-empty file.
        if not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
            decoded_err = (
                err.decode("utf-8", errors="ignore")
                if isinstance(err, (bytes, bytearray))
                else str(err)
            )
            raise VideoProcessingError(
                f"ffmpeg optimization produced empty output for {input_path}: {decoded_err}"
            )

        # Read optimized bytes from disk.
        with open(out_path, "rb") as f:
            out_bytes = f.read()

        # Successful optimization; wrap in Video object, attaching usage metadata.
        return Video(
            data=out_bytes,
            model=model,
            mime_type="video/mp4",
            optimized=True,
            num_pixels=num_pixels,
            num_frames=frames if frames > 0 else None,
            estimated_num_tokens=estimated_tokens,
        )
    finally:
        if temp_file is not None:
            temp_name = temp_file.name
            try:
                temp_file.close()
            except Exception:
                pass
            if temp_name and os.path.exists(temp_name):
                try:
                    os.unlink(temp_name)
                except OSError:
                    pass
        # Clean up temporary output file if it was created.
        try:
            if "out_path" in locals() and os.path.exists(out_path):
                os.unlink(out_path)
        except OSError:
            pass
