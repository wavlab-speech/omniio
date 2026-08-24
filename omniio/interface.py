import os
from typing import Optional
from omniio.definitions import ArchiveRead, AudioRead, TextRead, VideoRead, ImageRead
from omniio.audio.read import audio_read_local, audio_read_remote
from omniio.text.read import text_read_local, text_read_remote
from omniio.video.read import video_read_local, video_read_remote
from omniio.image.read import image_read_local, image_read_remote

def audio_read(
    archive_path: str, 
    start_offset: int, 
    file_size: int, 
    start_time: int = None, 
    end_time: int = None
) -> AudioRead:

    if os.path.exists(archive_path):
        return audio_read_local(
            archive_path,
            start_offset,
            file_size,
            start_time,
            end_time
        )

    else:
        return audio_read_remote(
            archive_path,
            start_offset,
            file_size,
            start_time,
            end_time
        )

def text_read(
    archive_path: str,
    start_offset: int,
    file_size: int,
) -> TextRead:

    if os.path.exists(archive_path):
        return text_read_local(
            archive_path,
            start_offset,
            file_size,
        )

    else:
        return text_read_remote(
            archive_path,
            start_offset,
            file_size,
        )

def video_read(
    archive_path: str,
    start_offset: int,
    file_size: int,
    start_frame: Optional[int] = None,
    end_frame: Optional[int] = None,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None
) -> VideoRead:
    """
    Read video from archive, automatically routing to local or remote.

    Args:
        archive_path: Path or URL to the archive bin file
        start_offset: Byte offset where video entry begins
        file_size: Number of bytes for this entry
        start_frame: Optional start frame for slicing
        end_frame: Optional end frame for slicing
        start_time: Optional start time in seconds for slicing
        end_time: Optional end time in seconds for slicing

    Returns:
        VideoRead object with video and audio data
    """
    if os.path.exists(archive_path):
        return video_read_local(
            archive_path,
            start_offset,
            file_size,
            start_frame,
            end_frame,
            start_time,
            end_time
        )
    else:
        return video_read_remote(
            archive_path,
            start_offset,
            file_size,
            start_frame,
            end_frame,
            start_time,
            end_time
        )

def image_read(
    archive_path: str,
    start_offset: int,
    file_size: int,
) -> ImageRead:
    if os.path.exists(archive_path):
        return image_read_local(archive_path, start_offset, file_size)
    else:
        return image_read_remote(archive_path, start_offset, file_size)