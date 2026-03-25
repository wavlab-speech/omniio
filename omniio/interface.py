import os
from omniio.definitions import ArchiveRead, AudioRead, TextRead
from omniio.audio.read import audio_read_local, audio_read_remote
from omniio.text.read import text_read_local, text_read_remote

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
            start_time,
            end_time
        )

    else:
        return text_read_remote(
            archive_path,
            start_offset,
            file_size,
            start_time,
            end_time
        )