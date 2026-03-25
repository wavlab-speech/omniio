from omniio.audio.write import audio_write
from omniio.text.write import text_write
from omniio.video.write import video_write

modality_writer = {
    'audio': audio_write,
    'text': text_write,
    'video': video_write,
}