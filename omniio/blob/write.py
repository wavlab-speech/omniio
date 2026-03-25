from omniio.audio.write import audio_write
from omniio.text.write import text_write

modality_writer = {
    'audio': audio_write,
    'text':  text_write
}