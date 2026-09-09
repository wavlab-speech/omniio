from omniio.modalities.audio.write import audio_write
from omniio.modalities.text.write import text_write
from omniio.modalities.video.write import video_write
from omniio.modalities.image.write import image_write
from omniio.modalities.midi.write import midi_write

modality_writer = {
    'audio': audio_write,
    'text': text_write,
    'video': video_write,
    'image': image_write,
    'midi': midi_write,
}