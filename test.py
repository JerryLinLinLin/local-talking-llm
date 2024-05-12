import requests
import numpy as np
import sounddevice as sd
from pydub import AudioSegment
from io import BytesIO
import json


def play_audio_from_url(url):
    # Download the audio file content from the URL
    response = requests.get(url)
    response.raise_for_status()  # Check if the download was successful

    # Load the audio file from the in-memory bytes
    audio = AudioSegment.from_file(BytesIO(response.content), format="mp3")

    # Convert audio to numpy array
    samples = np.array(audio.get_array_of_samples())
    
    # Check if stereo and reshape accordingly
    if audio.channels == 2:
        samples = samples.reshape((-1, 2))

    # Play audio
    sd.play(samples, audio.frame_rate)
    sd.wait()  # Wait until audio has finished playing

url = 'https://api.ttsmaker.cn/v1/create-tts-order'
headers = {'Content-Type': 'application/json; charset=utf-8'}
params = {
    'token': 'ttsmaker_demo_token',
    'text': "正在帮你占卜哦，请等等，马上就好喵",
    'voice_id': 1513,
    'audio_format': 'mp3',
    'audio_speed': 1.0,
    'audio_volume': 0,
    'text_paragraph_pause_time': 0
}
response = requests.post(url, headers=headers, data=json.dumps(params))
print(response.json())

play_audio_from_url(response.json()['audio_file_url'])

