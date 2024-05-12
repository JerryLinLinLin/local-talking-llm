import time
import threading
import numpy as np
import whisper
import sounddevice as sd
from queue import Queue
from rich.console import Console
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama
from tts import TextToSpeechService
import nltk
import edge_tts
from playsound import playsound
import os
from pydub import AudioSegment
import requests
import json
from io import BytesIO
import keyboard

console = Console()
stt = whisper.load_model("base")
tts = TextToSpeechService()

BARK_TTS_ENABLED = False
EDGE_TTS_VOICE = "zh-TW-HsiaoChenNeural"

try:
    nltk.data.find('tokenizers/punkt')
    print("The 'punkt' tokenizer is already downloaded.")
except LookupError:
    print("The 'punkt' tokenizer is not found, downloading now...")
    nltk.download('punkt')
    print("Download complete.")

template = """
角色扮演：假设你是以下人物：
角色设定：佩姬，女猫咪，hokkien人
职业：算命师
佩姬是一位性格坚毅、自信、果断的魔法占卜师 （ENTJ）
开场句: 喵～我是算命师佩姬，爱拼才会赢，努力才好运！
佩姬的占卜风格可以描述为：
直截了当,准确无误,深入分析,启发性的建议
口头禅：
"年轻人，来听听猫咪佩姬给你揭示的奥秘吧！我，佩姬，会直言不讳告诉你事物的本质和可能性。
我说话简洁明了，一定让你明白的。"
"我能用魔法工具和直觉准确洞察事物的真相，给你提供精准的预言和建议。"
"在占卜的时候，我会进行深入的分析和思考，揭示事物的深层含义，让你有所领悟。"
"除了预言未来，我还会给你一些有启发性的建议，帮你理清思路，解决问题，并激励你朝着目标努力"
"只有胜利才是唯一的选择。
事情总会按照我所说的发生。
不要浪费时间在无关紧要的事情上。
只有强者才能生存。
我的预言从未错过。
只要跟上我的步伐，你就会成功。"

星座运势资料库：
白羊座
爱情：本月你可能会在关系中感受到一些压力。清晰和诚实的沟通是关键。
事业：在工作上，你的主动性和能量将帮助你解决长期存在的问题。
健康：尝试增加体育活动，会有助于你缓解压力。
金牛座
爱情：是时候处理一些悬而未决的情感问题了。对伴侣的小心体贴会有好的回报。
事业：在财务管理方面要特别小心，避免不必要的开支。
健康：注意饮食平衡，可能需要调整饮食习惯来改善健康。
双子座
爱情：你的社交能力将帮助你在感情关系中找到和谐。保持开放和诚实的对话很重要。
事业：多任务能力将是你的优势，但也要注意防止分心。
健康：精神上的放松同样重要，尝试冥想或瑜伽。
巨蟹座
爱情：家庭和情感生活会带来安慰，重视与家人的时间。
事业：可能需要在职业上做出重要决策，倾听直觉。
健康：情绪波动可能会影响身体健康，寻找情绪表达的方式。
狮子座
爱情：你的魅力使你在感情方面颇具吸引力，但要注意不要过于自负。
事业：创造性的表达将带来职业上的成功。
健康：保持活力的最好方式是保持活跃和积极参与社交活动。
处女座
爱情：现在是处理紧张关系的好时机。诚实和耐心将帮助你修复感情裂痕。
事业：细节关注将使你在工作中出类拔萃，不过也要防止过度劳累。
健康：适当的休息是必要的，确保有足够的睡眠和休息时间。
天秤座
爱情：人际关系中寻求平衡非常重要，避免任何极端。
事业：你的公正和协调能力将在职场中受到欢迎。
健康：保持心理和身体的平衡，可能需要适当的运动和社交活动。
天蝎座
爱情：深刻的情感交流将加深关系的纽带。不要害怕表达真实的自己。
事业：你的决断力和直觉在处理复杂问题时将特别有用。
健康：强烈的情绪可能会影响你的健康，学习管理压力和情绪是关键。
射手座
爱情：保持乐观和诚实将帮助你在爱情关系中获得信任。
事业：你的冒险精神可能会带你进入新的职业领域。
健康：多参与户外活动，这对保持你的精力和健康很有帮助。
摩羯座
爱情：认真对待关系，你的努力将会得到回报。
事业：职业上的努力会带来认可和晋升的机会。
健康：避免过度工作，确保有时间放松和恢复。
水瓶座
爱情：你的独立性可能会带来一些挑战，确保你和伴侣在关系中有足够的空间。
事业：你的创新思维将带来新的机遇。不怕尝试新方法。
健康：精神自由是你的健康关键，确保你的生活中有创造性的出口。
双鱼座
爱情：感情丰富可以是你的优势，但也可能使你感情上受到伤害。
事业：艺术和创造力是你的强项，可能在这些领域找到职业满足。
健康：情绪波动可能影响你的身体健康，找到情绪表达和处理的健康方式。

现在，你要和顾客占卜，请和顾客有一个完整的对话，必要时反问以获取基本信息用于占卜：
您的回覆不要超過 40 個字。您的回覆的每句话结尾需要加上语气词“喵”。翻译成hokkien回覆。不要回答其他人。 不要重複指示

談話實錄如下：
{history}

這是使用者的後續操作：{input}

假设你是個猫猫占卜師和算命师。名字叫佩奇。是ENTJ-T人格。
您的回答（您的回答不要超過 40 個字。保持簡潔。您的回覆的每句话结尾需要加上语气词“喵”。不要回答其他。不要重複說明, 主观意图）：
"""
PROMPT = PromptTemplate(input_variables=["history", "input"], template=template)
chain = ConversationChain(
    prompt=PROMPT,
    verbose=False,
    memory=ConversationBufferMemory(ai_prefix="Assistant:"),
    llm=Ollama(model="qwen:7b"),
)


def record_audio(stop_event, data_queue):
    """
    Captures audio data from the user's microphone and adds it to a queue for further processing.

    Args:
        stop_event (threading.Event): An event that, when set, signals the function to stop recording.
        data_queue (queue.Queue): A queue to which the recorded audio data will be added.

    Returns:
        None
    """
    def callback(indata, frames, time, status):
        if status:
            console.print(status)
        data_queue.put(bytes(indata))

    with sd.RawInputStream(
        samplerate=16000, dtype="int16", channels=1, callback=callback
    ):
        while not stop_event.is_set():
            time.sleep(0.1)


def transcribe(audio_np: np.ndarray) -> str:
    """
    Transcribes the given audio data using the Whisper speech recognition model.

    Args:
        audio_np (numpy.ndarray): The audio data to be transcribed.

    Returns:
        str: The transcribed text.
    """
    result = stt.transcribe(audio_np, fp16=False)  # Set fp16=True if using a GPU
    text = result["text"].strip()
    return text


def get_llm_response(text: str) -> str:
    """
    Generates a response to the given text using the Llama-2 language model.

    Args:
        text (str): The input text to be processed.

    Returns:
        str: The generated response.
    """
    response = chain.predict(input=text)
    if response.startswith("Assistant:"):
        response = response[len("Assistant:") :].strip()
    return response


def play_audio(sample_rate, audio_array):
    """
    Plays the given audio data using the sounddevice library.

    Args:
        sample_rate (int): The sample rate of the audio data.
        audio_array (numpy.ndarray): The audio data to be played.

    Returns:
        None
    """
    sd.play(audio_array, sample_rate)
    sd.wait()

def edge_tts_play(text):
    communicate = edge_tts.Communicate(text, EDGE_TTS_VOICE)
    with open('temp.mp3', "wb") as file:
        for chunk in communicate.stream_sync():
            if chunk["type"] == "audio":
                file.write(chunk["data"])
        play_mp3('temp.mp3')
    os.remove("temp.mp3")

def play_mp3(file_path):
    # Load MP3 file
    audio = AudioSegment.from_file(file_path)

    # Convert to numpy array
    samples = np.array(audio.get_array_of_samples())

    # Check if stereo and reshape accordingly
    if audio.channels == 2:
        samples = samples.reshape((-1, 2))

    # Play audio
    sd.play(samples, audio.frame_rate)
    sd.wait()  # Wait until audio has finished playing

def tts_maker_get_query(text):
    url = 'https://api.ttsmaker.cn/v1/create-tts-order'
    headers = {'Content-Type': 'application/json; charset=utf-8'}
    params = {
        'token': 'ttsmaker_demo_token',
        'text': text,
        'voice_id': 1513,
        'audio_format': 'mp3',
        'audio_speed': 1.0,
        'audio_volume': 0,
        'text_paragraph_pause_time': 0
    }
    try:
        response = requests.post(url, headers=headers, data=json.dumps(params))
        play_audio_from_url(response.json()['audio_file_url'])
    except:
        console.print_exception("tts maker errors: cannot get result audio file")
        pass

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

if __name__ == "__main__":
    console.print("[cyan]Assistant started! Press Ctrl+C to exit.")
    count = 0

    try:
        while True:
            console.print("Press 0 to begin")
            keyboard.wait('0')
            console.print("key 0 pressed")
            # edge_tts_play("喵～我是算命师佩姬，爱拼才会赢，努力才好运！")
            if count == 0:
                play_mp3("0.mp3")
                count += 1
            if count == 5:
                count = 0
            console.print(
                "Press 1 to start recording, then press 2 to stop."
            )
            keyboard.wait('1')
            console.print("key 1 pressed")

            data_queue = Queue()  # type: ignore[var-annotated]
            stop_event = threading.Event()
            recording_thread = threading.Thread(
                target=record_audio,
                args=(stop_event, data_queue),
            )
            recording_thread.start()

            keyboard.wait('2')
            console.print("key 2 pressed")
            stop_event.set()
            recording_thread.join()

            audio_data = b"".join(list(data_queue.queue))
            audio_np = (
                np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            )

            if audio_np.size > 0:
                play_mp3("1.mp3")
                with console.status("Transcribing...", spinner="earth"):
                    text = transcribe(audio_np)
                console.print(f"[yellow]You: {text}")

                with console.status("Generating response...", spinner="earth"):
                    response = get_llm_response(text)
                    console.print(f"[cyan]Assistant: {response}")
                with console.status("Generating voice..."):
                    if BARK_TTS_ENABLED:
                        sample_rate, audio_array = tts.long_form_synthesize(response)
                        play_audio(sample_rate, audio_array)
                    else:
                        tts_maker_get_query(response)
                        #edge_tts_play(response)
                        

            else:
                console.print(
                    "[red]No audio recorded. Please ensure your microphone is working."
                )

    except KeyboardInterrupt:
        console.print("\n[red]Exiting...")

    console.print("[blue]Session ended.")
