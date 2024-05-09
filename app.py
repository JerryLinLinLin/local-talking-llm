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

console = Console()
stt = whisper.load_model("base")
tts = TextToSpeechService()

BARK_TTS_ENABLED = False
EDGE_TTS_VOICE = "zh-TW-HsiaoYuNeural"

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

现在，你要和顾客占卜：
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


if __name__ == "__main__":
    console.print("[cyan]Assistant started! Press Ctrl+C to exit.")

    try:
        while True:
            console.input(
                "Press Enter to begin"
            )
            edge_tts_play("你好呀，请问今天需要占卜什么呢")
            console.input(
                "Press Enter to start recording, then press Enter again to stop."
            )

            data_queue = Queue()  # type: ignore[var-annotated]
            stop_event = threading.Event()
            recording_thread = threading.Thread(
                target=record_audio,
                args=(stop_event, data_queue),
            )
            recording_thread.start()

            input()
            stop_event.set()
            recording_thread.join()

            audio_data = b"".join(list(data_queue.queue))
            audio_np = (
                np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            )

            if audio_np.size > 0:
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
                        edge_tts_play(response)
                        

            else:
                console.print(
                    "[red]No audio recorded. Please ensure your microphone is working."
                )

    except KeyboardInterrupt:
        console.print("\n[red]Exiting...")

    console.print("[blue]Session ended.")
