"""Real-time multimodal AI demo with Gemma 4 E2B + Kokoro TTS (mlx-audio)."""

import asyncio
import base64
import json
import os
import re
import tempfile
import time
from pathlib import Path

import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

import litert_lm

MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    os.path.expanduser("~/workspace/LiteRT-LM/run_dir/gemma-4-E2B-it.litertlm"),
)
SYSTEM_PROMPT = (
    "You are a friendly, conversational AI assistant. The user is talking to you "
    "through a microphone and showing you their camera. "
    "You MUST always use the respond_to_user tool to reply. "
    "First transcribe exactly what the user said, then write your response."
)

SENTENCE_SPLIT_RE = re.compile(r'(?<=[.!?])\s+')

app = FastAPI()
engine = None
tts_model = None
tts_sample_rate = 24000


def load_models():
    global engine, tts_model, tts_sample_rate
    print(f"Loading Gemma 4 E2B from {MODEL_PATH}...")
    engine = litert_lm.Engine(
        MODEL_PATH,
        backend=litert_lm.Backend.GPU,
        vision_backend=litert_lm.Backend.GPU,
        audio_backend=litert_lm.Backend.CPU,
    )
    engine.__enter__()
    print("Engine loaded.")

    from mlx_audio.tts.generate import load_model
    print("Loading mlx-audio Kokoro...")
    tts_model = load_model("mlx-community/Kokoro-82M-bf16")
    tts_sample_rate = tts_model.sample_rate
    # Warmup to trigger pipeline init (phonemizer, spacy, etc.)
    list(tts_model.generate(text="Hello", voice="af_heart", speed=1.0))
    print(f"Kokoro TTS loaded (mlx-audio, sample_rate={tts_sample_rate}).")


def save_temp(data: bytes, suffix: str) -> str:
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    tmp.write(data)
    tmp.close()
    return tmp.name


def split_sentences(text: str) -> list[str]:
    """Split text into sentences for streaming TTS."""
    parts = SENTENCE_SPLIT_RE.split(text.strip())
    return [s.strip() for s in parts if s.strip()]


def tts_sentence(text: str, voice: str = "af_heart", speed: float = 1.1) -> np.ndarray:
    """Generate audio for a single sentence using mlx-audio."""
    results = list(tts_model.generate(text=text, voice=voice, speed=speed))
    return np.concatenate([np.array(r.audio) for r in results])


@app.on_event("startup")
async def startup():
    await asyncio.get_event_loop().run_in_executor(None, load_models)


@app.get("/")
async def root():
    return HTMLResponse(content=(Path(__file__).parent / "index.html").read_text())


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()

    # Per-connection tool state captured via closure
    tool_result = {}

    def respond_to_user(transcription: str, response: str) -> str:
        """Respond to the user's voice message.

        Args:
            transcription: Exact transcription of what the user said in the audio.
            response: Your conversational response to the user. Keep it to 1-4 short sentences.
        """
        tool_result["transcription"] = transcription
        tool_result["response"] = response
        return "OK"

    conversation = engine.create_conversation(
        messages=[{"role": "system", "content": SYSTEM_PROMPT}],
        tools=[respond_to_user],
    )
    conversation.__enter__()

    interrupted = asyncio.Event()
    msg_queue = asyncio.Queue()

    async def receiver():
        """Receive messages from WebSocket and route them."""
        try:
            while True:
                raw = await ws.receive_text()
                msg = json.loads(raw)
                if msg.get("type") == "interrupt":
                    interrupted.set()
                    print("Client interrupted")
                else:
                    await msg_queue.put(msg)
        except WebSocketDisconnect:
            await msg_queue.put(None)

    recv_task = asyncio.create_task(receiver())

    try:
        while True:
            msg = await msg_queue.get()
            if msg is None:
                break

            audio_path = image_path = None
            interrupted.clear()

            try:
                if msg.get("audio"):
                    audio_path = save_temp(base64.b64decode(msg["audio"]), ".wav")
                if msg.get("image"):
                    image_path = save_temp(base64.b64decode(msg["image"]), ".jpg")

                # Build multimodal content
                content = []
                if audio_path:
                    content.append({"type": "audio", "path": os.path.abspath(audio_path)})
                if image_path:
                    content.append({"type": "image", "path": os.path.abspath(image_path)})

                if audio_path and image_path:
                    content.append({"type": "text", "text": "The user just spoke to you (audio) while showing their camera (image). Respond to what they said, referencing what you see if relevant."})
                elif audio_path:
                    content.append({"type": "text", "text": "The user just spoke to you. Respond to what they said."})
                elif image_path:
                    content.append({"type": "text", "text": "The user is showing you their camera. Describe what you see."})
                else:
                    content.append({"type": "text", "text": msg.get("text", "Hello!")})

                # LLM inference
                t0 = time.time()
                tool_result.clear()
                response = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: conversation.send_message({"role": "user", "content": content})
                )
                llm_time = time.time() - t0

                # Extract response from tool call or fallback to raw text
                if tool_result:
                    strip = lambda s: s.replace('<|"|>', "").strip()
                    transcription = strip(tool_result.get("transcription", ""))
                    text_response = strip(tool_result.get("response", ""))
                    print(f"LLM ({llm_time:.2f}s) [tool] heard: {transcription!r} → {text_response}")
                else:
                    transcription = None
                    text_response = response["content"][0]["text"]
                    print(f"LLM ({llm_time:.2f}s) [no tool]: {text_response}")

                if interrupted.is_set():
                    print("Interrupted after LLM, skipping response")
                    continue

                reply = {"type": "text", "text": text_response, "llm_time": round(llm_time, 2)}
                if transcription:
                    reply["transcription"] = transcription
                await ws.send_text(json.dumps(reply))

                if interrupted.is_set():
                    print("Interrupted before TTS, skipping audio")
                    continue

                # Streaming TTS: split into sentences and send chunks progressively
                sentences = split_sentences(text_response)
                if not sentences:
                    sentences = [text_response]

                tts_start = time.time()

                # Signal start of audio stream
                await ws.send_text(json.dumps({
                    "type": "audio_start",
                    "sample_rate": tts_sample_rate,
                    "sentence_count": len(sentences),
                }))

                for i, sentence in enumerate(sentences):
                    if interrupted.is_set():
                        print(f"Interrupted during TTS (sentence {i+1}/{len(sentences)})")
                        break

                    # Generate audio for this sentence
                    pcm = await asyncio.get_event_loop().run_in_executor(
                        None, lambda s=sentence: tts_sentence(s)
                    )

                    if interrupted.is_set():
                        break

                    # Convert to 16-bit PCM and send as base64
                    pcm_int16 = (pcm * 32767).clip(-32768, 32767).astype(np.int16)
                    await ws.send_text(json.dumps({
                        "type": "audio_chunk",
                        "audio": base64.b64encode(pcm_int16.tobytes()).decode(),
                        "index": i,
                    }))

                tts_time = time.time() - tts_start
                print(f"TTS ({tts_time:.2f}s): {len(sentences)} sentences")

                if not interrupted.is_set():
                    await ws.send_text(json.dumps({
                        "type": "audio_end",
                        "tts_time": round(tts_time, 2),
                    }))

            finally:
                for p in [audio_path, image_path]:
                    if p and os.path.exists(p):
                        os.unlink(p)

    except WebSocketDisconnect:
        print("Client disconnected")
    finally:
        recv_task.cancel()
        conversation.__exit__(None, None, None)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
