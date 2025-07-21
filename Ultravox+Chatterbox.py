import asyncio, json, logging, warnings
from concurrent.futures import ThreadPoolExecutor
from aiohttp import web, WSMsgType
from chatterbox.tts import ChatterboxTTS
from transformers import pipeline
from silero_vad import VADIterator
import uvloop

# Use uvloop
try:
    uvloop.install()
except ImportError:
    pass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("voice-agent")

# Thread pool for blocking calls
executor = ThreadPoolExecutor(max_workers=4)

# Load models globally
stt = pipeline("automatic-speech-recognition", model="Ultravox/ultravox-large")
tts = ChatterboxTTS("chatterbox/tts-large")
vad = VADIterator(model="silero_vad/v2")

# Maintain conversation history per user
conversations = {}  # {user_id: [msgs...]}

async def handle_ws(request):
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    user = request.remote
    conversations.setdefault(user, [])

    async for msg in ws:
        if msg.type == WSMsgType.BINARY:
            audio = msg.data
            # Voice activity detection
            segments = list(vad(audio))
            if not segments:
                continue
            # Convert to text
            future_text = asyncio.get_event_loop().run_in_executor(executor, stt, {"array": audio})
            text = (await future_text)["text"]
            # Append user message
            conversations[user].append({"speaker": "user", "text": text})
            # Generate response
            # Here simple echo; replace with LLM if needed
            reply_text = "You said: " + text
            conversations[user].append({"speaker": "agent", "text": reply_text})
            # Synthesize speech
            future_audio = asyncio.get_event_loop().run_in_executor(executor, tts, reply_text)
            audio_out = await future_audio
            await ws.send_bytes(audio_out)

    return ws

app = web.Application()
app.router.add_get("/ws", handle_ws)

if __name__ == "__main__":
    web.run_app(app, port=7860)
