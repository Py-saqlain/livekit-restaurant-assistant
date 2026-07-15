"""
edge_tts_plugin.py - Custom LiveKit TTS wrapper using Microsoft Edge TTS.

Why edge-tts:
- Completely free, no API key, no account, no rate limits
- Uses Microsoft's neural voices (same ones in Edge browser)
- No monthly credit cap — zero risk of running out mid-demo

Usage:
    from edge_tts_plugin import EdgeTTS
    tts = EdgeTTS(voice="en-US-JennyNeural")
"""

import io
import logging

import edge_tts
from livekit.agents import APIConnectOptions, tts
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS

logger = logging.getLogger("edge-tts-plugin")

# Good English voices to choose from:
# en-US-JennyNeural       - warm, friendly female (good for greeter)
# en-US-GuyNeural         - clear male voice (good for takeaway)
# en-US-AriaNeural        - expressive female (good for reservation)
# en-US-DavisNeural       - confident male (good for checkout)
# en-GB-SoniaNeural       - British female (sounds professional)


class EdgeTTS(tts.TTS):
    def __init__(self, voice: str = "en-US-JennyNeural", rate: str = "+0%", volume: str = "+0%"):
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=24000,
            num_channels=1,
        )
        self._voice = voice
        self._rate = rate
        self._volume = volume

    def synthesize(
        self, text: str, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> "EdgeTTSStream":
        return EdgeTTSStream(
            tts=self,
            input_text=text,
            conn_options=conn_options,
            voice=self._voice,
            rate=self._rate,
            volume=self._volume,
        )


class EdgeTTSStream(tts.ChunkedStream):
    def __init__(
        self,
        *,
        tts: EdgeTTS,
        input_text: str,
        conn_options: APIConnectOptions,
        voice: str,
        rate: str,
        volume: str,
    ):
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._voice = voice
        self._rate = rate
        self._volume = volume

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        output_emitter.initialize(
            request_id=str(id(self)),
            sample_rate=24000,
            num_channels=1,
            mime_type="audio/pcm",
        )
        try:
            communicate = edge_tts.Communicate(
                text=self._input_text,
                voice=self._voice,
                rate=self._rate,
                volume=self._volume,
            )

            mp3_buffer = io.BytesIO()
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    mp3_buffer.write(chunk["data"])

            mp3_bytes = mp3_buffer.getvalue()
            if not mp3_bytes:
                raise Exception("edge-tts returned no audio data")

            # edge-tts returns MP3 - decode to raw PCM (16-bit, mono, 24kHz)
            mp3_buffer.seek(0)
            import av

            container = av.open(mp3_buffer, format="mp3")
            resampler = av.AudioResampler(format="s16", layout="mono", rate=24000)

            for frame in container.decode(audio=0):
                for resampled in resampler.resample(frame):
                    pcm_bytes = bytes(resampled.planes[0])
                    output_emitter.push(pcm_bytes)

            # Flush any samples buffered inside the resampler - skipping this
            # can cause clicking/popping artifacts, especially near the end
            for resampled in resampler.resample(None):
                pcm_bytes = bytes(resampled.planes[0])
                output_emitter.push(pcm_bytes)

            output_emitter.end_input()

        except Exception as e:
            logger.error(f"edge-tts synthesis failed: {e}")
            raise