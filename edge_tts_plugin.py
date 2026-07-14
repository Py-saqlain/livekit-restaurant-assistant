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

import asyncio
import io
import logging
from dataclasses import dataclass

import edge_tts
from livekit.agents import tts
from livekit.agents.tts import SynthesizedAudio, SynthesizeStream
from livekit.agents.utils import AudioBuffer
import numpy as np

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

    def synthesize(self, text: str) -> "EdgeTTSStream":
        return EdgeTTSStream(
            tts=self,
            input_text=text,
            voice=self._voice,
            rate=self._rate,
            volume=self._volume,
        )


class EdgeTTSStream(tts.ChunkedStream):
    def __init__(self, *, tts: EdgeTTS, input_text: str, voice: str, rate: str, volume: str):
        super().__init__(tts=tts, input_text=input_text)
        self._voice = voice
        self._rate = rate
        self._volume = volume

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        try:
            communicate = edge_tts.Communicate(
                text=self._input_text,
                voice=self._voice,
                rate=self._rate,
                volume=self._volume,
            )

            audio_buffer = io.BytesIO()
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_buffer.write(chunk["data"])

            audio_bytes = audio_buffer.getvalue()
            if not audio_bytes:
                raise Exception("edge-tts returned no audio data")

            # edge-tts returns MP3 — decode to PCM using pydub/av
            audio_buffer.seek(0)
            import av
            container = av.open(audio_buffer, format="mp3")
            resampler = av.AudioResampler(
                format="s16",
                layout="mono",
                rate=24000,
            )

            pcm_frames = b""
            for frame in container.decode(audio=0):
                for resampled in resampler.resample(frame):
                    pcm_frames += bytes(resampled.planes[0])

            # Emit in ~100ms chunks (2400 samples @ 24kHz, 2 bytes each = 4800 bytes)
            chunk_size = 4800
            for i in range(0, len(pcm_frames), chunk_size):
                chunk = pcm_frames[i : i + chunk_size]
                if len(chunk) < chunk_size:
                    # pad last chunk
                    chunk = chunk + b"\x00" * (chunk_size - len(chunk))
                samples = np.frombuffer(chunk, dtype=np.int16)
                output_emitter.push(
                    SynthesizedAudio(
                        request_id=self._request_id,
                        frame=tts.AudioFrame(
                            data=samples.tobytes(),
                            sample_rate=24000,
                            num_channels=1,
                            samples_per_channel=len(samples),
                        ),
                    )
                )

        except Exception as e:
            logger.error(f"edge-tts synthesis failed: {e}")
            raise