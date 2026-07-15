import asyncio
from edge_tts_plugin import EdgeTTS


async def test():
    t = EdgeTTS(voice="en-US-JennyNeural")
    stream = t.synthesize("Hello, this is a test.")
    async for _ in stream:
        pass
    print("SUCCESS")


asyncio.run(test())