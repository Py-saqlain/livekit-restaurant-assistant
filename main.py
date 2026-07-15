"""
Main entrypoint for the restaurant multi-agent voice system.

Builds all 4 agents, wires them into a single AgentSession sharing one
UserData instance, and starts the session with Greeter as the first
agent the customer talks to.
"""

import logging

from dotenv import load_dotenv

load_dotenv()

from livekit.agents import AgentServer, AgentSession, JobContext, cli, inference, llm
from livekit.plugins import google, groq, openai

from agents.checkout import Checkout
from agents.greeter import Greeter
from agents.reservation import Reservation
from agents.takeaway import Takeaway
from shared.user_data import UserData

logger = logging.getLogger("restaurant-agent")
logger.setLevel(logging.INFO)

# 2-provider LLM fallback: Groq (primary, fast) → Cerebras (backup, separate quota)
# If Groq hits its daily token cap, Cerebras picks up seamlessly.
session_llm = llm.FallbackAdapter(
    [
        openai.LLM.with_cerebras(model="gpt-oss-120b", temperature=0.1),
        groq.LLM(model="llama-3.3-70b-versatile", temperature=0.1),
        openai.LLM.with_openrouter(model="meta-llama/llama-3.3-70b-instruct:free", temperature=0.1),
        google.LLM(model="gemini-2.0-flash", temperature=0.1),
    ]
)

server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: JobContext):
    userdata = UserData()
    userdata.agents.update(
        {
            "greeter": Greeter(),
            "reservation": Reservation(),
            "takeaway": Takeaway(),
            "checkout": Checkout(),
        }
    )

    session = AgentSession[UserData](
        userdata=userdata,
        stt=groq.STT(model="whisper-large-v3-turbo"),
        llm=session_llm,
        vad=inference.VAD(
            model="silero",
            activation_threshold=0.7,
            min_speech_duration=0.3,
            min_silence_duration=0.6,
        ),
        max_tool_steps=5,
    )

    await session.start(
        agent=userdata.agents["greeter"],
        room=ctx.room,
    )


if __name__ == "__main__":
    cli.run_app(server)