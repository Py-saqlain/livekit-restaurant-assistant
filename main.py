"""
Main entrypoint for the restaurant multi-agent voice system.

Builds all 4 agents, wires them into a single AgentSession sharing one
UserData instance, and starts the session with Greeter as the first
agent the customer talks to.
"""

import logging

from dotenv import load_dotenv

load_dotenv()

from livekit.agents import AgentServer, AgentSession, JobContext, cli, inference
from livekit.plugins import groq

from agents.checkout import Checkout
from agents.greeter import Greeter
from agents.reservation import Reservation
from agents.takeaway import Takeaway
from shared.user_data import UserData

logger = logging.getLogger("restaurant-agent")
logger.setLevel(logging.INFO)

MENU = "Pizza: $10, Salad: $5, Ice Cream: $3, Coffee: $2"

server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: JobContext):
    userdata = UserData()
    userdata.agents.update(
        {
            "greeter": Greeter(MENU),
            "reservation": Reservation(),
            "takeaway": Takeaway(MENU),
            "checkout": Checkout(MENU),
        }
    )

    session = AgentSession[UserData](
        userdata=userdata,
        stt=groq.STT(model="whisper-large-v3-turbo"),
        llm=groq.LLM(model="llama-3.3-70b-versatile", temperature=0.3),
        vad=inference.VAD(
            model="silero",
            activation_threshold=0.7,  # higher = needs clearer speech, ignores faint noise
            min_speech_duration=0.3,   # ignores tiny blips shorter than this
            min_silence_duration=0.6,  # waits a bit longer before deciding you're done talking
        ),
        max_tool_steps=5,
    )

    await session.start(
        agent=userdata.agents["greeter"],
        room=ctx.room,
    )


if __name__ == "__main__":
    cli.run_app(server)