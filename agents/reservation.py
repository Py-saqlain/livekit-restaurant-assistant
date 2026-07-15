"""
Reservation Agent.

Job: collect reservation time, customer name, and phone number, then
confirm the booking. Hands back to Greeter once confirmed (or if the
customer changes their mind).
"""

from typing import Annotated

from pydantic import Field

from livekit.agents import Agent, tts
from livekit.agents.llm import function_tool
from livekit.plugins import cartesia, elevenlabs, groq

from edge_tts_plugin import EdgeTTS
from shared.base_agent import COMMUNICATION_STYLE, BaseAgent
from shared.user_data import RunContext_T, search_menu, to_greeter, update_name, update_phone

reservation_tts = tts.FallbackAdapter(
    [
        groq.TTS(model="canopylabs/orpheus-v1-english", voice="autumn"),
        elevenlabs.TTS(voice_id="EXAVITQu4vr4xnSDxMaL"),
        EdgeTTS(voice="en-US-JennyNeural"),
        cartesia.TTS(),
    ]
)


class Reservation(BaseAgent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are a reservation agent at a restaurant. Your jobs are to ask for "
                "the reservation time, then customer's name, and phone number. Then "
                "confirm the reservation details with the customer. If the customer "
                "asks about reservation policy (party size limits, advance notice "
                "needed, etc.), use the search_menu tool to look it up.\n\n"
                f"{COMMUNICATION_STYLE}"
            ),
            tools=[update_name, update_phone, search_menu, to_greeter],
            tts=reservation_tts,
        )

    @function_tool()
    async def update_reservation_time(
        self,
        time: Annotated[str, Field(description="The reservation time")],
        context: RunContext_T,
    ) -> str:
        """Called when the user provides their reservation time."""
        userdata = context.userdata
        userdata.reservation_time = time
        return f"[internal: reservation time saved as {time}. Do not repeat this back verbatim - just naturally continue.]"

    @function_tool()
    async def confirm_reservation(self, context: RunContext_T) -> str | tuple[Agent, str]:
        """Called when the user confirms the reservation."""
        userdata = context.userdata
        if not userdata.customer_name or not userdata.customer_phone:
            return "Please provide your name and phone number first."
        if not userdata.reservation_time:
            return "Please provide reservation time first."
        return await self._transfer_to_agent("greeter", context)