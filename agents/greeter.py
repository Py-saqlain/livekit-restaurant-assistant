"""
Greeter Agent - the customer's first point of contact.

Job: welcome the caller, understand whether they want to make a
reservation or place a takeaway order, and route them to the right
specialist agent. Does not collect any customer data itself.
"""

from livekit.agents import Agent, RunContext, tts
from livekit.agents.llm import function_tool
from livekit.plugins import elevenlabs, groq

from shared.base_agent import BaseAgent
from shared.user_data import RunContext_T

GREETER_VOICE_ID = "21m00Tcm4TlvDq8ikWAM"  # Rachel

# Primary: ElevenLabs (higher quality). Fallback: Groq (if ElevenLabs
# websocket streaming hiccups - a known intermittent issue in the plugin).
greeter_tts = tts.FallbackAdapter(
    [
        elevenlabs.TTS(voice_id=GREETER_VOICE_ID),
        groq.TTS(model="canopylabs/orpheus-v1-english", voice="troy"),
    ]
)


class Greeter(BaseAgent):
    def __init__(self, menu: str) -> None:
        super().__init__(
            instructions=(
                f"You are a friendly restaurant receptionist. The menu is: {menu}\n"
                "Your jobs are to greet the caller and understand if they want to "
                "make a reservation or order takeaway. Guide them to the right agent using tools."
            ),
            llm=groq.LLM(model="llama-3.3-70b-versatile"),
            tts=greeter_tts,
        )
        self.menu = menu

    @function_tool()
    async def to_reservation(self, context: RunContext_T) -> tuple[Agent, str]:
        """Called when user wants to make or update a reservation."""
        return await self._transfer_to_agent("reservation", context)

    @function_tool()
    async def to_takeaway(self, context: RunContext_T) -> tuple[Agent, str]:
        """Called when the user wants to place a takeaway order."""
        return await self._transfer_to_agent("takeaway", context)