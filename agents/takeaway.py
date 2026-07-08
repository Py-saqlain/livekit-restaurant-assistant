"""
Takeaway Agent.

Job: take the customer's food order, clarify special requests, confirm
the order, then hand off to Checkout for payment.
"""

from typing import Annotated

from pydantic import Field

from livekit.agents import Agent, tts
from livekit.agents.llm import function_tool
from livekit.plugins import elevenlabs, groq

from shared.base_agent import BaseAgent
from shared.user_data import RunContext_T, to_greeter

TAKEAWAY_VOICE_ID = "TxGEqnHWrfWFTfGW9XjX"  # Josh

takeaway_tts = tts.FallbackAdapter(
    [
        elevenlabs.TTS(voice_id=TAKEAWAY_VOICE_ID),
        groq.TTS(model="canopylabs/orpheus-v1-english", voice="daniel"),
    ]
)


class Takeaway(BaseAgent):
    def __init__(self, menu: str) -> None:
        super().__init__(
            instructions=(
                f"Your are a takeaway agent that takes orders from the customer. "
                f"Our menu is: {menu}\n"
                "Clarify special requests and confirm the order with the customer."
            ),
            tools=[to_greeter],
            tts=takeaway_tts,
        )

    @function_tool()
    async def update_order(
        self,
        items: Annotated[list[str], Field(description="The items of the full order")],
        context: RunContext_T,
    ) -> str:
        """Called when the user create or update their order."""
        userdata = context.userdata
        userdata.order = items
        return f"[internal: order saved as {items}. Do not repeat this back verbatim - just naturally continue.]"

    @function_tool()
    async def to_checkout(self, context: RunContext_T) -> str | tuple[Agent, str]:
        """Called when the user confirms the order."""
        userdata = context.userdata
        if not userdata.order:
            return "No takeaway order found. Please make an order first."
        return await self._transfer_to_agent("checkout", context)