"""
Takeaway Agent.

Job: take the customer's food order, clarify special requests, confirm
the order, then hand off to Checkout for payment.
"""

from typing import Annotated

from pydantic import Field

from livekit.agents import Agent, tts
from livekit.agents.llm import function_tool
from livekit.plugins import cartesia

from edge_tts_plugin import EdgeTTS
from shared.base_agent import COMMUNICATION_STYLE, BaseAgent
from shared.user_data import RunContext_T, search_menu, to_greeter

takeaway_tts = tts.FallbackAdapter(
    [
        EdgeTTS(voice="en-US-GuyNeural"),
        cartesia.TTS(),
    ]
)


class Takeaway(BaseAgent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are a takeaway agent that takes orders from the customer at "
                "their chosen restaurant. Use the search_menu tool whenever the "
                "customer asks what's available, mentions a dish, or asks about "
                "prices - never guess menu items or prices from memory. Clarify "
                "special requests and confirm the order with the customer.\n\n"
                f"{COMMUNICATION_STYLE}"
            ),
            tools=[search_menu, to_greeter],
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