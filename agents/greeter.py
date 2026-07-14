"""
Greeter Agent - the customer's first point of contact.

Job: welcome the caller, understand whether they want to make a
reservation or place a takeaway order, and route them to the right
specialist agent. Does not collect any customer data itself.
"""

from livekit.agents import Agent, RunContext, tts, llm as lk_llm
from livekit.agents.llm import function_tool
from livekit.plugins import cartesia, groq, openai

from edge_tts_plugin import EdgeTTS
from shared.base_agent import COMMUNICATION_STYLE, BaseAgent
from shared.user_data import RESTAURANTS, RunContext_T, select_restaurant

# edge-tts: unlimited, no API key, no rate limits — primary for all turns
# cartesia: fallback only if edge-tts fails (should be rare)
greeter_tts = tts.FallbackAdapter(
    [
        EdgeTTS(voice="en-US-JennyNeural"),
        cartesia.TTS(),
    ]
)

# 2-layer LLM fallback: Groq → Cerebras
greeter_llm = lk_llm.FallbackAdapter(
    [
        groq.LLM(model="llama-3.3-70b-versatile", temperature=0.1),
        openai.LLM.with_cerebras(model="llama-3.3-70b", temperature=0.1),
    ]
)


class Greeter(BaseAgent):
    def __init__(self) -> None:
        restaurant_list = " and ".join(RESTAURANTS.values())
        super().__init__(
            instructions=(
                "You are a friendly restaurant receptionist for two restaurants: "
                f"{restaurant_list}.\n"
                "First, ask which restaurant the customer wants - use the "
                "select_restaurant tool once they answer. Then ask if they want to "
                "make a reservation or order takeaway, and route them using the "
                "right tool. If they ask about the menu or policies before choosing "
                "a restaurant, ask them to pick a restaurant first.\n\n"
                f"{COMMUNICATION_STYLE}"
            ),
            tools=[select_restaurant],
            llm=greeter_llm,
            tts=greeter_tts,
        )

    @function_tool()
    async def to_reservation(self, context: RunContext_T) -> tuple[Agent, str]:
        """Called when user wants to make or update a reservation."""
        return await self._transfer_to_agent("reservation", context)

    @function_tool()
    async def to_takeaway(self, context: RunContext_T) -> tuple[Agent, str]:
        """Called when the user wants to place a takeaway order."""
        return await self._transfer_to_agent("takeaway", context)