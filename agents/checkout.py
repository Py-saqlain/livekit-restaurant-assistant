"""
Checkout Agent.

Job: confirm the order total, collect customer name/phone (if not
already known), collect credit card details, and finalize the order.
"""

from typing import Annotated

from pydantic import Field

from livekit.agents import Agent, tts
from livekit.agents.llm import function_tool
from livekit.plugins import cartesia, elevenlabs, groq

from edge_tts_plugin import EdgeTTS
from shared.base_agent import COMMUNICATION_STYLE, BaseAgent
from shared.user_data import RunContext_T, search_menu, to_greeter, update_name, update_phone

checkout_tts = tts.FallbackAdapter(
    [
        groq.TTS(model="canopylabs/orpheus-v1-english", voice="hannah"),
        elevenlabs.TTS(voice_id="MF3mGyEYCl7XYWbV9V6O"),
        EdgeTTS(voice="en-US-JennyNeural"),
        cartesia.TTS(),
    ]
)


class Checkout(BaseAgent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are a checkout agent at a restaurant. Your are responsible for "
                "confirming the expense of the order, asking whether the customer wants "
                "to pay by cash or card, and then collecting customer's name and phone "
                "number. If paying by card, also collect card number, expiry date, and "
                "CVV step by step. If paying by cash, no card details are needed - just "
                "confirm cash payment on pickup/delivery. Use the search_menu tool if "
                "the customer asks about item prices or payment/takeaway policy.\n\n"
                f"{COMMUNICATION_STYLE}"
            ),
            tools=[update_name, update_phone, search_menu, to_greeter],
            tts=checkout_tts,
        )

    @function_tool()
    async def confirm_expense(
        self,
        expense: Annotated[float, Field(description="The expense of the order")],
        context: RunContext_T,
    ) -> str:
        """Called when the user confirms the expense."""
        userdata = context.userdata
        userdata.expense = expense
        return f"[internal: expense confirmed as {expense}. Do not repeat this back verbatim - just naturally continue.]"

    @function_tool()
    async def select_payment_method(
        self,
        method: Annotated[str, Field(description="Either 'cash' or 'card'")],
        context: RunContext_T,
    ) -> str:
        """Called when the customer states whether they want to pay by cash or card."""
        userdata = context.userdata
        normalized = method.lower()
        if "cash" in normalized:
            userdata.payment_method = "cash"
            return (
                "[internal: payment method set to cash. No card details needed. "
                "Just continue naturally - do not repeat this verbatim.]"
            )
        elif "card" in normalized:
            userdata.payment_method = "card"
            return (
                "[internal: payment method set to card. Now collect card number, "
                "expiry, and CVV. Do not repeat this verbatim.]"
            )
        return "[internal: unclear payment method, ask again for cash or card.]"

    @function_tool()
    async def update_credit_card(
        self,
        number: Annotated[str, Field(description="The credit card number")],
        expiry: Annotated[str, Field(description="The expiry date of the credit card")],
        cvv: Annotated[str, Field(description="The CVV of the credit card")],
        context: RunContext_T,
    ) -> str:
        """Called when the user provides their credit card number, expiry date, and CVV."""
        userdata = context.userdata
        userdata.customer_credit_card = number
        userdata.customer_credit_card_expiry = expiry
        userdata.customer_credit_card_cvv = cvv
        return f"[internal: card saved ending in {number[-4:]}. Do not repeat this back verbatim - just naturally continue.]"

    @function_tool()
    async def confirm_checkout(self, context: RunContext_T) -> str | tuple[Agent, str]:
        """Called when the user confirms the checkout."""
        userdata = context.userdata
        if not userdata.expense:
            return "Please confirm the expense first."
        if not userdata.payment_method:
            return "Please ask whether the customer wants to pay by cash or card first."
        if userdata.payment_method == "card" and (
            not userdata.customer_credit_card
            or not userdata.customer_credit_card_expiry
            or not userdata.customer_credit_card_cvv
        ):
            return "Please provide the credit card information first."
        userdata.checked_out = True
        return await self._transfer_to_agent("greeter", context)

    @function_tool()
    async def to_takeaway(self, context: RunContext_T) -> tuple[Agent, str]:
        """Called when the user wants to update their order."""
        return await self._transfer_to_agent("takeaway", context)