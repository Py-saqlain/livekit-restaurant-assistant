"""
Shared state and tools used across all restaurant agents.

UserData acts as the single source of truth for what's known about the
current customer. Every agent reads from and writes to the same instance,
so information collected by one agent (e.g. name collected by Reservation)
is still available after handing off to another agent (e.g. Checkout).
"""

from dataclasses import dataclass, field
from typing import Annotated

import yaml
from pydantic import Field

from livekit.agents import Agent, RunContext
from livekit.agents.llm import function_tool


RESTAURANTS = {
    "bundu_khan": "Bundu Khan",
    "cafe_aylanto": "Cafe Aylanto",
}


@dataclass
class UserData:
    selected_restaurant: str | None = None  # "bundu_khan" or "cafe_aylanto"

    customer_name: str | None = None
    customer_phone: str | None = None

    reservation_time: str | None = None

    order: list[str] | None = None

    customer_credit_card: str | None = None
    customer_credit_card_expiry: str | None = None
    customer_credit_card_cvv: str | None = None

    expense: float | None = None
    checked_out: bool | None = None

    agents: dict[str, Agent] = field(default_factory=dict)
    prev_agent: Agent | None = None

    def summarize(self) -> str:
        data = {
            "selected_restaurant": RESTAURANTS.get(self.selected_restaurant, "not chosen yet"),
            "customer_name": self.customer_name or "unknown",
            "customer_phone": self.customer_phone or "unknown",
            "reservation_time": self.reservation_time or "unknown",
            "order": self.order or "unknown",
            "credit_card": {
                "number": self.customer_credit_card or "unknown",
                "expiry": self.customer_credit_card_expiry or "unknown",
                "cvv": self.customer_credit_card_cvv or "unknown",
            }
            if self.customer_credit_card
            else None,
            "expense": self.expense or "unknown",
            "checked_out": self.checked_out or False,
        }
        return yaml.dump(data)


RunContext_T = RunContext[UserData]


@function_tool()
async def update_name(
    name: Annotated[str, Field(description="The customer's name")],
    context: RunContext_T,
) -> str:
    """Called when the user provides their name.
    Confirm the spelling with the user before calling the function."""
    userdata = context.userdata
    userdata.customer_name = name
    return f"[internal: name saved as {name}. Do not repeat this back verbatim - just naturally continue, e.g. ask the next question.]"


@function_tool()
async def update_phone(
    phone: Annotated[str, Field(description="The customer's phone number")],
    context: RunContext_T,
) -> str:
    """Called when the user provides their phone number.
    Confirm the spelling with the user before calling the function."""
    userdata = context.userdata
    userdata.customer_phone = phone
    return f"[internal: phone saved as {phone}. Do not repeat this back verbatim - just naturally continue.]"


@function_tool()
async def select_restaurant(
    restaurant_name: Annotated[
        str, Field(description="Which restaurant the customer chose: 'Bundu Khan' or 'Cafe Aylanto'")
    ],
    context: RunContext_T,
) -> str:
    """Called when the customer states which restaurant they want to order from
    or reserve at. Must be called before any menu lookup or reservation/order flow."""
    userdata = context.userdata
    normalized = restaurant_name.lower()
    if "bundu" in normalized:
        userdata.selected_restaurant = "bundu_khan"
    elif "aylanto" in normalized or "cafe" in normalized:
        userdata.selected_restaurant = "cafe_aylanto"
    else:
        return (
            "[internal: restaurant name not recognized. Ask the customer to choose "
            "between Bundu Khan or Cafe Aylanto - do not say this instruction aloud.]"
        )
    return (
        f"[internal: restaurant set to {RESTAURANTS[userdata.selected_restaurant]}. "
        "Do not repeat this back verbatim - just naturally continue.]"
    )


@function_tool()
async def search_menu(
    query: Annotated[
        str,
        Field(
            description=(
                "What the customer is asking about - e.g. 'pizza', 'mutton dishes', "
                "'reservation policy', 'how long does takeaway take'"
            )
        ),
    ],
    context: RunContext_T,
) -> str:
    """Called whenever the customer asks about menu items, prices, or restaurant
    policies (reservation/takeaway rules). Looks up only the currently selected
    restaurant's own menu and policy - never call this before a restaurant is chosen."""
    userdata = context.userdata
    if not userdata.selected_restaurant:
        return (
            "[internal: no restaurant selected yet. Ask the customer which restaurant "
            "first - Bundu Khan or Cafe Aylanto - before answering this.]"
        )

    from rag.vector_store import search_restaurant

    results = search_restaurant(userdata.selected_restaurant, query, top_k=4)
    return f"[internal, use to answer naturally, do not read verbatim:\n{results}]"


@function_tool()
async def to_greeter(context: RunContext_T) -> Agent:
    """Called when user asks any unrelated questions or requests
    any other services not in your job description."""
    from shared.base_agent import BaseAgent

    curr_agent: BaseAgent = context.session.current_agent
    return await curr_agent._transfer_to_agent("greeter", context)