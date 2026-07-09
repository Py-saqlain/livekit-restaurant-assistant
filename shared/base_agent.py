"""
BaseAgent: shared parent class for every restaurant agent.

Handles two things every agent needs identically:
1. on_enter() - runs the moment an agent takes over the conversation.
   Pulls a trimmed slice of the previous agent's chat history so context
   isn't lost, and injects current UserData into the system prompt.
2. _transfer_to_agent() - the actual handoff mechanic: swaps the active
   agent in the session and remembers who was talking before.
"""

import logging

from livekit.agents import Agent

from shared.user_data import RunContext_T, UserData

logger = logging.getLogger("restaurant-agent")
logger.setLevel(logging.INFO)

# Every agent prepends this to its own instructions. Reinforces the
# anti-verbatim rule at the instruction level (not just per-tool-return),
# since LLMs don't follow a single reminder with 100% consistency.
COMMUNICATION_STYLE = (
    "Speak like a real human receptionist on a phone call - warm, brief, natural. "
    "NEVER say phrases like 'has been updated', 'is confirmed to be', 'is saved as', "
    "or read back any internal system confirmation word-for-word. Instead, just "
    "naturally acknowledge (e.g. 'Got it', 'Perfect', 'Sounds good') and move to the "
    "next question or step. Keep responses to 1-2 short sentences unless summarizing "
    "a final confirmation the customer explicitly asked to hear back."
)


class BaseAgent(Agent):
    async def on_enter(self) -> None:
        agent_name = self.__class__.__name__
        logger.info(f"entering task {agent_name}")

        userdata: UserData = self.session.userdata
        chat_ctx = self.chat_ctx.copy()

        if isinstance(userdata.prev_agent, Agent):
            truncated_chat_ctx = userdata.prev_agent.chat_ctx.copy(
                exclude_instructions=True,
                exclude_function_call=False,
                exclude_handoff=True,
                exclude_config_update=True,
            ).truncate(max_items=6)
            existing_ids = {item.id for item in chat_ctx.items}
            items_copy = [
                item for item in truncated_chat_ctx.items if item.id not in existing_ids
            ]
            chat_ctx.items.extend(items_copy)

        chat_ctx.add_message(
            role="system",
            content=(
                f"You are {agent_name} agent. Current user data is {userdata.summarize()}\n"
                "When a tool returns a confirmation string, do not read it back word-for-word. "
                "Use it as internal context only, and respond to the user naturally and briefly, "
                "as a human receptionist would - e.g. just acknowledge and ask the next question."
            ),
        )
        await self.update_chat_ctx(chat_ctx)
        self.session.generate_reply(tool_choice="none")

    async def _transfer_to_agent(self, name: str, context: RunContext_T) -> tuple[Agent, str]:
        userdata = context.userdata
        current_agent = context.session.current_agent
        next_agent = userdata.agents[name]
        userdata.prev_agent = current_agent

        return next_agent, f"Transferring to {name}."