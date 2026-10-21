from typing import List, Sequence

from src.schemas import ConversationMessage


def limit_history(
    history: Sequence[ConversationMessage], max_messages: int, max_chars: int,
) -> List[ConversationMessage]:
    """Keep recent turns in order, within both limits and starting with a user."""
    if max_messages <= 0 or max_chars <= 0:
        return []

    messages = [message for message in history if message.content.strip()][-max_messages:]
    turns: List[List[ConversationMessage]] = []
    for message in messages:
        if message.role == "user":
            turns.append([message])
        elif turns:
            turns[-1].append(message)

    selected: List[List[ConversationMessage]] = []
    remaining = max_chars
    for turn in reversed(turns):
        size = sum(len(message.content) for message in turn)
        if size > remaining:
            if not selected:
                # Preserve the latest question and answer even if the answer is huge.
                per_message = remaining // len(turn)
                if per_message:
                    selected.append([
                        ConversationMessage(role=message.role, content=message.content[:per_message])
                        for message in turn
                    ])
            break
        selected.append(turn)
        remaining -= size

    return [message for turn in reversed(selected) for message in turn]
