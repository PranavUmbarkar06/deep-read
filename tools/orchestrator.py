from typing import TypedDict, Literal


def orchestrator(state: AgentState) -> dict:
    """
    Phase 1: hardcoded to 'discover'. Real version (phase 2) will use an LLM call
    to classify intent + detect whether files are attached, per the router design
    from earlier in the conversation.
    """
    print(f"[orchestrator] routing query: {state['query']!r}")
    return {"intent": "discover"}