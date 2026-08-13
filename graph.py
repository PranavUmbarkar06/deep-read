from typing import Literal
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from structures.agent_state import Paper, AgentState
from tools import find_papers, extract
from langgraph.checkpoint.memory import MemorySaver
from graph_wrapper import (
    orchestrator,
    set_papers,
    display_papers,
    ask_upload_node,
    ask_single_file_node,
    extract_formatted_node,
    summarise_paper_node,
    summary_critic_node,
    return_with_caveat_node,
    finalize_summary_node,
    check_pdf_input,
    evaluate_critic_result,
    compare_router,
    ask_compare_upload_node,
    summarize_comparison_node,
    compare_papers_node,
    compatibility_node,
    validate_and_query
)

def build_graph():
    builder = StateGraph(AgentState)

    # 1. Register All Nodes
    builder.add_node("orchestrator", orchestrator)
    builder.add_node("display_papers", display_papers)
    builder.add_node("ask_upload", ask_upload_node)
    builder.add_node("ask_single_file", ask_single_file_node)
    builder.add_node("extract_formatted", extract_formatted_node)
    builder.add_node("summarise_paper", summarise_paper_node)
    builder.add_node("summary_critic", summary_critic_node)
    builder.add_node("return_with_caveat", return_with_caveat_node)
    builder.add_node("finalize_summary", finalize_summary_node)
    builder.add_node("set_papers", set_papers)
    builder.add_node("check_pdf_input_node", lambda state: state)
    
    # Registered comparison pipeline nodes
    builder.add_node("ask_compare_upload", ask_compare_upload_node)
    builder.add_node("check_compatibility", compatibility_node)
    builder.add_node("compare_papers", compare_papers_node)
    builder.add_node("summarize_comparison", summarize_comparison_node)

    # Validator node
    builder.add_node("validate_and_query", validate_and_query)

    # Virtual Router Node for Compare Branch
    builder.add_node("compare_router_node", lambda state: state)

    # 2. Entry Point
    builder.add_edge(START, "orchestrator")  

    # 3. Orchestrator Conditional Routing
    builder.add_conditional_edges(
        "orchestrator",
        lambda state: state["intent"],
        {
            "summarize": "check_pdf_input_node",
            "discover": "set_papers",
            "compare": "compare_router_node",
            "validate": "set_papers"
        }
    )

    # 4. Compare Router Logic
    builder.add_conditional_edges(
        "compare_router_node",
        compare_router,
        {
            "ask_upload": "ask_compare_upload",
            "uploaded": "check_compatibility"
        }
    )
    
    # FIX 1: Point ask_compare_upload to check_compatibility instead of END
    builder.add_edge("ask_compare_upload", "check_compatibility")

    # 5. Set Papers Routing
    builder.add_conditional_edges(
        "set_papers",
        lambda state: state['intent'],
        {
            "discover": "display_papers",
            "compare": "check_compatibility",
            "validate": "validate_and_query"
        }
    )

    builder.add_edge("display_papers", END)

    # 6. PDF Check Loop (Summarization Workflow)
    builder.add_conditional_edges(
        "check_pdf_input_node",
        check_pdf_input,
        {
            "ask_upload": "ask_upload",
            "ask_single_file": "ask_single_file",
            "process_file": "extract_formatted"
        }
    )

    builder.add_edge("ask_upload", "check_pdf_input_node")
    builder.add_edge("ask_single_file", "check_pdf_input_node")

    # 7. Paper Summarization Pipeline
    builder.add_edge("extract_formatted", "summarise_paper")
    builder.add_edge("summarise_paper", "summary_critic")

    builder.add_conditional_edges(
        "summary_critic",
        evaluate_critic_result,
        {
            "pass": "finalize_summary",
            "retry": "summarise_paper",
            "max_reached": "return_with_caveat"
        }
    )

    builder.add_edge("finalize_summary", END)
    builder.add_edge("return_with_caveat", END)

    # 8. Comparison Pipeline Flow
    def route_after_compatibility(state: AgentState) -> str:
        if state.get("is_compatible"):
            return "compare"
        return "incompatible"

    builder.add_conditional_edges(
        "check_compatibility",
        route_after_compatibility,
        {
            "compare": "compare_papers",
            "incompatible": END
        }
    )

    builder.add_edge("compare_papers", "summarize_comparison")
    builder.add_edge("summarize_comparison", END)

    checkpointer = MemorySaver()
    
    # FIX 2: Interrupt BEFORE downstream processing nodes so state updates cleanly pause execution
    return builder.compile(
        checkpointer=checkpointer,
        interrupt_before=["check_compatibility", "extract_formatted"]
    )

if __name__ == "__main__":
    graph = build_graph()
    config = {"configurable": {"thread_id": "test_session_101"}}

    initial_input = {
        "query": "find some papers about attention in transformers",
        "pdf_paths": []
    }
    
    print("--- Step 1: Initial Run ---")
    res1 = graph.invoke(initial_input, config)
    
    current_state = graph.get_state(config)
    print("Next nodes waiting to run:", current_state.next)

    print("\n--- Step 2: Update State & Resume ---")
    # Mutate state to supply the required uploaded files
    graph.update_state(
        config,
        {"pdf_paths": ["downloaded_papers/paper1.pdf", "downloaded_papers/paper2.pdf"]}
    )

    # Resume graph execution
    final_result = graph.invoke(None, config)
    print("\n[Final State Values]:", final_result)