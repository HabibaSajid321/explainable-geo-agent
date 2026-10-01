"""
LangGraph wiring for the TrustGeoAgent pipeline.

    Discovery Agent -> Evaluation Agent -> Explainability Agent -> Human Review Agent

Each node reads/writes a shared TypedDict state, same pattern as the
SOC-assistant graph: specialized agents, explicit state, a human gate
before anything is treated as "final."
"""
from typing import TypedDict, Any
from langgraph.graph import StateGraph, END

from agents.discovery_agent import run_discovery
from agents.evaluation_agent import run_evaluation
from agents.explainability_agent import explain_results
from agents.review_agent import review


class PipelineState(TypedDict, total=False):
    split: dict
    discovery_out: dict
    eval_out: dict
    explanation: str
    review_out: dict
    interactive_review: bool
    scaling_factor: float
    recalibrate_attempts: int


def _discovery_node(state: PipelineState) -> PipelineState:
    X_train, y_train = state["split"]["train"]
    state["discovery_out"] = run_discovery(X_train, y_train)
    if "recalibrate_attempts" not in state:
        state["recalibrate_attempts"] = 0
    if "scaling_factor" not in state:
        state["scaling_factor"] = 1.0
    return state


def _evaluation_node(state: PipelineState) -> PipelineState:
    sf = state.get("scaling_factor", 1.0)
    state["eval_out"] = run_evaluation(
        state["discovery_out"]["model_cls"],
        state["split"],
        scaling_factor=sf,
    )
    return state


def _explainability_node(state: PipelineState) -> PipelineState:
    state["explanation"] = explain_results(state["discovery_out"], state["eval_out"])
    return state


def _review_node(state: PipelineState) -> PipelineState:
    attempts = state.get("recalibrate_attempts", 0)
    review_res = review(
        state["discovery_out"], state["eval_out"],
        interactive=state.get("interactive_review", False),
    )
    
    if review_res.get("recalibrate_needed") and attempts < 1:
        state["recalibrate_attempts"] = attempts + 1
        state["scaling_factor"] = review_res.get("suggested_scaling_factor", 1.0)
    else:
        # Stop loop after 1 recalibration attempt or if not needed
        review_res["recalibrate_needed"] = False

    state["review_out"] = review_res
    return state


def _should_recalibrate(state: PipelineState) -> str:
    if state.get("review_out", {}).get("recalibrate_needed", False):
        return "evaluation"
    return "explainability"



def build_graph():
    g = StateGraph(PipelineState)
    g.add_node("discovery", _discovery_node)
    g.add_node("evaluation", _evaluation_node)
    g.add_node("explainability", _explainability_node)
    g.add_node("review", _review_node)

    g.set_entry_point("discovery")
    g.add_edge("discovery", "evaluation")
    g.add_edge("evaluation", "review")
    g.add_conditional_edges("review", _should_recalibrate, {"evaluation": "evaluation", "explainability": "explainability"})
    g.add_edge("explainability", END)
    return g.compile()


