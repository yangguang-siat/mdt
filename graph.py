from langgraph.graph import StateGraph, START, END
from state import MDTState
from agents import ultrasound_node, radiology_node, pathology_node, breast_surgery_node, oncology_node, coordinator_node

def build_mdt_workflow(selected_experts: list[str] = None):
    if selected_experts is None:
        selected_experts = ["Ultrasound", "Radiology", "Pathology", "BreastSurgery", "Oncology"]
        
    workflow = StateGraph(MDTState)

    expert_funcs = {
        "Ultrasound": ultrasound_node,
        "Radiology": radiology_node,
        "Pathology": pathology_node,
        "BreastSurgery": breast_surgery_node,
        "Oncology": oncology_node
    }

    for expert in selected_experts:
        workflow.add_node(expert, expert_funcs[expert])
        
    workflow.add_node("Coordinator", coordinator_node)

    if not selected_experts:
        workflow.add_edge(START, "Coordinator")
        workflow.add_edge("Coordinator", END)
        return workflow.compile()

    diagnostics = [e for e in selected_experts if e in ["Ultrasound", "Radiology", "Pathology"]]
    
    if "BreastSurgery" in selected_experts:
        converge_node = "BreastSurgery"
    elif "Oncology" in selected_experts:
        converge_node = "Oncology"
    else:
        converge_node = "Coordinator"

    if diagnostics:
        for d in diagnostics:
            workflow.add_edge(START, d)
            workflow.add_edge(d, converge_node)
    else:
        workflow.add_edge(START, converge_node)
        
    if "BreastSurgery" in selected_experts:
        if "Oncology" in selected_experts:
            workflow.add_edge("BreastSurgery", "Oncology")
            workflow.add_edge("Oncology", "Coordinator")
        else:
            workflow.add_edge("BreastSurgery", "Coordinator")
    elif "Oncology" in selected_experts:
        workflow.add_edge("Oncology", "Coordinator")
        
    workflow.add_edge("Coordinator", END)

    return workflow.compile()
