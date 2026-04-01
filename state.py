from typing import TypedDict, List, Annotated
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage

class MDTState(TypedDict):
    patient_case: str
    ultrasound_report: str
    radiology_report: str
    pathology_report: str
    breast_surgery_opinion: str
    oncology_opinion: str
    final_mdt_report: str
    messages: Annotated[List[BaseMessage], add_messages]
