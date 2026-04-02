import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from state import MDTState

def get_llm(api_key=None, model_name=None):
    if not api_key:
        api_key = os.getenv("GOOGLE_API_KEY", "dummy_key")
    if not model_name:
        model_name = os.getenv("GOOGLE_MODEL_NAME", "gemini-1.5-pro")
    return ChatGoogleGenerativeAI(
        model=model_name, 
        temperature=0.1,
        google_api_key=api_key
    )

def ultrasound_node(state: MDTState):
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是超声科专家。请基于病历提取乳腺及区域淋巴结的超声表现（如结节大小、边界、血流、BI-RADS分级等）。"),
        ("user", "病历：\n{patient_case}\n\n请给出超声科评估结论：")
    ])
    response = (prompt | get_llm(state.get("api_key"), state.get("model_name"))).invoke({"patient_case": state["patient_case"]})
    return {"ultrasound_report": response.content}

def radiology_node(state: MDTState):
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是放射科专家。请基于病历提取乳腺钼靶或MRI的影像学表现及全身转移排查情况。"),
        ("user", "病历：\n{patient_case}\n\n请给出放射科评估结论：")
    ])
    response = (prompt | get_llm(state.get("api_key"), state.get("model_name"))).invoke({"patient_case": state["patient_case"]})
    return {"radiology_report": response.content}

def pathology_node(state: MDTState):
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是病理科专家。请提取穿刺或手术病理结果，包括组织学分级、免疫组化（ER, PR, HER2, Ki-67等）及分子分型。"),
        ("user", "病历：\n{patient_case}\n\n请给出病理科评估结论：")
    ])
    response = (prompt | get_llm(state.get("api_key"), state.get("model_name"))).invoke({"patient_case": state["patient_case"]})
    return {"pathology_report": response.content}

def breast_surgery_node(state: MDTState):
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是乳腺外科专家。结合前置检查报告（如超声、放射、病理），评估手术指征，决定保乳或全切、以及腋窝淋巴结处理策略（前哨或清扫）。"),
        ("user", "病历：\n{patient_case}\n超声：\n{ultrasound_report}\n放射：\n{radiology_report}\n病理：\n{pathology_report}\n\n请给出外科手术建议：")
    ])
    response = (prompt | get_llm(state.get("api_key"), state.get("model_name"))).invoke({
        "patient_case": state["patient_case"],
        "ultrasound_report": state.get("ultrasound_report", "未参与会诊"),
        "radiology_report": state.get("radiology_report", "未参与会诊"),
        "pathology_report": state.get("pathology_report", "未参与会诊")
    })
    return {"breast_surgery_opinion": response.content}

def oncology_node(state: MDTState):
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是肿瘤内科专家。结合病历、病理分子分型及外科意见，制定新辅助或辅助的系统性治疗方案（化疗、内分泌治疗、靶向治疗）。"),
        ("user", "病历：\n{patient_case}\n病理：\n{pathology_report}\n外科意见：\n{breast_surgery_opinion}\n\n请给出肿瘤内科系统治疗方案：")
    ])
    response = (prompt | get_llm(state.get("api_key"), state.get("model_name"))).invoke({
        "patient_case": state["patient_case"],
        "pathology_report": state.get("pathology_report", "未参与会诊"),
        "breast_surgery_opinion": state.get("breast_surgery_opinion", "未参与会诊")
    })
    return {"oncology_opinion": response.content}

class MDTSummary(BaseModel):
    final_decision: str = Field(description="MDT的最终综合决定，必须是一句极简的话，严格按照时间顺序说明治疗步骤。例如：'建议先行新辅助化疗+双靶治疗，后续行左乳保乳术+前哨淋巴结活检，术后继续靶向治疗及内分泌治疗。'")

def coordinator_node(state: MDTState):
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是乳腺癌MDT首席协调员。综合各专科参与者的意见，输出最终版《MDT综合诊疗方案》。你的任务是将方案浓缩为【极简的一句话】。如果有科室未参与，请忽略该科室。"),
        ("user", "超声：\n{ultrasound_report}\n放射：\n{radiology_report}\n病理：\n{pathology_report}\n外科：\n{breast_surgery_opinion}\n内科：\n{oncology_opinion}\n\n请输出最终的一句话MDT结论。")
    ])
    structured_llm = get_llm(state.get("api_key"), state.get("model_name")).with_structured_output(MDTSummary)
    response = (prompt | structured_llm).invoke({
        "ultrasound_report": state.get("ultrasound_report", "未参与会诊"),
        "radiology_report": state.get("radiology_report", "未参与会诊"),
        "pathology_report": state.get("pathology_report", "未参与会诊"),
        "breast_surgery_opinion": state.get("breast_surgery_opinion", "未参与会诊"),
        "oncology_opinion": state.get("oncology_opinion", "未参与会诊")
    })
    return {"final_mdt_report": response.final_decision}
