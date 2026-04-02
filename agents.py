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
        ("system", "你是超声科专家。请基于病历提取与宫颈癌相关的妇科超声或盆腔超声信息，重点关注宫颈病灶范围、宫旁受侵、子宫及附件情况、盆腔淋巴结线索，以及是否存在与病情相关的超声提示。若病历未提供超声信息，请明确说明。"),
        ("user", "病历：\n{patient_case}\n\n请给出超声科评估结论：")
    ])
    response = (prompt | get_llm(state.get("api_key"), state.get("model_name"))).invoke({"patient_case": state["patient_case"]})
    return {"ultrasound_report": response.content}


def radiology_node(state: MDTState):
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是放射科专家。请基于病历提取宫颈癌相关影像学信息，重点总结盆腔MRI、CT或PET-CT中的病灶大小、局部浸润范围、阴道/宫旁/盆壁受累情况、淋巴结转移线索及远处转移评估。"),
        ("user", "病历：\n{patient_case}\n\n请给出放射科评估结论：")
    ])
    response = (prompt | get_llm(state.get("api_key"), state.get("model_name"))).invoke({"patient_case": state["patient_case"]})
    return {"radiology_report": response.content}


def pathology_node(state: MDTState):
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是病理科专家。请提取宫颈癌相关病理结果，包括组织学类型、分化程度、浸润情况，以及免疫组化和分子检测信息，如 p16、p53、Ki-67、HPV 分型等；若信息不足请明确指出。"),
        ("user", "病历：\n{patient_case}\n\n请给出病理科评估结论：")
    ])
    response = (prompt | get_llm(state.get("api_key"), state.get("model_name"))).invoke({"patient_case": state["patient_case"]})
    return {"pathology_report": response.content}


def gynecology_node(state: MDTState):
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是妇科肿瘤专家。请结合病历、超声、放射及病理信息，对宫颈癌的临床分期倾向、局部可切除性、是否适合手术、是否更倾向同步放化疗等核心治疗路径提出妇科专科意见，并说明关键依据。"),
        ("user", "病历：\n{patient_case}\n超声：\n{ultrasound_report}\n放射：\n{radiology_report}\n病理：\n{pathology_report}\n\n请给出妇科专科治疗建议：")
    ])
    response = (prompt | get_llm(state.get("api_key"), state.get("model_name"))).invoke({
        "patient_case": state["patient_case"],
        "ultrasound_report": state.get("ultrasound_report", "未参与会诊"),
        "radiology_report": state.get("radiology_report", "未参与会诊"),
        "pathology_report": state.get("pathology_report", "未参与会诊")
    })
    return {"gynecology_opinion": response.content}


def oncology_node(state: MDTState):
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是肿瘤内科专家。请结合病历、病理结果及妇科意见，为宫颈癌患者提出系统治疗相关建议，重点说明是否建议同步放化疗、诱导/辅助治疗或后续全身治疗，并保持结论清晰可执行。"),
        ("user", "病历：\n{patient_case}\n病理：\n{pathology_report}\n妇科意见：\n{gynecology_opinion}\n\n请给出肿瘤内科治疗方案：")
    ])
    response = (prompt | get_llm(state.get("api_key"), state.get("model_name"))).invoke({
        "patient_case": state["patient_case"],
        "pathology_report": state.get("pathology_report", "未参与会诊"),
        "gynecology_opinion": state.get("gynecology_opinion", "未参与会诊")
    })
    return {"oncology_opinion": response.content}


class MDTSummary(BaseModel):
    final_decision: str = Field(description="MDT 的最终综合决定，必须是一句极简的话，严格按照时间顺序概括宫颈癌患者的推荐诊疗路径。")


def coordinator_node(state: MDTState):
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是宫颈癌 MDT 首席协调员。请综合各专科参与者的意见，输出最终版《MDT 综合诊疗方案》。你的任务是将方案浓缩为【极简的一句话】；如果有科室未参与，请自动忽略该科室的信息。"),
        ("user", "超声：\n{ultrasound_report}\n放射：\n{radiology_report}\n病理：\n{pathology_report}\n妇科：\n{gynecology_opinion}\n肿瘤内科：\n{oncology_opinion}\n\n请输出最终的一句话 MDT 结论。")
    ])
    structured_llm = get_llm(state.get("api_key"), state.get("model_name")).with_structured_output(MDTSummary)
    response = (prompt | structured_llm).invoke({
        "ultrasound_report": state.get("ultrasound_report", "未参与会诊"),
        "radiology_report": state.get("radiology_report", "未参与会诊"),
        "pathology_report": state.get("pathology_report", "未参与会诊"),
        "gynecology_opinion": state.get("gynecology_opinion", "未参与会诊"),
        "oncology_opinion": state.get("oncology_opinion", "未参与会诊")
    })
    return {"final_mdt_report": response.final_decision}
