import streamlit as st
import os
from dotenv import load_dotenv
from graph import build_mdt_workflow
import time
import requests

load_dotenv()

st.set_page_config(page_title="MDT 多学科会诊 AI Agent (宫颈癌专科)", page_icon="🎀", layout="wide")

st.title("🎀 MDT 多学科会诊 AI Agent (宫颈癌专科)")
st.markdown("基于 **LangGraph** 构建的多智能体系统，支持**动态编排**参与会诊的专家网络。")

with st.sidebar:
    st.header("⚙️ 大模型配置")
    st.markdown("本次会诊专家由 **Google Gemini** 模型大脑驱动。请提供一个免费的 [Google AI Studio](https://aistudio.google.com/) API Key。")
    api_key = st.text_input("Google AI Studio API Key", value="", type="password")

    available_models = ["gemini-1.5-pro", "gemini-1.5-flash", "gemini-2.0-flash", "gemini-2.5-flash"]
    if api_key and "your_api_key" not in api_key:
        try:
            response = requests.get(f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}", timeout=5)
            if response.status_code == 200:
                models_data = response.json().get("models", [])
                fetched_models = [
                    m["name"].replace("models/", "")
                    for m in models_data
                    if "generateContent" in m.get("supportedGenerationMethods", [])
                ]
                if fetched_models:
                    preferred = [m for m in fetched_models if "gemini-2.0" in m or "gemini-1.5" in m]
                    others = [m for m in fetched_models if m not in preferred]
                    available_models = preferred + others
        except Exception:
            pass

    model_name = st.selectbox("模型名称 (Model)", available_models, index=0)

    st.markdown("---")
    st.markdown("### 👩‍⚕️ 选择参与会诊的专家")
    expert_options = {
        "📉 超声科专家 (Ultrasound)": "Ultrasound",
        "🩺 放射科专家 (Radiology)": "Radiology",
        "🔬 病理科专家 (Pathology)": "Pathology",
        "🔪 妇科专家 (Gynecology)": "Gynecology",
        "💊 肿瘤内科专家 (Oncology)": "Oncology"
    }

    selected_labels = st.multiselect(
        "请选择需要的科室（MDT协调员将默认参与总结）：",
        list(expert_options.keys()),
        default=list(expert_options.keys())
    )
    selected_experts = [expert_options[label] for label in selected_labels]

# 宫颈癌默认病历
default_case = """# 模拟病历：宫颈占位性病变

## 基本信息
- **性别**：女
- **年龄**：52岁
- **主诉**：接触性阴道出血3个月，伴不规则阴道流血1个月。

## 影像与病理检查
- **盆腔MRI**：宫颈后唇可见一团块状异常信号影，大小约 3.5 x 3.0 x 2.8 cm，呈长T1、稍长T2信号，弥散受限（DWI高信号）。病灶向下累及阴道穹窿及阴道上1/3，向两侧浸润深层宫颈间质，未见明显宫旁组织受累。双侧盆腔（髂内外血管旁及闭孔区）可见多枚肿大淋巴结，最大短径约 1.2 cm。
- **阴道镜检查**：宫颈显著肥大，表面可见菜花样肿物，伴有坏死及异味分泌物。醋白试验见厚白上皮，边界清楚，血管形态异常，碘试验阴性，触之极易出血。
- **活检病理（宫颈多点活检及颈管搔刮）**：
  - 宫颈组织：浸润性鳞状细胞癌，中分化。
  - 颈管搔刮物（ECC）：血凝块中查见少许鳞癌细胞。
  - 免疫组化及分子检测：p16(弥漫强+)，p53(野生型)，Ki-67(约 75%+)；高危型HPV分型检测：HPV 16 (+)
"""

st.header("📋 患者病历输入")
patient_case = st.text_area("请输入患者的详细病历、影像报告等：", value=default_case, height=250)

if st.button("🚀 启动 MDT 专家组会诊", type="primary"):
    if not api_key or "your_api_key" in api_key:
        st.error("⚠️ 请先在左侧配置有效的 API Key！")
    elif not selected_experts:
        st.error("⚠️ 请至少在左侧选择一位专科医生参与会诊！")
    else:
        st.markdown("---")
        st.header("🔄 会诊实时进展")

        status_placeholder = st.empty()

        try:
            app = build_mdt_workflow(selected_experts)
            initial_state = {"patient_case": patient_case, "api_key": api_key, "model_name": model_name}

            with st.spinner("专家组正在紧张阅片和讨论中..."):
                status_texts = []
                for output in app.stream(initial_state):
                    for node_name, state_update in output.items():
                        node_map = {
                            "Ultrasound": "📉 超声科专家",
                            "Radiology": "🩺 放射科专家",
                            "Pathology": "🔬 病理科专家",
                            "Gynecology": "🔪 妇科专家",
                            "Oncology": "💊 肿瘤内科专家",
                            "Coordinator": "📋 MDT 协调员 (生成最终报告)"
                        }
                        cn_name = node_map.get(node_name, node_name)
                        status_texts.append(f"✅ **{cn_name}** 已完成评估并提交意见。")
                        status_placeholder.markdown("\n\n".join(status_texts))
                        time.sleep(0.5)

                final_state = app.invoke(initial_state)

            st.success("🎉 MDT 会诊结束！")

            st.markdown("---")
            st.header("🏥 MDT 综合诊疗方案")
            cn_experts = [label.split(" (")[0] for label in selected_labels]
            st.info(f"（本次会诊由 **{model_name}** 模型驱动，参与科室：{', '.join(cn_experts)}）\n\n以下是协调员汇总各科室意见后出具的最终**一句话决策**：")
            st.markdown(f"### 💡 {final_state['final_mdt_report']}")

            with st.expander("🔍 查看各科室详细原始意见 (点击展开)"):
                if "Ultrasound" in selected_experts:
                    st.markdown("### 📉 超声科意见")
                    st.markdown(final_state.get("ultrasound_report", "暂无"))
                    st.markdown("---")
                if "Radiology" in selected_experts:
                    st.markdown("### 🩺 放射科意见")
                    st.markdown(final_state.get("radiology_report", "暂无"))
                    st.markdown("---")
                if "Pathology" in selected_experts:
                    st.markdown("### 🔬 病理科意见")
                    st.markdown(final_state.get("pathology_report", "暂无"))
                    st.markdown("---")
                if "Gynecology" in selected_experts:
                    st.markdown("### 🔪 妇科意见")
                    st.markdown(final_state.get("gynecology_opinion", "暂无"))
                    st.markdown("---")
                if "Oncology" in selected_experts:
                    st.markdown("### 💊 肿瘤内科意见")
                    st.markdown(final_state.get("oncology_opinion", "暂无"))

        except Exception as e:
            st.error(f"会诊过程中发生错误: {str(e)}\n请检查 API Key 额度或网络连通性。")
