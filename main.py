import os
from dotenv import load_dotenv
from graph import build_mdt_workflow

# 加载环境变量
load_dotenv()


def main():
    print("🚀 正在初始化 MDT（多学科会诊）Agent 工作流...\n")

    if not os.getenv("GOOGLE_API_KEY") or "your_api_key" in os.getenv("GOOGLE_API_KEY"):
        print("⚠️ 警告: 未配置有效的 GOOGLE_API_KEY。请在 .env 文件中填入 API Key 后重试。")
        return

    try:
        with open("data/patient_case.md", "r", encoding="utf-8") as f:
            case_data = f.read()
    except FileNotFoundError:
        case_data = "患者病历加载失败"

    app = build_mdt_workflow()

    initial_state = {"patient_case": case_data}

    print("=" * 40 + " 会诊开始 " + "=" * 40)
    for output in app.stream(initial_state):
        for node_name, state_update in output.items():
            print(f"✅ [{node_name}] 节点已完成评估。")

    final_state = app.invoke(initial_state)

    print("\n" + "=" * 40 + " 🏥 MDT 综合报告 " + "=" * 40)
    print(final_state["final_mdt_report"])
    print("=" * 94)


if __name__ == "__main__":
    main()
