# 乳腺 MDT AI Agent (Breast Cancer MDT AI Agent)

这是一个基于 **LangGraph** 和 **Streamlit** 构建的多智能体（Multi-Agent）系统，旨在模拟真实的乳腺癌多学科会诊（MDT, Multi-Disciplinary Team）流程。该系统利用 Google Gemini 大模型（通过 LangChain 接入），为乳腺癌患者提供综合的诊疗建议。

## 项目特点

- **多学科协同**：模拟真实的 MDT 流程，包含超声科、放射科、病理科、乳腺外科和肿瘤内科的虚拟专家。
- **动态工作流**：基于 LangGraph 构建的状态图，支持动态选择参与会诊的科室。诊断科室并行处理，随后汇总至外科和内科，最终由首席协调员（Coordinator）出具最终极简版 MDT 综合诊疗方案。
- **可视化界面**：提供基于 Streamlit 的 Web 用户界面，可以实时追踪各个智能体的分析过程和最终结论。
- **CLI 支持**：除了 Web 界面，也提供命令行脚本（`main.py`）进行快速测试和运行。

## 项目结构

- `app.py`: Streamlit Web 应用的主入口，处理 UI 交互、API 密钥配置以及会诊流程的实时展示。
- `main.py`: 命令行界面的主入口点，用于在终端中运行 MDT 工作流。
- `graph.py`: 定义 LangGraph 的状态图（StateGraph）和工作流逻辑，控制各个 Agent 节点的执行顺序和数据流向。
- `agents.py`: 定义了各个领域专家 Agent 的节点函数（Node）和 Prompt 模板，并调用 Gemini API 生成专业意见。
- `state.py`: 定义了 `MDTState` 状态类，用于在图中各个节点之间传递患者病历和各科室的评估报告。
- `data/patient_case.md`: 示例患者病历数据。
- `requirements.txt`: 项目依赖包列表。

## 安装与配置

1. **环境要求**：建议使用 Python 3.10+。
2. **安装依赖**：
   ```bash
   pip install -r requirements.txt
   ```
3. **配置环境变量**：
   在项目根目录下创建一个 `.env` 文件，并添加你的 Google Gemini API Key（或者在 Streamlit 页面左侧边栏中直接输入）：
   ```env
   GOOGLE_API_KEY=your_google_api_key_here
   GOOGLE_MODEL_NAME=gemini-1.5-pro
   ```

## 运行方式

### Web 界面模式 (推荐)
使用 Streamlit 运行交互式 Web 界面：
```bash
streamlit run app.py
```
启动后，在浏览器中打开提供的本地链接，输入病历信息，选择参与的专家，然后点击“启动 MDT 诊疗流程”即可查看实时会诊结果。

### 命令行模式
如果仅需在终端中快速测试，可以运行：
```bash
python main.py
```
这会读取 `data/patient_case.md` 中的病历，并输出完整的 MDT 分析报告及最终的诊疗建议。

## 工作流说明 (Workflow)

整个工作流通过 LangGraph 进行编排：
1. **输入阶段**：系统接收患者的初始病历数据。
2. **诊断阶段 (并行)**：如果选择了超声科、放射科和病理科，这几个节点会并行提取和分析病历中的相关信息。
3. **外科评估**：乳腺外科节点汇总诊断阶段的结果，给出手术相关的建议。
4. **内科评估**：肿瘤内科节点结合病理和外科意见，给出全身系统性治疗方案。
5. **综合汇总**：首席协调员（Coordinator）节点根据所有专家的意见，总结出最终的“极简一句话”综合诊疗方案。
