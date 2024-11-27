![](libs/chatchat-server/chatchat/img/logo-long-langraph-chatchat.jpg)
<a href="https://trendshift.io/repositories/329" target="_blank"><img src="https://trendshift.io/api/badge/repositories/329" alt="chatchat-space%2FLangchain-Chatchat | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>

[![Generic badge](https://img.shields.io/badge/python-3.9%7C3.10%7C3.11%7C3.12-blue.svg)](https://pypi.org/project/pypiserver/)

📃 **LangGraph-Chatchat**

基于 ChatGLM 等大语言模型与 LangGraph 等应用框架实现，开源、可离线部署的 RAG 与 Agent 应用项目。

---

## 概述

🤖️ 一种利用 [LangGraph](https://langchain-ai.github.io/langgraph/)
思想实现的基于本地知识库的问答应用，目标期望建立一套对中文场景与开源模型支持友好、可离线运行的知识库问答解决方案。

💡 受 [GanymedeNil](https://github.com/GanymedeNil) 的项目 [document.ai](https://github.com/GanymedeNil/document.ai)
和 [AlexZhangji](https://github.com/AlexZhangji)
创建的 [ChatGLM-6B Pull Request](https://github.com/THUDM/ChatGLM-6B/pull/216)
启发，建立了全流程可使用开源模型实现的本地知识库问答应用。本项目的最新版本中可使用 [Xinference](https://github.com/xorbitsai/inference)、[Ollama](https://github.com/ollama/ollama)
等框架接入 [GLM-4-Chat](https://github.com/THUDM/GLM-4)、 [Qwen2-Instruct](https://github.com/QwenLM/Qwen2)、 [Llama3](https://github.com/meta-llama/llama3)
等模型，使用基于 [Streamlit](https://github.com/streamlit/streamlit) 的 WebUI 进行操作。

![](docs/img/langchain_chatchat_0.3.0.png)

✅ 本项目支持市面上主流的开源 LLM、 Embedding 模型与向量数据库，可实现全部使用**开源**模型**离线私有部署**。与此同时，本项目也支持
OpenAI GPT API 的调用，并将在后续持续扩充对各类模型及模型 API 的接入。

⛓️ 本项目实现原理如下图所示，过程包括加载文件 -> 读取文本 -> 文本分割 -> 文本向量化 -> 问句向量化 ->
在文本向量中匹配出与问句向量最相似的 `top k`个 -> 匹配出的文本作为上下文和问题一起添加到 `prompt`中 -> 提交给 `LLM`生成回答。

📺 [原理介绍视频](https://www.bilibili.com/video/BV13M4y1e7cN/?share_source=copy_web&vd_source=e6c5aafe684f30fbe41925d61ca6d514)

![实现原理图](docs/img/langchain+chatglm.png)

从文档处理角度来看，实现流程如下：

![实现原理图2](docs/img/langchain+chatglm2.png)

🚩 本项目未涉及微调、训练过程，但可利用微调或训练对本项目效果进行优化。

🧑‍💻 如果你想对本项目做出贡献，欢迎提交 pr。

## 快速上手

### Docker 安装部署 (一定要看)

查看 [Docker 安装指南](docs/install/README_docker_install.md)

### 源码安装部署/开发部署

查看 [开发部署指南](docs/install/README_dev_install.md)

### 旧版本迁移

- 首先按照 `安装部署` 中的步骤配置运行环境，修改配置文件
- 将 `Langchain-Chatchat` 项目的 `knowledge_base` 目录拷贝到配置的 `DATA` 目录下

---

## 项目里程碑

+ `2023年4月`: `Langchain-ChatGLM 0.1.0` 发布，支持基于 ChatGLM-6B 模型的本地知识库问答。
+ `2023年8月`: `Langchain-ChatGLM` 改名为 `Langchain-Chatchat`，发布 `0.2.0` 版本，使用 `fastchat` 作为模型加载方案，支持更多的模型和数据库。
+ `2023年10月`: `Langchain-Chatchat 0.2.5` 发布，推出 Agent 内容，开源项目在`Founder Park & Zhipu AI & Zilliz`
  举办的黑客马拉松获得三等奖。
+ `2023年12月`: `Langchain-Chatchat` 开源项目获得超过 **20K** stars.
+ `2024年6月`: `Langchain-Chatchat 0.3.0` 发布，带来全新项目架构。
+ `2024年11月`: `LangGraph-Chatchat 0.3.0` 发布，带来全新项目架构。

+ 🔥 让我们一起期待未来 Chatchat 的故事 ···

---

## 协议

本项目代码遵循 [Apache-2.0](LICENSE) 协议。

## 联系我们

### Telegram

[![Telegram](https://img.shields.io/badge/Telegram-2CA5E0?style=for-the-badge&logo=telegram&logoColor=white "langchain-chatchat")](https://t.me/+RjliQ3jnJ1YyN2E9)

### 项目交流群

<img src="docs/img/wx_01.jpg" alt="二维码" width="300" />

🎉 LangGraph-Chatchat 项目微信交流群，如果你也对本项目感兴趣，欢迎加入群聊参与讨论交流。

## 引用

如果本项目有帮助到您的研究，请引用我们：

```
@software{LangGraph-Chatchat,
    title        = {{LangGraph-Chatchat}},
    author       = {Liu, Qian and Zhang, Yuehua, and Song, Jinke, and liunux4odoo, and glide-the, and Huang, Zhiguo, and Zhang, Yuxuan},
    year         = 2024,
    journal      = {GitHub repository},
    publisher    = {GitHub},
    howpublished = {\url{https://github.com/chatchat-space/LangGraph-Chatchat}}
}
```
