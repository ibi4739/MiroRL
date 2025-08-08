<h1 align="center">
<em>MiroRL</em>: An MCP-first Reinforcement Learning Framework for Deep Research Agent
</h1>

<p align="center">
<a href="https://huggingface.co/miromind-ai"><img src="https://img.shields.io/badge/-gery?style=social&label=%F0%9F%A4%97%20Huggingface" alt="HuggingFace" style="height: 20px;"></a>
<a href="https://x.com/miromind_ai"><img src="https://img.shields.io/badge/-grey?style=social&logo=x&label=MiroMindAI" alt="X" style="height: 20px;"></a>
<a href="https://www.xiaohongshu.com/user/profile/663098830000000003033edc"><img src="https://img.shields.io/badge/-grey?style=social&logo=red&label=RedNote" alt="小红书" style="height: 20px;"></a>
<a href="https://discord.gg/GPqEnkzQZd"><img src="https://img.shields.io/badge/-grey?style=social&logo=discord&label=Discord" alt="Discord" style="height: 20px;"></a>
<a href="https://github.com/user-attachments/assets/214ab129-a880-4882-8ae3-2702c0ed850b"><img src="https://img.shields.io/badge/-grey?style=social&logo=wechat&label=WeChat" alt="WeChat" style="height: 20px;"></a>
<a href="https://miromind.ai"><img src="https://img.shields.io/badge/-grey?style=social&logo=google-chrome&label=miromind.ai" alt="miromind.ai" style="height: 20px;"></a>
</p>



<p align="center">
<a href="#overview"><b>📖 Overview</b></a> | <a href="#installation"><b>🛠️ Installation</b></a> | <a href="#quick-start"><b>🚀 Quick Start</b></a> | <a href="#your-custom-mcp-tool"><b>📚 Custom MCP Tool</b></a> | <a href="#license"><b>📄 License</b></a>
</p>


## News
<strong>[2025/08/08]</strong> We release the full training recipe for MiroRL-14B-SingleAgent-Preview-v0.1, a 14B Deep Research Agent trained with MiroRL that achieves 40.29%(Avg@8) on the GAIA-text-103 subset, featuring strong and reproducible results for open-source models. 
- 🤗 HF Dataset [`GenQA`](https://huggingface.co/datasets/miromind-ai/MiroRL-GenQA) for RL training.
- 🤗 HF SFT Coldstart Checkpoint [`MiroRL-14B-SFT-SingleAgent-v0.1`](https://huggingface.co/miromind-ai/MiroRL-14B-SFT-SingleAgent-v0.1)
- 📄 [Training Scripts](https://github.com/MiroMindAI/MiroRL/blob/main/mirorl/recipe/mcp/run_mirorl_14b_8xgpu.sh)

<strong>[2025/08/08]</strong> MiroRL-v0.1 is released.

## Overview

**MiroRL** is the first reinforcement learning framework to support multi-turn **MCP tool calls** for deep research agents. 
During training, MCP tools provide the agent with seamless access to search engines, webpage content, local file systems, Python interpreters, and Linux shells, all while maintaining training stability, efficiency, and large-scale extensibility.

The MiroRL package provides:

- 🔧 **Flexibility**: Seamless integration of multiple tools through MCP (Model Context Protocol), currently supporting various research tools including web search, data scraping, and code execution capabilities

- 🚀 **High-Performance GRPO Training**: Advanced asynchronous rollouts for multi-turn conversations with MCP tool calling, supporting both partial and streaming rollouts with more than 2x end-to-end training performance improvements compared to default settings

- 💾 **Memory Efficiency**: Intelligent selection of memory-efficient Triton kernels, sequence parallelism, and CPU-offloading techniques to reduce memory footprint for long-context training. **Supports running GRPO to train 14B-scale LLMs with 64K context length on a single GPU node**

- 🎯 **User-Friendly**: One-click script deployment for multi-GPU training of deep research models with comprehensive tooling and monitoring capabilities

## Installation
Before trying out our RL training, you need to set up the environment properly. We STRONLGY SUGGEST using our pre-built Docker image whenever possible. ONLY consider installing MiroRL manually if Docker is ABSOLUTELY not an option for your enviroment.

### 1. Install from Docker Image (Recommended)

We provide a pre-built Docker image with all dependencies installed, which is the recommended way to get started quickly:

```bash
# Pull the MiroRL Docker image
docker pull miromind/mirorl:v0.1.0

# Run the container with GPU support
docker run --gpus all --shm-size=8g -it --rm \
    -v $(pwd):/workspace \
    -w /workspace \
    miromind/mirorl:v0.1.0
```

<details>
<summary>Docker image contains:</summary>

- verl 0.4 framework
- CUDA 12.4 support
- cuDNN 9.8
- PyTorch 2.6.0
- FlashAttention 2.7.4.post1
- Node.js 24.2.0 for MCP support
- All required Python dependencies
</details>

### 2. Manual Installation

If you prefer to install manually, we recommend following the detailed installation guide in [install.md](docs/install.md) which provides step-by-step instructions. ONLY consider installing MiroRL manually if Docker is ABSOLUTELY not an option for your enviroment.

## Quick Start

### Single Node Training with MCP Tool Calling

#### Step 1: Prepare the dataset

```bash
# Download dataset from hugging face
huggingface-cli download --repo-type dataset miromind-ai/MiroRL-GenQA --local-dir data/
```

#### Step 2: Download the SFT model for RL-training

```bash
# Download model from hugging face
huggingface-cli download miromind-ai/MiroRL-14B-SFT-SingleAgent-v0.1 --local-dir models/
```

#### Step 3: Perform GPRO training with MCP Tool Calling

**Required Online Services**
Training a deep research model inevitably requires several online sevices including
1. Search Engine: we use `serper.dev` for Google search engine service.
2. Web Page Fetcher: we use `jina.ai/reader` for scraping webpage and PDFs as markdown.
3. Web Page Summary LLM: we self-hosted an SGLang endpoint of Qwen/Qwen3-14B-128k for summarizing web page scraped by jina.
4. LLM as judge: we use a `siliconflow` endpoint of Qwen/Qwen2.5-72B or an `OpenAI` endpoint of GPT-4.1 for calculating reward during training.
5. Logging: we use `wandb.ai` for logging.
6. (Optional) Python and Linux Shell: we use `e2b.dev` for code sandboxing.

> [!NOTE]
> In a single training step, the approximate costs for Serper API, Jina API, and OpenAI API are $0.85, $1.23, and $0.20, respectively. If you are a student doing research in Deep Research Agent RL, we are happy to give away free credits of these online services. Please reach out to us via the Discord or the WeChat group.

For training MiroRL-14B-SingleAgent-Preview-v0.1, you need to export :

```bash
export SERPER_API_KEY="your_serper_api_key"
export JINA_API_KEY="your_jina_api_key"
export SUMMARY_LLM_URL="your_llm_endpoint_url_for_summarization"
export SUMMARY_LLM_NAME="your_llm_name_for_summarization"
export OPENAI_API_KEY="your_openai_api_key"
export WANDB_API_KEY="your_wandb_api_key"
export HTTPS_PROXY="http://your_proxy:port"  # (Optional) Use if your network could not directly connect to serper.dev and jina.ai, e.g. an air-gapped cluster
```

<details>
<summary>Environment Setup (Manual Installation Only)</summary>

```bash
# If using manual installation, activate your environment
conda activate /your/mirorl/env
```
</details>

**Download the mirorl source code, including submodule verl**
```bash
git clone https://github.com/MiroMindAsia/mirorl.git --recursive
cd mirorl
```

**Run MCP Example on Single Node with 8GPUs**

Update the data and model path in demo config `mirorl/recipe/mcp/run_mirorl_14b_8xgpu.sh`:

```bash
DATA_HOME=/your/dataset/path
MODEL_PATH=/your/model/path

python3 -m mirorl.recipe.mcp.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name='mcp_trainer' \
    algorithm.adv_estimator=grpo \
    data.train_batch_size=64 \
    data.max_prompt_length=3072 \
    data.max_response_length=62464 \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.rollout.multi_turn.tool_config_path=$TOOL_CONFIG_PATH \
    data.train_files=$DATA_HOME/train.parquet \
    data.val_files=$DATA_HOME/test.parquet \
    trainer.total_epochs=1 $@
```

Runing the grpo training demo with mcp tool calling:

```bash
# launch the script
bash mirorl/recipe/mcp/run_mirorl_14b_8xgpu.sh
```

### Example Configuration

MCP tool configurations are located in `mirorl/recipe/mcp/config/tool_config/` directory:

```
mirorl/recipe/mcp/config/tool_config/
└── mcp_tool_config.yaml # Jina scraping + Serper search configuration
```

An example of web search tool config in `mcp_tool_config.yaml`:

```yaml
tools:
  - class_name: "mirorl.tools.mcp_tool.MCPTool"
    config:
      command: "python"
      args: ["mirorl/tools/serper_search.py"]
      env: ["SERPER_API_KEY", "HTTPS_PROXY"]
      server_name: "search_and_scrape_webpage"
    tool_schema:
      type: "mcp"
      function: {}
  - class_name: "mirorl.tools.mcp_tool.MCPTool"
    config:
      command: "python"
      args: ["mirorl/tools/jina_scrape_llm_summary.py"]
      env: ["JINA_API_KEY", "HTTPS_PROXY", "NO_PROXY", "SUMMARY_LLM_URL", "SUMMARY_LLM_NAME"]
      server_name: "jina_scrape_llm_summary"
      execution_timeout: 600
    tool_schema:
      type: "mcp"
      function: {}
```

## Your Custom MCP Tool

If you want to define your custom mcp tool, please refer to the [define_your_mcp_tool](docs/custom_mcp_tool.md) document.

## Acknowledgements

- MiroRL framework is built on top of [verl](https://github.com/volcengine/verl), an open-source RLHF library.


## Citation

```bibtex
@misc{2025mirorl,
    title={MiroRL: An MCP-first Reinforcement Learning Framework for Deep Research Agent},
    author={ MiroMind Foundation Model Team and MiroMind AI Infra Team},
    howpublished = {\url{https://github.com/MiroMindAI/MiroRL}},
    year={2025}
}
```


## License

This project is released under the [Apache License 2.0](LICENSE). Please also adhere to the Licenses of models and datasets being used.
