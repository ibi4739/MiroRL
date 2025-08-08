## Define Your Custom MCP Tool

MiroRL supports two main approaches for adding custom MCP tools:

### 1. Using npx/uvx (Node.js-based MCP tools)

For Node.js-based MCP tools that can be installed via npm and executed with `npx` or `uvx`, add your tool configuration file:

```yaml
tools:
  - class_name: "mirorl.tools.mcp_tool.MCPTool"
    config:
      command: "npx"  # or "uvx"
      args: ["your-mcp-tool-name"]
      env: ["YOUR_API_KEY", "HTTPS_PROXY"]  # Required environment variables
      server_name: "your_tool_name"
      max_retries: 5
      delay_between_retries: 1
      connection_timeout: 5
      execution_timeout: 60
    tool_schema:
      type: "mcp"
      function: {}
```

### 2. Using Python-based MCP tools

For Python-based MCP tools using the FastMCP framework, add your tool configuration file:

```yaml
tools:
  - class_name: "mirorl.tools.mcp_tool.MCPTool"
    config:
      command: "python"
      args: ["path/to/your_tool.py"]
      env: ["YOUR_API_KEY", "HTTPS_PROXY"]  # Required environment variables
      server_name: "your_tool_name"
      max_retries: 5
      delay_between_retries: 1
      connection_timeout: 5
      execution_timeout: 60
    tool_schema:
      type: "mcp"
      function: {}
```

### 3. Configuration Parameters

All MCP tools support the following configuration parameters:

| Parameter | Description | Default | Example |
|-----------|-------------|---------|---------|
| `command` | MCP server command to execute | - | `"npx"`, `"python"` |
| `args` | Command line arguments | `[]` | `["your-tool"]`, `["tool.py"]` |
| `env` | Environment variables | `[]` | `["API_KEY", "PROXY"]` |
| `server_name` | MCP server identifier | - | `"your_tool_name"` |
| `blacklist` | Forbidden operations | `[]` | `["scrape"]` |
| `max_retries` | Maximum retry attempts | `5` | `3` |
| `delay_between_retries` | Retry delay in seconds | `1` | `2` |
| `connection_timeout` | Connection timeout in seconds | `5` | `10` |
| `execution_timeout` | Execution timeout in seconds | `60` | `120` |

### 4. Integration with Training

To use your custom MCP tool in training:

1. **Create your tool configuration file** in `mirorl/recipe/mcp/config/tool_config/`
2. **Set environment variables** required by your tool
3. **Update training command** to use your tool config:

```bash
python3 -m mirorl.recipe.mcp.main_ppo \
    --config-path="mirorl/recipe/mcp/config" \
    --config-name='mcp_trainer' \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="path/to/your_tool_config.yaml" \
    data.train_batch_size=256 \
    trainer.n_gpus_per_node=8
```
