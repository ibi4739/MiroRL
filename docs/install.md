## Installation

### Requirements

First of all, to manage environment, we recommend using conda:

```bash
conda create -n mirorl python==3.10.4
conda activate mirorl
```

For prerequisites installation (CUDA, cuDNN, Apex), we recommend following the [verl prerequisites guide](https://verl.readthedocs.io/en/latest/start/install.html#pre-requisites) which provides detailed instructions for:

- CUDA: Version >= 12.4
- cuDNN: Version >= 9.8.0
- Apex

### Install Dependencies

#### Install python dependencies

For python dependencies installation (sglang, flash-attn, flashinfer), execute the `install.sh` script that we provided in `mirorl/scripts`:

```bash
# Make sure you have activated verl conda env
# only for FSDP backend and sglang engine
bash scripts/install.sh
```

If you encounter errors in this step, please check the script and manually follow the steps in the script.

#### Install Node.js for MCP Support

MCP (Model Context Protocol) requires Node.js to run MCP servers. Node.js version 18+ is recommended for optimal compatibility with MCP tools.

```bash
# Download Node.js binary (example for Linux x64)
wget https://nodejs.org/dist/v24.2.0/node-v24.2.0-linux-x64.tar.xz

# Extract to your target path
tar -xf node-v24.2.0-linux-x64.tar.xz -C /your/target/path

# Add to PATH
export NODEJS_HOME=/your/target/path/node-v24.2.0-linux-x64
export PATH=$NODEJS_HOME/bin:$PATH
export NODE_SHARED=/your/target/path/node-shared/node_modules
export PATH=$NODE_SHARED/.bin:$PATH

# Verify installation
node --version
npm --version

# Install serper mcp server
mkdir -p /your/target/path/node-shared
cd /your/target/path/node-shared
npm init -y
npm install serper-search-scrape-mcp-server
```

### Install verl

For installing the target version of verl, the best way is to clone and install it from source.

```bash
git clone https://github.com/MiroMindAsia/mirorl.git --recursive
cd mirorl/verl
pip install --no-deps -v .
```
