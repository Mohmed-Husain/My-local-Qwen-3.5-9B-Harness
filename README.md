# My Local Qwen 3.5 9B Agent Harness

An autonomous terminal and coding assistant harness that connects local LLMs running in **LM Studio** (such as Qwen 2.5 / 3 / 3.5 Coder, Llama 3, Mistral) directly to your local terminal and filesystem.

---

## ✨ Features

- **⚡ 4 Core Agentic Tools**:
  - `edit_file`: Precise search-and-replace for existing files (generates targeted diffs without rewriting entire files, saving massive context).
  - `write_file`: Clean multi-line file writer without shell quoting issues.
  - `read_file`: Line-numbered, context-safe file reader.
  - `execute_terminal_command`: Runs shell commands (tests, scripts, git, package installs).
- **🌊 Real-Time Token Streaming**: Token-by-token streaming with instant **`Ctrl+C`** interrupt support to stop generation without exiting.
- **🛡️ Auto Context Compression**: Automatically monitors context usage and compresses old conversation history at 85% capacity, allowing sessions to run indefinitely.
- **📋 Multi-Line Paste Support**:
  - Auto-detects paste bursts from clipboard.
  - Dedicated `/paste` mode for large code blocks.
  - Triple-quote (`"""`) multiline blocks.
- **🔌 Auto-Port Detection**: Automatically detects LM Studio on port `11434` or `1234`.
- **💬 Interactive Chat REPL**: Built-in status commands (`/status`, `/context`, `/compress`, `/approve`, `/clear`, `/help`, `/exit`).

---

## 🚀 Quickstart

### 1. Requirements
```bash
pip install openai
```

### 2. Start LM Studio Local Server
1. Open **LM Studio**.
2. Go to the **Developer / Local Server** tab (`<->` icon).
3. Load your model (e.g., `qwen/qwen3.5-9b` or `qwen3.5-4b`).
4. Click **Start Server** (default ports: `1234` or `11434`).

### 3. Run the Agent
```bash
python lm_terminal_agent.py
```

### Options
```bash
# Auto-approve all tool actions without prompt
python lm_terminal_agent.py -y

# Specify a custom model or host
python lm_terminal_agent.py --model "qwen/qwen3.5-9b" --host "http://127.0.0.1:11434/v1"

# Adjust context compression threshold
python lm_terminal_agent.py --compress-at 0.80
```

---

## 🛠️ Slash Commands
- `/status` — Ping LM Studio and test latency.
- `/context` — View current token usage and visual context meter.
- `/compress` — Manually trigger conversation compression.
- `/approve` — Toggle auto-approval of terminal commands on/off.
- `/paste` — Multi-line paste mode (finish with `END`).
- `/clear` — Reset conversation history and clear context.
- `/exit` — Quit the session.
