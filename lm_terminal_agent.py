import os, sys, json, argparse, subprocess, urllib.request, time
from openai import OpenAI

# Force UTF-8 stdout/stderr so fancy glyphs (bars, emoji) don't crash when the
# agent's output is piped/captured (e.g. by eval_agent.py) on Windows.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# Enable ANSI escape sequences on Windows
if os.name == "nt":
    os.system("")

# ---------------------------------------------------------------------------
# Headless / eval support
# Allows the agent to be driven non-interactively from eval_agent.py so that
# models can be benchmarked objectively.
# ---------------------------------------------------------------------------
HEADLESS = {
    "enabled": False,
    "queue": [],          # prompts to feed instead of input()
    "json_out": None,
    "quiet": False,
    "stats": {
        "tool_calls": [],  # [{"name": str, "ok": bool, "detail": str}]
        "loops": 0,
        "invalid_json": 0,
        "narration": 0,
        "truncations": 0,
        "python_syntax_errors": 0,
        "assistant_messages": 0,
        "start_time": None,
        "end_time": None,
        "error": None,
    },
}

# Safety cap: how many model turns one task may take before we stop it.
MAX_AGENT_TURNS = 25


def _rec_tool(name, ok, detail=""):
    if HEADLESS["enabled"]:
        HEADLESS["stats"]["tool_calls"].append(
            {"name": name, "ok": bool(ok), "detail": str(detail)[:400]}
        )


def _rec_stat(key):
    if HEADLESS["enabled"]:
        HEADLESS["stats"][key] = HEADLESS["stats"].get(key, 0) + 1


def _write_headless_result():
    """Emit the structured result JSON for the eval runner."""
    if not HEADLESS["enabled"] or not HEADLESS["json_out"]:
        return
    st = HEADLESS["stats"]
    st["end_time"] = time.time()
    duration = 0.0
    if st["start_time"]:
        duration = round(st["end_time"] - st["start_time"], 2)
    calls = st["tool_calls"]
    result = {
        "tool_calls": calls,
        "tool_call_count": len(calls),
        "failed_tool_calls": sum(1 for c in calls if not c["ok"]),
        "tools_used": sorted({c["name"] for c in calls}),
        "loops": st["loops"],
        "invalid_json": st["invalid_json"],
        "narration": st["narration"],
        "truncations": st["truncations"],
        "python_syntax_errors": st["python_syntax_errors"],
        "assistant_messages": st["assistant_messages"],
        "duration_sec": duration,
        "error": st["error"],
    }
    try:
        with open(HEADLESS["json_out"], "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
    except Exception:
        pass

# Styling helpers
class C:
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    DIM     = "\033[2m"
    GREEN   = "\033[92m"
    YELLOW  = "\033[93m"
    RED     = "\033[91m"
    CYAN    = "\033[96m"
    MAGENTA = "\033[95m"
    BLUE    = "\033[94m"
    GRAY    = "\033[90m"

# Tools definition
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Edit a specific section or single line of a file by finding `target` and replacing it with `replacement`. PREFER THIS over write_file when modifying existing files because it saves tokens and avoids rewriting the whole file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path of the file to edit."},
                    "target": {"type": "string", "description": "The exact existing line(s) or text block to be replaced."},
                    "replacement": {"type": "string", "description": "The new replacement text."}
                },
                "required": ["path", "target", "replacement"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "execute_terminal_command",
            "description": "Execute a shell command in the terminal and return its stdout, stderr, and exit code. Use this for running code, testing, git, installing packages, or inspecting system status.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Shell command to execute."}
                },
                "required": ["command"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write or overwrite content directly to a file on disk. Use this tool whenever you create or edit code, scripts, configs, or text files instead of echoing via shell.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Relative or absolute path of the file to write."},
                    "content": {"type": "string", "description": "The full text content to write into the file."},
                    "append": {"type": "boolean", "description": "If true, append to the file instead of overwriting. Default is false."}
                },
                "required": ["path", "content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read content from a file on disk with line limits to protect the context window.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Relative or absolute path of the file to read."},
                    "start_line": {"type": "integer", "description": "Line number to start reading from (1-indexed). Default is 1."},
                    "max_lines": {"type": "integer", "description": "Maximum number of lines to read. Default is 120."}
                },
                "required": ["path"]
            }
        }
    }
]

SYSTEM_PROMPT = """You are an autonomous local coding and terminal agent running directly on the user's computer.
You have direct disk and shell access via 4 tools:
1. `edit_file`: (PRIMARY FOR EXISTING FILES) Edit specific lines or code blocks in existing files.
2. `write_file`: (FOR NEW FILES) Create brand new files or write complete new scripts from scratch.
3. `read_file`: Inspect file contents and line numbers before modifying them.
4. `execute_terminal_command`: Run tests, execute scripts, check git, install libraries.

Operating System: {os_name} | Shell: {shell} | Directory: {cwd}

STRICT OPERATIONAL RULES:
1. NEVER WRITE CODE BLOCKS IN CHAT: When asked to write code, create a script, add a feature, or fix a bug, DO NOT dump the code into your chat response. You MUST invoke `write_file` or `edit_file` to write the code directly to the file on disk. The user cannot run code from chat bubbles.
2. PREFER `edit_file` OVER `write_file`: When editing existing files, use `edit_file` to replace only the specific line or block. Never rewrite 100 lines just to change 1 line.
3. CHAT IS FOR CONCISE STATUS ONLY: Use chat only to explain in 1-2 sentences what was modified (e.g. "Created server.py" or "Updated line 14 in app.py").
4. If unsure of file contents, use `read_file` first, then make edits.
"""

COMPRESS_PROMPT = """Summarize the following conversation between a user and a terminal AI assistant.
Focus on:
- What tasks the user asked for
- What files were created or modified
- What commands were run and their outcomes
- Current state of the project

Be concise. Use bullet points. Keep it under 300 words.

CONVERSATION:
{conversation}"""

def detect_port():
    for port in [11434, 1234]:
        try:
            url = f"http://127.0.0.1:{port}/v1/models"
            req = urllib.request.urlopen(url, timeout=0.8)
            if req.status == 200:
                return f"http://127.0.0.1:{port}/v1"
        except Exception:
            pass
    return "http://127.0.0.1:11434/v1"

def ping_lm_studio(host: str) -> tuple:
    try:
        t0 = time.time()
        url = f"{host.rstrip('/')}/models"
        req = urllib.request.urlopen(url, timeout=1.5)
        latency = int((time.time() - t0) * 1000)
        return (True, latency)
    except Exception:
        return (False, -1)

def estimate_tokens(messages: list) -> int:
    raw_chars = 0
    for m in messages:
        if m.get("content"):
            raw_chars += len(str(m["content"]))
        if m.get("tool_calls"):
            raw_chars += len(json.dumps(m["tool_calls"]))
    return max(1, int(raw_chars / 3.8))

def render_context_bar(current_tokens: int, max_tokens: int = 8192, bar_len: int = 15) -> str:
    pct = min(1.0, current_tokens / max_tokens)
    filled = int(bar_len * pct)
    bar = "█" * filled + "░" * (bar_len - filled)
    if pct < 0.6:
        color = C.GREEN
    elif pct < 0.85:
        color = C.YELLOW
    else:
        color = C.RED
    return f"{color}[{bar}] {current_tokens:,} / {max_tokens:,} tokens ({int(pct*100)}%){C.RESET}"

def truncate_text(text: str, max_chars: int = 2000) -> str:
    if not text or len(text) <= max_chars:
        return text
    head_len = int(max_chars * 0.6)
    tail_len = int(max_chars * 0.4)
    cut = len(text) - (head_len + tail_len)
    return f"{text[:head_len]}\n\n[... {cut:,} chars truncated ...]\n\n{text[-tail_len:]}"

def _fix_escaped_newlines(text: str) -> tuple:
    """
    Safety net for small models (Qwen3 8B etc.).

    Small local models sometimes emit JSON-string content where intended newlines show
    up as the two literal characters backslash+n instead of real line breaks. When those
    literals appear BETWEEN complete statements they can be safely split. But a literal
    backslash+n INSIDE a quoted string / f-string is ambiguous (it may be intentional or
    may be the model trying to wrap text) and splitting it produces broken Python.

    So this helper is conservative:
      - It splits a line on literal '\\n' only when every occurrence is OUTSIDE quotes.
      - If any occurrence is inside a quote, it leaves the text alone (the caller will
        get a clear diagnostic instead of a silently corrupted file).

    Returns (fixed_text, num_lines_fixed).
    """
    if not text:
        return text, 0
    literal = chr(92) + "n"   # the two chars: backslash n
    if literal not in text:
        return text, 0

    def outside_quotes(s: str, pos: int) -> bool:
        """True if position `pos` in string `s` is not inside a single/double quote."""
        in_s = in_d = False
        i = 0
        while i < pos:
            ch = s[i]
            if ch == chr(92):  # skip over escaped char inside quotes
                i += 2
                continue
            if ch == "'" and not in_d:
                in_s = not in_s
            elif ch == chr(34) and not in_s:
                in_d = not in_d
            i += 1
        return not (in_s or in_d)

    out_lines = []
    fixed = 0
    for line in text.split(chr(10)):
        idxs, search = [], 0
        while True:
            j = line.find(literal, search)
            if j == -1:
                break
            idxs.append(j)
            search = j + len(literal)
        if not idxs:
            out_lines.append(line)
            continue
        # Only fix if ALL occurrences sit outside quoted strings.
        if all(outside_quotes(line, j) for j in idxs):
            out_lines.extend(line.split(literal))
            fixed += 1
        else:
            out_lines.append(line)
    if fixed:
        return chr(10).join(out_lines), fixed
    return text, 0


def _has_ambiguous_escapes(text: str) -> bool:
    """True if `text` contains literal '\\n' inside a quoted string: too ambiguous to
    auto-fix safely (splitting would corrupt Python). Used to emit a clear diagnostic."""
    literal = chr(92) + "n"
    if literal not in text:
        return False
    for line in text.split(chr(10)):
        pos = 0
        while True:
            j = line.find(literal, pos)
            if j == -1:
                break
            # count quotes before j to know if we're inside a string
            prefix = line[:j]
            in_s = prefix.count("'") % 2 == 1
            in_d = prefix.count(chr(34)) % 2 == 1
            if in_s or in_d:
                return True
            pos = j + 2
    return False
    if fixed:
        return chr(10).join(out_lines), fixed
    return text, 0


def _find_target_block(original: str, target: str):
    """Locate a target with a little tolerance: exact, newline-normalized, or
    escaped-newline-normalized. Returns (index, matched_len, used_target) or (None, 0, None)."""
    n = chr(10)
    rn = chr(13) + chr(10)
    variants = [(original, target)]
    variants.append((original.replace(rn, n), target.replace(rn, n)))
    fixed_t, _ = _fix_escaped_newlines(target)
    if fixed_t != target:
        variants.append((original.replace(rn, n), fixed_t.replace(rn, n)))
    for hay, needle in variants:
        if needle and needle in hay:
            idx = hay.find(needle)
            return idx, len(needle), needle
    return None, 0, None


def _context_around(original: str, target: str, radius: int = 2) -> str:
    """Give a small model a concrete preview of the current file around the edit point,
    with escapes visible so it can see e.g. literal \\n corruption."""
    n = chr(10)
    lines = original.split(n)
    tlines = [ln for ln in target.split(n) if ln.strip()]
    probe = tlines[0][:60] if tlines else ""
    hit = -1
    for i, ln in enumerate(lines):
        if probe and probe in ln:
            hit = i
            break
    if hit == -1:
        # fall back to first 20 lines so the model still sees the file shape
        shown = lines[:20]
        start = 1
    else:
        start = max(1, hit + 1 - radius)
        shown = lines[start - 1: hit + 1 + radius]

    def vis(s: str) -> str:
        return s.replace(chr(13), "\\r").replace(chr(92) + 't', "    ")

    body = "\n".join(f"{start + i:4d} | {vis(ln)}" for i, ln in enumerate(shown))
    return body


def _strip_wrapping_quotes(s: str) -> str:
    """Windows 'Copy as path' wraps paths in single quotes; strip them."""
    s = s.strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in ("'", '"'):
        s = s[1:-1].strip()
    return s


def _looks_like_tool_call_json(snippet: str) -> bool:
    """True if a code fence contains JSON describing a tool call instead of real code.
    Small/mid models sometimes 'narrate' a tool call in a ```json block rather than calling it."""
    t = snippet.strip()
    if not (t.startswith("{") and t.endswith("}")):
        return False
    try:
        obj = json.loads(t, strict=False)
    except Exception:
        return False
    if not isinstance(obj, dict):
        return False
    has_name = any(k in obj for k in ("name", "tool", "function", "tool_name"))
    has_args = any(k in obj for k in ("arguments", "args", "parameters", "input"))
    return has_name and has_args


def tool_edit_file(path: str, target: str, replacement: str, cwd: str = ".") -> dict:
    try:
        target_path = path if os.path.isabs(path) else os.path.normpath(os.path.join(cwd, path))
        if not os.path.exists(target_path):
            return {"success": False, "message": f"File not found: '{target_path}'"}

        with open(target_path, "r", encoding="utf-8") as f:
            original = f.read()

        rn = chr(13) + chr(10)
        n = chr(10)

        # Repair ONLY safe literal '\n' escapes in the replacement; note ambiguous ones.
        ambiguous = _has_ambiguous_escapes(replacement)
        fixed_replacement, fixed_count = _fix_escaped_newlines(replacement)
        replacement = fixed_replacement

        idx, mlen, used_target = _find_target_block(original, target)
        if idx is None:
            ctx = _context_around(original, target)
            note = ""
            if fixed_count:
                note = f" (Note: your replacement contained {fixed_count} line(s) with literal '\\n' text that would break the file.)"
            return {
                "success": False,
                "message": (
                    f"Target string not found in '{os.path.basename(target_path)}'.{note}\n"
                    "Use one of the EXACT lines below as `target` (copy it verbatim, including indentation).\n"
                    f"Current file looks like:\n{ctx}"
                )
            }

        # Work on newline-normalized copies so matching is consistent
        norm_orig = original.replace(rn, n)
        norm_target = used_target.replace(rn, n)
        norm_repl = replacement.replace(rn, n)

        count = norm_orig.count(norm_target)
        updated = norm_orig.replace(norm_target, norm_repl, 1)

        with open(target_path, "w", encoding="utf-8", newline=n) as f:
            f.write(updated)

        msgs = [f"Successfully edited '{target_path}'" + (f" (1 of {count} occurrences)" if count > 1 else "") + "."]
        if fixed_count:
            msgs.append(f"Auto-repaired literal '\\n' escapes in {fixed_count} line(s) of your replacement.")
        if ambiguous:
            msgs.append("WARNING: replacement contained literal '\\n' inside quoted strings; verify the file.")
        return {"success": True, "message": " ".join(msgs)}
    except Exception as e:
        return {"success": False, "message": f"Error editing '{path}': {str(e)}"}

def tool_write_file(path: str, content: str, append: bool = False, cwd: str = ".") -> dict:
    try:
        target_path = path if os.path.isabs(path) else os.path.normpath(os.path.join(cwd, path))
        parent = os.path.dirname(target_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        # Repair ONLY safe literal '\n' escapes (between statements). Ambiguous ones
        # inside strings are reported back so the model rewrites instead of corrupting.
        ambiguous = _has_ambiguous_escapes(content)
        content, fixed_count = _fix_escaped_newlines(content)

        mode = "a" if append else "w"
        with open(target_path, mode, encoding="utf-8", newline=chr(10)) as f:
            f.write(content)
        lines = len(content.splitlines())
        action = "Appended to" if append else "Successfully wrote"
        msg = f"{action} '{target_path}' ({lines} lines, {len(content):,} bytes)."
        if fixed_count:
            msg += f" Auto-repaired literal '\\n' escapes in {fixed_count} line(s)."

        # Validate Python files immediately so the model gets the syntax error NOW,
        # at write time, instead of looping later with edit_file.
        if target_path.lower().endswith(".py"):
            try:
                with open(target_path, "r", encoding="utf-8") as vf:
                    source = vf.read()
                compile(source, target_path, "exec")
            except SyntaxError as se:
                return {
                    "success": False,
                    "message": (
                        f"WROTE '{target_path}' BUT IT IS INVALID PYTHON. Line {se.lineno}: {se.msg}\n"
                        f"Offending line: {str(se.text).rstrip() if se.text else '(unavailable)'}\n\n"
                        "The file was saved but cannot run. Fix it by REWRITING the WHOLE file with "
                        "write_file (not edit_file). Use REAL line breaks (press Enter) between statements; "
                        "NEVER type backslash-n to separate statements. If a message must be multi-line, "
                        "build it with separate string concatenation or keep it on one line."
                    )
                }

        if ambiguous:
            msg += ("\nWARNING: some of your content had literal '\\n' INSIDE quoted strings, "
                    "which may be a mistake. If the file looks broken, rewrite it using REAL line "
                    "breaks (press Enter) instead of typing backslash-n.")
        return {"success": True, "message": msg}
    except Exception as e:
        return {"success": False, "message": f"Error writing file '{path}': {str(e)}"}

def tool_read_file(path: str, start_line: int = 1, max_lines: int = 120, cwd: str = ".") -> dict:
    try:
        target_path = path if os.path.isabs(path) else os.path.normpath(os.path.join(cwd, path))
        if not os.path.exists(target_path):
            return {"success": False, "message": f"File not found: '{target_path}'"}
        
        with open(target_path, "r", encoding="utf-8", errors="replace") as f:
            all_lines = f.readlines()
        
        total = len(all_lines)
        start_idx = max(0, start_line - 1)
        end_idx = min(total, start_idx + max_lines)
        selected = all_lines[start_idx:end_idx]
        
        numbered = []
        suspicious = 0
        literal = chr(92) + "n"
        for i, line in enumerate(selected, start=start_idx):
            body = line.rstrip()
            marker = ""
            # Flag lines that contain literal backslash-n text glued into a long line:
            # this is the corruption small models produce, and seeing it explicitly helps them fix it.
            if literal in body and len(body) > 120:
                marker = "   <-- WARNING: literal '\\n' text detected, this line is likely broken"
                suspicious += 1
            numbered.append(f"{i+1:4d} | {body}{marker}")
        output = "\n".join(numbered)

        meta = f"[Lines {start_idx+1}-{end_idx} of {total} total in '{os.path.basename(target_path)}']\n"
        if suspicious:
            meta += (f"[!] {suspicious} line(s) contain literal '\\n' text instead of real newlines. "
                     "If you edit them, replace the whole broken line with properly split lines.]\n")
        return {"success": True, "message": meta + output}
    except Exception as e:
        return {"success": False, "message": f"Error reading file '{path}': {str(e)}"}

def tool_run_command(command, cwd, timeout=60, max_chars=2000):
    cmd_strip = command.strip()
    if cmd_strip.startswith("cd ") or cmd_strip == "cd":
        target = cmd_strip[3:].strip() or os.path.expanduser("~")
        new_path = os.path.abspath(os.path.join(cwd, target))
        if os.path.isdir(new_path):
            return {"stdout": f"Directory changed to {new_path}", "stderr": "", "code": 0, "cwd": new_path}
        return {"stdout": "", "stderr": f"Directory not found: {target}", "code": 1, "cwd": cwd}

    shell_cmd = ["powershell", "-NoProfile", "-Command", command] if os.name == "nt" else ["/bin/bash", "-c", command]
    try:
        p = subprocess.run(shell_cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout, errors="replace")
        return {
            "stdout": truncate_text(p.stdout.strip(), max_chars=max_chars),
            "stderr": truncate_text(p.stderr.strip(), max_chars=max_chars),
            "code": p.returncode,
            "cwd": cwd
        }
    except subprocess.TimeoutExpired:
        return {"stdout": "", "stderr": f"Command timed out after {timeout}s", "code": -1, "cwd": cwd}
    except Exception as e:
        return {"stdout": "", "stderr": str(e), "code": -1, "cwd": cwd}

def trim_message_history(messages: list, max_messages: int = 18) -> list:
    if len(messages) <= max_messages + 1:
        return messages
    system_msg = messages[0]
    tail = messages[-(max_messages):]
    while tail and tail[0].get("role") == "tool":
        tail.pop(0)
    if tail and tail[0].get("role") == "assistant" and tail[0].get("tool_calls"):
        expected_ids = {tc.get("id") for tc in tail[0].get("tool_calls", [])}
        found_ids = {m.get("tool_call_id") for m in tail[1:] if m.get("role") == "tool"}
        if not expected_ids.issubset(found_ids):
            tail.pop(0)
    return [system_msg] + tail

def compress_context(client, model: str, messages: list, keep_recent: int = 4) -> list:
    system_msg = messages[0]
    if len(messages) <= keep_recent + 1:
        return messages

    old_messages = messages[1:-(keep_recent)]
    recent_messages = messages[-(keep_recent):]

    while recent_messages and recent_messages[0].get("role") == "tool":
        recent_messages.pop(0)

    conversation_text = ""
    for m in old_messages:
        role = m.get("role", "unknown").upper()
        content = m.get("content", "")
        if role == "TOOL":
            content = truncate_text(str(content), max_chars=400)
            conversation_text += f"[TOOL RESULT]: {content}\n"
        elif role == "ASSISTANT" and m.get("tool_calls"):
            calls = m.get("tool_calls", [])
            for tc in calls:
                fn = tc.get("function", {})
                conversation_text += f"ASSISTANT called tool: {fn.get('name', '?')}({fn.get('arguments', '')[:100]}...)\n"
            if content:
                conversation_text += f"ASSISTANT: {content}\n"
        else:
            conversation_text += f"{role}: {content}\n"

    conversation_text = truncate_text(conversation_text, max_chars=3000)
    print(f"\n{C.CYAN}[⟳ Auto-compressing context...]{C.RESET}", end=" ", flush=True)

    try:
        compress_response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a precise conversation summarizer. Output only the summary, nothing else."},
                {"role": "user", "content": COMPRESS_PROMPT.format(conversation=conversation_text)}
            ],
            temperature=0.1,
            max_tokens=500
        )
        summary = compress_response.choices[0].message.content.strip()
    except Exception as e:
        print(f"{C.RED}[Compression failed: {e}. Falling back to trim.]{C.RESET}")
        return trim_message_history(messages)

    before_tokens = estimate_tokens(messages)
    compressed_messages = [
        system_msg,
        {"role": "assistant", "content": f"[CONVERSATION SUMMARY - Auto-compressed from earlier chat]\n{summary}"}
    ] + recent_messages
    after_tokens = estimate_tokens(compressed_messages)
    saved = before_tokens - after_tokens

    print(f"{C.GREEN}Done! Freed ~{saved:,} tokens ({before_tokens:,} → {after_tokens:,}){C.RESET}\n")
    return compressed_messages

def get_loaded_model(client):
    try:
        models = client.models.list()
        if models.data:
            return models.data[0].id
    except Exception:
        pass
    return "local-model"

def get_multiline_input(prompt_label: str) -> str:
    # Headless / eval mode: feed prompts from a queue instead of stdin.
    if HEADLESS["enabled"]:
        if HEADLESS["queue"]:
            return HEADLESS["queue"].pop(0)
        return "/exit"

    first_line = input(prompt_label)

    if first_line.strip().lower() == "/paste":
        print(f"{C.CYAN}╭── [PASTE MODE] ──────────────────────────────────────────────╮{C.RESET}")
        print(f"{C.CYAN}│ Paste your text/code below.                                  │{C.RESET}")
        print(f"{C.CYAN}│ When finished, type {C.BOLD}`END`{C.RESET}{C.CYAN} on a new line and press Enter.         │{C.RESET}")
        print(f"{C.CYAN}╰──────────────────────────────────────────────────────────────╯{C.RESET}")
        paste_lines = []
        while True:
            try:
                line = input()
                if line.strip() == "END":
                    break
                paste_lines.append(line)
            except (KeyboardInterrupt, EOFError):
                break
        return "\n".join(paste_lines).strip()

    if first_line.strip().startswith('"""') or first_line.strip().startswith("'''"):
        quote_tag = first_line.strip()[:3]
        clean_first = first_line.strip()[3:]
        if clean_first.endswith(quote_tag) and len(clean_first) >= 3:
            return clean_first[:-3].strip()
        block_lines = [clean_first] if clean_first else []
        print(f"{C.GRAY}... (multi-line mode, finish with {quote_tag}){C.RESET}")
        while True:
            try:
                nxt = input(f"{C.GRAY}... {C.RESET}")
                if nxt.strip().endswith(quote_tag):
                    block_lines.append(nxt.strip()[:-3])
                    break
                block_lines.append(nxt)
            except (KeyboardInterrupt, EOFError):
                break
        return "\n".join(block_lines).strip()

    lines = [first_line]
    if os.name == "nt":
        import msvcrt
        time.sleep(0.04)
        while msvcrt.kbhit():
            try:
                more = sys.stdin.readline()
            except (KeyboardInterrupt, EOFError):
                break
            # Strip stray control characters (e.g. ^A from terminal escape sequences)
            # that Windows paste/line-editing can inject into stdin.
            more = "".join(ch for ch in more if ch == "\t" or ord(ch) >= 32).rstrip("\r\n")
            if not more.strip():
                break
            lines.append(more)
            time.sleep(0.01)

    result = "\n".join(lines).strip()
    if len(lines) > 1:
        print(f"{C.GRAY}[Captured {len(lines)} lines of pasted text]{C.RESET}")
    return result

def print_help():
    print(f"\n{C.BOLD}Available Tools for AI:{C.RESET}")
    print(f"  {C.GREEN}write_file{C.RESET}               - Direct file writer (clean code writing, no shell quotes)")
    print(f"  {C.GREEN}read_file{C.RESET}                - Safe file reader with line numbers")
    print(f"  {C.GREEN}execute_terminal_command{C.RESET} - Shell execution for tests, git, installs, and runs")
    print(f"\n{C.BOLD}Interactive Slash Commands:{C.RESET}")
    print(f"  {C.CYAN}/paste{C.RESET}     - Multi-line paste mode (type 'END' on new line to finish)")
    print(f"  {C.CYAN}\"\"\"...\"\"\"{C.RESET}  - Triple quote input block")
    print(f"  {C.CYAN}/status{C.RESET}    - Check if LM Studio is live and ping latency")
    print(f"  {C.CYAN}/context{C.RESET}   - View exact context window stats and message breakdown")
    print(f"  {C.CYAN}/compress{C.RESET}  - Manually trigger context compression now")
    print(f"  {C.CYAN}/approve{C.RESET}   - Toggle auto-approval on/off")
    print(f"  {C.CYAN}/clear{C.RESET}     - Reset chat history and wipe context memory")
    print(f"  {C.CYAN}Ctrl+C{C.RESET}     - Instantly interrupt streaming generation without quitting")
    print(f"  {C.CYAN}/exit{C.RESET}      - Quit the agent\n")

def main():
    detected_host = detect_port()
    parser = argparse.ArgumentParser(description="LM Studio Local LLM Terminal Agent")
    parser.add_argument("--host", default=detected_host, help=f"LM Studio endpoint (default: {detected_host})")
    parser.add_argument("--model", default=None, help="Model ID")
    parser.add_argument("-y", "--auto-approve", action="store_true", help="Auto-run commands/file edits without prompt")
    parser.add_argument("--context-limit", type=int, default=8192, help="Context window limit (default: 8192)")
    parser.add_argument("--compress-at", type=float, default=0.85, help="Auto-compress when context reaches this fraction (default: 0.85)")
    parser.add_argument("--max-output", type=int, default=2000, help="Max output chars per command (default: 2000)")
    parser.add_argument("--max-history", type=int, default=20, help="Max history messages to retain (default: 20)")
    parser.add_argument("--max-response", type=int, default=4096,
                        help="Max tokens the model may generate per turn (default: 4096). "
                             "Too low truncates large write_file calls mid-string.")
    parser.add_argument("--task", default=None,
                        help="Headless: run this single task then exit (for evals).")
    parser.add_argument("--json-out", default=None,
                        help="Headless: write a structured JSON result to this path.")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress banner and per-tool chatter (headless).")
    args = parser.parse_args()

    # --- Headless setup ---
    if args.task:
        HEADLESS["enabled"] = True
        HEADLESS["json_out"] = args.json_out
        HEADLESS["quiet"] = bool(args.quiet)
        HEADLESS["queue"] = [args.task, "/exit"]
        HEADLESS["stats"]["start_time"] = time.time()
        args.auto_approve = True

    client = OpenAI(base_url=args.host, api_key="lm-studio")
    model = args.model or get_loaded_model(client)
    cwd = os.getcwd()
    auto_approve = args.auto_approve

    is_live, latency = ping_lm_studio(args.host)
    status_tag = f"{C.GREEN}● LIVE ({latency}ms){C.RESET}" if is_live else f"{C.RED}● OFFLINE{C.RESET}"

    sys_prompt = SYSTEM_PROMPT.format(
        os_name="Windows" if os.name == "nt" else sys.platform,
        shell="PowerShell" if os.name == "nt" else "Bash",
        cwd=cwd
    )
    messages = [{"role": "system", "content": sys_prompt}]

    if not HEADLESS["quiet"]:
        print(f"\n{C.BOLD}{C.CYAN}╭──────────────────────────────────────────────────────────╮{C.RESET}")
        print(f"{C.BOLD}{C.CYAN}│         LOCAL LLM CODING & TERMINAL AGENT                │{C.RESET}")
        print(f"{C.BOLD}{C.CYAN}├──────────────────────────────────────────────────────────┤{C.RESET}")
        print(f"  Status      : {status_tag}")
        print(f"  Model       : {C.BOLD}{model}{C.RESET}")
        print(f"  Tools       : {C.GREEN}edit_file, write_file, read_file, execute_terminal{C.RESET}")
        print(f"  Endpoint    : {C.GRAY}{args.host}{C.RESET}")
        print(f"  Auto-run    : {C.MAGENTA}{'ON' if auto_approve else 'OFF (asks before action)'}{C.RESET}")
        print(f"  Compress at : {C.GREEN}{int(args.compress_at * 100)}% context usage (auto){C.RESET}")
        print(f"{C.BOLD}{C.CYAN}╰──────────────────────────────────────────────────────────╯{C.RESET}")
        print(f"{C.GRAY}Commands: /paste /status /context /compress /approve /clear /help /exit{C.RESET}\n")

    while True:
        live, _ = ping_lm_studio(args.host)
        live_dot = f"{C.GREEN}●{C.RESET}" if live else f"{C.RED}● (offline){C.RESET}"

        tokens_used = estimate_tokens(messages)
        cbar = render_context_bar(tokens_used, max_tokens=args.context_limit, bar_len=10)

        try:
            print(f"{C.GRAY}┌─[{live_dot} Agent]──[{cbar}]{C.RESET}")
            prompt = get_multiline_input(f"{C.GRAY}└─{C.RESET}{C.BOLD}{C.CYAN} You > {C.RESET}")
        except KeyboardInterrupt:
            print(f"\n{C.YELLOW}Type '/exit' or press Ctrl+C again to quit.{C.RESET}")
            try:
                time.sleep(0.3)
                continue
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
        except EOFError:
            break

        if not prompt:
            continue

        cmd_lower = prompt.lower().strip()
        if cmd_lower in ["/exit", "/quit", "exit", "quit"]:
            print("Goodbye!")
            break
        if cmd_lower in ["/help", "help"]:
            print_help()
            continue
        if cmd_lower == "/clear":
            messages = [{"role": "system", "content": sys_prompt}]
            print(f"{C.GREEN}[✓] Chat history and context memory reset.{C.RESET}\n")
            continue
        if cmd_lower == "/approve":
            auto_approve = not auto_approve
            print(f"{C.MAGENTA}[✓] Auto-approval: {'ON' if auto_approve else 'OFF'}{C.RESET}\n")
            continue
        if cmd_lower == "/status":
            live, lat = ping_lm_studio(args.host)
            if live:
                print(f"{C.GREEN}● LIVE ({lat}ms). Model: {model}{C.RESET}\n")
            else:
                print(f"{C.RED}● UNREACHABLE at {args.host}{C.RESET}\n")
            continue
        if cmd_lower == "/context":
            toks = estimate_tokens(messages)
            pct = (toks / args.context_limit) * 100
            print(f"\n{C.BOLD}Context Window:{C.RESET}")
            print(f"  Usage    : {toks:,} / {args.context_limit:,} tokens ({pct:.1f}%)")
            print(f"  Messages : {len(messages)} (System + {len(messages)-1} chat/tool)")
            print(f"  Meter    : {render_context_bar(toks, args.context_limit, bar_len=20)}")
            print(f"  Auto-zip : Triggers at {int(args.compress_at * 100)}%\n")
            continue
        if cmd_lower == "/compress":
            if len(messages) <= 3:
                print(f"{C.GRAY}[Nothing to compress yet.]{C.RESET}\n")
            else:
                messages = compress_context(client, model, messages, keep_recent=4)
            continue

        if cmd_lower.startswith("/"):
            print(f"{C.RED}[!] Unknown command '{prompt.split()[0]}'. Type /help to see valid commands.{C.RESET}\n")
            continue

        messages.append({"role": "user", "content": prompt})

        recent_calls = []  # loop guard: tracks (tool, raw_args) signatures to break 8B repetition loops
        recent_fuzzy = []  # loop guard: whitespace-stripped arg signatures to catch near-identical repeats
        fail_counts = {}   # per-file count of write_file results that were invalid Python
        narration_fails = 0  # per-turn count of narrated tool-call blocks
        tools_executed = 0   # number of real tool calls that ran in this turn
        inner_turns = 0      # safety cap for headless/agentic turns

        while True:
            inner_turns += 1
            if HEADLESS["enabled"] and inner_turns > MAX_AGENT_TURNS:
                HEADLESS["stats"]["error"] = f"max_turns_exceeded({MAX_AGENT_TURNS})"
                print(f"\n{C.RED}[!] Hit turn cap ({MAX_AGENT_TURNS}); stopping to avoid an infinite loop.{C.RESET}")
                _rec_stat("loops")
                break
            messages = trim_message_history(messages, max_messages=args.max_history)

            # Auto-compress if nearing threshold
            current_tokens = estimate_tokens(messages)
            threshold = int(args.context_limit * args.compress_at)
            if current_tokens >= threshold and len(messages) > 5:
                messages = compress_context(client, model, messages, keep_recent=4)

            try:
                response_stream = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    tools=TOOLS,
                    tool_choice="auto",
                    stream=True,
                    temperature=0.1,
                    max_tokens=args.max_response
                )
            except Exception as e:
                print(f"\n{C.RED}[!] Error from LM Studio: {e}{C.RESET}\n")
                break

            full_content = ""
            tool_calls_acc = {}
            first_text_chunk = True
            interrupted = False

            try:
                for chunk in response_stream:
                    if not chunk.choices:
                        continue
                    delta = chunk.choices[0].delta

                    if delta.content:
                        if first_text_chunk:
                            print(f"\n{C.BOLD}{C.GREEN}AI > {C.RESET}", end="", flush=True)
                            first_text_chunk = False
                        print(delta.content, end="", flush=True)
                        full_content += delta.content

                    if delta.tool_calls:
                        for tc_chunk in delta.tool_calls:
                            idx = tc_chunk.index
                            if idx not in tool_calls_acc:
                                tool_calls_acc[idx] = {
                                    "id": tc_chunk.id or f"call_{idx}",
                                    "type": "function",
                                    "function": {"name": "", "arguments": ""}
                                }
                            if tc_chunk.id:
                                tool_calls_acc[idx]["id"] = tc_chunk.id
                            if tc_chunk.function:
                                if tc_chunk.function.name:
                                    tool_calls_acc[idx]["function"]["name"] += tc_chunk.function.name
                                if tc_chunk.function.arguments:
                                    tool_calls_acc[idx]["function"]["arguments"] += tc_chunk.function.arguments

            except KeyboardInterrupt:
                interrupted = True
                print(f"\n\n{C.YELLOW}[⏹ Generation interrupted (Ctrl+C)]{C.RESET}\n")

            if not first_text_chunk:
                print()

            if interrupted:
                if full_content:
                    messages.append({"role": "assistant", "content": full_content})
                break

            # --- Narration rescue -------------------------------------------------
            # Mid/small models sometimes DESCRIBE a tool call in a ```json block instead
            # of calling it. Text nudges rarely fix a stuck pattern, so instead we parse
            # --- Native Qwen tool-call parsing -----------------------------------
            # The Qwen chat template instructs the model to emit tool calls as:
            #   <tool_call>
            #   {"name": ..., "arguments": {...}}
            #   </tool_call>
            # LM Studio's OpenAI-compat layer does NOT translate these into
            # delta.tool_calls, so they arrive as plain text. We must parse them.
            if not tool_calls_acc and "<tool_call>" in full_content:
                import re as _re
                _valid_names = {t["function"]["name"] for t in TOOLS}
                rescued = []
                for m_tc in _re.finditer(r"<tool_call>\s*\{.*?\}\s*</tool_call>", full_content, _re.DOTALL):
                    block = m_tc.group(0)
                    inner = block[len("<tool_call>"):-len("</tool_call>")].strip()
                    try:
                        obj = json.loads(inner, strict=False)
                    except Exception:
                        # sometimes the arguments are a JSON string; try a lenient repair
                        continue
                    if not isinstance(obj, dict):
                        continue
                    name = obj.get("name") or obj.get("tool") or obj.get("function")
                    args = obj.get("arguments", obj.get("args", obj.get("parameters", obj.get("input"))))
                    if isinstance(name, dict):
                        args = name.get("arguments", args)
                        name = name.get("name")
                    if isinstance(name, str) and name in _valid_names and args is not None:
                        rescued.append({
                            "id": f"qtool_{len(rescued)}_{int(time.time()*1000)%1000}",
                            "type": "function",
                            "function": {
                                "name": name,
                                "arguments": json.dumps(args) if not isinstance(args, str) else args,
                            },
                        })
                if rescued:
                    print(f"\n{C.CYAN}[!] Parsed {len(rescued)} native <tool_call> block(s) from model text.{C.RESET}")
                    tool_calls_acc = {
                        i: {"id": r["id"], "type": "function",
                            "function": {"name": r["function"]["name"], "arguments": r["function"]["arguments"]}}
                        for i, r in enumerate(rescued)
                    }
                    # strip the tool_call text so it isn't mistaken for chat output
                    full_content = _re.sub(r"<tool_call>\s*\{.*?\}\s*</tool_call>", "", full_content, flags=_re.DOTALL).strip()

            # --- Markdown-fence rescue (fallback) --------------------------------
            # Some models/prompts produce ```json blocks instead of <tool_call> tags.
            # Execute the described call for the model.
            if not tool_calls_acc and "```" in full_content:
                import re as _re
                fences = _re.findall(r"```([A-Za-z0-9_+-]*)\n(.*?)```", full_content, _re.DOTALL)
                if not fences:
                    fences = [("", p.strip()) for p in full_content.split("```")[1:-1:2]]
                _valid_names = {t["function"]["name"] for t in TOOLS}
                rescued = []
                for lang, body in fences:
                    if lang.lower() not in ("json", "jsonc", ""):
                        continue
                    t = body.strip()
                    if not t.startswith("{"):
                        continue
                    try:
                        obj = json.loads(t, strict=False)
                    except Exception:
                        continue
                    if not isinstance(obj, dict):
                        continue
                    name = obj.get("name") or obj.get("tool") or obj.get("function")
                    args = obj.get("arguments", obj.get("args", obj.get("parameters", obj.get("input"))))
                    if isinstance(name, dict):
                        args = name.get("arguments", args)
                        name = name.get("name")
                    if isinstance(name, str) and name in _valid_names and args is not None:
                        rescued.append({
                            "id": f"rescued_{len(rescued)}_{int(time.time()*1000)}%1000",
                            "type": "function",
                            "function": {
                                "name": name,
                                "arguments": json.dumps(args) if not isinstance(args, str) else args,
                            }
                        })
                if rescued:
                    print(f"\n{C.YELLOW}[!] Model described a tool call in text; harness is executing it directly.{C.RESET}")
                    tool_calls_acc = {
                        i: {"id": r["id"], "type": "function",
                            "function": {"name": r["function"]["name"], "arguments": r["function"]["arguments"]}}
                        for i, r in enumerate(rescued)
                    }
                    # strip the fence blocks so they aren't mistaken for chat output
                    full_content = _re.sub(r"```[A-Za-z0-9_+-]*\n.*?```", "", full_content, flags=_re.DOTALL).strip()

            if tool_calls_acc:
                tool_calls_list = list(tool_calls_acc.values())
                messages.append({
                    "role": "assistant",
                    "content": full_content or None,
                    "tool_calls": tool_calls_list
                })

                for tc in tool_calls_list:
                    func_name = tc["function"]["name"]
                    raw_args = tc["function"]["arguments"]

                    # Loop guard: if the model repeats the same call, nudge it to change approach.
                    # Two signals: (a) exact repeat, (b) self-similar repeat (same tool + same hash of
                    # whitespace-stripped args) so re-emitting the same broken edit still trips the guard.
                    call_sig = (func_name, raw_args)
                    fuzzy_sig = (func_name, "".join(raw_args.split()))
                    exact_hits = recent_calls.count(call_sig)
                    fuzzy_hits = recent_fuzzy.count(fuzzy_sig)
                    if exact_hits >= 2 or fuzzy_hits >= 3:
                        _rec_stat("loops")
                        print(f"\n{C.RED}[!] Loop detected: '{func_name}' repeated with the same or near-identical arguments. Nudging model to change approach.{C.RESET}\n")
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tc["id"],
                            "name": func_name,
                            "content": (
                                "ERROR: You have already made this tool call (or a near-identical one) several times "
                                "and it did not succeed. STOP repeating it. You MUST change strategy now:\n"
                                "- If a target string was not found, use read_file first and copy the EXACT current line.\n"
                                "- If you keep writing literal '\\n' text, write the whole file again with write_file "
                                "using real line breaks.\n"
                                "- Or tell the user you are stuck and ask for guidance."
                            )
                        })
                        continue
                    recent_calls.append(call_sig)
                    recent_fuzzy.append(fuzzy_sig)

                    # Parse tool arguments; if the 8B emitted malformed/truncated JSON, tell it explicitly
                    # so it can self-correct instead of silently receiving {} and looping forever.
                    try:
                        args_data = json.loads(raw_args, strict=False)
                    except Exception as e:
                        _rec_stat("invalid_json")
                        print(f"\n{C.RED}[!] Invalid JSON in '{func_name}' arguments: {e}{C.RESET}\n")
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tc["id"],
                            "name": func_name,
                            "content": (
                                f"ERROR: Your tool arguments were not valid JSON ({e}). "
                                "You likely cut off mid-output or escaped characters incorrectly. "
                                "Retry with valid JSON. If the content is very long, split it into "
                                "multiple smaller write_file calls using append=true."
                            )
                        })
                        continue

                    # 0. EDIT_FILE (Single-line / targeted replacement)
                    if func_name == "edit_file":
                        fpath = args_data.get("path", "")
                        ftarget = args_data.get("target", "")
                        freplacement = args_data.get("replacement", "")

                        print(f"\n{C.CYAN}⚡ [Edit File Tool (Diff)]:{C.RESET} {C.BOLD}{fpath}{C.RESET}")
                        print(f"{C.RED}- {ftarget}{C.RESET}")
                        print(f"{C.GREEN}+ {freplacement}{C.RESET}\n")

                        if not auto_approve:
                            try:
                                confirm = input(f"{C.YELLOW}Apply this edit? [y/n/a(always)]: {C.RESET}").strip().lower()
                            except KeyboardInterrupt:
                                confirm = "n"

                            if confirm in ["a", "always"]:
                                auto_approve = True
                            elif confirm not in ["y", "yes", ""]:
                                messages.append({
                                    "role": "tool",
                                    "tool_call_id": tc["id"],
                                    "name": func_name,
                                    "content": f"User skipped edit on '{fpath}'."
                                })
                                print(f"{C.GRAY}Skipped.{C.RESET}\n")
                                continue

                        res = tool_edit_file(fpath, ftarget, freplacement, cwd=cwd)
                        res_str = res["message"]
                        res_color = C.GREEN if res["success"] else C.RED
                        print(f"{res_color}[{res_str}]{C.RESET}\n")
                        _rec_tool("edit_file", res["success"], fpath)
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tc["id"],
                            "name": func_name,
                            "content": res_str
                        })

                    # 1. WRITE_FILE (Full file create / overwrite)
                    elif func_name == "write_file":
                        fpath = args_data.get("path", "")
                        fcontent = args_data.get("content", "")
                        fappend = args_data.get("append", False)
                        num_lines = len(fcontent.splitlines())
                        preview = fcontent[:200] + ("..." if len(fcontent) > 200 else "")

                        print(f"\n{C.MAGENTA}⚡ [Write File Tool]:{C.RESET} {C.BOLD}{fpath}{C.RESET} ({num_lines} lines)")
                        print(f"{C.GRAY}Content preview:{C.RESET}\n{preview}\n")

                        if not auto_approve:
                            try:
                                confirm = input(f"{C.YELLOW}Write this file? [y/n/a(always)]: {C.RESET}").strip().lower()
                            except KeyboardInterrupt:
                                confirm = "n"

                            if confirm in ["a", "always"]:
                                auto_approve = True
                            elif confirm not in ["y", "yes", ""]:
                                messages.append({
                                    "role": "tool",
                                    "tool_call_id": tc["id"],
                                    "name": func_name,
                                    "content": f"User skipped writing to '{fpath}'."
                                })
                                print(f"{C.GRAY}Skipped.{C.RESET}\n")
                                continue

                        res = tool_write_file(fpath, fcontent, append=fappend, cwd=cwd)
                        res_str = res["message"]
                        res_color = C.GREEN if res["success"] else C.RED
                        print(f"{res_color}[{res_str}]{C.RESET}\n")
                        _rec_tool("write_file", res["success"], fpath)
                        if "INVALID PYTHON" in res_str:
                            _rec_stat("python_syntax_errors")
                        # Escalate if the same file keeps failing to compile.
                        if not res["success"] and "INVALID PYTHON" in res_str:
                            key = os.path.basename(fpath or "")
                            fail_counts[key] = fail_counts.get(key, 0) + 1
                            if fail_counts[key] >= 2:
                                res_str += (
                                    f"\n\nSTOP. You have now written '{key}' with invalid Python "
                                    f"{fail_counts[key]} times. Abandon the approach that keeps breaking. "
                                    "Write a SIMPLER version: avoid multi-line f-strings entirely, keep each "
                                    "assert on ONE physical line, and do not use backslash escapes for newlines."
                                )
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tc["id"],
                            "name": func_name,
                            "content": res_str
                        })

                    # 2. READ_FILE
                    elif func_name == "read_file":
                        fpath = args_data.get("path", "")
                        sline = args_data.get("start_line", 1)
                        mline = args_data.get("max_lines", 120)

                        print(f"\n{C.BLUE}⚡ [Read File Tool]:{C.RESET} {C.BOLD}{fpath}{C.RESET} (from line {sline})")
                        res = tool_read_file(fpath, start_line=sline, max_lines=mline, cwd=cwd)
                        res_str = res["message"]
                        _rec_tool("read_file", res["success"], fpath)
                        preview = res_str[:300] + ("..." if len(res_str) > 300 else "")
                        print(f"{C.GRAY}{preview}{C.RESET}\n")
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tc["id"],
                            "name": func_name,
                            "content": res_str
                        })

                    # 3. EXECUTE_TERMINAL_COMMAND
                    elif func_name == "execute_terminal_command":
                        cmd = args_data.get("command", raw_args)
                        print(f"\n{C.MAGENTA}⚡ [Terminal Command]:{C.RESET} {C.BOLD}{cmd}{C.RESET}")
                        if not auto_approve:
                            try:
                                confirm = input(f"{C.YELLOW}Execute? [y/n/a(always)]: {C.RESET}").strip().lower()
                            except KeyboardInterrupt:
                                confirm = "n"

                            if confirm in ["a", "always"]:
                                auto_approve = True
                            elif confirm not in ["y", "yes", ""]:
                                messages.append({
                                    "role": "tool",
                                    "tool_call_id": tc["id"],
                                    "name": func_name,
                                    "content": "User canceled the command."
                                })
                                print(f"{C.GRAY}Skipped.{C.RESET}\n")
                                continue

                        print(f"{C.GRAY}Running...{C.RESET}")
                        out = tool_run_command(cmd, cwd, max_chars=args.max_output)
                        cwd = out["cwd"]
                        output_display = out["stdout"] or out["stderr"] or "(Done)"
                        exit_color = C.GREEN if out["code"] == 0 else C.RED
                        print(f"{C.GRAY}┌─ Output ({exit_color}Exit {out['code']}{C.GRAY}) ─────────────────────{C.RESET}")
                        print(output_display)
                        print(f"{C.GRAY}└───────────────────────────────────────────{C.RESET}\n")

                        tool_res = f"STDOUT: {out['stdout']}\nSTDERR: {out['stderr']}\nEXIT CODE: {out['code']}"
                        _rec_tool("execute_terminal_command", out["code"] == 0, cmd)
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tc["id"],
                            "name": func_name,
                            "content": tool_res
                        })
                narration_fails = 0  # a real tool call succeeded; reset narration counter
                tools_executed += len(tool_calls_list)
                continue
            else:
                messages.append({"role": "assistant", "content": full_content})
                # Catch if the model dumped code/tool-call JSON into chat instead of using tools
                if "```" in full_content:
                    # Find every fenced block: ```lang\n...```
                    import re as _re
                    fences = _re.findall(r"```([A-Za-z0-9_+-]*)\n(.*?)```", full_content, _re.DOTALL)
                    if not fences:
                        fences = [("", parts.strip()) for parts in full_content.split("```")[1:-1:2]]

                    tool_call_blobs = []
                    code_blocks = []
                    truncated_tool_calls = 0
                    for lang, body in fences:
                        body = body.rstrip()
                        is_json = lang.lower() in ("json", "jsonc", "")
                        if is_json and _looks_like_tool_call_json(body):
                            tool_call_blobs.append(body)
                        elif is_json and ("\"name\"" in body[:80] or "\"arguments\"" in body or "\"path\"" in body):
                            # JSON fence that looks like a tool call but did not parse:
                            # almost always because generation was cut off mid-string.
                            truncated_tool_calls += 1
                        else:
                            code_blocks.append((lang, body))

                    # Case 0: a tool-call JSON fence that could not parse = truncated generation.
                    if truncated_tool_calls:
                        _rec_stat("truncations")
                        print(f"\n{C.RED}[!] A tool call was CUT OFF mid-output (max tokens reached). "
                              f"Content too large for one response.{C.RESET}")
                        messages.append({
                            "role": "user",
                            "content": (
                                "Your last tool call was cut off before it finished, so it could not be parsed. "
                                "This means the content was too large for a single response. Do NOT retry the same "
                                "huge call. Instead:\n"
                                "- Write a SMALLER file (under ~40 lines), then use write_file with append=true "
                                "for the rest, OR\n"
                                "- Write a simpler/shorter implementation.\n"
                                "Then call the tool again."
                            )
                        })
                        print(f"{C.GRAY}Asked model to write in smaller chunks. Retrying...{C.RESET}\n")
                        continue

                    # Case A: the model NARRATED a tool call instead of making one.
                    # (Normally rescued earlier; if we still get here the JSON was malformed.)
                    if tool_call_blobs:
                        narration_fails += 1
                        _rec_stat("narration")
                        print(f"\n{C.YELLOW}[!] The model described a tool call in text instead of calling the tool.{C.RESET}")
                        if narration_fails >= 2:
                            print(f"{C.RED}[!] Repeated narration. Stopping this turn and asking the model to try a simpler, direct action.{C.RESET}\n")
                            messages.append({
                                "role": "user",
                                "content": (
                                    "STOP describing tool calls. Reply with a single short sentence of what you will do, "
                                    "then immediately invoke ONE tool. Do not print any JSON or code blocks."
                                )
                            })
                            break
                        messages.append({
                            "role": "user",
                            "content": (
                                "Do not describe tool calls or print their JSON. Invoke the tool directly. "
                                "If you cannot, reply in plain text without any code blocks."
                            )
                        })
                        print(f"{C.GRAY}Nudged model to actually call the tool. Retrying...{C.RESET}\n")
                        continue

                    # Case B: real source code was dumped in chat.
                    # Skip the nag if tools already ran this turn (this is just a summary)
                    # and skip shell transcripts like "$ python calc.py" which are examples, not files.
                    if tools_executed == 0:
                        for lang, snippet in code_blocks:
                            raw_lines = snippet.splitlines()
                            # Heuristic: shell-example transcript (mostly lines starting with $ or >)
                            body_lines = [l for l in raw_lines if l.strip()]
                            shellish = bool(body_lines) and sum(
                                1 for l in body_lines if l.lstrip().startswith(("$", "> "))
                            ) >= max(1, len(body_lines) // 2)
                            if shellish:
                                continue
                            lines = snippet.splitlines()
                            if lines and lines[0].strip().lower() in ["python", "bash", "powershell", "javascript", "js", "html", "css", "json", "sh", "py", "ps1"]:
                                snippet = "\n".join(lines[1:]).strip()
                            if not snippet.strip():
                                continue
                            print(f"\n{C.YELLOW}[!] The model printed code in chat instead of saving to a file.{C.RESET}")
                            try:
                                save_dest = _strip_wrapping_quotes(
                                    input(f"{C.BOLD}Enter file path to save this code (or press Enter to skip): {C.RESET}")
                                )
                            except (KeyboardInterrupt, EOFError):
                                save_dest = ""
                            if save_dest:
                                res = tool_write_file(save_dest, snippet, cwd=cwd)
                                res_color = C.GREEN if res["success"] else C.RED
                                print(f"{res_color}[{res['message']}]{C.RESET}\n")
                print()
                break

    # Headless: emit the structured result JSON for the eval runner.
    _write_headless_result()

if __name__ == "__main__":
    main()
