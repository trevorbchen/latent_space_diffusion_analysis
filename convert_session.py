"""Convert a Claude Code session JSONL to readable Markdown."""
import json, sys, os

def to_text(content):
    """Extract readable text from a message 'content' field."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content)
    parts = []
    for block in content:
        if not isinstance(block, dict):
            parts.append(str(block))
            continue
        btype = block.get("type")
        if btype == "text":
            parts.append(block.get("text", ""))
        elif btype == "tool_use":
            name = block.get("name", "?")
            inp = block.get("input", {})
            # Truncate long tool inputs
            inp_str = json.dumps(inp, indent=2)
            if len(inp_str) > 2000:
                inp_str = inp_str[:2000] + "\n...[truncated]..."
            parts.append(f"\n**[tool: {name}]**\n```json\n{inp_str}\n```")
        elif btype == "tool_result":
            out = block.get("content", "")
            if isinstance(out, list):
                out = to_text(out)
            out = str(out)
            if len(out) > 2000:
                out = out[:2000] + "\n...[truncated]..."
            parts.append(f"\n**[tool result]**\n```\n{out}\n```")
        elif btype == "thinking":
            # Skip thinking blocks to reduce clutter
            continue
        elif btype == "image":
            parts.append("[image]")
        else:
            parts.append(f"[{btype}]")
    return "\n".join(parts)

def main(in_path, out_path):
    with open(in_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    out = []
    out.append(f"# Claude Code Session Transcript\n")
    out.append(f"Source: `{os.path.basename(in_path)}`\n\n---\n")

    for line in lines:
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        etype = entry.get("type")
        if etype not in ("user", "assistant"):
            continue
        msg = entry.get("message", {})
        role = msg.get("role", etype)
        content = msg.get("content", "")
        text = to_text(content).strip()
        if not text:
            continue
        ts = entry.get("timestamp", "")
        out.append(f"\n## {role.title()} — {ts}\n\n{text}\n")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(out))

    print(f"Wrote {out_path}")
    print(f"  Size: {os.path.getsize(out_path) / 1024:.1f} KB")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python convert_session.py <input.jsonl> <output.md>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
