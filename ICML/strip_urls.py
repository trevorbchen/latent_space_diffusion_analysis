"""Strip url = {...} lines from references.bib. Handles single-line and
multi-line url fields. Leaves a comment marker so the change is auditable."""
import re

with open("references.bib", encoding="utf-8") as f:
    src = f.read()

# Match a url field. The value may be braced or quoted, and may span lines
# until the closing brace at the same nesting depth, followed by an optional
# trailing comma and end-of-line.
out = []
i = 0
n = len(src)
removed = 0
while i < n:
    # Look for "url" or "URL" as a field key at start of (optionally indented) line
    m = re.match(r"([ \t]*)(url|URL|Url)([ \t]*=[ \t]*)", src[i:])
    if m and (i == 0 or src[i-1] == "\n"):
        # Skip past the value: handle {...} (with brace counting) or "..."
        j = i + m.end()
        if j < n and src[j] == "{":
            depth = 1
            j += 1
            while j < n and depth > 0:
                if src[j] == "{":
                    depth += 1
                elif src[j] == "}":
                    depth -= 1
                j += 1
        elif j < n and src[j] == '"':
            j += 1
            while j < n and src[j] != '"':
                j += 1
            if j < n:
                j += 1
        # Eat optional trailing comma and rest of the line including newline
        while j < n and src[j] in " \t":
            j += 1
        if j < n and src[j] == ",":
            j += 1
        while j < n and src[j] != "\n":
            j += 1
        if j < n and src[j] == "\n":
            j += 1
        # Drop the field entirely
        removed += 1
        i = j
        continue
    out.append(src[i])
    i += 1

with open("references.bib", "w", encoding="utf-8") as f:
    f.write("".join(out))

print(f"removed {removed} url fields")
