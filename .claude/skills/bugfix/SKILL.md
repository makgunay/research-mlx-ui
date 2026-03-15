---
name: bugfix
description: |
  Verify and fix issues reported by external code review tools (CodeRabbit, Cursor Bugbot, etc.).
  Use when pasting bug reports with: "/bugfix <issue details>" or "verify and fix this issue".
  Validates issues are real, researches best practices, and applies fixes following project conventions.
---

# Bugfix — External Issue Verification & Resolution

## Purpose

Handle bug reports from external code review tools (CodeRabbit, Cursor Bugbot, Copilot, etc.) by:
1. Verifying the issue is real (not a false positive)
2. Researching best practices from project docs and Ref MCP
3. Applying fixes that follow project conventions
4. Explaining what was changed and why

**Critical Principle**: External tools can produce false positives. Always verify before fixing.

---

## Phase 1: Parse the Issue

Extract from the pasted issue:
- **File path(s)**: Where the issue is reported
- **Line number(s)**: Specific location if provided
- **Issue type**: Bug, security, performance, style, etc.
- **Tool source**: CodeRabbit, Cursor Bugbot, Copilot, etc.
- **Suggested fix**: What the tool recommends (if any)

---

## Phase 2: Verify the Issue Exists

### 2.1 Locate the Code

```
1. Read the file(s) mentioned in the issue
2. Navigate to the specific line numbers
3. Understand the surrounding context (read 20-30 lines around)
```

### 2.2 Confirm or Reject

Ask yourself:
```
□ Does this code actually exist as described?
□ Is the issue a real problem or a tool misunderstanding?
□ Could this be intentional behavior?
□ Is the current code actually correct for this context?
```

**If the issue is a false positive**:
- Explain why the current code is correct
- Note any context the tool missed
- Stop here (no fix needed)

**If the issue is real**: Continue to Phase 3

---

## Phase 3: Research Best Practices

### 3.1 Check Project CLAUDE.md (Priority 1)

Load `CLAUDE.md` at the project root for:
- Key constraints (train.py line limit, MLX idioms, compile patterns)
- Server module reference
- Frontend component reference
- Known issues / tech debt

### 3.2 Check HANDOFF.md (Priority 2)

Load `docs/HANDOFF.md` for:
- Architecture decisions and rationale
- MLX-specific patterns and anti-patterns
- API specifications (REST + WebSocket protocol)
- Design decisions that should not be changed

### 3.3 Check External Best Practices (Priority 3)

Use Ref MCP for external library/framework guidance:

```
mcp__Ref__ref_search_documentation: "MLX <topic>" or "FastAPI <topic>" or "React <topic>"
mcp__Ref__ref_read_url: <url from search>
```

**When to use Ref MCP**:
- MLX framework patterns (compile, optimizers, lazy eval)
- FastAPI async patterns (WebSocket, lifespan, middleware)
- React patterns (hooks, state management)
- Recharts API usage

---

## Phase 4: Apply the Fix

### 4.1 Plan the Fix

Before changing code:
```
1. Identify the minimal change needed
2. Consider side effects
3. Check if related code needs updates
4. Verify fix doesn't break the build
```

### 4.2 Implement

Follow project conventions:

**Python (server/)**:
- Use `uv` for all package management (never pip)
- FastAPI async patterns
- Pydantic models for request validation
- No shell injection — use `create_subprocess_exec` not `create_subprocess_shell`
- Type hints where appropriate

**Python (train.py / prepare.py)**:
- MLX idioms: single `mx.eval()` per step, lazy computation
- `@partial(mx.compile, inputs=state, outputs=state)` for compiled train step
- Official `mlx.optimizers.Muon` + `MultiOptimizer` (not custom)
- Keep train.py under ~600 lines
- Never modify prepare.py

**JavaScript (client/)**:
- React functional components with hooks
- `useRef` for WebSocket callbacks (not in dependency arrays)
- Tailwind classes, no custom CSS except in index.css
- Recharts for charts

### 4.3 Verify the Fix

#### Build & Lint
```bash
# Python — check imports and syntax
uv run python -c "import server.main"

# Frontend — build check
cd client && npm run build

# Run server health check
uv run uvicorn server.main:app --host 127.0.0.1 --port 8000 &
sleep 2
curl -s http://127.0.0.1:8000/api/health
kill %1
```

#### Post-Fix Review Checklist

**Code Quality**
```
□ Follows CLAUDE.md patterns
□ No hardcoded secrets or credentials
□ Functions focused and readable
□ No over-engineering (fix the issue, don't refactor surroundings)
```

**MLX Specifics** (if training code)
```
□ Single mx.eval() per training step
□ No premature evaluation
□ Pure functions for mx.compile
□ Warmup step excluded from training budget
□ train.py under ~600 lines
```

**Server Specifics** (if server code)
```
□ No command injection (subprocess_exec not subprocess_shell)
□ Pydantic models for API inputs
□ WebSocket errors handled gracefully
□ CORS configured for frontend origin
```

**Frontend Specifics** (if React code)
```
□ No reconnection loops in useWebSocket (use refs for callbacks)
□ Experiments deduped in reducer
□ Error states shown to user (not swallowed)
□ Builds without errors (npm run build)
```

**Security**
```
□ No hardcoded API keys
□ Agent commands validated against ALLOWED_AGENTS allowlist
□ User input sanitized before use in subprocess/file paths
```

**If any check fails**: Fix it before completing the bugfix.

---

## Phase 5: Report Results

### If Issue Was Fixed

```markdown
## Issue Verified & Fixed

**Source**: [Tool name]
**Location**: `path/to/file:123`
**Issue Type**: [Bug/Security/Performance/etc.]

### Problem
[What the issue was]

### Research
- Checked: [CLAUDE.md / HANDOFF.md / Ref MCP source]
- Guidance: [What best practices recommend]

### Fix Applied
[Before/after code]

### Verification
- [x] Build passes
- [x] Server starts and responds
- [ ] Frontend builds (if applicable)

### Post-Fix Review
- [x] Follows CLAUDE.md patterns
- [x] No security issues
- [x] Minimal change applied
```

### If Issue Was False Positive

```markdown
## Issue Rejected (False Positive)

**Source**: [Tool name]
**Location**: `path/to/file:123`

### Tool's Claim
[What the tool said was wrong]

### Why It's Correct
[Explanation of why current code is fine]

### Context Missed
[What the tool didn't understand about this codebase]
```

---

## Quick Reference

### Reference Priority

1. **CLAUDE.md** — Project constraints and conventions
2. **docs/HANDOFF.md** — Architecture decisions and specifications
3. **Ref MCP** — External framework documentation

### Common False Positive Patterns

| Tool Says | Often Actually |
|-----------|----------------|
| "Unused import" | Used by MLX lazy evaluation or type hints |
| "Dangerous subprocess" | Already uses create_subprocess_exec with allowlist |
| "Missing error handling" | Error handled by FastAPI exception handler |
| "Deprecated API" | Tool has outdated MLX knowledge |
| "Unused state variable" | Used by WebSocket ref pattern |
| "React hook dependency missing" | Intentionally excluded to prevent reconnection loops |

### Files That Need Extra Care

- `train.py` — Must stay under ~600 lines, agent modifies this
- `prepare.py` — FROZEN, never modify
- `server/process_manager.py` — Security-sensitive (subprocess spawning)
- `pyproject.toml` — Dependency changes affect all users

---

## Invocation

**Standard usage**:
```
/bugfix

[Paste issue from CodeRabbit/Bugbot/etc.]
```

**With context**:
```
/bugfix This is from CodeRabbit PR review:

[Paste issue]
```
