"""Call Claude API to generate a plain-language summary of experiment results."""

import json

import anthropic


async def generate_summary(experiments: list[dict]) -> dict:
    if not experiments:
        return {"summary": "No experiments completed yet.", "insights": []}

    kept = [e for e in experiments if e["status"] == "keep"]
    best = min(kept, key=lambda e: e["val_bpb"]) if kept else None

    experiments_text = "\n".join(
        f"- [{e['status'].upper()}] val_bpb={e['val_bpb']:.4f}: {e['description']}"
        for e in experiments
    )

    prompt = f"""You are summarizing an autonomous ML research session on Apple Silicon.
The agent ran {len(experiments)} experiments to minimize val_bpb (lower is better).

Experiments:
{experiments_text}

Write a concise plain-language summary (3-4 sentences) of:
1. What the agent discovered about this hardware
2. What types of changes helped vs. hurt
3. The best result achieved

Then list 2-3 key insights as short bullet points.
Keep the language accessible to someone who doesn't know ML deeply.
Format as JSON: {{"summary": "...", "insights": ["...", "...", "..."]}}"""

    client = anthropic.Anthropic()
    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}],
    )

    try:
        return json.loads(message.content[0].text)
    except Exception:
        return {
            "summary": message.content[0].text,
            "insights": [],
        }
