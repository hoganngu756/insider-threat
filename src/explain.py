import anthropic


def explain_alert(evidence: dict) -> str:
    """Ask the LLM why this alert is suspicious and what the analyst should check next."""
    client = anthropic.Anthropic()  # needs ANTHROPIC_API_KEY
    msg = client.messages.create(
        model="claude-haiku-5-5",
        max_tokens=300,
        messages=[{
            "role": "user",
            "content": f"Security alert evidence: {evidence}\n"
                       "In 3 sentences: why is this suspicious, and what should an analyst investigate next?",
        }],
    )
    return msg.content[0].text
