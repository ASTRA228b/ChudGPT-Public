"""Select a neural conversation checkpoint by topic, never by generated output."""
import re
from chudlm.emoji_awareness import strip_emoji_context

TOPIC = re.compile(r"\b(?:gay|lesbian|bisexual|pansexual|asexual|aromantic|agender|genderfluid|trans|transgender|nonbinary|non-binary|intersex|queer|femboys?|lgbt(?:q(?:ia)?)?|sexuality|sexual orientation|gender identity|coming out|come out)\b", re.I)


def lgbtq_conversation(message, history=()):
    text = strip_emoji_context(message).strip()
    # Explicit tasks still use the normal model, even when they mention a label.
    if re.search(r"```|\b(?:write|code|debug|translate|poem|function|script|mod|game)\b", text, re.I):
        return False
    if TOPIC.search(text) or re.search(r"\b(?:im|i'm|i am|are you) (?:bi|pan|ace)\b", text, re.I):
        return True
    if re.fullmatch(r"(?:why|how so|what do you mean|really|okay|ok|okay cool|thanks|and you|what about you)[?.! ]*", text, re.I):
        previous = next((t['content'] for t in reversed(history) if t.get('role') == 'user'), '')
        return bool(TOPIC.search(strip_emoji_context(previous)))
    return False
