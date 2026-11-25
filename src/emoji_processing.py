"""Emoji conditions for the ablation. Only posts that contain an emoji are changed.

Earlier versions removed every ':' and '_' and collapsed whitespace in every post, so
the "described" and "removed" conditions also altered posts with no emoji. Here each
emoji is replaced in place, and whitespace is normalized only in posts that had one.
"""
import emoji


def _edit(text, replacement):
    found = emoji.emoji_list(text)
    if not found:
        return text  # posts without emoji are left byte-identical
    out, last = [], 0
    for item in found:
        out.append(text[last:item["match_start"]])
        out.append(replacement(item["emoji"]))
        last = item["match_end"]
    out.append(text[last:])
    return " ".join("".join(out).split())


def _describe(char):
    name = emoji.demojize(char).strip(":").replace("_", " ")
    return f" {name} "


def convert_emojis_to_text(text):
    """Replace each emoji with its English name, e.g. 😀 -> 'grinning face'."""
    return _edit(text, _describe)


def remove_emojis(text):
    """Delete each emoji."""
    return _edit(text, lambda char: " ")
