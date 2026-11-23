import emoji

def convert_emojis_to_text(text):
    text = emoji.demojize(text)

    text = text.replace(":", " ")
    text = text.replace("_", " ")

    return " ".join(text.split())


def remove_emojis(text):
    text =  emoji.replace_emoji(text, replace="")

    return " ".join(text.split())