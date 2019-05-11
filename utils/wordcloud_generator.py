import base64
from io import BytesIO
from wordcloud import WordCloud


def wordcloud_generator(text):
    wc = WordCloud(background_color="white", max_words=2000)
    wc.generate(text)
    image = wc.to_image()

    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue())

    return img_str
