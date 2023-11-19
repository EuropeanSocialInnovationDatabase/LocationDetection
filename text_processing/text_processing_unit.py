import html
import re

import nltk
import unicodedata
from cleantext.sklearn import CleanTransformer

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
from nltk.tokenize import sent_tokenize


def clean_text(raw_text):
    def is_pua(c):
        return unicodedata.category(c) == 'Co'

    if not isinstance(raw_text, str):
        return ''

    try:
        text = html.unescape(raw_text)
    except:
        text = raw_text
    text = text.replace('\xa0', ' ').replace('\r', ' ').replace('&quot', ' ').replace('#', '').replace(
        '=', '')
    text = re.sub(r'(\{|\<|\[)(.*)(\>|\]|\})', '', text)  # text between [], (), {}
    text = re.sub(r'\b\w{20,}\b', '', text)  # long text
    sentences = sent_tokenize(text)
    new_sents = []

    # remove dup sent:
    for sent in sentences:
        if sent not in new_sents:
            new_sents.append(sent)
    if new_sents:
        sentences = new_sents
    text = ' '.join(sentences)
    text = [x for x in text.split('\n') if len(x.split()) > 2]
    text = ' '.join(text)
    text = re.sub('\n', ' ', text)
    text = text.replace('\n', '. ').replace('\t', '')

    text = "".join([char for char in text if not is_pua(char)])

    return text


class TextProcessingUnit:
    # max text length supported by BERT.
    BERT_MAX = 512

    def __init__(self):
        # Load text summarization model

        self.cleaner = CleanTransformer(
            fix_unicode=True,  # fix various unicode errors
            to_ascii=True,  # transliterate to closest ASCII representation
            lower=False,  # lowercase text
            no_line_breaks=False,  # fully strip line breaks as opposed to only normalizing them
            no_urls=True,  # replace all URLs with a special token
            no_emails=True,  # replace all email addresses with a special token
            no_phone_numbers=True,  # replace all phone numbers with a special token
            no_numbers=True,  # replace all numbers with a special token
            no_digits=True,  # replace all digits with a special token
            no_currency_symbols=True,  # replace all currency symbols with a special token
            no_punct=False,  # remove punctuations
            replace_with_punct="",  # instead of removing punctuations you may replace them
            replace_with_url="<URL>",
            replace_with_email="<EMAIL>",
            replace_with_phone_number="<PHONE>",
            replace_with_number="<NUMBER>",
            replace_with_digit="0",
            replace_with_currency_symbol="<CUR>",
            lang="en")

    def clean_text(self, text):

        if not isinstance(text, str):
            return ''

        try:
            text = html.unescape(text)
        except:
            text = text

        text = self.cleaner.transform([text])[0]

        text = text.replace('\xa0', ' ').replace('\r', ' ').replace('&quot', ' ').replace('#', '').replace(
            '=', '')
        text = re.sub(r'(\{|\[)(.*)(|\]|\})', '', text)  # text between [], (), {}
        text = re.sub(r'\b\w{20,}\b', '', text)  # long text

        return text


if __name__ == "__main__":
    clr = TextProcessingUnit()
    te = clr.clean_text('This is some text to Beee cleanned')
    print(te)
