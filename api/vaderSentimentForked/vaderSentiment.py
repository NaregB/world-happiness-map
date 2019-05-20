# coding: utf-8
# Author: C.J. Hutto
# Thanks to George Berry for reducing the time complexity from something like O(N^4) to O(N).
# Thanks to Ewan Klein and Pierpaolo Pantone for bringing VADER into NLTK. Those modifications were awesome.
# For license information, see LICENSE.TXT

"""
If you use the VADER sentiment analysis tools, please cite:
Hutto, C.J. & Gilbert, E.E. (2014). VADER: A Parsimonious Rule-based Model for
Sentiment Analysis of Social Media Text. Eighth International Conference on
Weblogs and Social Media (ICWSM-14). Ann Arbor, MI, June 2014.
"""
import math
import os
import re
import string
from inspect import getsourcefile
from io import open
from itertools import product

# ##Constants##

# (empirically derived mean sentiment intensity rating increase for booster words)
B_INCR = 0.293
B_DECR = -0.293

# (empirically derived mean sentiment intensity rating increase for using ALLCAPs to emphasize a word)
C_INCR = 0.733
N_SCALAR = -0.74

# for removing punctuation
REGEX_REMOVE_PUNCTUATION = re.compile('[%s]' % re.escape(string.punctuation))

PUNC_LIST = [".", "!", "?", ",", ";", ":", "-", "'", "\"",
             "!!", "!!!", "??", "???", "?!?", "!?!", "?!?!", "!?!?"]
PUNC_LIST_STOP_NEGATION = [".", "!", "?", ",", ";", ":", "-",
                           "!!", "!!!", "??", "???", "?!?", "!?!", "?!?!", "!?!?"]

NEGATE = \
    ["aint", "arent", "cannot", "cant", "couldnt", "darent", "didnt", "doesnt",
     "ain't", "aren't", "can't", "couldn't", "daren't", "didn't", "doesn't",
     "dont", "hadnt", "hasnt", "havent", "isnt", "mightnt", "mustnt", "neither",
     "don't", "hadn't", "hasn't", "haven't", "isn't", "mightn't", "mustn't",
     "neednt", "needn't", "never", "none", "nope", "no", "nor", "not", "nothing", "nowhere",
     "oughtnt", "shant", "shouldnt", "uhuh", "wasnt", "werent",
     "oughtn't", "shan't", "shouldn't", "uh-uh", "wasn't", "weren't",
     "without", "wont", "wouldnt", "won't", "wouldn't", "rarely", "seldom", "despite", "least"]

# booster/dampener 'intensifiers' or 'degree adverbs'
# http://en.wiktionary.org/wiki/Category:English_degree_adverbs

BOOSTER_DICT = \
    {"absolutely": B_INCR, "amazingly": B_INCR, "awfully": B_INCR, "completely": B_INCR, "considerably": B_INCR,
     "decidedly": B_INCR, "deeply": B_INCR, "effing": B_INCR, "enormously": B_INCR,
     "entirely": B_INCR, "especially": B_INCR, "exceptionally": B_INCR, "extremely": B_INCR,
     "fabulously": B_INCR, "flipping": B_INCR, "flippin": B_INCR,
     "fricking": B_INCR, "frickin": B_INCR, "frigging": B_INCR, "friggin": B_INCR, "fully": B_INCR, "fucking": B_INCR,
     "greatly": B_INCR, "hella": B_INCR, "highly": B_INCR, "hugely": B_INCR, "incredibly": B_INCR,
     "intensely": B_INCR, "majorly": B_INCR, "more": B_INCR, "most": B_INCR, "particularly": B_INCR,
     "purely": B_INCR, "quite": B_INCR, "really": B_INCR, "remarkably": B_INCR,
     "so": B_INCR, "substantially": B_INCR, "without doubt": B_INCR,
     "thoroughly": B_INCR, "totally": B_INCR, "tremendously": B_INCR,
     "uber": B_INCR, "unbelievably": B_INCR, "unusually": B_INCR, "utterly": B_INCR,
     "very": B_INCR, "sofa king": B_INCR,
     "almost": B_DECR, "barely": B_DECR, "hardly": B_DECR, "just enough": B_DECR,
     "kind of": B_DECR, "kinda": B_DECR, "kindof": B_DECR, "kind-of": B_DECR,
     "less": B_DECR, "little": B_DECR, "marginally": B_DECR, "occasionally": B_DECR, "partly": B_DECR,
     "scarcely": B_DECR, "slightly": B_DECR, "somewhat": B_DECR,
     "sort of": B_DECR, "sorta": B_DECR, "sortof": B_DECR, "sort-of": B_DECR}

# check for idioms
IDIOM_DICT = {"cut the mustard": 2, "hand to mouth": -2,
              "back handed": -2, "blow smoke": -2, "blowing smoke": -2,
              "upper hand": 1, "break a leg": 2,
              "cooking with gas": 2, "in the black": 2, "in the red": -2,
              "on the ball": 2, "under the weather": -2, "the shit": 3, "the bomb": 3, "bad ass": 1.5, "yeah right": -2,
              "kiss of death": -1.5}


# #Static methods# #

def negated(input_words, include_nt=True):
    """
    Determine if input contains negation words
    """
    input_words = [str(w).lower() for w in input_words]
    neg_words = []
    neg_words.extend(NEGATE)
    for word in neg_words:
        if word in input_words:
            return True
    if include_nt:
        for word in input_words:
            if "n't" in word:
                return True
    if "least" in input_words:
        i = input_words.index("least")
        if i > 0 and input_words[i - 1] != "at":
            return True
    return False


def normalize(score, alpha=15):
    """
    Normalize the score to be between -1 and 1 using an alpha that
    approximates the max expected value
    """
    norm_score = score / math.sqrt((score * score) + alpha)
    if norm_score < -1.0:
        return -1.0
    elif norm_score > 1.0:
        return 1.0
    else:
        return norm_score


def allcap_differential(words):
    """
    Check whether just some words in the input are ALL CAPS
    :param list words: The words to inspect
    :returns: `True` if some but not all items in `words` are ALL CAPS
    """
    is_different = False
    allcap_words = 0
    for word in words:
        if word.isupper():
            allcap_words += 1
    cap_differential = len(words) - allcap_words
    if 0 < cap_differential < len(words):
        is_different = True
    return is_different


def scalar_inc_dec(word, valence, is_cap_diff):
    """
    Check if the preceding words increase, decrease, or negate/nullify the
    valence
    """
    scalar = 0.0
    word_lower = word.lower()
    if word_lower in BOOSTER_DICT:
        scalar = BOOSTER_DICT[word_lower]
        if valence < 0:
            scalar *= -1
        # check if booster/dampener word is in ALLCAPS (while others aren't)
        if word.isupper() and is_cap_diff:
            if valence > 0:
                scalar += C_INCR
            else:
                scalar -= C_INCR
    return scalar


class SentiText(object):
    """
    Identify sentiment-relevant string-level properties of input text.
    """

    def __init__(self, text):
        if not isinstance(text, str):
            text = str(text).encode('utf-8')
        self.text = text
        self.shortened_text = []
        for text in text.split():
            if len(text) > 1 or text.endswith(tuple(PUNC_LIST_STOP_NEGATION)):
                self.shortened_text.append(text)
        self.words_and_emoticons = self._words_and_emoticons()
        # doesn't separate words from\
        # adjacent punctuation (keeps emoticons & contractions)
        self.is_cap_diff = allcap_differential(self.words_and_emoticons)

    def _words_plus_punc(self):
        """
        Returns mapping of form:
        {
            'cat,': 'cat',
            ',cat': 'cat',
        }
        """
        no_punc_text = REGEX_REMOVE_PUNCTUATION.sub('', self.text)
        # removes punctuation (but loses emoticons & contractions)
        words_only = no_punc_text.split()
        # remove singletons
        words_only = set(w for w in words_only if len(w) > 1)
        # the product gives ('cat', ',') and (',', 'cat')
        punc_before = {''.join(p): p[1] for p in product(PUNC_LIST, words_only)}
        punc_after = {''.join(p): p[0] for p in product(words_only, PUNC_LIST)}
        words_punc_dict = punc_before
        words_punc_dict.update(punc_after)
        return words_punc_dict

    def _words_and_emoticons(self):
        """
        Removes leading and trailing puncutation
        Leaves contractions and most emoticons
            Does not preserve punc-plus-letter emoticons (e.g. :D)
        """
        text = self.text

        if "kind of" in text.lower():
            pattern = re.compile("kind of", re.IGNORECASE)
            text = pattern.sub("kinda", text)

        if "sort of" in text.lower():
            pattern = re.compile("sort of", re.IGNORECASE)
            text = pattern.sub("sorta", text)

        if "without doubt" in text.lower():
            pattern = re.compile("without doubt", re.IGNORECASE)
            text = pattern.sub("absolutely", text)

        wes = text.split()

        for idiom in IDIOM_DICT:
            if idiom in text.lower():
                index = wes.index(idiom.split(' ', 1)[0])
                length = len(idiom.split())
                wes[index] = idiom
                for i in range(length - 1):
                    del wes[index + 1]

        words_punc_dict = self._words_plus_punc()
        wes = [we for we in wes if len(we) > 1]
        for i, we in enumerate(wes):
            if we in words_punc_dict:
                wes[i] = words_punc_dict[we]
        return wes


class SentimentIntensityAnalyzer(object):
    """
    Give a sentiment intensity score to sentences.
    """

    def __init__(self, lexicon_file="vader_lexicon.txt", emoji_lexicon="emoji_utf8_lexicon.txt"):
        _this_module_file_path_ = os.path.abspath(getsourcefile(lambda: 0))
        lexicon_full_filepath = os.path.join(os.path.dirname(_this_module_file_path_), lexicon_file)
        with open(lexicon_full_filepath, encoding='utf-8') as f:
            self.lexicon_full_filepath = f.read()
        self.lexicon = self.make_lex_dict()

        emoji_full_filepath = os.path.join(
            os.path.dirname(_this_module_file_path_), emoji_lexicon)
        with open(emoji_full_filepath, encoding='utf-8') as f:
            self.emoji_full_filepath = f.read()
        self.emojis = self.make_emoji_dict()

    def make_lex_dict(self):
        """
        Convert lexicon file to a dictionary
        """
        lex_dict = {}
        for line in self.lexicon_full_filepath.split('\n'):
            (word, measure) = line.strip().split('\t')[0:2]
            lex_dict[word] = float(measure)
        return lex_dict

    def make_emoji_dict(self):
        """
        Convert emoji lexicon file to a dictionary
        """
        emoji_dict = {}
        for line in self.emoji_full_filepath.split('\n'):
            (emoji, description) = line.strip().split('\t')[0:2]
            emoji_dict[emoji] = description
        return emoji_dict

    def polarity_scores(self, text):
        """
        Return a float for sentiment strength based on the input text.
        Positive values are positive valence, negative value are negative
        valence.
        """
        # convert emojis to their textual descriptions
        text_token_list = text.split()
        text_no_emoji_lst = []
        for token in text_token_list:
            if token in self.emojis:
                # get the textual description
                description = self.emojis[token]
                text_no_emoji_lst.append(description)
            else:
                text_no_emoji_lst.append(token)
        text = " ".join(x for x in text_no_emoji_lst)

        sentitext = SentiText(text)

        sentiments = []
        words_and_emoticons = sentitext.words_and_emoticons
        for item in words_and_emoticons:
            valence = 0
            if item.lower() in IDIOM_DICT:
                sentiments.append(IDIOM_DICT[item.lower()])
                continue
            else:
                sentiments.append(valence)
                continue

        sentiments = self.sentiment_val(sentitext, sentiments)

        sentiments = self.booster_check(sentitext, sentiments)

        sentiments = self.negation_check(sentitext, sentiments)

        sentiments = self._but_check(words_and_emoticons, sentiments)

        valence_dict = self.score_valence(sentiments, text)

        return valence_dict

    def sentiment_val(self, sentitext, sentiments):
        is_cap_diff = sentitext.is_cap_diff
        words_and_emoticons = sentitext.words_and_emoticons

        for i, word in enumerate(words_and_emoticons):
            word_lower = word.lower()
            valence = 0
            if word_lower in self.lexicon:
                valence = self.lexicon[word_lower]
                if word.isupper and is_cap_diff:
                    if valence > 0:
                        valence += C_INCR
                    else:
                        valence -= C_INCR
            sentiments[i] = sentiments[i] + valence

        return sentiments

    def booster_check(self, sentitext, sentiments):
        is_cap_diff = sentitext.is_cap_diff
        words_and_emoticons = sentitext.words_and_emoticons

        for i, score in enumerate(sentiments):
            if score != 0:
                valence = score
                for start_i in range(0, 3):
                    # dampen the scalar modifier of preceding words and emoticons
                    # (excluding the ones that immediately preceed the item) based
                    # on their distance from the current item.
                    if i > start_i and words_and_emoticons[i - (start_i + 1)].lower() not in self.lexicon:
                        s = scalar_inc_dec(words_and_emoticons[i - (start_i + 1)], valence, is_cap_diff)
                        if start_i == 1 and s != 0:
                            s = s * 0.95
                        if start_i == 2 and s != 0:
                            s = s * 0.9
                        valence = valence + s
                        sentiments[i] = valence
        return sentiments

    def negation_check(self, sentitext, sentiments):
        words_with_punc = sentitext.shortened_text
        words_and_emoticons = sentitext.words_and_emoticons

        i = 0
        while i < len(words_and_emoticons) - 1:
            word_unchanged = words_and_emoticons[i].lower()
            # remove punctuation for contractions
            word_lower = re.sub("[^a-zA-Z' ]+", '', word_unchanged)
            if word_lower in NEGATE or word_lower.endswith("n't"):
                if word_lower != 'least' and word_lower != 'without':
                    start_i = i
                    for j in range(0, len(words_with_punc) - 1):
                        if re.sub("[^a-zA-Z' ]+", '', words_with_punc[j]) == re.sub("[^a-zA-Z' ]+", '', word_unchanged):
                            start_i = j
                            break
                    if start_i < i:
                        start_i = i
                    end_i = len(words_and_emoticons) - 1

                    for k in range(start_i + 1, len(words_with_punc)):
                        word = words_with_punc[k]
                        if word.endswith(tuple(PUNC_LIST_STOP_NEGATION)) or word.startswith(
                                tuple(PUNC_LIST_STOP_NEGATION)):
                            end_i = k
                            break
                    i = end_i
                    if end_i > len(words_and_emoticons) - 1:
                        end_i = len(words_and_emoticons) - 1
                    while start_i <= end_i:
                        if sentiments[start_i] != 0:
                            sentiments[start_i] = sentiments[start_i] * N_SCALAR
                        start_i += 1
                elif word_lower == 'least' or word_lower == 'without':
                    if word_lower == 'least' and (words_and_emoticons[i - 1] != 'at' and words_and_emoticons[i - 1] != "very"):#checked case
                        start_i = i
                        for j in range(0, len(words_with_punc) - 1):
                            if re.sub("[^a-zA-Z' ]+", '', words_with_punc[j]) == re.sub("[^a-zA-Z' ]+", '',
                                                                                        word_unchanged):
                                start_i = j
                                break
                        if start_i < i:
                            start_i = i
                        end_i = len(words_and_emoticons) - 1

                        for k in range(start_i + 1, len(words_with_punc)):
                            word = words_with_punc[k]
                            if word.endswith(tuple(PUNC_LIST_STOP_NEGATION)) or word.startswith(
                                    tuple(PUNC_LIST_STOP_NEGATION)):
                                end_i = k
                                break
                        i = end_i
                        if end_i > len(words_and_emoticons) - 1:
                            end_i = len(words_and_emoticons) - 1
                        while start_i <= end_i:
                            if sentiments[start_i] != 0:
                                sentiments[start_i] = sentiments[start_i] * N_SCALAR
                            start_i += 1
                    elif word_lower == 'without' and (words_and_emoticons[i + 1] != "doubt" and words_and_emoticons[i + 2] != "doubt"):
                        start_i = i
                        for j in range(0, len(words_with_punc) - 1):
                            if re.sub("[^a-zA-Z' ]+", '', words_with_punc[j]) == re.sub("[^a-zA-Z' ]+", '',
                                                                                        word_unchanged):
                                start_i = j
                                break
                        if start_i < i:
                            start_i = i
                        end_i = len(words_and_emoticons) - 1

                        for k in range(start_i + 1, len(words_with_punc)):
                            word = words_with_punc[k]
                            if word.endswith(tuple(PUNC_LIST_STOP_NEGATION)) or word.startswith(
                                    tuple(PUNC_LIST_STOP_NEGATION)):
                                end_i = k
                                break
                        i = end_i
                        if end_i > len(words_and_emoticons) - 1:
                            end_i = len(words_and_emoticons) - 1
                        while start_i <= end_i:
                            if sentiments[start_i] != 0:
                                sentiments[start_i] = sentiments[start_i] * N_SCALAR
                            start_i += 1
                    else:
                        i += 1
            else:
                i += 1

        return sentiments

    @staticmethod
    def _but_check(words_and_emoticons, sentiments):
        # check for modification in sentiment due to contrastive conjunction 'but'
        words_and_emoticons_lower = [str(w).lower()
                                     for w in words_and_emoticons]
        if 'but' in words_and_emoticons_lower:
            bi = words_and_emoticons_lower.index('but')
            for sentiment in sentiments:
                si = sentiments.index(sentiment)
                if si < bi:
                    sentiments.pop(si)
                    sentiments.insert(si, sentiment * 0.5)
                elif si > bi:
                    sentiments.pop(si)
                    sentiments.insert(si, sentiment * 1.5)
        return sentiments

    def _punctuation_emphasis(self, text):
        # add emphasis from exclamation points and question marks
        ep_amplifier = self._amplify_ep(text)
        qm_amplifier = self._amplify_qm(text)
        punct_emph_amplifier = ep_amplifier + qm_amplifier
        return punct_emph_amplifier

    @staticmethod
    def _amplify_ep(text):
        # check for added emphasis resulting from exclamation points (up to 4 of them)
        ep_count = text.count("!")
        if ep_count > 4:
            ep_count = 4
        # (empirically derived mean sentiment intensity rating increase for
        # exclamation points)
        ep_amplifier = ep_count * 0.292
        return ep_amplifier

    @staticmethod
    def _amplify_qm(text):
        # check for added emphasis resulting from question marks (2 or 3+)
        qm_count = text.count("?")
        qm_amplifier = 0
        if qm_count > 1:
            if qm_count <= 3:
                # (empirically derived mean sentiment intensity rating increase for
                # question marks)
                qm_amplifier = qm_count * 0.18
            else:
                qm_amplifier = 0.96
        return qm_amplifier

    @staticmethod
    def _sift_sentiment_scores(sentiments):
        # want separate positive versus negative sentiment scores
        pos_sum = 0.0
        neg_sum = 0.0
        neu_count = 0
        for sentiment_score in sentiments:
            if sentiment_score > 0:
                # compensates for neutral words that are counted as 1
                pos_sum += (float(sentiment_score) + 1)
            if sentiment_score < 0:
                # when used with math.fabs(), compensates for neutrals
                neg_sum += (float(sentiment_score) - 1)
            if sentiment_score == 0:
                neu_count += 1
        return pos_sum, neg_sum, neu_count

    def score_valence(self, sentiments, text):
        if sentiments:
            sum_s = float(sum(sentiments))
            # compute and add emphasis from punctuation in text
            punct_emph_amplifier = self._punctuation_emphasis(text)
            if sum_s > 0:
                sum_s += punct_emph_amplifier
            elif sum_s < 0:
                sum_s -= punct_emph_amplifier

            compound = normalize(sum_s)
            # discriminate between positive, negative and neutral sentiment scores
            pos_sum, neg_sum, neu_count = self._sift_sentiment_scores(
                sentiments)

            if pos_sum > math.fabs(neg_sum):
                pos_sum += punct_emph_amplifier
            elif pos_sum < math.fabs(neg_sum):
                neg_sum -= punct_emph_amplifier

            total = pos_sum + math.fabs(neg_sum) + neu_count
            pos = math.fabs(pos_sum / total)
            neg = math.fabs(neg_sum / total)
            neu = math.fabs(neu_count / total)

        else:
            compound = 0.0
            pos = 0.0
            neg = 0.0
            neu = 0.0

        sentiment_dict = \
            {"neg": round(neg, 3),
             "neu": round(neu, 3),
             "pos": round(pos, 3),
             "compound": round(compound, 4)}

        return sentiment_dict
