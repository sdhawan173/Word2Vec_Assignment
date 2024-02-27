import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from gensim import corpora
from gensim.models import LdaModel


try:
    nltk.data.find('tokenizers/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

STOP_WORDS = set(stopwords.words('english'))
STOP_WORDS.add('n\'t')
STOP_WORDS.add('\'s')
STOP_WORDS.add('\'m')
STOP_WORDS.add('\'ll')
STOP_WORDS.add('like')
STOP_WORDS.add('good')
STOP_WORDS.add('also')
STOP_WORDS.add('every')

PUNCTUATION = set(string.punctuation)
PUNCTUATION.add('br')
PUNCTUATION.add('<br>')
PUNCTUATION.add('<\\br>')
PUNCTUATION.add('``')
PUNCTUATION.add('\'\'')
PUNCTUATION.add('...')
PUNCTUATION.add('.........')
PUNCTUATION.add('..........')


def split_sentence(text_data):
    """
    Splits each comment from list of comments into sentences
    :param text_data: list of comments
    :return: list of lists of comments split into sentences
    """
    # Make sure 'punkt' is installed
    print('\nSplitting {} comments by sentences ...'.format(len(text_data)))
    text_split_sentence = text_data
    for index in range(len(text_data)):
        text_split_sentence[index] = nltk.sent_tokenize(text_data[index][0])
    return text_split_sentence


def replace_multiwords(word_list):
    """
    replaces individual words in list with words that may be
    :param word_list: list of words
    :return: list of words with multiword combinations replaced by a single string
    """
    finder = BigramCollocationFinder.from_words(word_list)
    finder.apply_freq_filter(3)
    collocations = finder.nbest(BigramAssocMeasures.pmi, 5)
    for collocation_to_merge in collocations:
        merged_words = []
        i = 0
        while i < len(word_list):
            if i < len(word_list) - 1 and (word_list[i], word_list[i + 1]) == collocation_to_merge:
                merged_words.append(' '.join(collocation_to_merge))
                i += 2
            else:
                merged_words.append(word_list[i])
                i += 1
        word_list = merged_words.copy()
    return word_list


def exclusion_filter(word_list):
    exclusion_list = []
    for word in word_list:
        if word.lower() not in STOP_WORDS and word not in PUNCTUATION:
            exclusion_list.append(word.lower())
    return exclusion_list


def split_words(text_data, exclusion=False):
    """
    Splits each comment from list of comments into words
    :param text_data: list of comments
    :param exclusion: Boolean to exclude STOP_WORDS and PUNCTUATION
    :return: list of lists of comments split into words
    """
    print('\nSplitting {} comments by words ...'.format(len(text_data)))
    if exclusion:
        print('Excluding stop words and punctuation ...')
    text_split_word = []
    for index in range(len(text_data)):
        comment_words = []
        temp = []
        for sentence in text_data[index]:
            if not exclusion:
                temp = nltk.word_tokenize(sentence)
            elif exclusion:
                temp = exclusion_filter(nltk.word_tokenize(sentence))
            for word in temp:
                comment_words.append(word.lower())
        text_split_word.append(comment_words)
    return text_split_word


def lemmatization(text_split_sentence, exclusion=False, multiword=False):
    """
    lemmatizes each comment from list of comments into lemmatized words
    :param text_split_sentence: list of lists of comments split into sentences
    :param exclusion: Boolean to exclude STOP_WORDS and PUNCTUATION
    :param multiword: Boolean to activate replace_multiwords function
    :return: list of lists of comments split into lemmatized words
    """
    print('\nLemmatizing {} comments ... '.format(format(len(text_split_sentence))))
    if exclusion:
        print('Excluding stop words and punctuation ...')
    if multiword:
        print('Replacing multiwords ...')
    lemmatizer = WordNetLemmatizer()
    all_comments_lemmatized = []
    if not multiword:
        for comment in text_split_sentence:
            comment_lemmatized = []
            for sentence in comment:
                words = [word.lower() for word in word_tokenize(sentence)]
                if exclusion:
                    words = exclusion_filter(words.copy())
                for word in words:
                    comment_lemmatized.append(lemmatizer.lemmatize(word))
            all_comments_lemmatized.append(comment_lemmatized)
    elif multiword:
        for comment in text_split_sentence:
            word_list = []
            comment_lemmatized = []
            for sentence in comment:
                word_list += [word.lower() for word in word_tokenize(sentence)]
            word_list = exclusion_filter(word_list.copy())
            word_list = replace_multiwords(word_list.copy())
            for word in word_list:
                comment_lemmatized.append(lemmatizer.lemmatize(word))
            all_comments_lemmatized.append(comment_lemmatized)
    return all_comments_lemmatized


def stemming(text_split_word, exclusion=False, multiword=False):
    """
    stems each comment from list of comments into stemmed words
    :param text_split_word: list of lists of comments split into sentences
    :param exclusion: Boolean to exclude STOP_WORDS and PUNCTUATION
    :param multiword: Boolean to activate replace_multiwords function
    :return: list of comments split into stemmed words
    """
    print('\nStemming {} comments ... '.format(format(len(text_split_word))))
    if exclusion:
        print('Excluding stop words and punctuation ...')
    if multiword:
        print('Replacing multiwords ...')
    stemmer = nltk.PorterStemmer()
    all_comments_stemmed = []
    for word_list in text_split_word:
        temp = [stemmer.stem(word) for word in word_list]
        if exclusion:
            temp = exclusion_filter(temp.copy())
        if multiword:
            temp = replace_multiwords(temp.copy())
        for word in word_list:
            temp.append(stemmer.stem(word))
        all_comments_stemmed.append(temp)
    return all_comments_stemmed


def lda(all_comments, n):
    """

    :param all_comments:
    :param n:
    :return:
    """
    print('\nPerforming Latent Drichlet Allocation ({} topics)...'.format(n))
    lda_dict = corpora.Dictionary(all_comments)
    lda_corpus = []
    for comment in all_comments:
        lda_corpus.append(lda_dict.doc2bow(comment))
    lda_model = LdaModel(lda_corpus, n, id2word=lda_dict)
    for output in lda_model.print_topics():
        print(output)
    return lda_model
