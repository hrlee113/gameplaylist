from gameplaylist.prep.game_prep import clear_genres
import re
import nltk
# nltk.download('stopwords')
# nltk.download('opinion_lexicon')
from nltk.corpus import stopwords, opinion_lexicon



def pre_sentiment(review):
    # 알파벳, 숫자만 남기기
    review_list = []
    for data in review['content']:
        data = re.sub('[^a-zA-Z]', ' ', data)
        data = re.sub('  ', ' ', data)
        review_list.append(data)
    # 소문자로 변환
    lowercase = []
    for data in review_list:
        lowercase.append(data.lower().split())
    # 불용어 제거
    stop = set(stopwords.words('english'))
    clean_review = []
    for data in lowercase:
        no_stops = [word for word in data if not word in stop]
        clean_review.append(no_stops)
    return clean_review

def extract_ugan(clean_review):
    stemmer = nltk.stem.SnowballStemmer('english')
    stemmer_words = []
    for review in clean_review:
        words = [stemmer.stem(word) for word in review]
        stemmer_words.append(words)
    return stemmer_words

def content_len(stemmer_words):
    data = re.sub('[^a-zA-Z0-9]', ' ', stemmer_words)
    data = data.split()
    return len(data)

def sentiment_analysis(stemmer_words):
    pos_list = set(opinion_lexicon.positive()) # dictionary에서 긍정적 단어
    neg_list = set(opinion_lexicon.negative()) # dictionary에서 부정적 단어
    senti = 0
    for word in stemmer_words:
        if word in pos_list:
            senti += 1
        elif word in neg_list:
            senti -= 1
    return senti

def sentiment_score(review):
    clean_review = pre_sentiment(review)
    stemmer_words = extract_ugan(clean_review)
    sentiment = sentiment_analysis(stemmer_words)
    content_length = content_len(stemmer_words)
    
    scaled_sentiment = []
    for t in range(len(sentiment)):
        score = sentiment[t] / content_length[t]
        scaled_sentiment.append(score)
    review['scaled_sentiment'] = scaled_sentiment
    
    return review
    