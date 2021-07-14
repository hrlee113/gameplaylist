import pandas as pd
from prep.game_prep import clear_genres, remove_title
from sklearn.model_selection import train_test_split
from gensim.corpora.dictionary import Dictionary
from gensim.models.ldamodel import LdaModel
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')



def make_dic_corpus(texts):
    # dictionary 
    docs = []
    for text in texts:
        doc = text.split(', ')
        docs.append(doc)
    dic = Dictionary(docs)
    dic.filter_extremes(no_below=10, no_above=0.8)
    # corpus
    corpus = [dic.doc2bow(doc) for doc in docs]
    return dic, corpus


def topic_lda(text):
    dic, corpus = make_dic_corpus(text)
    train_corpus, valid_corpus = train_test_split(corpus, test_size=0.1, random_state=1234)
    model = LdaModel(corpus=train_corpus, id2word=dic, num_topics=18, random_state=1234, passes = 32)   
    topic = []
    for i in range(18): # Total number of topics : 18
        res = []
        for j in range(20): # One topic includes 20 genres
            res.append(model.show_topic(i, 20)[j][0])
        topic.append(res)
    topic = pd.DataFrame(topic).T
    topic.columns = ['Topic {}'.format(i) for i in range(18)]
    topic_table = pd.DataFrame()
    for i, topic_list in enumerate(model[corpus]):
        doc = topic_list[0] if model.per_word_topics else topic_list            
        doc = sorted(doc, key=lambda x: (x[1]), reverse=True)
        for j, (topic_num, prop_topic) in enumerate(doc):
            if j == 0:  
                topic_table = topic_table.append(pd.Series([int(topic_num), round(prop_topic,4), topic_list]), ignore_index=True)
            else:
                break
    return topic, topic_table


def genre_lda(game):
    # 게임 장르 LDA -> 18개의 topic (장르 묶음) 생성
    removed_data = clear_genres(game)
    topic, topic_table = topic_lda(removed_data)
    # genre_topic = [game_id, topic_num, topic]
    # ----- game_id : 게임 고유번호
    # ----- topic_num : topic 고유번호
    # ----- topic : 해당 게임과 가장 관련성이 높은 genre 집합
    genre_topic = game[['game_id']]
    topic_num = topic_table.iloc[:,0]
    genre_topic['topic_num'] = topic_num
    genre_topic['topic'] = ''
    f = topic[:5]
    for i in range(len(genre_topic)):
        genre_topic.loc[i, 'topic'] = list(f.iloc[:,genre_topic.loc[i, 'topic_num']])
    return genre_topic
    

def content_lda(game):
    # 게임 설명 LDA -> 110개의 topic (설명 단어 묶음) 생성
    removed_data = remove_title(game)
    tokens = []
    for sen in removed_data:
        token = nltk.word_tokenize(sen)
        tagged = nltk.pos_tag(token)
        tokens.append(tagged)
    noun_verb = []
    for sen in tokens:
        sentence = [word for word, pos in sen if pos in ['NN', 'NNS', 'VB']]
        noun_verb.append(sentence)
    topic, content_topic = topic_lda(noun_verb)
    content_topic = content_topic.reset_index() 
    content_topic.columns = ['game_index', '1st_topic', '1st_topic_ratio', 'topics_ratio']
    content_topic['1st_topic_ratio'] = content_topic['1st_topic_ratio'].astype(int)
    return content_topic


def review_lda(review, content_topic, genre_topic):
    review['game_id'] = ''
    review['content_lda'] = ''
    review['game_id'] = content_topic['game_index'].apply(lambda x : review.loc[x, 'game_id'])
    for i in range(len(review)):
        topic = content_topic[content_topic.game_id==review.loc[i, 'game_id']]
        review['content_lda'] = topic.loc[0, '1st_topic']
    review['genre_lda'] = review.merge(genre_topic[['game_id']], on='game_id', how='left')
    return review