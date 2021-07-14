import numpy as np
import tensorflow as tf
tf.keras.backend.set_floatx('float32')
from tensorflow.keras.metrics import BinaryAccuracy, AUC, Precision, Recall
from model.model_prep import get_data



class FM_layer(tf.keras.layers.Layer):
    def __init__(self, num_feature, num_field, embedding_size, field_index, mul_feature):
        super(FM_layer, self).__init__()
        self.embedding_size = embedding_size    # 임베딩 벡터의 차원(크기)
        self.num_feature = num_feature          # 원래 feature 개수
        self.num_field = num_field              # grouped field 개수
        self.field_index = field_index          # 인코딩된 X의 칼럼들이 본래 어디 소속이었는지
        self.mul_feature = mul_feature          # 게임 임베딩 차원(크기)

        # Parameters of FM Layer
        # w: 유저, 게임 메타정보간 선형 관계 파악 (1st interactions term)
        # V: 유저, 게임 메타정보간 비선형 관계 파악 (2nd interactions term)
        # m: 게임 임베딩간 선형 관계 파악 (1st interactions term)
        # M: 게임 임베딩간 선형 관계 파악 (2nd interactions term)

        self.w = tf.Variable(tf.random.normal(shape=[num_feature],
                                              mean=0.0, stddev=1.0), name='w')
        self.V = tf.Variable(tf.random.normal(shape=(num_field, embedding_size),
                                              mean=0.0, stddev=0.01), name='V')
        self.m = tf.Variable(tf.random.normal(shape=[mul_feature],
                                              mean=0.0, stddev=1.0, name='m'))
        self.M = tf.Variable(tf.random.normal(shape=[mul_feature],
                                              mean=0.0, stddev=1.0, name='M'))

    def call(self, input):

        inputs = input[:, :self.num_feature]
        multi_inputs = input[:, self.num_feature:]

        x_batch = tf.reshape(inputs, [-1, self.num_feature, 1])
        # Parameter V를 field_index에 맞게 복사하여 num_feature에 맞게 늘림
        embeds = tf.nn.embedding_lookup(params=self.V, ids=self.field_index)

        # Deep Component에서 쓸 Input
        # (batch_size, num_feature, embedding_size)
        new_inputs = tf.math.multiply(x_batch, embeds)
        multi_inputs = tf.math.multiply(self.M, multi_inputs)

        # (batch_size, )
        linear_terms = tf.reduce_sum(
            tf.math.multiply(self.w, inputs), axis=1, keepdims=False) # element-wise

        # (batch_size, )
        interactions = 0.5 * tf.subtract(
            tf.square(tf.reduce_sum(new_inputs, [1, 2])),
            tf.reduce_sum(tf.square(new_inputs), [1, 2])
        )

        # (batch_size, )
        multi_terms = tf.reduce_sum(
            tf.math.multiply(self.m, multi_inputs), axis=1, keepdims=False
        )

        # (batch_size, )
        multiintersections = 0.5 * tf.subtract(
            tf.square(tf.reduce_sum(multi_inputs, 1)),
            tf.reduce_sum(tf.square(multi_inputs), 1)
        )


        linear_terms = tf.reshape(linear_terms, [-1, 1]) # 유저, 게임 정보 컬럼의 선형 관계 term
        interactions = tf.reshape(interactions, [-1, 1]) # 유저, 게임 정보 컬럼의 비선형 관계 term
        multimodal_terms = tf.reshape(multi_terms, [-1, 1]) # 게임 임베딩 벡터의 선형 관계 term
        multimodal_inters = tf.reshape(multiintersections, [-1,1]) # (batch_size, 1) # 게임 임베딩 벡터의 비선형 관계 term

        y_fm = tf.concat([multimodal_terms, multimodal_inters, linear_terms, interactions], 1) # (batch_size, 4)

        return y_fm, new_inputs # new_inputs: Deep Component의 input으로 쓰일 개체


class DeepFM(tf.keras.Model):

    def __init__(self, num_feature, num_field, embedding_size, field_index):
        super(DeepFM, self).__init__()
        self.embedding_size = embedding_size    # k: 임베딩 벡터의 차원(크기)
        self.num_feature = num_feature          # f: 원래 feature 개수
        self.num_field = num_field              # m: grouped field 개수
        self.field_index = field_index          # 인코딩된 X의 칼럼들이 본래 어디 소속이었는지

        self.fm_layer = FM_layer(num_feature, num_field, embedding_size, field_index, mul_feature=556)

        self.layers1 = tf.keras.layers.Dense(units=64, activation='relu')
        self.dropout1 = tf.keras.layers.Dropout(rate=0.1)
        self.layers2 = tf.keras.layers.Dense(units=16, activation='relu')
        self.dropout2 = tf.keras.layers.Dropout(rate=0.1)
        self.layers3 = tf.keras.layers.Dense(units=4, activation='relu')

        self.final = tf.keras.layers.Dense(units=1, activation='sigmoid')

    def __repr__(self):
        return "DeepFM Model: #Field: {}, #Feature: {}, ES: {}".format(
            self.num_field, self.num_feature, self.embedding_size)

    def call(self, inputs):
        # 1) FM Component: (num_batch, 2)
        y_fm, new_inputs = self.fm_layer(inputs)

        # retrieve Dense Vectors: (num_batch, num_feature*embedding_size)
        new_inputs = tf.reshape(new_inputs, [-1, self.num_feature*self.embedding_size])

        # 2) Deep Component
        y_deep = self.layers1(new_inputs)
        y_deep = self.dropout1(y_deep)
        y_deep = self.layers2(y_deep)
        y_deep = self.dropout2(y_deep)
        y_deep = self.layers3(y_deep)

        # Concatenation
        y_pred = tf.concat([y_fm, y_deep], 1)
        y_pred = self.final(y_pred)
        y_pred = tf.reshape(y_pred, [-1, ])

        return y_pred


def train_on_batch(model, optimizer, acc, auc, pc, rc, x_train, y_train): # Batch 단위 학습
    with tf.GradientTape() as tape:
        y_pred = model(x_train)
        loss = tf.keras.losses.binary_crossentropy(from_logits=False, y_true=y_train, y_pred=y_pred)

    grads = tape.gradient(target=loss, sources=model.trainable_variables)

    # apply_gradients()를 통해 processed gradients를 적용함
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    # accuracy & auc
    acc.update_state(y_train, y_pred)
    auc.update_state(y_train, y_pred)
    pc.update_state(y_train, y_pred)
    rc.update_state(y_train, y_pred)

    return loss


def f1_score(precision, recall, eps=1e-6):
    f1 = (2 * precision * recall) / (precision + recall + eps)
    return f1


def deepfm_run(train_modified, val_modified, test_modified, gamevec):
    # 전체 feature
    ALL_FIELDS = train_modified.columns
    # 범주형 feature
    CAT_FIELDS = ['topic', 'publisher_100', 'developer_100', 'preference_rec_genre']
    # ID feature
    ID_FIELDS = ['label_encode_game_id', 'label_encode_user_id', 'recommended']
    # 연속형 feature
    CONT_FIELDS = list(set(ALL_FIELDS).difference(CAT_FIELDS).difference(ID_FIELDS))

    # Hyper-parameters for Experiment
    NUM_BIN = 10
    BATCH_SIZE = 256
    EMBEDDING_SIZE = 5

    train_ds, val_ds, test_ds, field_dict, field_index = get_data(train_modified, val_modified, test_modified, gamevec, CONT_FIELDS, CAT_FIELDS, BATCH_SIZE)

    model = DeepFM(embedding_size=EMBEDDING_SIZE, num_feature=len(field_index), num_field=len(field_dict), field_index=field_index)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.05)
    epochs = 30

    # 학습
    print("Start Training: Batch Size: {}, Embedding Size: {}".format(BATCH_SIZE, EMBEDDING_SIZE))
    pred_list = list()

    for i in range(epochs):
        acc = BinaryAccuracy(threshold=0.5) 
        auc = AUC()
        pc = Precision()
        rc = Recall()
        loss_history = []
        for x_train, y_train in train_ds:
            loss = train_on_batch(model, optimizer, acc, auc, pc, rc, x_train, y_train)
            loss_history.append(loss)
            f1 = f1_score(pc.result().numpy(), rc.result().numpy())
        print("Epoch {:03d}: 누적 Loss: {:.4f}, Acc: {:.4f}, AUC: {:.4f}, F1: {:.4f}".format(i, np.mean(loss_history),
                                                                            acc.result().numpy(), auc.result().numpy(), f1))

    # 평가
    test_acc = BinaryAccuracy(threshold=0.5)
    test_auc = AUC()
    test_pc = Precision()
    test_rc = Recall()
    eps = 1e-6

    for x, y in test_ds:
        y_pred = model(x)
        test_acc.update_state(y, y_pred)  
        test_auc.update_state(y, y_pred)
        test_pc.update_state(y, y_pred)  
        test_rc.update_state(y, y_pred)
    
    test_f1 = f1_score(test_pc.result().numpy(), test_rc.result().numpy())

    return test_acc.result().numpy(), test_auc.result().numpy(), test_f1