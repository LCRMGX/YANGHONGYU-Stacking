from sklearn import metrics
from keras.utils import np_utils
import numpy as np
import pandas as pd
from keras.models import Model, Sequential, load_model
from keras.layers import Input, Dense, Activation, Conv1D, MaxPooling1D, Flatten, Dropout, BatchNormalization, \
     Add, LSTM
from sklearn.model_selection import KFold
from keras import optimizers
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.linear_model import LogisticRegression
import warnings

warnings.filterwarnings('ignore')
tf.random.set_seed(6)
np.random.seed(6)


def read_data(file_path):
    df = pd.read_csv(file_path)
    features = df[['SOIL_ID', 'NDVI_MEAN', 'DEM_ADJ', 'ROUGH_MEAN', 'SLOPE_MEAN', 'SLOPE_VAR',
                   'PLANCURV', 'POU_WAM', 'R_Index']].values
    labels = df['is_prototype'].values
    return features, labels


# 读取训练和测试数据
train_x, train_y_1D = read_data('training_samples.csv')
test_x, test_y_1D = read_data('test_samples.csv')

num_classes = 2
train_y = np_utils.to_categorical(train_y_1D, num_classes)

# 扩展维度以适应CNN输入
train_x = np.expand_dims(train_x, axis=2)
test_x = np.expand_dims(test_x, axis=2)

k = 5
kf = KFold(n_splits=k, shuffle=True, random_state=6)

auc_scores = []
best_auc = 0
best_auc_model = None
best_accuracy = 0
best_accuracy_model = None


def create_cnn_model(input_shape, num_classes):
    input_layer = Input(shape=input_shape)
    x = Conv1D(32, kernel_size=3, padding='same')(input_layer)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    def residual_block(x, filters):
        shortcut = x
        x = Conv1D(filters, kernel_size=3, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv1D(filters, kernel_size=3, padding='same')(x)
        x = BatchNormalization()(x)
        x = Add()([x, shortcut])
        x = Activation('relu')(x)
        return x

    for _ in range(3):
        x = residual_block(x, 32)

    x = Conv1D(64, kernel_size=3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(pool_size=2)(x)

    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)

    output_layer = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=input_layer, outputs=output_layer)
    optimizer = optimizers.Adam()
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


def create_rnn_model(input_shape, num_classes):
    model = Sequential()
    model.add(LSTM(50, input_shape=input_shape, return_sequences=False))
    model.add(Dense(num_classes, activation='softmax'))
    optimizer = optimizers.Adam(learning_rate=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


for train_index, val_index in kf.split(train_x):
    X_train, X_val = train_x[train_index], train_x[val_index]
    y_train, y_val = train_y[train_index], train_y[val_index]

    model_cnn = create_cnn_model(X_train.shape[1:], num_classes)
    callbacks = [ModelCheckpoint('best_model_cnn.h5', monitor='val_accuracy', save_best_only=True, mode='max'),
                 EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)]
    model_cnn.fit(X_train, y_train, validation_data=(X_val, y_val), callbacks=callbacks, batch_size=64, epochs=100)

    model_rnn = create_rnn_model(X_train.shape[1:], num_classes)
    model_rnn.fit(X_train, y_train, validation_data=(X_val, y_val), callbacks=callbacks, batch_size=64, epochs=50)

    train_pred_cnn = model_cnn.predict(X_train)
    train_pred_rnn = model_rnn.predict(X_train)
    val_pred_cnn = model_cnn.predict(X_val)
    val_pred_rnn = model_rnn.predict(X_val)

    train_pred_stack = np.hstack((train_pred_cnn, train_pred_rnn))
    val_pred_stack = np.hstack((val_pred_cnn, val_pred_rnn))

    meta_model = LogisticRegression(max_iter=1000, solver='liblinear', multi_class='ovr')
    meta_model.fit(train_pred_stack, y_train.argmax(axis=1))

    val_pred_final = meta_model.predict_proba(val_pred_stack)
    auc = metrics.roc_auc_score(y_val.argmax(axis=1), val_pred_final[:, 1])
    val_accuracy = metrics.accuracy_score(y_val.argmax(axis=1), val_pred_final.argmax(axis=1))

    auc_scores.append(auc)

    if auc > best_auc:
        best_auc = auc
        best_auc_model = (model_cnn, model_rnn, meta_model)

    if val_accuracy > best_accuracy:
        best_accuracy = val_accuracy
        best_accuracy_model = (model_cnn, model_rnn, meta_model)

# 使用准确率最高的模型在测试集上进行预测
best_cnn, best_rnn, best_meta_model = best_accuracy_model
test_pred_cnn = best_cnn.predict(test_x)
test_pred_rnn = best_rnn.predict(test_x)
test_pred_stack = np.hstack((test_pred_cnn, test_pred_rnn))
test_pred_final = best_meta_model.predict_proba(test_pred_stack)

# 评估测试集上的性能
test_auc = metrics.roc_auc_score(test_y_1D, test_pred_final[:, 1])
test_accuracy = metrics.accuracy_score(test_y_1D, test_pred_final.argmax(axis=1))
print(f"Test AUC: {test_auc}")
print(f"Test Accuracy: {test_accuracy}")