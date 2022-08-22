import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import LSTM, Dense, Activation, Dropout
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint  # EarlyStopping
from sklearn.preprocessing import MinMaxScaler
import os


from math import sqrt

from keras.optimizers import Adam


# load data

def drow_TG(data_path, sheet):

    data_TG = pd.read_excel(data_path, sheet_name=sheet, header=0)
    data_TG = data_TG.values
    # plot each column
    plt.figure(figsize=(10, 10))
    plt.scatter(data_TG[:, 1], data_TG[:, 0], s=20, c='r', marker='+')

    #plt.ion()
    plt.show()


def read_data_exl(data_path, sheet, mms):
    dataset = pd.read_excel(data_path, sheet_name=sheet, header=0)
    values = dataset.values
    TG_values = values.astype('float64')
    TG_value = mms.fit_transform(TG_values)
    return TG_value


def train_test(TG_value, train_percentage, FEATURE_NUMBER, REACTION_FRACTION):  
    if REACTION_FRACTION == 1:
        line = len(TG_value)
        for i in range(line):
            a = line - i - 1
            if TG_value[a, 0] == 1:
                TG_value = np.delete(TG_value, a, axis=0)
    np.random.shuffle(TG_value)
    np.random.shuffle(TG_value)
    sum =TG_value.shape[0]
    trian_num = int(sum * train_percentage)
    train = TG_value[:trian_num, :]
    test = TG_value[trian_num:, :]
    plt.figure(figsize=(10, 10)) 
    train_X, train_y = train[:, 1:], train[:, 0]
    test_X, test_y = test[:, 1:], test[:, 0]   
    train_X = train_X.reshape((train_X.shape[0], FEATURE_NUMBER, int(train_X.shape[1] / FEATURE_NUMBER) ))
    test_X = test_X.reshape((test_X.shape[0], FEATURE_NUMBER, int(test_X.shape[1] / FEATURE_NUMBER)))
    print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
    return train_X, train_y, test_X, test_y


def fit_network(train_X, train_y, test_X, test_y, mms, save_path, EPOCHS, BATCH_SIZE, LEARN_RATE, DROPOUT_val, net_name, FEATURE_NUMBER, train_percentage):
    model = Sequential()
    model.add(LSTM(800, activation='tanh', input_shape=(train_X.shape[1], train_X.shape[2]), return_sequences=True))
    model.add(LSTM(500, activation='relu'))
    model.add(Dropout(DROPOUT_val))
    model.add(Dense(1, activation='relu'))
    model.compile(optimizer=Adam(lr=LEARN_RATE), loss='mse')
    history = model.fit(train_X, train_y, epochs=EPOCHS, batch_size=BATCH_SIZE,
                        validation_data=(test_X, test_y), verbose=2, shuffle=False,                        
                        )
    model.save_weights(save_path+net_name)
    train_loss = history.history['loss']
    np_train_loss = np.array(train_loss).reshape((len(train_loss), 1))
    np_epochs_num = np.array(np.arange(1, len(train_loss)+1)).reshape((len(train_loss), 1))
    plt.plot(history.history['loss'], label='train')
    if train_percentage != 1:
        plt.plot(history.history['val_loss'], label='test')
        test_loss = history.history['val_loss']
        np_test_loss = np.array(test_loss).reshape((len(test_loss), 1))
        np_out = np.concatenate([np_epochs_num, np_train_loss, np_test_loss], axis=1)
    else:
        np_out = np.concatenate([np_epochs_num, np_train_loss], axis=1)
    plt.legend()
    plt.show()

    np.savetxt(save_path+'loss.data', np_out)
    print('loss, Done!')

    y_pred = model.predict(train_X)   
    x_pred = train_X[:, 0]
    train_X = train_X.reshape((train_X.shape[0], train_X.shape[2] * FEATURE_NUMBER))
    all_pred = np.concatenate((y_pred, train_X), axis=1)
    
    shape_plus=train_X.shape[1]
    
    train_y = train_y.reshape(-1, 1)
    test_y = test_y.reshape(-1, 1)
    train_y_data = np.pad(train_y, ((0, 0), (0, shape_plus)), 'constant', constant_values=((0, 0), (0, 0)))

    #####################################
    y_pred_train = mms.inverse_transform(all_pred)
    train_y_data = mms.inverse_transform(train_y_data)
    plt.scatter(x_pred[:, 0], y_pred_train[:, 0], s=20, c='b', marker='+')

    #plt.ion()
    plt.show()

    np_y_pred_train = np.array(y_pred_train[:, 0]).reshape((len(y_pred), 1))
    np_train_y = np.array(train_y_data[:, 0]).reshape((len(train_y_data), 1))
    np_out_train = np.concatenate([np_y_pred_train, np_train_y], axis=1)
    np.savetxt(save_path+'pre_train.data', np_out_train)


    if train_percentage != 1:
        y_pred_test = model.predict(test_X)
        y_pred_test = np.pad(y_pred_test, ((0, 0), 
                         (0, shape_plus)), 
                         'constant', constant_values=((0, 0), (0, 0)))
        test_y_data = np.pad(test_y, ((0, 0), (0, shape_plus)), 'constant', constant_values=((0, 0), (0, 0)))
        y_pred_test = mms.inverse_transform(y_pred_test)
        test_y_data = mms.inverse_transform(test_y_data)
        np_y_pred_test = np.array(y_pred_test[:, 0]).reshape((len(y_pred_test), 1))
        np_test_y = np.array(test_y_data[:, 0]).reshape((len(test_y_data), 1))
        np_out_test = np.concatenate([np_y_pred_test, np_test_y], axis=1)
        np.savetxt(save_path + 'pre_test.data', np_out_test)

    print('y_pre, Done!!!')



if __name__ == '__main__':
    data_path = r"C:\Users\test.xlsx"
    sheet = 'Sheet1'
    save_path = r"C:\Users\TG-LSTM/"
    net_name = "/test.h5"
    drow_TG(data_path, sheet)
    mms = MinMaxScaler(feature_range=(0, 1))
    TG_value = read_data_exl(data_path, sheet, mms)
    train_percentage = 0.8
    EPOCHS = 50
    BATCH_SIZE = 64
    LEARN_RATE = 1e-3
    FEATURE_NUMBER = 3
    REACTION_FRACTION = 1
    DROPOUT_val = 0.2
    train_X, train_y, test_X, test_y = train_test(TG_value, train_percentage, FEATURE_NUMBER, REACTION_FRACTION)
    fit_network(train_X, train_y, test_X, test_y, mms, save_path, EPOCHS, BATCH_SIZE, LEARN_RATE, DROPOUT_val, net_name,
                FEATURE_NUMBER, train_percentage)