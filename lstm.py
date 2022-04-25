from locale import normalize
import pandas as pd
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten, LSTM, TimeDistributed
from keras.callbacks import EarlyStopping

class MODEL:
    def __init__(self):
        self.date_slice = 5
        self.date_gap = 1
        self.offset_rate = 0.1
        self.save_name = 'lstm_model'
        self.epoch = 20

    def lstm_stock_model(self, shape):
        model = Sequential()
        model.add(LSTM(512, input_shape=(shape[1], shape[2]), return_sequences=True))
        model.add(Dropout(rate=0.5))
        model.add(LSTM(512, return_sequences=True))
        model.add(Dropout(rate=0.5))
        model.add(LSTM(256, return_sequences=True))
        model.add(Dropout(rate=0.5))
        model.add(LSTM(256, return_sequences=True))
        model.add(Dropout(rate=0.5))
        model.add(TimeDistributed(Dense(1)))
        model.add(Flatten())
        model.add(Dense(1,activation='relu'))
        model.compile(loss="mean_absolute_error", optimizer="adam",metrics=['mean_absolute_error'])
        model.summary()
        return model

    def normalize_xy(self, x, y = 0):
        Max = np.max(x)
        Min = np.min(x)
        diff = Max - Min
        x = (x - Min) / diff
        if (y != 0):
            y = (y - Min) / diff
            return [x, y]
        return x

    def training_process(self, train_file):
        df = pd.read_csv(train_file, header = None)
        train_data_count = (len(df.values) - (self.date_slice - self.date_gap) - 1) // self.date_gap
        train_x = []
        train_y = []
        for c in range(train_data_count):
            each_x = []
            for n in range(self.date_slice):
                each_x.append([df.values[c * self.date_gap + n][0]])
                each_x.append([df.values[c * self.date_gap + n][3]])
            each_y = [[df.values[c * self.date_gap + self.date_slice][0]]]
            # each_x, each_y = self.normalize_xy(each_x, each_y)
            train_x.append(each_x)
            train_y.append(each_y)
        train_x = np.array(train_x)
        train_y = np.array(train_y)
        return [train_x, train_y]

    def compute_offset(self, model, x, y):
        pred = model.predict(x)
        offset = np.average(y) - np.average(pred)     # we hope (pred + offset = real) will be true
        return offset

    def model_training(self, train_x, train_y):
        
        offset_num = int(len(train_x) * self.offset_rate)
        train_x = train_x[:-offset_num]
        train_y = train_y[:-offset_num]
        offset_x = train_x[-offset_num:]
        offset_y = train_y[-offset_num:]
        model = self.lstm_stock_model(train_x.shape)
        callback = EarlyStopping(monitor="mean_absolute_error", patience=8, verbose=1, mode="auto")
        history = model.fit(train_x, train_y, epochs=self.epoch, batch_size=5, validation_split=0.1, callbacks=[callback], shuffle=True)
        offset = self.compute_offset(model, offset_x, offset_y)
        return [model, offset]

        # model.save(save_name)
        # model = load_model(save_name)

    def testing_process(self, test_file, train_file):
        
        df = pd.read_csv(train_file, header = None)
        test_df = pd.read_csv(test_file, header = None)
        test_x = []
        for i in range(len(test_df.values)):
            each_x = []
            for d in range(self.date_slice):
                if (i + d < self.date_slice):
                    each_x.append([df.values[-(self.date_slice - d - i)][0]])
                    each_x.append([df.values[-(self.date_slice - d - i)][3]])
                else:
                    each_x.append([test_df.values[d - (self.date_slice - i)][0]])
                    each_x.append([test_df.values[d - (self.date_slice - i)][3]])
            # each_x = self.normalize_xy(each_x)
            test_x.append(each_x)
            # test_y.append([test_df.values[i][0]])
        test_x = np.array(test_x)

        return test_x

    def action_pred(self, pred):
        state = 0
        action = []
        for i in range(len(pred)):
            if state == pred[i]:
                action.append(0)
            else:
                if state == 0:
                    state = pred[i]
                else:
                    state = 0
                action.append(pred[i])
        return action

    def make_pred(self, test_x, offset, model):
        pred = []
        for i in range(len(test_x) - 1):
            pred_x1 = model.predict(np.array([test_x[i]]))
            pred_x1 += offset
            
            test_x2 = np.append(test_x[i][1:], pred_x1)
            pred_x2 = model.predict(np.array([test_x2]))
            pred_x2 += offset

            test_x3 = np.append(test_x2[1:], pred_x2)
            pred_x3 = model.predict(np.array([test_x3]))
            pred_x3 += offset

            if pred_x1 > pred_x3:
                pred.append(-1)
            else:
                pred.append(1)
        return self.action_pred(pred) 



# pred = model.predict(test_x)
# pred2 = pred + offset

# offset_y = np.reshape(offset_y, (len(offset_y), 1))
# plt.plot(offset_y)
# plt.plot(pred_offset)

# plt.figure(figsize=(10,12))
# plt.plot(pred)
# plt.plot(pred2)
# plt.plot(pred3)
# plt.plot(test_y)
# plt.legend(["first day", "second day", "third day", "test"], loc ="lower left")
# plt.show()


# result = pd.DataFrame()
# result['action'] = action
# result.to_csv('out.csv',index=0,header=None)

# buyandholdalgo = [1] + [0 for i in range(18)]
# bah = pd.DataFrame()
# bah['action'] = buyandholdalgo
# bah.to_csv('buyandhold.csv', index=0, header=None)
