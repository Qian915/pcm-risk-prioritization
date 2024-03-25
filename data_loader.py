import io
import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import utils
from sklearn import preprocessing


class LogsDataLoader:
    def __init__(self, log_name):
        self._processed_path = f"data/{log_name}/processed"
        self._data_name = log_name

    def prepare_data(self, df, activity_dict, max_case_length, time_scaler=None, y_scaler=None, shuffle = False):

        x_activity = df["prefix"].values    # 1d array
        x_time = df["latest_time"].values.astype(np.float32)
        y_activity = df["next_activity"].values
        y_time = df["next_time"].values.astype(np.float32)
        if shuffle:
            x_activity, x_time, y_activity, y_time = utils.shuffle(x_activity, x_time, y_activity, y_time)
    
        x_act = list()
        for _x in x_activity:
            x_act.append([activity_dict[s] for s in _x.split(", ")])    # seperator: ','

        if time_scaler is None:
            time_scaler = preprocessing.StandardScaler()
            x_time = time_scaler.fit_transform(
                x_time.reshape(-1, 1)).astype(np.float32)
        else:
            x_time = time_scaler.transform(
                x_time.reshape(-1, 1)).astype(np.float32)

        y_act = list()
        for _y in y_activity:
            y_act.append(activity_dict[_y])

        if y_scaler is None:
            y_scaler = preprocessing.StandardScaler()
            y_time = y_scaler.fit_transform(
                y_time.reshape(-1, 1)).astype(np.float32)
        else:
            y_time = y_scaler.transform(
                y_time.reshape(-1, 1)).astype(np.float32)

        # padding with 0
        x_act = tf.keras.preprocessing.sequence.pad_sequences(
            x_act, maxlen=max_case_length)
        
        x_act = np.array(x_act, dtype=np.float32)
        x_time = np.array(x_time, dtype=np.float32).reshape(-1, 1)
        y_act = np.array(y_act, dtype=np.float32).reshape(-1, 1)
        y_time = np.array(y_time, dtype=np.float32).reshape(-1, 1)
        return x_act, x_time, y_act, y_time, time_scaler, y_scaler
    
    def get_max_case_length(self, prefixes_train, prefixes_test):
        len_train = [len(_prefix.split(', ')) for _prefix in prefixes_train]
        len_test = [len(_prefix.split(', ')) for _prefix in prefixes_test]
        max_train = max(len_train)
        max_test = max(len_test)
        if max_train > max_test:
            print(f"max case length from training set: {max_train}")
            return max_train
        elif max_train == max_test:
            print(f"the same max case length: {max_train}")
            return max_train
        else:
            print(f"max case length from test set: {max_test}")
            return max_test


    def load_data(self):
        # read processed data
        train_df = pd.read_csv(f"{self._processed_path}/{self._data_name}_train.csv")
        test_df = pd.read_csv(f"{self._processed_path}/{self._data_name}_test.csv")

        with open(f"{self._processed_path}/metadata.json", "r") as json_file:
            metadata = json.load(json_file)

        activity_dict = metadata
        vocab_size = len(activity_dict)
        max_case_length = self.get_max_case_length(train_df["prefix"].values, test_df["prefix"].values)
        print(f"vocab_size: {vocab_size}")

        return (train_df, test_df, activity_dict, max_case_length, vocab_size)