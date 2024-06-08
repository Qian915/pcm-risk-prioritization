import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
import argparse
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import tensorflow as tf
import data_loader as loader
import model
import time
import matplotlib.pyplot as plt
import seaborn as sns
import csv
from itertools import zip_longest
from datetime import timedelta
from datetime import datetime
from dateutil.relativedelta import relativedelta
from itertools import islice
from sklearn.preprocessing import MinMaxScaler
from ast import literal_eval
from sklearn import metrics 
from cf_matrix import make_confusion_matrix


parser = argparse.ArgumentParser(description="Predictive Compliance Monitoring with Risk Prioritization.")
parser.add_argument("--dataset", default="bpic2011_c1", type=str, help="dataset name")
parser.add_argument("--granularity", default="DAY", type=str, help="granularity of temporal features")
parser.add_argument("--model_dir", default="./models", type=str, help="model directory")
parser.add_argument("--result_dir", default="./results", type=str, help="results directory")
parser.add_argument("--epochs", default=100, type=int, help="number of total epochs")    
parser.add_argument("--batch_size", default=16, type=int, help="batch size")		
parser.add_argument("--learning_rate", default=0.0001, type=float, help="learning rate")		# o2c: 0.00001
parser.add_argument("--gpu", default=0, type=int, help="gpu id")

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

def train_model(train_df, activity_dict, max_case_length, vocab_size, model_path):
    # Prepare training data
    (train_act_x, train_time_x, 
        train_act_y, train_time_y, time_scaler, y_scaler) = data_loader.prepare_data(train_df, activity_dict, max_case_length)
    
    # Train the prediction model (a transformer)
    transformer_model = model.get_prediction_model(max_case_length, vocab_size)
    transformer_model.compile(optimizer=tf.keras.optimizers.Adam(args.learning_rate),
        loss={'output_act': 'sparse_categorical_crossentropy', 'output_time': 'mean_squared_error'},
        metrics={'output_act': 'accuracy', 'output_time': 'mae'})
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=model_path, monitor='val_loss',
        save_weights_only=False, save_best_only=True)
    # Define EarlyStopping callback with patience
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)	
    transformer_model.fit([train_act_x, train_time_x], [train_act_y, train_time_y], 
        epochs=args.epochs, batch_size=args.batch_size, validation_split=0.2, 
        verbose=2, callbacks=[model_checkpoint_callback, early_stopping])
    return transformer_model, y_scaler, time_scaler

def update_granularity(_time_diff, granularity):
    if granularity == "HOUR":
        time_diff = [timedelta(hours=int(hour[0])) for hour in _time_diff]
    else:   # granularity: DAY
        time_diff = [timedelta(days=int(day[0])) for day in _time_diff]
    return time_diff

def check_coded_value(activity_dict, key=None, value=None):
    for k, v in activity_dict.items():
        if key is not None and key == k:
            return v
        if value is not None and value == v:
            return k
    return key
    
def check_df_relation(df_constraint, traces):
    result, degree_sat, degree_vio = [],[],[]
    pre = df_constraint.iloc[0]["predecessor"]
    suc = df_constraint.iloc[0]["successor"]
    for _, trace in traces.iterrows():
        suffix, ts = trace["suffix"], trace["timestamps"]   # trace["timestamps"] is an nparray of type string
        if pre in suffix and suc in suffix:
            idx_pre = np.where(suffix==pre)[0][0]
            idx_suc = np.where(suffix==suc)[0][0]
            # check time perspective
            if idx_suc == idx_pre + 1:               
                if df_constraint.iloc[0]["time_type"] == "DURATION":
                    time_actual = relativedelta(datetime.strptime(ts[idx_suc], "%Y-%m-%d %H:%M:%S"), datetime.strptime(ts[idx_pre], "%Y-%m-%d %H:%M:%S"))
                    time_actual = update_duration_granularity(df_constraint.iloc[0]["granularity"], time_actual)
                else:
                    time_actual = datetime.strptime(ts[idx_suc], "%Y-%m-%d %H:%M:%S")
                    time_actual = update_time_granularity(df_constraint.iloc[0]["granularity"], time_actual)
                time_limit = df_constraint.iloc[0]["time_value"]
                time_to_be_scaled = check_time(df_constraint, time_actual, time_limit)
                # constraint satisfied
                if time_to_be_scaled >= 0:
                    result.append("sat")
                    degree_sat.append(time_to_be_scaled)
                # constraint violated: root cause - time
                else:
                    result.append("vio_time")
                    degree_vio.append(time_to_be_scaled)
            # constraint violated: root cause - successor does not occurs immediately after the predecessor
            else:
                result.append("vio_act")
        elif pre not in suffix and suc not in suffix:
            # wrong prediction for the next activity 
            result.append("vio_act")
        # constraint violated: root cause - predecessor and successor don't occur simultaneously
        else:
            result.append("vio_act") 
    return result, degree_vio, degree_sat   

def check_ef_relation(ef_constraint, traces):
    result, degree_sat, degree_vio = [],[],[]
    pre = ef_constraint.iloc[0]["predecessor"]
    suc = ef_constraint.iloc[0]["successor"]
    for _, trace in traces.iterrows():
        suffix, ts = trace["suffix"], trace["timestamps"]
        if pre in suffix and suc in suffix:
            idx_pre = np.where(suffix==pre)[0][0]   # return the first occurrence! -> delete duplicates in the log?
            idx_suc = np.where(suffix==suc)[0][0]
            # check time perspective
            if idx_suc > idx_pre:
                if ef_constraint.iloc[0]["time_type"] == "DURATION":
                    time_actual = relativedelta(datetime.strptime(ts[idx_suc], "%Y-%m-%d %H:%M:%S"), datetime.strptime(ts[idx_pre], "%Y-%m-%d %H:%M:%S"))
                    time_actual = update_duration_granularity(ef_constraint.iloc[0]["granularity"], time_actual)
                else:
                    time_actual = datetime.strptime(ts[idx_suc], "%Y-%m-%d %H:%M:%S")
                    time_actual = update_time_granularity(ef_constraint.iloc[0]["granularity"], time_actual)
                time_limit = ef_constraint.iloc[0]["time_value"]
                time_to_be_scaled = check_time(ef_constraint, time_actual, time_limit)
                # constraint satisfied
                if time_to_be_scaled >= 0:
                    result.append("sat")
                    degree_sat.append(time_to_be_scaled)
                # constraint violated: root cause - time
                else:
                    result.append("vio_time")
                    degree_vio.append(time_to_be_scaled)
            # constraint violated: root cause - successor occurs before the predecessor
            else:
                result.append("vio_act")
        elif pre not in suffix and suc not in suffix:
            # wrong prediction for the next activity 
            result.append("vio_act")
        # constraint violated: root cause - predecessor and successor don't occur simultaneously
        else:
            result.append("vio_act") 
    return result, degree_vio, degree_sat

def check_coexist_relation(coexist_constraint, traces):
    result, degree_sat, degree_vio = [],[],[]
    pre = coexist_constraint.iloc[0]["predecessor"]
    suc = coexist_constraint.iloc[0]["successor"]
    for _, trace in traces.iterrows():
        suffix, ts = trace["suffix"], trace["timestamps"]
        # check time perspective
        if pre in suffix and suc in suffix:
            idx_pre = np.where(suffix==pre)[0][0]
            idx_suc = np.where(suffix==suc)[0][0]
            if idx_pre > idx_suc:
                tmp = idx_pre
                idx_pre = idx_suc
                idx_suc = tmp           
            if coexist_constraint.iloc[0]["time_type"] == "DURATION":
                    time_actual = relativedelta(datetime.strptime(ts[idx_suc], "%Y-%m-%d %H:%M:%S"), datetime.strptime(ts[idx_pre], "%Y-%m-%d %H:%M:%S"))
                    time_actual = update_duration_granularity(coexist_constraint.iloc[0]["granularity"], time_actual)
            else:
                time_actual = datetime.strptime(ts[idx_suc], "%Y-%m-%d %H:%M:%S")
                time_actual = update_time_granularity(coexist_constraint.iloc[0]["granularity"], time_actual)
            time_limit = coexist_constraint.iloc[0]["time_value"]
            time_to_be_scaled = check_time(coexist_constraint, time_actual, time_limit)
            # constraint satisfied
            if time_to_be_scaled >= 0:
                result.append("sat")
                degree_sat.append(time_to_be_scaled)
            # constraint violated: root cause - time
            else:
                result.append("vio_time")
                degree_vio.append(time_to_be_scaled)
        elif pre not in suffix and suc not in suffix:
            continue
        # constraint violated: root cause - predecessor and successor don't occur simultaneously
        else:
            result.append("vio_act") 
    return result, degree_vio, degree_sat

def check_exist_relation(exist_constraint, traces):
    result, degree_sat, degree_vio = [],[],[]
    act = exist_constraint.iloc[0]["predecessor"]
    for _, trace in traces.iterrows():
        suffix, ts = trace["suffix"], trace["timestamps"]
        # check time perspective
        _time_last = datetime.strptime(ts[-1], "%Y-%m-%d %H:%M:%S")
        time_last = update_time_granularity(exist_constraint.iloc[0]["granularity"], _time_last)
        time_limit = exist_constraint.iloc[0]["time_value"]
        time_last_to_be_scaled = check_time(exist_constraint, time_last, time_limit)
        if act in suffix:
            idx = np.where(suffix==act)[0][0]
            _time_actual = datetime.strptime(ts[idx], "%Y-%m-%d %H:%M:%S")
            time_actual = update_time_granularity(exist_constraint.iloc[0]["granularity"], _time_actual)
            time_to_be_scaled = check_time(exist_constraint, time_actual, time_limit)
            # constraint satisfied
            if time_to_be_scaled >= 0:
                result.append("sat")
                degree_sat.append(time_to_be_scaled)
            # constraint violated: root cause - time
            else:
                result.append("vio_time")
                degree_vio.append(time_to_be_scaled)
        # the condition of presence is not fulfilled
        elif act not in suffix and time_last_to_be_scaled < 0:
            continue
        # constraint violated: root cause - activity does not occur
        else:
            result.append("vio_act") 
    return result, degree_vio, degree_sat

def check_time(constraint,time_actual, time_limit):
    if constraint.iloc[0]["relation_time"] == "interval":
        time_to_be_scaled = calculate_value_interval(time_actual, time_limit)
    elif constraint.iloc[0]["relation_time"] == "max":
        time_to_be_scaled = calculate_value_max(time_actual, time_limit)
    elif constraint.iloc[0]["relation_time"] == "min":
        time_to_be_scaled = calculate_value_min(time_actual, time_limit)
    else:
        time_to_be_scaled = calculate_value_exactlyAt(time_actual, time_limit)
    return time_to_be_scaled

def update_time_granularity(granularity, time):
    if granularity == "MONTH":
        return time.month
    if granularity == "DAY":
        return time.day
    if granularity == "WDAY":
        return time.isoweekday()
    if granularity == "HOUR":
        return time.hour
    if granularity == "MINUTE":
        return time.minute

def update_duration_granularity(granularity, duration):
    if granularity == "DAY":
        if duration.days:
            diff = duration.days + duration.hours / 24 + duration.minutes / (24 * 60) + duration.seconds / (24 * 60 * 60)
        else:
            diff = duration.hours / 24 + duration.minutes / (24 * 60) + duration.seconds / (24 * 60 * 60)
    if granularity == "HOUR":
        if duration.days:
            diff = duration.days * 24 + duration.hours + duration.minutes / 60 + duration.seconds / 3600
        else:
            diff = duration.hours + duration.minutes / 60 + duration.seconds / 3600
    return diff    

def calculate_value_interval(value_actual, value_limit):
    value_limit = literal_eval(value_limit) if '(' in value_limit else int(value_limit)
    lower_value = value_limit[0]
    upper_value = value_limit[1]

    center = (lower_value + upper_value) / 2
    width = (upper_value - lower_value) / 2
    
    distance_to_center = np.abs(value_actual - center)
    # time satisfied: value >= 0
    if distance_to_center <= width:
        return (distance_to_center - width) ** 2
    # time violated: value < 0
    else:
        return -((distance_to_center - width) ** 2)
    
def calculate_value_max(value_actual, value_max):
    value_max = int(value_max)
    distance_to_max = value_actual - value_max

    if value_actual <= value_max:
        return distance_to_max**2
    else:
        return -distance_to_max**2
    
def calculate_value_min(value_actual, value_min):
    value_min = int(value_min)
    distance_to_min = value_actual - value_min

    if value_actual >= value_min:
        return distance_to_min**2
    else:
        return -distance_to_min**2

def calculate_value_exactlyAt(value_actual, value_center):
    value_center = int(value_center)
    distance_to_center = np.abs(value_actual - value_center)
    epsilon = 1e-10
    
    if np.abs(distance_to_center) < epsilon:    # resolve floating-point precision issues via a small epsilon value
        return 5
    else: 
        return -distance_to_center**2
    
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def calculate_degree(degrees_original):
    degree_vio = degrees_original[0]
    degree_sat = degrees_original[1]
    scalers = [MinMaxScaler(feature_range=(-5, -1e-5)), MinMaxScaler(feature_range=(0, 5))]
    # Transform values to compliance degrees via sigmoid function
    if len(degree_vio) > 0:
        degree_vio = sigmoid(scalers[0].fit_transform(np.array(degree_vio).reshape(-1, 1)))
        degree_vio = degree_vio.flatten().tolist()
    if len(degree_sat) > 0:
        degree_sat = sigmoid(scalers[1].fit_transform(np.array(degree_sat).reshape(-1, 1)))
        degree_sat = degree_sat.flatten().tolist()
    return degree_vio, degree_sat

if __name__ == "__main__":
    model_path = f"{args.model_dir}/{args.dataset}"
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    model_path = f"{model_path}/model.h5"

    result_path = f"{args.result_dir}/{args.dataset}"
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    result_path = f"{result_path}/"
    result = None

    # Load data
    data_loader = loader.LogsDataLoader(args.dataset)
    (train_df, _test_df, activity_dict, max_case_length, vocab_size) = data_loader.load_data()

    # Load constraints
    constraint = pd.read_excel(f"constraints/{args.dataset}.xlsx")   # constraint is a df with one row!
    pre, suc, relation_act = "predecessor", "successor", "relation_activity"

    # Filter cases in test_df for evaluation and compliance checking
    pref, next_act = "prefix", "next_activity"
    test_df = pd.DataFrame(columns=_test_df.columns)
    unique_cases = _test_df["case_id"].unique()
    # for df/ef constraint
    if constraint.at[0, relation_act] == "df" or constraint.at[0, relation_act] == "ef":
        for _, case in enumerate(unique_cases):   
            trace = _test_df[_test_df["case_id"] == case].copy()
            # only select the first occurrence to avoid duplicates
            filtered_case = trace[trace[next_act] == constraint.at[0, suc]].head(1)
            test_df = pd.concat([test_df, filtered_case], ignore_index=True)
    # for coexist constraint
    if constraint.at[0, relation_act] == "coexist":
        for _, case in enumerate(unique_cases):   
            trace = _test_df[_test_df["case_id"] == case].copy()
            # pre -> suc
            filtered_case = trace[trace[next_act] == constraint.at[0, suc]].head(1)
            if not filtered_case.empty and constraint.at[0, pre] in filtered_case.iloc[0][pref]:
                test_df = pd.concat([test_df, filtered_case], ignore_index=True)
            # suc -> pre
            filtered_case = trace[trace[next_act] == constraint.at[0, pre]].head(1)
            if not filtered_case.empty and constraint.at[0, suc] in filtered_case.iloc[0][pref]:
                test_df = pd.concat([test_df, filtered_case], ignore_index=True)    
    test_df.to_csv(f"data/{args.dataset}/test_updated.csv", index=False)
    
    # Train the prediction model based on the training set
    transformer_model, y_scaler, time_scaler = train_model(train_df, activity_dict, max_case_length, vocab_size, model_path)

    # Evaluate over filtered test_df
    x_act, x_time, _y_act_true, _y_time_true, _, _ = data_loader.prepare_data(test_df, activity_dict, max_case_length, time_scaler, y_scaler)
    _y_act_pred, _y_time_pred = transformer_model.predict([x_act, x_time], verbose=0)
        # metrics for activity prediction
    y_act_true = _y_act_true.flatten()      # y_act_true: 1d
    y_act_pred = np.argmax(_y_act_pred, axis=1)     # y_act_pred: 1d 
    accuracy = metrics.accuracy_score(y_act_true, y_act_pred)
    precision, recall, fscore, _ = metrics.precision_recall_fscore_support(y_act_true, y_act_pred, average="weighted")
    print('Accuracy:', accuracy)
    print('F-score:', fscore)
    print('Precision:', precision)
    print('Recall:', recall)
        # metrics for time prediction
    y_time_true = y_scaler.inverse_transform(_y_time_true).flatten()    # y_time_true: 1d
    y_time_pred = y_scaler.inverse_transform(_y_time_pred)      # y_time_pred: 1d
    mae = metrics.mean_absolute_error(y_time_true, y_time_pred)
    mse = metrics.mean_squared_error(y_time_true, y_time_pred)
    rmse = np.sqrt(metrics.mean_squared_error(y_time_true, y_time_pred))
    print('MAE:', mae)
    print('MSE:', mse)
    print('RMSE:', rmse)
    acc_fscore_mae = []
    acc_fscore_mae.extend([accuracy, fscore, mae])

    # check compliance for the corresponding constraint:
    # prepare data sets for compliance checking: add the next activity to prefix column; add the next time to timestamps column
    suffix = test_df[["case_id","timestamps"]]
    #suffix.columns = ["case_id", "suffix", "timestamps"]
    # convert timestamps to array
    _ts = suffix["timestamps"].values   
    ts = list()
    for t in _ts:
        ts.append([_t for _t in t.split(", ")])
    #ts = np.array(_ts)  # not a 2d array -> a 2d list!!!
    suffix_true = suffix.copy()
    suffix_pred = suffix.copy()

    s_true = np.concatenate((x_act, y_act_true.reshape(-1,1)), axis=1)
    s_true_list = [[elem for elem in inner_list if elem != 0] for inner_list in s_true.tolist()]
    suffix_true["suffix"] = s_true_list      # delete zeros
    s_pred = np.concatenate((x_act, y_act_pred.reshape(-1,1)), axis=1)
    s_pred_list = [[elem for elem in inner_list if elem != 0] for inner_list in s_pred.tolist()]
    suffix_pred["suffix"] = s_pred_list

    time_diff_true = update_granularity(y_time_true.reshape(-1,1), args.granularity)
    ts_true = list()
    for t, diff in zip(ts, time_diff_true):  # suffix["timestamps"] is a string of timestamps seperated by ','
        #ts = [timestamp.strip() for timestamp in ts_string.split(',')]     # convert the string of timestamps to a list ['...', '...']
        last_ts = datetime.strptime(t[-1], "%Y-%m-%d %H:%M:%S")
        append_ts = last_ts + diff
        ts_true.append(np.concatenate((t, [datetime.strftime(append_ts, "%Y-%m-%d %H:%M:%S")])))
    #ts_true = np.array(ts_true)   # not a 2d array -> a 2d list!!!
    suffix_true["timestamps"] = ts_true

    time_diff_pred = update_granularity(y_time_pred.reshape(-1,1), args.granularity)	#reshape to 2d!
    ts_pred = list()
    for t, diff in zip(ts, time_diff_pred):
        last_ts = datetime.strptime(t[-1], "%Y-%m-%d %H:%M:%S")
        append_ts = last_ts + diff
        ts_pred.append(np.concatenate((t, [datetime.strftime(append_ts, "%Y-%m-%d %H:%M:%S")])))
    #ts_pred = np.array(ts_pred)   # 2d
    suffix_pred["timestamps"] = ts_pred
    # save df for true and predicted values
    suffix_true.to_csv(f"data/suffix_true.csv", index=False)
    suffix_pred.to_csv(f"data/suffix_pred.csv", index=False)

    # start compliance checking
    constraint[pre] = constraint[pre].apply(lambda act: check_coded_value(activity_dict, key=act))
    constraint[suc] = constraint[suc].apply(lambda act: check_coded_value(activity_dict, key=act))
    # for df constraint
    if constraint.at[0, relation_act] == "df":
        result_pred, degree_vio_pred, degree_sat_pred = check_df_relation(constraint, suffix_pred)
        result_true, degree_vio_true, degree_sat_true = check_df_relation(constraint, suffix_true)
    # for ef constraint
    if constraint.at[0, relation_act] == "ef":
        print("######for prediction######")
        result_pred, degree_vio_pred, degree_sat_pred = check_ef_relation(constraint, suffix_pred)
        print("######for ground truth######")
        result_true, degree_vio_true, degree_sat_true = check_ef_relation(constraint, suffix_true)
    # for coexist constraint
    if constraint.at[0, relation_act] == "coexist":
        result_pred, degree_vio_pred, degree_sat_pred = check_coexist_relation(constraint, suffix_pred)
        result_true, degree_vio_true, degree_sat_true = check_coexist_relation(constraint, suffix_true)
    # for exist constraint
    if constraint.at[0, relation_act] == "exist":
        result_pred, degree_vio_pred, degree_sat_pred = check_exist_relation(constraint, suffix_pred)
        result_true, degree_vio_true, degree_sat_true = check_exist_relation(constraint, suffix_true)
    # convert values to degrees of compliance via sigmoid function
    degree_vio_pred, degree_sat_pred = calculate_degree([degree_vio_pred, degree_sat_pred])
    degree_vio_true, degree_sat_true = calculate_degree([degree_vio_true, degree_sat_true])
    
    # save all results
    combined_data = zip_longest(result_pred, degree_vio_pred, degree_sat_pred, result_true, degree_vio_true, degree_sat_true, acc_fscore_mae, fillvalue=None)
    with open(result_path+'output.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['result_pred', 'degree_vio_pred', 'degree_sat_pred', 'result_true', 'degree_vio_true', 'degree_sat_true', 'acc_fscore_mae'])
        writer.writerows(combined_data)
        
    ### evaluate performance of compliance predictions ###
        
    # -> confusion matrix for compliance states
    labels=["sat", "vio_act", "vio_time"]
    cm = metrics.confusion_matrix(result_true, result_pred, labels=labels)
    make_confusion_matrix(cm, categories=labels, cbar=False)
    plt.savefig(result_path+'heatmap.pdf', format='pdf', bbox_inches='tight')

    # -> scatter plot for compliance degrees
    degree_vio_pred_loaded = [x for x in degree_vio_pred if pd.notna(x)]
    degree_sat_pred_loaded = [x for x in degree_sat_pred if pd.notna(x)]
    degree_pred = degree_vio_pred_loaded + degree_sat_pred_loaded
    plt.figure(figsize=(4, 3))
    plt.scatter(degree_pred, np.arange(len(degree_pred)))
    plt.axvline(x=0.5, color='red', linestyle='--', label='Threshold')
    plt.xlabel('Degree of Compliance')
    plt.ylabel('Data Point Index')
    plt.legend()
    plt.savefig(result_path+'scatter_pred.pdf', format='pdf', bbox_inches='tight')
    plt.show()