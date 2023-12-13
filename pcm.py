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
from datetime import timedelta
from datetime import datetime
from dateutil.relativedelta import relativedelta
from itertools import islice
from sklearn.preprocessing import MinMaxScaler
from ast import literal_eval

parser = argparse.ArgumentParser(description="Predictive Compliance Monitoring with Risk Prioritization.")
parser.add_argument("--dataset", default="BPIC_2011", type=str, help="dataset name")
parser.add_argument("--granularity", default="DAY", type=str, help="granularity of temporal features")
parser.add_argument("--model_dir", default="./models", type=str, help="model directory")
parser.add_argument("--result_dir", default="./results", type=str, help="results directory")
parser.add_argument("--epochs", default=10, type=int, help="number of total epochs")    # epochs for O2C: 30!
parser.add_argument("--batch_size", default=12, type=int, help="batch size")
parser.add_argument("--learning_rate", default=0.01, type=float, help="learning rate")
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
    transformer_model.fit([train_act_x, train_time_x], [train_act_y, train_time_y], 
        epochs=args.epochs, batch_size=args.batch_size, validation_split=0.2, 
        verbose=2, callbacks=[model_checkpoint_callback])
    return transformer_model, y_scaler
    
def predict_suffix(test_df, k, prediction_model, activity_dict, max_case_length, y_scaler, granularity):
    print(f"predicting suffixes of {k}-prefixes")
    # Prepare test data of k-prefix
    test_k = test_df[test_df["k"]==k]
    ts = test_k["timestamps"].values   
    _ts = list()
    for t in ts:
        _ts.append([_t for _t in t.split(", ")])
    ts = np.array(_ts)  
    cases = test_k["case_id"].values
    x_act, x_time, y_act, y_time, _, _ = data_loader.prepare_data(test_k, activity_dict, max_case_length)
    # Suffix prediction till [eoc]
    suffix_k = pd.DataFrame(columns=["case_id", "k", "prefix", "suffix", "timestamps"])
    idx = 0
    while True:
        next_act, next_time, ts = predict_next(prediction_model, x_act, x_time, y_scaler, ts, granularity)
        # save completed cases
        idx_del = list()
        for i, (case, _x_act, _next_act, _t) in enumerate(zip(islice(cases, None), islice(x_act, None), islice(next_act, None),islice(ts, None))):
            if _next_act[0] == check_coded_value(activity_dict, key="[eoc]"):        
                suffix_k.at[idx, "case_id"] = case
                suffix_k.at[idx, "k"] = k
                suffix_k.at[idx, "prefix"] = _x_act[np.argmax(_x_act != 0):np.argmax(_x_act != 0)+k]
                s = np.concatenate((_x_act, _next_act))
                suffix_k.at[idx, "suffix"] = s[np.argmax(s != 0):]    # suffix contains the prefix for the ease of compliance checking!
                suffix_k.at[idx, "timestamps"] = _t
                idx += 1
                idx_del.append(i)
        if len(idx_del) != 0:
            cases = np.delete(cases, idx_del, axis=0)
            x_act = np.delete(x_act, idx_del, axis=0)
            x_time = np.delete(x_time, idx_del, axis=0)
            next_act = np.delete(next_act, idx_del, axis=0)
            next_time = np.delete(next_time, idx_del, axis=0)
            ts = np.delete(ts, idx_del, axis=0)
        if len(x_act) == 0:     # all cases ended already
            break
        if np.count_nonzero(x_act[0]) == max_case_length:     # maxlen is reached
            for case, _x_act, _next_act, _t in zip(islice(cases, None), islice(x_act, None), islice(next_act, None),islice(ts, None)):
                suffix_k.at[idx, "case_id"] = case
                suffix_k.at[idx, "k"] = k
                suffix_k.at[idx, "prefix"] = _x_act[np.argmax(_x_act != 0):np.argmax(_x_act != 0)+k]
                s = np.concatenate((_x_act, _next_act))
                suffix_k.at[idx, "suffix"] = s[np.argmax(s != 0):]
                suffix_k.at[idx, "timestamps"] = _t
                idx += 1
            break
        # Generate (k+1)-prefix
        x_act = np.concatenate((x_act, next_act), axis=1)
        x_act = tf.keras.preprocessing.sequence.pad_sequences(x_act, maxlen=max_case_length)
        x_time = next_time
    return suffix_k

def predict_next(prediction_model, x_act, x_time, y_scaler, timestamps, granularity):
    _next_act, next_time = prediction_model.predict([x_act, x_time], verbose=0)
    next_act = np.argmax(_next_act, axis=1).reshape(-1,1)     # 1d -> 2d
    _time_diff = y_scaler.inverse_transform(next_time)
    time_diff = update_granularity(_time_diff, granularity)
    ts_new = list()
    for ts, diff in zip(timestamps, time_diff):
        last_ts = datetime.strptime(ts[-1], "%Y-%m-%d %H:%M:%S")
        append_ts = last_ts + diff
        ts_new.append(np.concatenate((ts, [datetime.strftime(append_ts, "%Y-%m-%d %H:%M:%S")])))
    ts_new = np.array(ts_new)   # 2d
    return next_act, next_time, ts_new

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
    
def check_constraints(df, constraints, activity_dict, result):
    pre, suc, relation_act = "predecessor", "successor", "relation_activity"
    # replace activity names in constraints with coded numerical values
    constraints[pre] = constraints[pre].apply(lambda act: check_coded_value(activity_dict, key=act))
    constraints[suc] = constraints[suc].apply(lambda act: check_coded_value(activity_dict, key=act))

    df_constraints = constraints[constraints[relation_act]=="df"]
    ef_constraints = constraints[constraints[relation_act]=="ef"]
    exist_constraints = constraints[constraints[relation_act]=="exist"]
    coexist_constraints = constraints[constraints[relation_act]=="coexist"]
    
    if result == None:
        result = {"df": {}, "ef": {}, "exist": {}, "coexist": {},}    
    for _, case in df.iterrows():
        result["df"] = check_df_relation(df_constraints, case, result["df"])
        result["ef"] = check_ef_relation(ef_constraints, case, result["ef"])
        result["coexist"] = check_coexist_relation(coexist_constraints, case, result["coexist"])
        result["exist"] = check_exist_relation(exist_constraints, case, result["exist"])
    return result
    
def check_df_relation(df_constraints, trace, df_compliance):
    suffix, ts = trace["suffix"], trace["timestamps"]   # trace["timestamps"] is an nparray of type string
    sample_df = pd.DataFrame(columns=["case_id", "k", "prefix", "suffix", "timestamps", "degree"])
    # check every df constraint
    for _, constraint in df_constraints.iterrows():
        if len(df_compliance.get(constraint["constraint_id"], {})) == 0:
            df_compliance[constraint["constraint_id"]] = {"violations": {"activity": [], "time": sample_df.copy()}, "satisfactions": sample_df.copy()}
        df_compliance_k = df_compliance[constraint["constraint_id"]]
        pre = constraint["predecessor"]
        suc = constraint["successor"]
        if pre in suffix and suc in suffix:
            idx_pre = np.where(suffix==pre)[0][0]
            idx_suc = np.where(suffix==suc)[0][0]
            # check time perspective
            if idx_suc == idx_pre + 1:               
                if constraint["time_type"] == "DURATION":
                    time_actual = relativedelta(datetime.strptime(ts[idx_suc], "%Y-%m-%d %H:%M:%S"), datetime.strptime(ts[idx_pre], "%Y-%m-%d %H:%M:%S"))
                    time_actual = update_duration_granularity(constraint["granularity"], time_actual)
                else:
                    time_actual = datetime.strptime(ts[idx_suc], "%Y-%m-%d %H:%M:%S")
                    time_actual = update_time_granularity(constraint["granularity"], time_actual)
                time_limit = constraint["time_value"]
                time_to_be_scaled = check_time(constraint, time_actual, time_limit)
                trace["degree"] = time_to_be_scaled
                # constraint satisfied
                if time_to_be_scaled >= 0:
                    print(f"-> trace {trace['case_id']}_{trace['k']} satisfies constraint {constraint['constraint_id']}!")
                    df_compliance_k["satisfactions"] = pd.concat([df_compliance_k["satisfactions"], trace.to_frame().T], ignore_index=True)
                # constraint violated: root cause - time
                else:
                    print(f"-> trace {trace['case_id']}_{trace['k']} violates constraint {constraint['constraint_id']} on time!")
                    df_compliance_k["violations"]["time"] = pd.concat([df_compliance_k["violations"]["time"], trace.to_frame().T], ignore_index=True)
            # constraint violated: root cause - successor does not occurs immediately after the predecessor - only record the case id and length of the prefix/case
            else:
                print(f"-> trace {trace['case_id']}_{trace['k']} violates constraint {constraint['constraint_id']} on activity: other events between pre and suc!")
                df_compliance_k["violations"]["activity"].append([trace["case_id"], trace["k"]])       # list([case_0, k_1], [case_1,k_1],...)
        elif pre not in suffix and suc not in suffix:
            continue
        # constraint violated: root cause - predecessor and successor don't occur simultaneously
        else:
            print(f"-> trace {trace['case_id']}_{trace['k']} violates constraint {constraint['constraint_id']} on activity: only one occurs!")
            df_compliance_k["violations"]["activity"].append([trace["case_id"], trace["k"]]) 
    return df_compliance

def check_ef_relation(ef_constraints, trace, ef_compliance):
    suffix, ts = trace["suffix"], trace["timestamps"]
    sample_df = pd.DataFrame(columns=["case_id", "k", "prefix", "suffix", "timestamps", "degree"])
    # check every ef constraint
    for _, constraint in ef_constraints.iterrows():
        if len(ef_compliance.get(constraint["constraint_id"], {})) == 0:
            ef_compliance[constraint["constraint_id"]] = {"violations": {"activity": [], "time": sample_df.copy()}, "satisfactions": sample_df.copy()}
        ef_compliance_k = ef_compliance[constraint["constraint_id"]]
        pre = constraint["predecessor"]
        suc = constraint["successor"]
        if pre in suffix and suc in suffix:
            idx_pre = np.where(suffix==pre)[0][0]   # return the first occurrence! -> delete duplicates in the log?
            idx_suc = np.where(suffix==suc)[0][0]
            # check time perspective
            if idx_suc > idx_pre:
                if constraint["time_type"] == "DURATION":
                    time_actual = relativedelta(datetime.strptime(ts[idx_suc], "%Y-%m-%d %H:%M:%S"), datetime.strptime(ts[idx_pre], "%Y-%m-%d %H:%M:%S"))
                    time_actual = update_duration_granularity(constraint["granularity"], time_actual)
                else:
                    time_actual = datetime.strptime(ts[idx_suc], "%Y-%m-%d %H:%M:%S")
                    time_actual = update_time_granularity(constraint["granularity"], time_actual)
                time_limit = constraint["time_value"]
                time_to_be_scaled = check_time(constraint, time_actual, time_limit)
                trace["degree"] = time_to_be_scaled
                # constraint satisfied
                if time_to_be_scaled >= 0:
                    print(f"-> trace {trace['case_id']}_{trace['k']} satisfies constraint {constraint['constraint_id']}!")
                    ef_compliance_k["satisfactions"] = pd.concat([ef_compliance_k["satisfactions"], trace.to_frame().T], ignore_index=True)
                # constraint violated: root cause - time
                else:
                    print(f"-> trace {trace['case_id']}_{trace['k']} violates constraint {constraint['constraint_id']} on time!")
                    ef_compliance_k["violations"]["time"] = pd.concat([ef_compliance_k["violations"]["time"], trace.to_frame().T], ignore_index=True)
            # constraint violated: root cause - successor occurs before the predecessor - only record the case id and length of the prefix/case
            else:
                print(f"-> trace {trace['case_id']}_{trace['k']} violates constraint {constraint['constraint_id']} on activity: suc before pre!")
                ef_compliance_k["violations"]["activity"].append([trace["case_id"], trace["k"]])       # list([case_0, k_1], [case_1,k_1],...)
        elif pre not in suffix and suc not in suffix:
            continue
        # constraint violated: root cause - predecessor and successor don't occur simultaneously
        else:
            print(f"-> trace {trace['case_id']}_{trace['k']} violates constraint {constraint['constraint_id']} on activity: only one occurs!")
            ef_compliance_k["violations"]["activity"].append([trace["case_id"], trace["k"]]) 
    return ef_compliance

def check_coexist_relation(coexist_constraints, trace, coexist_compliance):
    suffix, ts = trace["suffix"], trace["timestamps"]
    sample_df = pd.DataFrame(columns=["case_id", "k", "prefix", "suffix", "timestamps", "degree"])
    # check every coexist constraint
    for _, constraint in coexist_constraints.iterrows():
        if len(coexist_compliance.get(constraint["constraint_id"], {})) == 0:
            coexist_compliance[constraint["constraint_id"]] = {"violations": {"activity": [], "time": sample_df.copy()}, "satisfactions": sample_df.copy()}
        coexist_compliance_k = coexist_compliance[constraint["constraint_id"]]
        pre = constraint["predecessor"]
        suc = constraint["successor"]
        # check time perspective
        if pre in suffix and suc in suffix:
            idx_pre = np.where(suffix==pre)[0][0]
            idx_suc = np.where(suffix==suc)[0][0]
            if idx_pre > idx_suc:
                tmp = idx_pre
                idx_pre = idx_suc
                idx_suc = tmp           
            if constraint["time_type"] == "DURATION":
                    time_actual = relativedelta(datetime.strptime(ts[idx_suc], "%Y-%m-%d %H:%M:%S"), datetime.strptime(ts[idx_pre], "%Y-%m-%d %H:%M:%S"))
                    time_actual = update_duration_granularity(constraint["granularity"], time_actual)
            else:
                time_actual = datetime.strptime(ts[idx_suc], "%Y-%m-%d %H:%M:%S")
                time_actual = update_time_granularity(constraint["granularity"], time_actual)
            time_limit = constraint["time_value"]
            time_to_be_scaled = check_time(constraint, time_actual, time_limit)
            trace["degree"] = time_to_be_scaled
            # constraint satisfied
            if time_to_be_scaled >= 0:
                print(f"-> trace {trace['case_id']}_{trace['k']} satisfies constraint {constraint['constraint_id']}!")
                coexist_compliance_k["satisfactions"] = pd.concat([coexist_compliance_k["satisfactions"], trace.to_frame().T], ignore_index=True)
            # constraint violated: root cause - time
            else:
                print(f"-> trace {trace['case_id']}_{trace['k']} violates constraint {constraint['constraint_id']} on time!")
                coexist_compliance_k["violations"]["time"] = pd.concat([coexist_compliance_k["violations"]["time"], trace.to_frame().T], ignore_index=True)
        elif pre not in suffix and suc not in suffix:
            continue
        # constraint violated: root cause - predecessor and successor don't occur simultaneously
        else:
            print(f"-> trace {trace['case_id']}_{trace['k']} violates constraint {constraint['constraint_id']} on activity: only one occurs!")
            coexist_compliance_k["violations"]["activity"].append([trace["case_id"], trace["k"]]) 
    return coexist_compliance

def check_exist_relation(exist_constraints, trace, exist_compliance):
    suffix, ts = trace["suffix"], trace["timestamps"]
    sample_df = pd.DataFrame(columns=["case_id", "k", "prefix", "suffix", "timestamps", "degree"])
    # check every exist constraint
    for _, constraint in exist_constraints.iterrows():
        if len(exist_compliance.get(constraint["constraint_id"], {})) == 0:
            exist_compliance[constraint["constraint_id"]] = {"violations": {"activity": [], "time": sample_df.copy()}, "satisfactions": sample_df.copy()}
        exist_compliance_k = exist_compliance[constraint["constraint_id"]]
        act = constraint["predecessor"]
        # check time perspective
        _time_last = datetime.strptime(ts[-1], "%Y-%m-%d %H:%M:%S")
        time_last = update_time_granularity(constraint["granularity"], _time_last)
        time_limit = constraint["time_value"]
        time_last_to_be_scaled = check_time(constraint, time_last, time_limit)
        if act in suffix:
            idx = np.where(suffix==act)[0][0]
            _time_actual = datetime.strptime(ts[idx], "%Y-%m-%d %H:%M:%S")
            time_actual = update_time_granularity(constraint["granularity"], _time_actual)
            time_to_be_scaled = check_time(constraint, time_actual, time_limit)
            trace["degree"] = time_to_be_scaled
            # constraint satisfied
            if time_to_be_scaled >= 0:
                print(f"-> trace {trace['case_id']}_{trace['k']} satisfies constraint {constraint['constraint_id']}!")
                exist_compliance_k["satisfactions"] = pd.concat([exist_compliance_k["satisfactions"], trace.to_frame().T], ignore_index=True)
            # constraint violated: root cause - time
            else:
                print(f"-> trace {trace['case_id']}_{trace['k']} violates constraint {constraint['constraint_id']} on time!")
                exist_compliance_k["violations"]["time"] = pd.concat([exist_compliance_k["violations"]["time"], trace.to_frame().T], ignore_index=True)
        # the condition of presence is not fulfilled
        elif act not in suffix and time_last_to_be_scaled < 0:
            continue
        # constraint violated: root cause - activity does not occur
        else:
            print(f"-> trace {trace['case_id']}_{trace['k']} violates constraint {constraint['constraint_id']} on activity: pre does not occur!")
            exist_compliance_k["violations"]["activity"].append([trace["case_id"], trace["k"]]) 
    return exist_compliance

def check_time(constraint,time_actual, time_limit):
    if constraint["relation_time"] == "interval":
        time_to_be_scaled = calculate_degree_interval(time_actual, time_limit)
    elif constraint["relation_time"] == "max":
        time_to_be_scaled = calculate_degree_max(time_actual, time_limit)
    elif constraint["relation_time"] == "min":
        time_to_be_scaled = calculate_degree_min(time_actual, time_limit)
    else:
        time_to_be_scaled = calculate_degree_exactlyAt(time_actual, time_limit)
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
    if granularity == "MONTH":
        return duration.months
    if granularity == "DAY":
        return duration.days
    if granularity == "HOUR":
        return duration.hours
    if granularity == "MINUTE":
        return duration.minutes    

def calculate_degree_interval(value_actual, value_limit):
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
    
def calculate_degree_max(value_actual, value_max):
    value_max = int(value_max)
    distance_to_max = value_actual - value_max

    if value_actual <= value_max:
        return distance_to_max**2
    else:
        return -distance_to_max**2
    
def calculate_degree_min(value_actual, value_min):
    value_min = int(value_min)
    distance_to_min = value_actual - value_min

    if value_actual >= value_min:
        return distance_to_min**2
    else:
        return -distance_to_min**2

def calculate_degree_exactlyAt(value_actual, value_center):
    value_center = int(value_center)
    distance_to_center = np.abs(value_actual - value_center)
    epsilon = 1e-10
    
    if np.abs(distance_to_center) < epsilon:    # resolve floating-point precision issues via a small epsilon value
        return 5
    else: 
        return -distance_to_center**2
    
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def calculate_probability(df_original):
    scalers = [MinMaxScaler(feature_range=(-5, -1e-5)), MinMaxScaler(feature_range=(0, 5))]
    # Transform values ("degree" column) to probabilities using sigmoid function
    for i, df in enumerate(df_original):
        if len(df) > 0:
            # Fit and transform using the scaler
            df['scaled_degree'] = scalers[i].fit_transform(df[['degree']])               
            # Apply sigmoid function to scaled values
            df['probability'] = sigmoid(df['scaled_degree'])
    return df_original[0], df_original[1]

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
    (train_df, test_df, activity_dict, max_case_length, vocab_size) = data_loader.load_data()
    '''
    (train_act_x, train_time_x, 
        train_act_y, train_time_y, time_scaler, y_scaler) = data_loader.prepare_data(train_df, activity_dict, max_case_length)
    '''
    
    # Load constraints
    constraints = pd.read_excel(f"constraints/{args.dataset}.xlsx")   # excel file

    # Train the prediction model based on the training set
    transformer_model, y_scaler = train_model(train_df, activity_dict, max_case_length, vocab_size, model_path)
    # Load model after training
    '''
    with tf.keras.utils.custom_object_scope({
        'TokenAndPositionEmbedding': model.TokenAndPositionEmbedding,
        'TransformerBlock': model.TransformerBlock}):
            transformer_model = tf.keras.models.load_model(model_path)
    '''
    
    # Prediction suffixes for all prefixes and monitor compliance against all constraints
    for k in range(1, max_case_length+1):
        start = time.time()
        if len(test_df[test_df["k"]==k]) > 0:
            suffix_k = predict_suffix(test_df, k, transformer_model, activity_dict, max_case_length, y_scaler, args.granularity)
            result = check_constraints(suffix_k, constraints, activity_dict, result)
        end = time.time()
        diff = end - start
        print("########## Time spent for {}-prefixes: {:.0f}h {:.0f}m ##########".format(k, diff // 3600, (diff % 3600) // 60))
    
    # Calculate the degree of compliance per constraint
    # df
    if len(result["df"]) > 0:
        for constraint, df_compliance in result["df"].items():
            num_sat = len(df_compliance["satisfactions"])
            num_vio = len(df_compliance["violations"]["activity"]) + len(df_compliance["violations"]["time"])
            healthiness = num_sat / (num_sat + num_vio)
            print(f"compliance degree with respect to {constraint} is {healthiness}")
            print(f"number of satisfactions: {num_sat}; number of time_violations: {len(df_compliance['violations']['time'])}; number of act_violations: {len(df_compliance['violations']['activity'])}")
            # Transform numerical values of "degree" to probabilities via sigmoid function
            df_compliance["violations"]["time"], df_compliance["satisfactions"] = \
                calculate_probability([df_compliance["violations"]["time"], df_compliance["satisfactions"]])
            # Save results
            df_compliance["satisfactions"].to_csv(result_path + constraint + "_satisfaction.csv", index=False)
            df_compliance["violations"]["time"].to_csv(result_path + constraint + "_violation.csv", index=False)
    # ef
    if len(result["ef"]) > 0:
        for constraint, ef_compliance in result["ef"].items():
            num_sat = len(ef_compliance["satisfactions"])
            num_vio = len(ef_compliance["violations"]["activity"]) + len(ef_compliance["violations"]["time"])
            healthiness = num_sat / (num_sat + num_vio)
            print(f"compliance degree with respect to {constraint} is {healthiness}")
            print(f"number of satisfactions: {num_sat}; number of time_violations: {len(ef_compliance['violations']['time'])}; number of act_violations: {len(ef_compliance['violations']['activity'])}")
            ef_compliance["violations"]["time"], ef_compliance["satisfactions"] = \
                calculate_probability([ef_compliance["violations"]["time"], ef_compliance["satisfactions"]])
            ef_compliance["satisfactions"].to_csv(result_path + constraint + "_satisfaction.csv", index=False)
            ef_compliance["violations"]["time"].to_csv(result_path + constraint + "_violation.csv", index=False)
    # exist
    if len(result["exist"]) > 0:
        for constraint, exist_compliance in result["exist"].items():
            num_sat = len(exist_compliance["satisfactions"])
            num_vio = len(exist_compliance["violations"]["activity"]) + len(exist_compliance["violations"]["time"])
            healthiness = num_sat / (num_sat + num_vio)
            print(f"compliance degree with respect to {constraint} is {healthiness}")
            print(f"number of satisfactions: {num_sat}; number of time_violations: {len(exist_compliance['violations']['time'])}; number of act_violations: {len(exist_compliance['violations']['activity'])}")
            exist_compliance["violations"]["time"], exist_compliance["satisfactions"] = \
                calculate_probability([exist_compliance["violations"]["time"], exist_compliance["satisfactions"]])
            exist_compliance["satisfactions"].to_csv(result_path + constraint + "_satisfaction.csv", index=False)
            exist_compliance["violations"]["time"].to_csv(result_path + constraint + "_violation.csv", index=False)
    # coexist
    if len(result["coexist"]) > 0:
        for constraint, coexist_compliance in result["coexist"].items():
            num_sat = len(coexist_compliance["satisfactions"])
            num_vio = len(coexist_compliance["violations"]["activity"]) + len(coexist_compliance["violations"]["time"])
            healthiness = num_sat / (num_sat + num_vio)
            print(f"compliance degree with respect to {constraint} is {healthiness}")
            print(f"number of satisfactions: {num_sat}; number of time_violations: {len(coexist_compliance['violations']['time'])}; number of act_violations: {len(coexist_compliance['violations']['activity'])}")
            coexist_compliance["violations"]["time"], coexist_compliance["satisfactions"] = \
                calculate_probability([coexist_compliance["violations"]["time"], coexist_compliance["satisfactions"]])
            coexist_compliance["satisfactions"].to_csv(result_path + constraint + "_satisfaction.csv", index=False)
            coexist_compliance["violations"]["time"].to_csv(result_path + constraint + "_violation.csv", index=False)
