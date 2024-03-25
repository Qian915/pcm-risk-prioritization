import os
import warnings
warnings.filterwarnings("ignore")
import json
import argparse
import pm4py
import pandas as pd
import numpy as np
import datetime
import time
from dateutil.relativedelta import relativedelta

parser = argparse.ArgumentParser(
    description="Data Processing")

parser.add_argument("--dataset", 
    type=str, 
    default="bpic2011_c1",
    help="dataset name")

parser.add_argument("--file_format", 
    type=str, 
    default="csv",
    help="format of event log")

parser.add_argument("--granularity", 
    type=str, 
    default="DAY",
    help="granularity of temporal features")

args = parser.parse_args()

def load_log(log_name, file_format):
    if file_format == "xes":
        xes_file_path = f"data/{log_name}/{log_name}.xes"
        log = pm4py.read_xes(xes_file_path)
        df = pm4py.convert_to_dataframe(log)
    else:
        csv_file_path = f"data/{log_name}/{log_name}.csv"
        df = pd.read_csv(csv_file_path)
        #df = df.sample(frac=1, random_state=42)         ##### shuffle df #####

    df = df[["case:concept:name", "concept:name","time:timestamp"]]
    # for BPIC 2011: delete empty spaces
    if log_name == "BPIC_2011":
        df['concept:name'] = df['concept:name'].apply(delte_empty_spaces)
        
    df["concept:name"] = df["concept:name"].str.lower()
    df.sort_values(by = ["time:timestamp"], inplace = True)	# sort df by timestamp
    df["time:timestamp"] = pd.to_datetime(df["time:timestamp"]).map(lambda x: x.strftime("%Y-%m-%d %H:%M:%S"))

    # add [eoc] to the end of each case
    df_new = pd.DataFrame(columns=df.columns)
    case_id, activity, timestamp = "case:concept:name", "concept:name","time:timestamp"
    unique_cases = df[case_id].unique()
    for _, case in enumerate(unique_cases):
        trace = df[df[case_id] == case]
        trace.sort_values(by = [timestamp], inplace = True)
        eoc = pd.Series({case_id: case, activity: "[eoc]", timestamp: trace[timestamp].iloc[-1]})
        trace = pd.concat([trace, eoc.to_frame().T], ignore_index=True)
        df_new = pd.concat([df_new, trace], ignore_index=True)
    return df_new


def delte_empty_spaces(value):
    # only for BPIC 2011
    if value == 'administratief tarief       - eerste pol':
        return 'administratief tarief - eerste pol'
    else:
        return value
    
def extract_logs_metadata(df, dir_path):
    # mapping for activities
    activities = list(df["concept:name"].unique())
    keys = ["[PAD]"]
    keys.extend(activities)
    val = range(len(keys))

    coded_activity = dict(zip(keys, val))
    coded_json = json.dumps(coded_activity)
    with open(f"{dir_path}/metadata.json", "w") as metadata_file:
        metadata_file.write(coded_json)

def process_df_train(df_train, dir_path, log_name, granularity):
    case_id, activity, timestamp = "case:concept:name", "concept:name","time:timestamp"
    processed_df = pd.DataFrame(columns=["case_id", "k", "prefix", "latest_time", "next_activity", "next_time"])
    idx = 0
    unique_cases = df_train[case_id].unique()
    for _, case in enumerate(unique_cases):
        act = df_train[df_train[case_id] == case][activity].to_list()
        time = df_train[df_train[case_id] == case][timestamp].str[:19].to_list()
        latest_diff = relativedelta()
        next_time = relativedelta()
        for i in range(len(act) - 1):
            prefix = np.where(i == 0, act[0], ", ".join(act[:i+1]))
            next_act = act[i + 1]
            if i > 0:
                latest_diff = relativedelta(datetime.datetime.strptime(time[i], "%Y-%m-%d %H:%M:%S"),
                                    datetime.datetime.strptime(time[i-1], "%Y-%m-%d %H:%M:%S"))

            latest_time = np.where(i == 0, 0, update_granularity(latest_diff, granularity))
            next_time = relativedelta(datetime.datetime.strptime(time[i+1], "%Y-%m-%d %H:%M:%S"),
                        datetime.datetime.strptime(time[i], "%Y-%m-%d %H:%M:%S"))

            processed_df.at[idx, "case_id"] = case
            processed_df.at[idx, "k"] = i+1
            processed_df.at[idx, "prefix"] = prefix
            processed_df.at[idx, "latest_time"] = latest_time
            processed_df.at[idx, "next_activity"] = next_act
            processed_df.at[idx, "next_time"] = update_granularity(next_time, granularity)
            idx = idx + 1
    processed_df.to_csv(f"{dir_path}/{log_name}_train.csv", index=False)

def process_df_test(df_test, dir_path, log_name, granularity):
    case_id, activity, timestamp = "case:concept:name", "concept:name","time:timestamp"
    processed_df = pd.DataFrame(columns=["case_id", "k", "prefix", "latest_time", "timestamps", "next_activity", "next_time"])
    idx = 0
    unique_cases = df_test[case_id].unique()
    for _, case in enumerate(unique_cases):
        act = df_test[df_test[case_id] == case][activity].to_list()
        time = df_test[df_test[case_id] == case][timestamp].str[:19].to_list()
        latest_diff = relativedelta()
        next_time = relativedelta()
        for i in range(len(act) - 1):
            prefix = np.where(i == 0, act[0], ", ".join(act[:i+1]))
            timestamps = np.where(i == 0, time[0], ", ".join(time[:i+1]))
            next_act = act[i + 1]
            if i > 0:
                latest_diff = relativedelta(datetime.datetime.strptime(time[i], "%Y-%m-%d %H:%M:%S"),
                                    datetime.datetime.strptime(time[i-1], "%Y-%m-%d %H:%M:%S"))
                
            latest_time = np.where(i == 0, 0, update_granularity(latest_diff, granularity))
            next_time = relativedelta(datetime.datetime.strptime(time[i+1], "%Y-%m-%d %H:%M:%S"),
                        datetime.datetime.strptime(time[i], "%Y-%m-%d %H:%M:%S"))

            processed_df.at[idx, "case_id"] = case
            processed_df.at[idx, "k"] = i+1
            processed_df.at[idx, "prefix"] = prefix
            processed_df.at[idx, "latest_time"] = latest_time
            processed_df.at[idx, "timestamps"] = timestamps    # an extra column to record timestamps of events
            processed_df.at[idx, "next_activity"] = next_act
            processed_df.at[idx, "next_time"] = update_granularity(next_time, granularity)
            idx = idx + 1
    processed_df.to_csv(f"{dir_path}/{log_name}_test.csv", index=False)

def update_granularity(time, granularity):
    if granularity == "HOUR":
        if time.days:
            time = time.days * 24 + time.hours
        else:
            time = time.hours
    else:   # granularity: DAY
        time = time.days
    return time

if __name__ == "__main__":
    if not os.path.exists(f"data/{args.dataset}/processed"):
        os.makedirs(f"data/{args.dataset}/processed")
    dir_path = f"data/{args.dataset}/processed"
    # load event log (default: xes file)
    df = load_log(args.dataset, args.file_format)
    # extract activity dict
    extract_logs_metadata(df, dir_path)
    # process training and test set
    train_test_ratio = int(abs(df["case:concept:name"].nunique()*0.8))   
    train_list = df["case:concept:name"].unique()[:train_test_ratio]
    test_list = df["case:concept:name"].unique()[train_test_ratio:]
    train_df = df[df["case:concept:name"].isin(train_list)]
    test_df = df[df["case:concept:name"].isin(test_list)]
    start = time.time()
    process_df_train(train_df, dir_path, args.dataset, args.granularity)
    process_df_test(test_df, dir_path, args.dataset, args.granularity)
    end = time.time()
    diff = end - start
    print("########## Time spent for data processing: {:.0f}h {:.0f}m ##########".format(diff // 3600, (diff % 3600) // 60))
