import json
import yaml
import pandas as pd
from pathlib import Path
import random
from sklearn.model_selection import LeavePOut

# define logging console
import logging

logging.basicConfig(
    format='%(asctime)s %(levelname)-3s ==> %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)

if __name__ == '__main__':

    # Read data path
    with open("../config/config.yml", "r") as stream:
        config = yaml.safe_load(stream)
    data_path   = str(Path(config["data_dir"]["commsense"]))
    output_path = Path(config["project_dir"])

    Path.mkdir(output_path.joinpath('train_split'), parents=True, exist_ok=True)
    train_list, dev_list, test_list = list(), list(), list()

    df_labels = pd.read_csv(data_path + "/metadata.csv", index_col=False)
    # 使用 apply 将小于 4 的值设为 "Neutral"，大于等于 4 的值设为 "Anxious"
    df_labels['state_anxiety'] = df_labels['state_anxiety'].apply(lambda x: 'neutral' if x < 4 else 'anxious')

    audio_paths = df_labels['file_name'].tolist()

    audio_paths = [data_path + '/' + path for path in audio_paths]

    pid_labels = df_labels['PID'].tolist()
    anxiety_labels = df_labels['state_anxiety'].tolist()
    eval_labels = df_labels['evaluative'].tolist()
    size_labels = df_labels['if_group'].tolist()

    # 获取唯一的说话者ID列表
    unique_pids = df_labels['PID'].unique()
    # Leave-5-Out Cross Validation
    lpo = LeavePOut(5)

    print(lpo.get_n_splits(unique_pids))
    # 创建一个带有音频路径和标签的字典
    data_dict = {
        "audio": audio_paths,
        "pid": pid_labels,
        "anxious": anxiety_labels,
        # "eval": eval_labels,
        # "size": size_labels
    }
    train_indices=[1,2,3,4,6,7,8,9,10,11,13,14,15,16,17,18,19,21, 22, 23, 24, 25, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45]
    test_indices=[0, 5, 12, 20, 26]
    # 遍历Leave-5-Out的拆分
    # for train_index, test_index in lpo.split(unique_pids):
    #     print(f"  Train: index={train_index}")
    #     print(f"  Test:  index={test_index}")
        # 获取当前拆分的训练PID和测试PID
    train_pids = []
    test_pids = []
    for index in train_indices:
        train_pids.append(unique_pids[index])
    for index in test_indices:
        test_pids.append(unique_pids[index])

    # 遍历 data_dict 字典
    for idx, pid in enumerate(data_dict["pid"]):
        # 检查当前的 pid 是否在训练集中
        if pid in train_pids:
            # 将对应的音频路径和焦虑标签添加到 train_list 中
            train_list.append([data_dict["audio"][idx], data_dict["anxious"][idx]])
        elif pid in test_pids:
            test_list.append([data_dict["audio"][idx], data_dict["anxious"][idx]])

    return_dict = dict()
    return_dict['train'], return_dict['dev'], return_dict['test'] = train_list, dev_list, test_list
    logging.info(f'-------------------------------------------------------')
    logging.info(f'Split distribution for Commsense dataset')
    for split in ['train', 'dev', 'test']:
        logging.info(f'Split {split}: Number of files {len(return_dict[split])}')
    logging.info(f'-------------------------------------------------------')

    # dump the dictionary
    jsonString = json.dumps(return_dict, indent=4)
    jsonFile = open(str(output_path.joinpath('train_split', f'commsense.json')), "w")
    jsonFile.write(jsonString)
    jsonFile.close()

