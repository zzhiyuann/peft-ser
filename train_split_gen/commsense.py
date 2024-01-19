import json
import yaml
import pandas as pd
from pathlib import Path
import random

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
    data_path   = Path(config["data_dir"]["commsense"])
    output_path = Path(config["project_dir"])

    Path.mkdir(output_path.joinpath('train_split'), parents=True, exist_ok=True)
    train_list, dev_list, test_list = list(), list(), list()

    df_labels = pd.read_csv("../data/dataset/transformer_sentence/metadata.csv", index_col=False)
    df_labels.state_anxiety[df_labels.state_anxiety < 4] = "neutral"
    df_labels.state_anxiety[df_labels.state_anxiety > 3] = "anxious"
    audio_paths = df_labels['file_name'].tolist()

    audio_paths = ["../data/dataset/transformer_sentence/" + path for path in audio_paths]

    pid_labels = df_labels['PID'].tolist()
    anxiety_labels = df_labels['state_anxiety'].tolist()
    eval_labels = df_labels['evaluative'].tolist()
    size_labels = df_labels['if_group'].tolist()
    # 创建一个带有音频路径和标签的字典
    data_dict = {
        "audio": audio_paths,
        "pid": pid_labels,
        "anxious": eval_labels,
        # "eval": eval_labels,
        # "size": size_labels
    }
    # 定义筛选函数
    # 定义划分数据集的比例
    train_ratio = 0.7  # 70% 训练集
    test_ratio = 0.2  # 20% 测试集
    validation_ratio = 0.1  # 10% 验证集

    # 获取数据集的长度
    num_rows = len(data_dict["audio"])

    # 生成随机索引用于划分数据集
    indices = list(range(num_rows))
    random.shuffle(indices)

    # 计算划分边界
    train_split = int(train_ratio * num_rows)
    test_split = int((train_ratio + test_ratio) * num_rows)

    # 划分数据集
    train_indices = indices[:train_split]
    test_indices = indices[train_split:test_split]
    validation_indices = indices[test_split:]

    # 将数据按照索引添加到对应的列表中
    for idx in train_indices:
        train_list.append([data_dict["audio"][idx], data_dict["pid"][idx], data_dict["anxious"][idx]])

    for idx in test_indices:
        test_list.append([data_dict["audio"][idx], data_dict["pid"][idx], data_dict["anxious"][idx]])

    for idx in validation_indices:
        dev_list.append([data_dict["audio"][idx], data_dict["pid"][idx], data_dict["anxious"][idx]])

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

