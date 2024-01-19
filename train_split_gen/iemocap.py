import json
import yaml
import re

from tqdm import tqdm
from pathlib import Path
from sklearn.model_selection import KFold

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
    data_path   = Path(config["data_dir"]["iemocap"])
    output_path = Path(config["project_dir"])

    session_list = ['Session1', 'Session2', 'Session3', 'Session4', 'Session5']

    kf = KFold(n_splits=5, random_state=None, shuffle=False)
    for fold_idx, (train_index, test_index) in enumerate(kf.split(session_list)):
        Path.mkdir(output_path.joinpath('train_split'), parents=True, exist_ok=True)
        train_list, dev_list, test_list = list(), list(), list()

        train_index, dev_index = train_index[:-1], train_index[-1:]
        # read sessions
        train_sessions = [session_list[idx] for idx in train_index]
        dev_sessions = [session_list[idx] for idx in dev_index]
        test_sessions = [session_list[idx] for idx in test_index]
        
        # iemocap 
        for session_id in session_list:
            ground_truth_path_list = list(Path(data_path).joinpath(session_id, 'dialog', 'EmoEvaluation').glob('*.txt'))
            for ground_truth_path in tqdm(ground_truth_path_list, ncols=100, miniters=100):
                with open(str(ground_truth_path)) as f:
                    file_content = f.read()
                    useful_regex = re.compile(r'\[.+\]\n', re.IGNORECASE)
                    label_lines = re.findall(useful_regex, file_content)
                    for line in label_lines:
                        if 'Ses' in line:
                            sentence_file = line.split('\t')[-3]
                            gender = sentence_file.split('_')[-1][0]
                            speaker_id = 'iemocap_' + sentence_file.split('_')[0][:-1] + gender
                            label = line.split('\t')[-2]

                            file_path = Path(data_path).joinpath(
                                session_id, 'sentences', 'wav', '_'.join(sentence_file.split('_')[:-1]), f'{sentence_file}.wav'
                            )
                            # [key, speaker id, gender, path, label]
                            gender_label = "female" if gender == "F" else "male"
                            file_data = [sentence_file, speaker_id, gender_label, str(file_path), label]
                            # append data
                            if session_id in test_sessions: test_list.append(file_data)
                            elif session_id in dev_sessions: dev_list.append(file_data)
                            else: train_list.append(file_data)

        return_dict = dict()
        return_dict['train'], return_dict['dev'], return_dict['test'] = train_list, dev_list, test_list
        logging.info(f'-------------------------------------------------------')
        logging.info(f'Split distribution for IEMOCAP dataset')
        for split in ['train', 'dev', 'test']: logging.info(f'Split {split}: Number of files {len(return_dict[split])}')
        logging.info(f'-------------------------------------------------------')
        
        # dump the dictionary
        jsonString = json.dumps(return_dict, indent=4)
        jsonFile = open(str(output_path.joinpath('train_split', f'iemocap_fold{fold_idx+1}.json')), "w")
        jsonFile.write(jsonString)
        jsonFile.close()

    