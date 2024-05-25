import os
import torch.backends.cudnn as cudnn
import yaml
from EasyOCR.trainer.train import train
from EasyOCR.trainer.utils import AttrDict
from change_config import update_yaml_parameters
import pandas as pd

cudnn.benchmark = True
cudnn.deterministic = False

def get_config(file_path):
    with open(file_path, 'r', encoding="utf8") as stream:
        opt = yaml.safe_load(stream)
    opt = AttrDict(opt)
    if opt.lang_char == 'None':
        characters = ''
        for data in opt['select_data'].split('-'):
            csv_path = os.path.join(opt['train_data'], data, 'labels.csv')
            df = pd.read_csv(csv_path, sep='^([^,]+),', engine='python', usecols=['filename', 'words'], keep_default_na=False)
            all_char = ''.join(df['words'])
            characters += ''.join(set(all_char))
        characters = sorted(set(characters))
        opt.character= ''.join(characters)
    else:
        opt.character = opt.number + opt.symbol + opt.lang_char
    os.makedirs(f'./saved_models/{opt.experiment_name}', exist_ok=True)
    return opt

experiment_name = 'new_experiment_models'
num_iter = 10000
valInterval = 100
train_data_name = 'dataset1'

input_file = r'EasyOCR/trainer/config_files/en_filtered_config.yaml'
path_to_new_yaml = rf'recognition_module/datasets/{train_data_name}/{train_data_name}_config_file.yaml'

updates = {
    'experiment_name': experiment_name,
    'num_iter': 10000,
    'valInterval': 500,
    'train_data': f'recognition_module/datasets/{train_data_name}',
    'valid_data': fr'recognition_module/datasets/{train_data_name}/validation',
    'select_data': fr'dataset',
    'saved_model': fr'recognition_module/models/{train_data_name}'
}

update_yaml_parameters(input_file, updates, train_data_name)

opt = get_config(path_to_new_yaml)
train(opt, amp=False)