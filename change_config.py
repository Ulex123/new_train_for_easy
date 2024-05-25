import yaml

class MyDumper(yaml.SafeDumper):
    def represent_str(self, data):
        return self.represent_scalar('tag:yaml.org,2002:str', data)

MyDumper.add_representer(str, MyDumper.represent_str)

def update_yaml_parameters(input_file, updates, name_file):

    with open(input_file, 'r', encoding='utf-8') as file:
        data = yaml.safe_load(file)

    # Список параметров, которые разрешено изменять
    allowed_keys = {'experiment_name', 'select_data', 'saved_model', 'num_iter', 'valInterval', 'train_data', 'valid_data', 'path_save_model'}

    # Обновление данных только для разрешенных параметров
    for key, value in updates.items():
        if key in allowed_keys:
            data[key] = value
    yaml_output = rf'{data["train_data"]}/{name_file}_config_file.yaml'
    # Сохранение обновленных данных в новый YAML файл
    with open(yaml_output, 'w', encoding='utf-8') as file:
        yaml.dump(data, file, allow_unicode=True, Dumper=MyDumper)
