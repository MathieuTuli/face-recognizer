'''ConfigParser Help'''

import os

import yaml


class ConfigParser:
    '''
    '''
    def __init__(self,
                 config_path: str):
        '''
        '''
        if not os.path.isfile(config_path):
            raise ValueError("Config file could not be found.")

        with open(config_path, 'r') as yaml_file:
            settings = yaml.load(yaml_file)
        print(settings)


__all__ = 'ConfigParser',
