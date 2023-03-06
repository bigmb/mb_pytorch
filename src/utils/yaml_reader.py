#file for reading all the data from yaml file and returning the data

import yaml
import os
from mb_utils.src.logging import logger

__all__ = ['YamlReader']

class YamlReader:
    """
    Yaml file reader class
    Input:
        Yaml: yaml file path
    Output:
        data: list of data from yaml file
    """
    def __init__(self,Yaml) -> None:
        if os.path.exists(Yaml):
            self.yaml = Yaml
        else:
            raise FileNotFoundError("Yaml file not found")
        self._data = None
    
    def data(self,logger=None):
        """
        read data from yaml file and return a list
        """
        assert self.yaml.endswith('.yaml') or self.yaml.endswith('.yml'), "Yaml file format wrong"
        if not self._data:
            with open(self.yaml,'r',encoding='utf-8') as f:
                self._data = list(yaml.safe_load_all(f))
        if logger:
            logger.info("Read data from yaml file: {}".format(self._data))
        return self._data[0]
    