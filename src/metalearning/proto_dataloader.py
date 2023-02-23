# Data load for proto net
# Path: src/metalearning/proto_dataloader.py

from src.dataloader.loader import data_fetcher
from src.utils.yaml_reader import YamlReader

__all__ = ['ProtoDataLoader']

class ProtoDataLoader(data_fetcher):
    """
    Data loader for proto net
    """
    def __init__(self, yaml, logger=None) -> None:
        super().__init__(yaml, logger)
        self.yaml = yaml
        self.logger = logger
        self._data = None
        self.data_dict = {}
        self.transforms_final = []
        self.data_dict = self.load_data_params()
        self.transforms_final = self.get_transforms()

    
    def load_embeddings(self):
        """
        load embeddings from yaml file
        """
        pass