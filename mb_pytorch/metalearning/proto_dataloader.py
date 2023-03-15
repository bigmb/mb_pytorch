# Data load for proto net
# Path: src/metalearning/proto_dataloader.py

from mb_pytorch.dataloader.loader import data_fetcher
from mb_pytorch.utils.yaml_reader import YamlReader
from mb_pandas.src.dfload import load_any_df

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
        self.embeddings = None
        self.data_dict = self.load_data_params()
        self.transforms_final = self.get_transforms()

    def load_embeddings(self):
        """
        load embeddings from yaml file
        Input:
            yaml: yaml file path (CSV or parquet)
        Output:
            embeddings: embeddings from yaml file as (CSV or parquet)
        """
        file = load_any_df(self.data_dict['data_train']['path'])
        self.embeddings = file['embeddings']
        return self.embeddings

