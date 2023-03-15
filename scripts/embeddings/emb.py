##File to create embeddings for the given dataset

from mb_pytorch.utils.generate_emb import EmbeddingGenerator
from mb_utils.src.logging import logger
import argparse

def main():
    data = args.file
    if args.log:
        logger = logger
    else:
        logger = None
    all_events = args.all

    emb = EmbeddingGenerator(data, logger=logger)
    emb_loader = emb.data_emb_loader(logger=logger)
    final_emb = emb.generate_emb(emb_loader)
    emb.file_save(final_emb,logger=logger)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, default='./scripts/embeddings/gen_emb_all.yaml', help='path to config file')
    parser.add_argument('--log', type=bool, default=False, help='path to log file')
    parser.add_argument('--all', type=bool, default=True, help='All files will be processed - Training and Validation.Otherwise only training file will be processed')
   
    args = parser.parse_args()
    main()


