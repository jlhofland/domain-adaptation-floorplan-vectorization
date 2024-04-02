import lmdb
import os
import pickle
import argparse
import logging
import numpy as np
from datetime import datetime
from tqdm import tqdm
from datasets.cubicasa5k.loaders.svg_loader import FloorplanSVG
import warnings


def main(args, logger):
    logger.info("Opening database...")
    env = lmdb.open(args.lmdb, map_size=int(200e9))

    logger.info("Creating data loader...")
    data = FloorplanSVG(args.data_path, args.txt, format='txt', original_size=True)

    logger.info("Parsing data...")
    if args.overwrite:
        for d in tqdm(data):
            key = d['folder']
            logger.info("Adding " + key)
            with env.begin(write=True, buffers=True) as txn:
                txn.put(key.encode('ascii'), pickle.dumps(d))
    else:
        folders = np.genfromtxt(args.data_path + args.txt, dtype='str')
        for i, f in tqdm(enumerate(folders), total=len(folders)):
            with env.begin(write=True, buffers=True) as txn:
                elem = txn.get(f.encode('ascii'))
                if not elem:
                    logger.info("Adding " + f)
                    elem = data[i]
                    txn.put(f.encode('ascii'), pickle.dumps(elem))
                else:
                    logger.info(f + ' already exists')

    logger.info("Database creation complete.")


if __name__ == '__main__':
    # Ignore UserWarning related to libpng
    warnings.filterwarnings("ignore", category=UserWarning, module=".*libpng.*")

    time_stamp = datetime.now().strftime("%Y-%m-%d-%H_%M_%S")
    parser = argparse.ArgumentParser(description='Script for creating lmdb database.')
    parser.add_argument('--txt', nargs='?', type=str, default='', required=True,
                        help='Path to text file containing file paths')
    parser.add_argument('--lmdb', nargs='?', type=str,
                        default='./data/cubi_lmdb/', help='Path to lmdb')
    parser.add_argument('--data-path', nargs='?', type=str, default='./data/',
                        help='Path to data directory')
    parser.add_argument('--log-path', nargs='?', type=str, default='runs',
                        help='Path to log directory')
    parser.add_argument('--overwrite', nargs='?', type=bool, default=False,
                        const=True, help='Overwrite existing data')
    args = parser.parse_args()

    log_dir = args.log_path + '/' + time_stamp + '/'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logger = logging.getLogger('lmdb')
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(log_dir+'/lmdb.log')
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    main(args, logger)
