from tqdm import tqdm
import os
import gc
import torch
import random
import numpy as np
from loader.MSVD import MSVD
from loader.MSRVTT import MSRVTT
from config import TrainConfig, load_graph_data, clear_graph_data
from model.model import VCModel
from model.modules.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader, RandomSampler
from utils import evaluate, get_lr, load_checkpoint, save_checkpoint, test, train, build_onlyonce_iter
from transformers import get_linear_schedule_with_warmup
from run import build_loader, run
import psutil
from tensorboardX import SummaryWriter
import pickle
import logging
import argparse

global logger
def get_logger(filename=None):
    logger = logging.getLogger('logger')
    logger.setLevel(logging.DEBUG)
    logging.basicConfig(format='%(asctime)s - %(levelname)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
    if filename is not None:
        handler = logging.FileHandler(filename)
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logging.getLogger().addHandler(handler)
    return logger

def build_loaders(C):
    global logger
    corpus = None
    if C.corpus == "MSVD":
        corpus = MSVD(C)
    elif C.corpus == "MSR-VTT":
        corpus = MSRVTT(C)
        
    logger.info('#vocabs: {} ({}), #words: {} ({}). Trim words which appear less than {} times.'.format(
        corpus.vocab.n_vocabs, corpus.vocab.n_vocabs_untrimmed, corpus.vocab.n_words,
        corpus.vocab.n_words_untrimmed, C.loader.min_count))
    
    return corpus.graph_data, corpus.train_data_loader, corpus.val_data_loader, corpus.test_data_loader, corpus.vocab

def build_model(vocab, C):
    model_state_dict = None
    cache_dir = C.transformer.cache_dir if C.transformer.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed')

    model = VCModel(vocab, model_state_dict, cache_dir, C.feat.feature_mode, C.transformer, C.feat.size, C.attention_mode)
    model.cuda()
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--attention', type=int, default=1, choices = [1,2,3])
    parser.add_argument('--model_name', type=str, default="MSVD_GBased+rel+videomask",
                        choices=[
                            "MSVD_GBased+OFeat+rel+videomask",
                            "MSR-VTT_GBased+OFeat+rel+videomask",
                            "MSVD_GBased+rel+videomask",
                            "MSR-VTT_GBased+rel+videomask",
                            "MSVD_GBased+videomask",
                            "MSR-VTT_GBased+videomask"
                        ],
                        help='Specify the model configuration')
    args = parser.parse_args()

    C = TrainConfig(model_name=args.model_name, n_gpus=args.n_gpus, do_train=False)
    C.attention_model = args.attention
    
    global logger
    logger = get_logger(filename="inference.txt")
          
    C.model_id = "MSR-VTT_GBased+OFeat+rel+videomask _ 2025-06-26 00_49_35"    
    logger.info("[BEST: {} SEED: {}]".format(best_epoch, seed))
    folder_path = "./result"
    os.makedirs(folder_path, exist_ok=True)
    f = open(os.path.join(folder_path, "{}.txt".format(C.model_id)), 'w')
    f.write('#vocabs: {} ({}), #words: {} ({}). Trim words which appear less than {} times.\n'.format(
    vocab.n_vocabs, vocab.n_vocabs_untrimmed, vocab.n_words, vocab.n_words_untrimmed, C.loader.min_count))
    f.write("Max caption length: {}\n".format(C.loader.max_caption_len))
    f.write("Heads: {}\n".format(C.transformer.n_heads))
    f.write("Small Heads: {}\n".format(C.transformer.n_heads_small))
    f.write("Big Heads: {}\n".format(C.transformer.n_heads_big))
    f.write("Model Dim: {}\n".format(C.transformer.d_model))
    if args.attention == 1:
        f.write("MHA for relation\n")
    elif args.attention == 2:
        f.write("MHA + pe for relation\n")
    elif args.attention == 3:
        f.write("FFN for relation\n")
    f.write(os.linesep)
    f.write("\n[BEST: {} SEED:{}]".format(best_epoch, seed) + os.linesep)
        
    path = "/workspace/AKG-sv"
    file =  os.path.join(path, f"checkpoints/{C.corpus}/{C.model_id}")
    ckpt_list = os.listdir(file)
    logger.info(file)
    logger.info(ckpt_list)
    logger.info('Build data_loader according to ' + ckpt_list[0])
    
    load_graph_data(C.corpus, 'test')
    test_iter, vocab, l2r_test_vid2GTs = build_loader(file + '/' + ckpt_list[0], device)
    onlyonce_iter = build_onlyonce_iter(test_iter, C.feat.feature_mode, C.transformer.num_object, C.loader.frame_sample_len, device, 'test')
        
    for i in range(len(ckpt_list)):
        if i + 1 <= 3:
            continue
        ckpt_fpath = file + '/' + str(i + 1) + '.ckpt'
        logger.info("Now is test in the " + ckpt_fpath)
        captioning_fpath = C.captioning_fpath_tpl.format(str(i + 1))
        run(ckpt_fpath, onlyonce_iter, vocab, str(i + 1) + '.ckpt', l2r_test_vid2GTs, f, captioning_fpath, C, device)

        logger.info("Memory usage after testing checkpoint {}:".format(ckpt_fpath))
        logger.info("  VRAM used     : {:.2f} MB".format(torch.cuda.memory_allocated() / 1024**2))
        logger.info("  VRAM reserved : {:.2f} MB".format(torch.cuda.memory_reserved() / 1024**2))
        logger.info("  RAM used      : {:.2f} MB".format(psutil.Process(os.getpid()).memory_info().rss / 1024**2))

    f.close()
    del test_iter, onlyonce_iter
    gc.collect()
    torch.cuda.empty_cache()
    clear_graph_data('all')
    
if __name__ == "__main__":
    main()