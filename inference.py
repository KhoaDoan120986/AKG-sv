from tqdm import tqdm
import os
import gc
import torch
import random
import numpy as np
from loader.MSVD import MSVD
from loader.MSRVTT import MSRVTT
from config import TrainConfig as C

from model.model import VCModel
from model.modules.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils import evaluate, get_lr, load_checkpoint, save_checkpoint, test, train
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

def build_loaders():
    global logger
    corpus = None
    if C.corpus == "MSVD":
        corpus = MSVD(C)
    elif C.corpus == "MSR-VTT":
        corpus = MSRVTT(C)
        
    logger.info('#vocabs: {} ({}), #words: {} ({}). Trim words which appear less than {} times.'.format(
        corpus.vocab.n_vocabs, corpus.vocab.n_vocabs_untrimmed, corpus.vocab.n_words,
        corpus.vocab.n_words_untrimmed, C.loader.min_count))
    if C.feat.feature_mode in ['grid-rel', 'object-rel', 'grid', 'object']:
        return corpus.graph_data, corpus.train_data_loader, corpus.val_data_loader, corpus.test_data_loader, corpus.vocab
    return None, corpus.train_data_loader, corpus.val_data_loader, corpus.test_data_loader, corpus.vocab

def build_model(vocab):
    # model_state_dict = torch.load(C.transformer.init_model, map_location='cpu')
    model_state_dict = None
    cache_dir = C.transformer.cache_dir if C.transformer.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed')

    model = VCModel(vocab, model_state_dict, cache_dir, C.feat.feature_mode, C.transformer, C.feat.size)
    model.cuda()
    return model

def main():
    parser = argparse.ArgumentParser()
    # Loader settings
    parser.add_argument('--max_caption_len', type=int, default=10)
    parser.add_argument('--frame_sample_len', type=int, default=50)

    # Transformer settings
    parser.add_argument('--n_heads_small', type=int, default=12)
    parser.add_argument('--n_heads_big', type=int, default=128)
    parser.add_argument('--d_model', type=int, default=640)
    parser.add_argument('--hidden_size', type=int, default=768)
    
    # Training settings
    parser.add_argument('--epochs', type=int, default=10)
    args = parser.parse_args()
    
    global logger
    logger = get_logger(filename="log.txt")
    
    C.loader.max_caption_len = args.max_caption_len
    C.loader.frame_sample_len = args.frame_sample_len 
    C.transformer.n_heads_small = args.n_heads_small
    C.transformer.n_heads_big = args.n_heads_big
    C.transformer.d_model = args.d_model
    C.transformer.hidden_size = args.hidden_size
    C.epochs = args.epochs

    model_id = "MSVD_GBased+OFeat+rel+videomask | 2025-05-31 11:59:21"
    
    graph_data, train_iter, val_iter, test_iter, vocab = build_loaders()
    
    model = build_model(vocab)
    optimizer = torch.optim.Adam(model.parameters(), lr=C.lr, weight_decay=C.weight_decay, amsgrad=True)
    lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=C.lr_decay_gamma,
                                    patience=C.lr_decay_patience, verbose=True)
    
    
    best_ckpt_fpath = "/media02/lnthanh01/phatkhoa/ZZZ/checkpoints/MSVD/MSVD_GBased+OFeat+rel+videomask | 2025-05-31 11:59:21/best.ckpt"
    best_model = load_checkpoint(model, best_ckpt_fpath)
    if graph_data is None:
        r2l_best_scores, l2r_best_scores = evaluate(test_iter, None, best_model, vocab, C.beam_size, C.loader.max_caption_len,
                                                C.feat.feature_mode)
    else:
        r2l_best_scores, l2r_best_scores = evaluate(test_iter, graph_data['test'], best_model, vocab, C.beam_size, C.loader.max_caption_len,
                                                    C.feat.feature_mode)
        
        
    folder_path = "./result"
    os.makedirs(folder_path, exist_ok=True)
    f = open(os.path.join(folder_path, "{}.txt".format(C.model_id)), 'w')
    f.write('#vocabs: {} ({}), #words: {} ({}). Trim words which appear less than {} times.'.format(
    vocab.n_vocabs, vocab.n_vocabs_untrimmed, vocab.n_words, vocab.n_words_untrimmed, C.loader.min_count))
    f.write("Max caption length: {}\n".format(C.loader.max_caption_len))
    f.write("Max frame: {}\n".format(C.loader.frame_sample_len))
    f.write("Heads: {}\n".format(C.transformer.n_heads))
    f.write("Small Heads: {}\n".format(C.transformer.n_heads_small))
    f.write("Big Heads: {}\n".format(C.transformer.n_heads_big))
    f.write("Model Dim: {}\n".format(C.transformer.d_model))
    f.write("Hidden Size: {}\n".format(C.transformer.hidden_size))
    f.write("Feature Mode: {}\n".format(C.feat.feature_mode))
    f.write(os.linesep)
    # f.write("\n[BEST: {} SEED:{}]".format(best_epoch, seed) + os.linesep)
    f.write("r2l scores: {}".format(r2l_best_scores))
    f.write(os.linesep)
    f.write("l2r scores: {}".format(l2r_best_scores))
    f.write(os.linesep)
    
    gc.collect()
    torch.cuda.empty_cache()
    
    file = "/media02/lnthanh01/phatkhoa/ZZZ/checkpoints/MSVD/MSVD_GBased+OFeat+rel+videomask | 2025-05-31 11:59:21"
    ckpt_list = os.listdir(file)

    if graph_data is None:
        test_iter, vocab, l2r_test_vid2GTs = build_loader(file + '/' + ckpt_list[0], None)
    else:
        test_iter, vocab, l2r_test_vid2GTs = build_loader(file + '/' + ckpt_list[0], graph_data['test'])
        
    for i in range(len(ckpt_list) - 1):  # because have a best.ckpt
        if i + 1 <= 3:
            continue
        ckpt_fpath = file + '/' + str(i + 1) + '.ckpt'
        logger.info("Now is test in the " + ckpt_fpath)
        captioning_fpath = C.captioning_fpath_tpl.format(str(i + 1))
        
        if graph_data is None:
            run(ckpt_fpath, test_iter, None, vocab, str(i + 1) + '.ckpt', l2r_test_vid2GTs, f, captioning_fpath)
        else: 
            run(ckpt_fpath, test_iter, graph_data['test'], vocab, str(i + 1) + '.ckpt', l2r_test_vid2GTs, f, captioning_fpath)
   
    f.close()
    
if __name__ == "__main__":
    main()