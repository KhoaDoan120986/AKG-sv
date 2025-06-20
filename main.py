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
    if C.feat.feature_mode in ['grid-rel', 'object-rel', 'grid', 'object', 'grid-rel-no_obj']:
        return corpus.graph_data, corpus.train_data_loader, corpus.val_data_loader, corpus.test_data_loader, corpus.vocab
    return None, corpus.train_data_loader, corpus.val_data_loader, corpus.test_data_loader, corpus.vocab

def build_model(vocab):
    # model_state_dict = torch.load(C.transformer.init_model, map_location='cpu')
    model_state_dict = None
    cache_dir = C.transformer.cache_dir if C.transformer.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed')

    model = VCModel(vocab, model_state_dict, cache_dir, C.feat.feature_mode, C.transformer, C.feat.size)
    model.cuda()
    return model


def log_train(summary_writer, e, loss, lr, reg_lambda, scores=None):
    global logger
    summary_writer.add_scalar(C.tx_train_loss, loss['total'], e)
    summary_writer.add_scalar(C.tx_train_r2l_cross_entropy_loss, loss['r2l_loss'], e)
    summary_writer.add_scalar(C.tx_train_l2r_cross_entropy_loss, loss['l2r_loss'], e)
    summary_writer.add_scalar(C.tx_lr, lr, e)
    
    
    logger.info("Train loss: {} = (1 - reg): {} * r2l_loss: {} + (reg):{} * l2r_loss: {} ".format(
        loss['total'], 1 - reg_lambda, loss['r2l_loss'], reg_lambda, loss['l2r_loss']))

    if scores is not None:
        for metric in C.metrics:
            summary_writer.add_scalar("TRAIN SCORE/{}".format(metric), scores[metric], e)
        logger.info("scores: {}".format(scores))


def log_val(summary_writer, e, loss, reg_lambda, r2l_scores, l2r_scores):
    global logger
    summary_writer.add_scalar(C.tx_val_loss, loss['total'], e)
    summary_writer.add_scalar(C.tx_val_r2l_cross_entropy_loss, loss['r2l_loss'], e)
    summary_writer.add_scalar(C.tx_val_l2r_cross_entropy_loss, loss['l2r_loss'], e)
    
    logger.info("Validation loss: {} = (1 - reg): {} * r2l_loss: {} + (reg):{} * l2r_loss: {} ".format(
        loss['total'], 1 - reg_lambda, loss['r2l_loss'], reg_lambda, loss['l2r_loss']))
    
    for metric in C.metrics:
        summary_writer.add_scalar("VAL R2L SCORE/{}".format(metric), r2l_scores[metric], e)
    for metric in C.metrics:
        summary_writer.add_scalar("VAL L2R SCORE/{}".format(metric), l2r_scores[metric], e)
        
    logger.info("r2l_scores: {}".format(r2l_scores))
    logger.info("l2r_scores: {}".format(l2r_scores))


def log_test(summary_writer, e, r2l_scores, l2r_scores):
    global logger
    
    for metric in C.metrics:
        summary_writer.add_scalar("TEST R2L SCORE/{}".format(metric), r2l_scores[metric], e) 
    logger.info("r2l_scores: {}".format(r2l_scores))
    
    for metric in C.metrics:
        summary_writer.add_scalar("TEST L2R SCORE/{}".format(metric), l2r_scores[metric], e)
    logger.info("l2r_scores: {}".format(l2r_scores))
    
def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

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
    
    C.loader.max_caption_len = args.max_caption_len
    C.loader.frame_sample_len = args.frame_sample_len 
    C.transformer.n_heads_small = args.n_heads_small
    C.transformer.n_heads_big = args.n_heads_big
    C.transformer.d_model = args.d_model
    C.transformer.hidden_size = args.hidden_size
    C.epochs = args.epochs

    global logger
    logger = get_logger(filename="log.txt")
    logger.info("MODEL ID: {}".format(C.model_id))
    summary_writer = SummaryWriter(C.log_dpath)
    # seed = 2
    # seed = 16
    # seed = 8
    # seed = 10
    # seed = 100
    # seed = 44
    # seed = 444  # for ResNet152 seems good
    seed = 904666
    # seed = 7242
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    logger.info("Max caption length: {}".format(C.loader.max_caption_len))
    logger.info("Max frame: {}".format(C.loader.frame_sample_len))
    logger.info("Heads: {}".format(C.transformer.n_heads))
    logger.info("Small Heads: {}".format(C.transformer.n_heads_small))
    logger.info("Big Heads: {}".format(C.transformer.n_heads_big))
    logger.info("Model Dim: {}".format(C.transformer.d_model))
    logger.info("Hidden Size: {}".format(C.transformer.hidden_size))
    logger.info("Feature Mode: {}".format(C.feat.feature_mode))
    graph_data, train_iter, val_iter, test_iter, vocab = build_loaders()
    
    print("[Memory when loading data]")
    tmp = round(torch.cuda.memory_allocated() / 1024**2)
    print("  VRAM used     : {} MB".format(tmp))
    
    tmp = round(torch.cuda.memory_reserved() / 1024**2)
    print("  VRAM reserved : {} MB".format(tmp))
    
    tmp = round(psutil.Process(os.getpid()).memory_info().rss / 1024**2)
    print("  RAM used      : {} MB".format(tmp))

    model = build_model(vocab)
    # C.lr_decay_start_from = 8 # ban đầu là 12
    # C.lr_decay_patience = 3 # ban đầu là 5
    # optimizer = torch.optim.Adam(model.parameters(), lr=C.lr, weight_decay=C.weight_decay, amsgrad=True)
    # # lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=C.lr_decay_gamma,
    # #                                 patience=C.lr_decay_patience, verbose=True) # monitor theo loss

    # lr_scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=C.lr_decay_gamma,
    #                                 patience=C.lr_decay_patience, verbose=True) # monitor theo metric

    gradient_accumulation_steps = 2
    optimizer = torch.optim.Adam(model.parameters(), lr=C.lr, weight_decay=C.weight_decay)
    num_training_steps = int(len(train_iter) / gradient_accumulation_steps) * C.epochs
    num_warmup_steps = int(0.1 * num_training_steps)
    lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
    
    best_val_CIDEr = -1
    best_epoch = None
    best_ckpt_fpath = None

    parameter_number = get_parameter_number(model)
    logger.info(parameter_number)
    
    vram_u = []
    vram_r = []
    ram = []
    for e in range(1, C.epochs + 1):
        ckpt_fpath = C.ckpt_fpath_tpl.format(e)
        
        """ Train """
        if graph_data is None:
            train_loss = train(e, model, optimizer, train_iter, None, vocab, 
                            C.reg_lambda, C.gradient_clip, C.feat.feature_mode, lr_scheduler)
        else:
            train_loss = train(e, model, optimizer, train_iter, graph_data['train'], vocab, 
                                C.reg_lambda, C.gradient_clip, C.feat.feature_mode, lr_scheduler)
        log_train(summary_writer, e, train_loss, get_lr(optimizer), C.reg_lambda)

        vram_u.append(round(torch.cuda.memory_allocated() / 1024**2))
        vram_r.append(round(torch.cuda.memory_reserved() / 1024**2))
        ram.append(round(psutil.Process(os.getpid()).memory_info().rss / 1024**2)) 

        """ Validation """
        if graph_data is None:
            val_loss = test(model, val_iter, None, vocab, C.reg_lambda, C.feat.feature_mode)
            r2l_val_scores, l2r_val_scores = evaluate(val_iter, None, model, vocab, C.beam_size, C.loader.max_caption_len,
                                                    C.feat.feature_mode)
        else: 
            val_loss = test(model, val_iter, graph_data['val'], vocab, C.reg_lambda, C.feat.feature_mode)
            r2l_val_scores, l2r_val_scores = evaluate(val_iter, graph_data['val'], model, vocab, C.beam_size, C.loader.max_caption_len,
                                                    C.feat.feature_mode)
        log_val(summary_writer, e, val_loss, C.reg_lambda, r2l_val_scores, l2r_val_scores)

        summary_writer.add_scalars("compare_loss/total_loss", {'train_total_loss': train_loss['total'],
                                                                'val_total_loss': val_loss['total']}, e)
        summary_writer.add_scalars("compare_loss/l2r_loss", {'train_l2r_loss': train_loss['l2r_loss'],
                                                            'val_l2r_loss': val_loss['l2r_loss']}, e)
        summary_writer.add_scalars("compare_loss/r2l_loss", {'train_r2l_loss': train_loss['r2l_loss'],
                                                            'val_r2l_loss': val_loss['r2l_loss']}, e)


        if e >= C.save_from and e % C.save_every == 0:
            logger.info("Saving checkpoint at epoch={} to {}".format(e, ckpt_fpath))
            save_checkpoint(e, model, ckpt_fpath, C)

        # if e >= C.lr_decay_start_from:
        #     # lr_scheduler.step(val_loss['total'])
        #     lr_scheduler.step(l2r_val_scores['CIDEr'])
        if l2r_val_scores['CIDEr'] > best_val_CIDEr:
            best_epoch = e
            best_val_CIDEr = l2r_val_scores['CIDEr']
            best_ckpt_fpath = ckpt_fpath
        
    print("[Memory when training]")
    
    tmp = np.mean(vram_u) 
    print("  VRAM used     : {} MB".format(tmp))
    
    tmp = np.mean(vram_r)
    print("  VRAM reserved : {} MB".format(tmp))
    
    tmp = np.mean(ram)
    print("  RAM used      : {} MB".format(tmp))
    
    """ Test with Best Model """
    gc.collect()
    torch.cuda.empty_cache()

    logger.info("[BEST: {} SEED: {}]".format(best_epoch, seed))
    best_model = load_checkpoint(model, best_ckpt_fpath)

    if graph_data is None:
        r2l_best_scores, l2r_best_scores = evaluate(test_iter, None, best_model, vocab, C.beam_size, C.loader.max_caption_len,
                                                C.feat.feature_mode)
    else:
        r2l_best_scores, l2r_best_scores = evaluate(test_iter, graph_data['test'], best_model, vocab, C.beam_size, C.loader.max_caption_len,
                                                    C.feat.feature_mode)
    
    logger.info("r2l scores: {}".format(r2l_best_scores))
    logger.info("l2r scores: {}".format(l2r_best_scores))

    for metric in C.metrics:
        summary_writer.add_scalar("BEST R2L SCORE/{}".format(metric), r2l_best_scores[metric], best_epoch)
    for metric in C.metrics:
        summary_writer.add_scalar("BEST L2R SCORE/{}".format(metric), l2r_best_scores[metric], best_epoch)
        
    save_checkpoint(best_epoch, best_model, C.ckpt_fpath_tpl.format("best"), C)
    
    
    folder_path = "./result"
    os.makedirs(folder_path, exist_ok=True)
    f = open(os.path.join(folder_path, "{}.txt".format(C.model_id)), 'w')
    f.write('#vocabs: {} ({}), #words: {} ({}). Trim words which appear less than {} times.'.format(
    vocab.n_vocabs, vocab.n_vocabs_untrimmed, vocab.n_words, vocab.n_words_untrimmed, C.loader.min_count))
    f.write("Max caption length: {}\n".format(C.loader.max_caption_len))
    f.write("Heads: {}\n".format(C.transformer.n_heads))
    f.write("Small Heads: {}\n".format(C.transformer.n_heads_small))
    f.write("Big Heads: {}\n".format(C.transformer.n_heads_big))
    f.write("Model Dim: {}\n".format(C.transformer.d_model))
    f.write(os.linesep)
    f.write("\n[BEST: {} SEED:{}]".format(best_epoch, seed) + os.linesep)
    f.write("r2l scores: {}".format(r2l_best_scores))
    f.write(os.linesep)
    f.write("l2r scores: {}".format(l2r_best_scores))
    f.write(os.linesep)
        
    summary_writer.close()
    del train_iter, val_iter, test_iter, vocab, best_model, model, optimizer, lr_scheduler
    del train_loss
    gc.collect()
    torch.cuda.empty_cache()
        
    file = C.ckpt_dpath
    ckpt_list = os.listdir(file)
    logger.info(file)
    logger.info(ckpt_list)
    logger.info('Build data_loader according to ' + ckpt_list[0])
    
    if graph_data is None:
        test_iter, vocab, l2r_test_vid2GTs = build_loader(file + '/' + ckpt_list[0], None)
    else:
        test_iter, vocab, l2r_test_vid2GTs = build_loader(file + '/' + ckpt_list[0], None)
        
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
