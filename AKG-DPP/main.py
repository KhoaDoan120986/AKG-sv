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
C = None
args = None 
torch.distributed.init_process_group(backend="nccl")
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_gpus", default=1, type=int, help="distribted training")
    parser.add_argument("--local_rank", default=0, type=int, help="distribted training")
    parser.add_argument('--attention', type=int, default=1, choices = [1,2,3])
    parser.add_argument('--model_id', type=str, default="MSVD_GBased+rel+videomask",
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
    args.local_rank = int(os.environ.get("LOCAL_RANK", 0))
    args.n_gpus = torch.distributed.get_world_size()
    return args

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
    
    train_iter, val_iter, vocab = corpus.train_data_loader, corpus.val_data_loader,  corpus.vocab
    del corpus
    gc.collect()
    return train_iter, val_iter, vocab

def build_model(vocab, C, device ,local_rank):
    model_state_dict = None
    cache_dir = C.transformer.cache_dir if C.transformer.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed')

    model = VCModel(vocab, model_state_dict, cache_dir, C.feat.feature_mode, C.transformer, C.feat.size, C.attention_mode, device)
    model = torch.nn.parallel.DistributedDataParallel(model.to(device), device_ids=[local_rank],
                                                      output_device=local_rank, find_unused_parameters=True)
    return model


def log_train(summary_writer, e, loss, lr, reg_lambda, scores, C):
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


def log_val(summary_writer, e, loss, reg_lambda, r2l_scores, l2r_scores, C):
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
    global args, C, logger, GRAPH_DATA_DICT
    args = get_args()
    torch.cuda.set_device(args.local_rank)
    device = torch.device(f'cuda:{args.local_rank}')
    logger = get_logger(filename="log.txt")

    seed = 904666
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    C = TrainConfig(args.model_id, args.n_gpus)
    C.attention_model = args.attention
    load_graph_data(C.corpus, 'train')
    load_graph_data(C.corpus, 'val')

    if args.local_rank == 0:
        # load_graph_data(C.corpus, 'train')
        # load_graph_data(C.corpus, 'val')
        summary_writer = SummaryWriter(C.log_dpath)
        logger.info("MODEL ID: {}".format(C.model_id))
        logger.info("Max caption length: {}".format(C.loader.max_caption_len))
        logger.info("Max frame: {}".format(C.loader.frame_sample_len))
        logger.info("Heads: {}".format(C.transformer.n_heads))
        logger.info("Small Heads: {}".format(C.transformer.n_heads_small))
        logger.info("Big Heads: {}".format(C.transformer.n_heads_big))
        logger.info("Model Dim: {}".format(C.transformer.d_model))
        logger.info("Feature Mode: {}".format(C.feat.feature_mode))
        logger.info("Epochs: {}".format(C.epochs))
        logger.info("Effective Batch Size: {}".format(C.batch_size * C.gradient_accumulation_steps))
        logger.info("GPUs: {}".format(C.n_gpus))
        if args.attention == 1:
            logger.info("MHA for relation")
        elif args.attention == 2:
            logger.info("MHA + pe for relation")
        elif args.attention == 3:
            logger.info("FFN for relation")

    train_iter, val_iter, vocab= build_loaders(C)
    if args.local_rank == 0:
        logger.info("[Memory when loading data]")
        logger.info("  VRAM used     : {:.2f} MB".format(round(torch.cuda.memory_allocated() / 1024**2)))
        logger.info("  VRAM reserved : {:.2f} MB".format(round(torch.cuda.memory_reserved() / 1024**2)))
        logger.info("  RAM used      : {:.2f} MB".format(round(psutil.Process(os.getpid()).memory_info().rss / 1024**2)))

    model = build_model(vocab, C, device , args.local_rank)
    if args.local_rank == 0:
        parameter_number = get_parameter_number(model)
        logger.info(parameter_number)
    

    optimizer = torch.optim.Adam(model.parameters(), lr=C.lr, weight_decay=C.weight_decay)
    num_training_steps = int(len(train_iter) / C.gradient_accumulation_steps) * C.epochs
    num_warmup_steps = int(0.1 * num_training_steps)
    lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
    
    best_val_CIDEr = -1
    best_epoch = None
    best_ckpt_fpath = None

    for e in range(1, C.epochs + 1):
        ckpt_fpath = C.ckpt_fpath_tpl.format(e)
        train_iter.sampler.set_epoch(e)
        """ Train """
        train_loss = train(e, model, optimizer, train_iter, vocab, 
                                C.reg_lambda, C.gradient_clip, C.feat.feature_mode, lr_scheduler,
                                C, device, args.local_rank)


        if args.local_rank == 0:
            log_train(summary_writer, e, train_loss, get_lr(optimizer), C.reg_lambda, None,C)

            """ Validation """
            val_loss = test(model, val_iter, vocab, C.reg_lambda, C.feat.feature_mode, C, device)
            r2l_val_scores, l2r_val_scores = evaluate(val_iter, model, vocab, C.beam_size, C.loader.max_caption_len,
                                                    C.feat.feature_mode, C, device, 'val')

            log_val(summary_writer, e, val_loss, C.reg_lambda, r2l_val_scores, l2r_val_scores, C)

            summary_writer.add_scalars("compare_loss/total_loss", {'train_total_loss': train_loss['total'],
                                                                    'val_total_loss': val_loss['total']}, e)
            summary_writer.add_scalars("compare_loss/l2r_loss", {'train_l2r_loss': train_loss['l2r_loss'],
                                                                'val_l2r_loss': val_loss['l2r_loss']}, e)
            summary_writer.add_scalars("compare_loss/r2l_loss", {'train_r2l_loss': train_loss['r2l_loss'],
                                                                'val_r2l_loss': val_loss['r2l_loss']}, e)

            logger.info("Epoch {} memory usage:".format(e))
            logger.info("  VRAM used     : {:.2f} MB".format(torch.cuda.memory_allocated() / 1024**2))
            logger.info("  VRAM reserved : {:.2f} MB".format(torch.cuda.memory_reserved() / 1024**2))
            logger.info("  RAM used      : {:.2f} MB".format(psutil.Process(os.getpid()).memory_info().rss / 1024**2))
            if e >= C.save_from and e % C.save_every == 0:
                logger.info("Saving checkpoint at epoch={} to {}".format(e, ckpt_fpath))
                save_checkpoint(e, model, ckpt_fpath, C)

            if l2r_val_scores['CIDEr'] > best_val_CIDEr:
                best_epoch = e
                best_val_CIDEr = l2r_val_scores['CIDEr']
                best_ckpt_fpath = ckpt_fpath

            del val_loss, r2l_val_scores, l2r_val_scores
            torch.cuda.empty_cache()
            gc.collect()
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
    if args.local_rank == 0:
        logger.info("[Memory after training]")
        logger.info("  VRAM used     : {:.2f} MB".format(torch.cuda.memory_allocated() / 1024**2))
        logger.info("  VRAM reserved : {:.2f} MB".format(torch.cuda.memory_reserved() / 1024**2))
        logger.info("  RAM used      : {:.2f} MB".format(psutil.Process(os.getpid()).memory_info().rss / 1024**2))
    
    """ Test with Best Model """
    del train_iter, val_iter, model, optimizer, lr_scheduler, train_loss
    summary_writer.close()
    gc.collect()
    torch.cuda.empty_cache()
    clear_graph_data('all')

    if args.local_rank == 0: 
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
            
        file = C.ckpt_dpath
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

    torch.distributed.destroy_process_group()
if __name__ == "__main__":
    main()
