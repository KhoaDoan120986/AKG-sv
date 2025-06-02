from __future__ import print_function
import os
import gc
import torch
from loader.MSVD import MSVD
from loader.MSRVTT import MSRVTT
from config import TrainConfig as C
from model.model import VCModel
from utils import dict_to_cls, get_predicted_captions, get_groundtruth_captions, save_result, score
import  logging 

logger = logging.getLogger(__name__)

def build_loader(ckpt_fpath, graph_data):
    checkpoint = torch.load(ckpt_fpath)
    config = dict_to_cls(checkpoint['config'])
    """ Build Data Loader """
    if config.corpus == "MSVD":
        corpus = MSVD(config)
    elif config.corpus == "MSR-VTT":
        corpus = MSRVTT(config)
    else:
        raise "无该数据集"

    graph_data, train_iter, val_iter, test_iter, vocab = corpus.graph_data, \
        corpus.train_data_loader, corpus.val_data_loader, corpus.test_data_loader, corpus.vocab
    r2l_test_vid2GTs, l2r_test_vid2GTs = get_groundtruth_captions(test_iter, graph_data['test'], vocab, config.feat.feature_mode)
    
    logger.info('#vocabs: {} ({}), #words: {} ({}). Trim words which appear less than {} times.'.format(
        vocab.n_vocabs, vocab.n_vocabs_untrimmed, vocab.n_words, vocab.n_words_untrimmed, config.loader.min_count))
    
    del train_iter, val_iter, r2l_test_vid2GTs
    corpus.clear()
    gc.collect()
    return graph_data['test'], test_iter, vocab, l2r_test_vid2GTs


def run(ckpt_fpath, test_iter, graph_data, vocab, ckpt, l2r_test_vid2GTs, f, captioning_fpath):
    captioning_dpath = os.path.dirname(captioning_fpath)

    if not os.path.exists(captioning_dpath):
        os.makedirs(captioning_dpath)

    checkpoint = torch.load(ckpt_fpath)
    """ Load Config """
    config = dict_to_cls(checkpoint['config'])

    """ Build Models """
    model_state_dict = None
    cache_dir = None
    model = VCModel(vocab, model_state_dict, cache_dir, C.feat.feature_mode, C.transformer, C.feat.size)
    
    model.load_state_dict(checkpoint['vc_model'])
    model = model.cuda()

    """ Test Set """
    logger.info('Finish the model load in CUDA. Try to enter Test Set.')
    r2l_test_vid2pred, l2r_test_vid2pred = get_predicted_captions(test_iter, graph_data, model, config.beam_size,
                                                                  config.loader.max_caption_len,
                                                                  config.feat.feature_mode)
    l2r_test_scores = score(l2r_test_vid2pred, l2r_test_vid2GTs)
    logger.info("[TEST L2R] in {} is {}".format(ckpt, l2r_test_scores))

    f.write(ckpt + " result: ")
    f.write("[TEST L2R] in {} is {}".format(ckpt, l2r_test_scores))
    f.write('\n')

    save_result(l2r_test_vid2pred, l2r_test_vid2GTs, captioning_fpath)