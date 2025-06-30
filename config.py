import os
import time


class FeatureConfig(object):
    # model = "MSVD_GBased+OFeat+rel+videomask"
    # model = "MSR-VTT_GBased+OFeat+rel+videomask"

    model = "MSVD_GBased+rel+videomask"
    # model = "MSR-VTT_GBased+rel+videomask"
    
    # model = "MSVD_GBased"
    # model = "MSR-VTT_GBased

    size = None
    feature_mode = None
    num_boxes = 60
    three_turple = 60
    if model == 'MSVD_GBased+OFeat+rel+videomask' or model == 'MSR-VTT_GBased+OFeat+rel+videomask':
        size = [1028, 300]
        feature_mode = 'grid-obj-rel'
    elif model == 'MSVD_GBased+rel+videomask' or model == 'MSR-VTT_GBased+rel+videomask':
        # size = [300]
        size = [512]
        feature_mode = 'grid-rel'
    elif model == 'MSVD_GBased+videomask' or model == 'MSR-VTT_GBased+videomask':
        feature_mode = 'grid'   
    else:
        raise NotImplementedError("Unknown model: {}".format(model))

    if model.split('_')[0] == "MSR-VTT":
        size = [s + 300 for s in size]


class VocabConfig(object):
    init_word2idx = {'<PAD>': 0, '<S>': 1}
    embedding_size = 512


class MSVDLoaderConfig(object):
    n_train = 1200
    n_val = 100
    n_test = 670
    
    total_caption_fpath = "/workspace/AKG-sv/data/MSVD/metadata/MSR Video Description Corpus.csv"
    train_caption_fpath = "/workspace/AKG-sv/data/MSVD/metadata/train.csv"
    val_caption_fpath = "/workspace/AKG-sv/data/MSVD/metadata/val.csv"
    test_caption_fpath = "/workspace/AKG-sv/data/MSVD/metadata/test.csv"
    min_count = 3
    max_caption_len = 15

    total_video_feat_fpath_tpl = "/workspace/AKG-sv/data/{}/features/{}.{}"
    phase_video_feat_fpath_tpl = "/workspace/AKG-sv/data/{}/features/{}_{}.{}"

    frame_sampling_method = 'uniform'
    assert frame_sampling_method in ['uniform', 'random']
    frame_sample_len = 50
    num_workers = 4


class MSRVTTLoaderConfig(object):
    n_train = 6513
    n_val = 497
    n_test = 2990

    total_caption_fpath = "/workspace/AKG-sv/data/MSR-VTT/metadata/total.json"
    train_caption_fpath = "/workspace/AKG-sv/data/MSR-VTT/metadata/train.json"
    val_caption_fpath = "/workspace/AKG-sv/data/MSR-VTT/metadata/val.json"
    test_caption_fpath = "/workspace/AKG-sv/data/MSR-VTT/metadata/test.json"
    min_count = 3
    max_caption_len = 15

    total_video_feat_fpath_tpl = "/workspace/AKG-sv/data/{}/features/{}.{}"
    phase_video_feat_fpath_tpl = "/workspace/AKG-sv/data/{}/features/{}_{}.{}"
    frame_sampling_method = 'uniform'
    assert frame_sampling_method in ['uniform', 'random']
    frame_sample_len = 60

    num_workers = 4


class TransformerConfig(object):
    d_model = 640
    n_heads = 10
    n_heads_small = 12
    
    d_ff = 2048
    n_layers = 4
    dropout = 0.1
    n_heads_big = 128

    max_frames = 50
    num_object = 9
    visual_num_hidden_layers = 4
    d_graph = 1024
    node_feat_dim = 512
    video_dim = 1024
    edge_dim = 1024
    hidden_size = 768
    visual_model = "visual-base"
    init_model = "./model/weight/univl.pretrained.bin"
    gnn_model_type = "transformer"
    cache_dir = ""
    local_rank = 0
    project_edge_dim = None
    no_skip = False
    last_average = False
    no_beta_transformer = False
    device = "cuda"
    select_num = 0  # if sn==0, automatic select num

class TrainConfig(object):
    feat = FeatureConfig
    msrvtt_dim = 1028
    rel_dim = 300

    vocab = VocabConfig
    corpus = feat.model.split('_')[0]
    loader = {
        'MSVD': MSVDLoaderConfig,
        'MSR-VTT': MSRVTTLoaderConfig
    }[corpus]
    transformer = TransformerConfig

    """ Optimization """
    epochs = {
        'MSVD': 30,
        'MSR-VTT': 18,
    }[corpus]

    batch_size = 32

    optimizer = "Adam"
    gradient_clip = 5.0  # None if not used
    lr = {
        'MSVD': 1e-4,
        'MSR-VTT': 3e-5,
    }[corpus]
    lr_decay_start_from = 12
    lr_decay_gamma = 0.5
    lr_decay_patience = 5
    weight_decay = 0.5e-5

    reg_lambda = 0.6  # weights of r2l

    beam_size = 5
    label_smoothing = 0.15

    """ Pretrained Model """
    pretrained_decoder_fpath = None

    """ Evaluate """
    metrics = ['Bleu_4', 'CIDEr', 'METEOR', 'ROUGE_L']

    """ ID """
    exp_id = "Transformer"
    feat_id = "FEAT {} fsl-{} mcl-{}".format(feat.model, loader.frame_sample_len, loader.max_caption_len)
    embedding_id = "EMB {}".format(vocab.embedding_size)
    transformer_id = "Transformer d-{}-N-{}-h-{}-h_big-{}-dp-{}-sn-{}".format(transformer.d_model, transformer.n_layers,
                                                                              transformer.n_heads,
                                                                              transformer.n_heads_big,
                                                                              transformer.dropout,
                                                                              transformer.select_num)
    optimizer_id = "OPTIM {} lr-{}-dc-{}-{}-{}-wd-{}-rg-{}".format(
        optimizer, lr, lr_decay_start_from, lr_decay_gamma, lr_decay_patience, weight_decay, reg_lambda)
    hyperparams_id = "bs-{}".format(batch_size)
    if gradient_clip is not None:
        hyperparams_id += " gc-{}".format(gradient_clip)

    timestamp = time.strftime("%Y-%m-%d %X", time.localtime(time.time()))
    # model_id = " | ".join(
    #     [timestamp, exp_id, corpus, feat_id, embedding_id, transformer_id, optimizer_id, hyperparams_id])
    model_id = " | ".join([feat.model, timestamp])

    """ Log """
    path = "/workspace/AKG-sv"
    log_dpath = os.path.join(path, "logs/{}/{}".format(corpus, model_id))
    ckpt_dpath = os.path.join(os.path.join(path,"checkpoints/{}".format(corpus)), model_id)
    captioning_dpath = os.path.join(os.path.join(path,"captioning/{}".format(corpus)), model_id)
    ckpt_fpath_tpl = os.path.join(ckpt_dpath, "{}.ckpt")
    captioning_fpath_tpl = os.path.join(captioning_dpath, "{}.csv")

    save_from = 1
    save_every = 1

    """ TensorboardX """
    tx_train_loss = "loss/train"
    tx_train_r2l_cross_entropy_loss = "loss/train/r2l_loss"
    tx_train_l2r_cross_entropy_loss = "loss/train/l2r_loss"
    tx_val_loss = "loss/val"
    tx_val_r2l_cross_entropy_loss = "loss/val/r2l_loss"
    tx_val_l2r_cross_entropy_loss = "loss/val/l2r_loss"
    tx_lr = "params/vc_model_LR"