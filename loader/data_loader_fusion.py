from __future__ import print_function, division

from collections import defaultdict

from tqdm import tqdm
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler
from torchvision import transforms
import pickle
import sparse
import gc

from loader.transform import UniformSample, RandomSample, ToTensor, TrimExceptAscii, Lowercase, \
    RemovePunctuation, SplitWithWhiteSpace, Truncate, PadFirst, PadLast, PadToLength, \
    ToIndex
from torch_geometric.data import Data

class CustomVocab(object):
    def __init__(self, caption_fpath, init_word2idx, min_count=1, transform=str.split):
        self.caption_fpath = caption_fpath
        self.min_count = min_count
        self.transform = transform

        self.word2idx = init_word2idx
        self.idx2word = {v: k for k, v in self.word2idx.items()}
        self.word_freq_dict = defaultdict(lambda: 0)
        self.n_vocabs = len(self.word2idx)
        self.n_words = self.n_vocabs
        self.max_sentence_len = -1

        self.build()

    def load_captions(self):
        raise NotImplementedError("You should implement this function.")
        # df = pd.read_csv(self.caption_fpath)
        # df = df[df['Language'] == 'English']
        # df = df[pd.notnull(df['Description'])]
        # captions = df['Description'].values
        # return captions

    def build(self):
        captions = self.load_captions()
        for caption in captions:
            words = self.transform(caption)
            self.max_sentence_len = max(self.max_sentence_len, len(words))
            for word in words:
                self.word_freq_dict[word] += 1
        self.n_vocabs_untrimmed = len(self.word_freq_dict)
        self.n_words_untrimmed = sum(list(self.word_freq_dict.values()))

        keep_words = [word for word, freq in self.word_freq_dict.items() if freq >= self.min_count]

        for idx, word in enumerate(keep_words, len(self.word2idx)):
            self.word2idx[word] = idx
            self.idx2word[idx] = word
        self.n_vocabs = len(self.word2idx)
        self.n_words = sum([self.word_freq_dict[word] for word in keep_words])


class CustomDataset(Dataset):
    """ Dataset """

    def __init__(self, C, phase, caption_fpath, transform_caption=None, transform_frame=None):
        self.C = C
        self.phase = phase
        self.caption_fpath = caption_fpath
        self.transform_frame = transform_frame
        self.transform_caption = transform_caption
        self.feature_mode = C.feat.feature_mode

        if self.feature_mode == 'btkg':
            self.image_video_feats = defaultdict(lambda: [])
            self.motion_video_feats = defaultdict(lambda: [])
            self.object_video_feats = defaultdict(lambda: [])
            self.rel_feats = defaultdict(lambda: [])
        elif self.feature_mode in ['grid', 'object']:
            self.object_video_feats = defaultdict(lambda: [])
            self.video_mask = defaultdict(lambda: [])       
        elif self.feature_mode in ['grid-rel','object-rel']:
            # self.grid_video_feats = defaultdict(lambda: {'x': None, 'edge_index': None, 'edge_attr': None})
            self.object_video_feats = defaultdict(lambda: [])
            self.rel_feats = defaultdict(lambda: [])
            self.video_mask = defaultdict(lambda: [])
        elif self.feature_mode in ['grid-rel-no_obj']:
            self.rel_feats = defaultdict(lambda: [])
            self.video_mask = defaultdict(lambda: [])

        self.r2l_captions = defaultdict(lambda: [])
        self.l2r_captions = defaultdict(lambda: [])
        self.data = []

        self.build_video_caption_pairs()
        
        if self.feature_mode == 'btkg':
            self.image_video_feats.clear()
            self.motion_video_feats.clear()
            self.object_video_feats.clear()
            self.rel_feats.clear()
        elif self.feature_mode in ['grid', 'object']:
            self.video_mask.clear()
            self.object_video_feats.clear()
        elif self.feature_mode in ['grid-rel','object-rel']:
            self.video_mask.clear()
            self.object_video_feats.clear()
            self.rel_feats.clear()
        elif self.feature_mode in ['grid-rel-no_obj']:
            self.video_mask.clear()
            self.rel_feats.clear()
        self.r2l_captions.clear()
        self.l2r_captions.clear()
        gc.collect()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.feature_mode == 'btkg':
            vid, image_video_feats, motion_video_feats, object_video_feats, rel_feats, r2l_caption, l2r_caption = \
                self.data[idx]
            if self.transform_frame:
                image_video_feats = [self.transform_frame(
                    feat) for feat in image_video_feats]
                motion_video_feats = [self.transform_frame(
                    feat) for feat in motion_video_feats]
                object_video_feats = [self.transform_frame(
                    feat) for feat in object_video_feats]
                rel_feats = [self.transform_frame(
                    feat) for feat in rel_feats]
            if self.transform_caption:
                r2l_caption = self.transform_caption(r2l_caption)
                l2r_caption = self.transform_caption(l2r_caption)
            return vid, image_video_feats, motion_video_feats, object_video_feats, rel_feats, r2l_caption, l2r_caption
        elif self.feature_mode in ['grid','object']:
            vid, video_mask, object_video_feats, r2l_caption, l2r_caption = self.data[idx]
            if self.transform_frame:
                object_video_feats = [self.transform_frame(feat) for feat in object_video_feats]
            if self.transform_caption:
                r2l_caption = self.transform_caption(r2l_caption)
                l2r_caption = self.transform_caption(l2r_caption)
            return vid, video_mask, object_video_feats, r2l_caption, l2r_caption
        elif self.feature_mode in ['grid-rel','object-rel']:
            vid, video_mask, object_video_feats, rel_feats, r2l_caption, l2r_caption = self.data[idx]
            if self.transform_frame:
                object_video_feats = [self.transform_frame(feat) for feat in object_video_feats]
                rel_feats = [self.transform_frame(feat) for feat in rel_feats]
            if self.transform_caption:
                r2l_caption = self.transform_caption(r2l_caption)
                l2r_caption = self.transform_caption(l2r_caption)
            return vid, video_mask, object_video_feats, rel_feats, r2l_caption, l2r_caption
        elif self.feature_mode in ['grid-rel-no_obj']:
            vid, video_mask, rel_feats, r2l_caption, l2r_caption = self.data[idx]
            if self.transform_frame:
                rel_feats = [self.transform_frame(feat) for feat in rel_feats]
            if self.transform_caption:
                r2l_caption = self.transform_caption(r2l_caption)
                l2r_caption = self.transform_caption(l2r_caption)
            return vid, video_mask, rel_feats, r2l_caption, l2r_caption
        
        else:
            raise NotImplementedError("Unknown feature mode: {}".format(self.feature_mode))

    def load_object_feats(self, frames, fin_o, fin_b, vid):  # It's too complex so write another function
        # vid = 'video122'
        # assert vid == vid1, "video id of OFeat and BFeat is not align"
        feats_b = fin_b[vid][()]
        feats_o = fin_o[vid][()]
        feats = np.concatenate((feats_b, feats_o), axis=1)
        num_paddings = frames - len(feats)
        if feats.size == 0:
            feats = np.zeros((frames, 1028))  # now just object feat may appear the feature is empty
        else:
            feats = feats.tolist() + [np.zeros_like(feats[0])
                                      for _ in range(num_paddings)]
        feats = np.asarray(feats)
        sampled_idxs = np.linspace(
            0, len(feats) - 1, frames, dtype=int)  # return evenly sapced number within the specified
        feats = feats[sampled_idxs]
        assert len(feats) == frames
        return feats

    def load_btkg_video_feats(self):
        models = self.C.feat.model.split('_')[1].split('+')
        print('\nEnter the load4 method.  data_loader_fusion.py--row279')
        for i in range(len(models)):
            frames = self.C.loader.frame_sample_len
            # i = 2
            if i == 2:
                frames = self.C.feat.num_boxes
            if i == 3:
                frames = self.C.feat.three_turple
            fpath = self.C.loader.phase_video_feat_fpath_tpl.format(self.C.corpus,
                                                                    self.C.corpus +
                                                                    '_' +
                                                                    models[i],
                                                                    self.phase, 'hdf5')
            fpath_b = self.C.loader.phase_video_feat_fpath_tpl.format(self.C.corpus,
                                                                      self.C.corpus +
                                                                      '_' +
                                                                      'BFeat',
                                                                      self.phase, 'hdf5')  # load two feats at the sames
            # time, there are some problems in efficiency
            fin = h5py.File(fpath, 'r')
            fin_b = h5py.File(fpath_b, 'r')
            for vid in tqdm(fin.keys(),desc=f'Load feature: {models[i]}'):
                # vid = 'video122'
                feats = fin[vid][()]
                if len(feats) < frames:
                    if i == 2:

                        feats = self.load_object_feats(frames=frames, fin_o=fin, fin_b=fin_b, vid=vid)
                        self.object_video_feats[vid].append(feats)
                        continue
                    num_paddings = frames - len(feats)
                    if feats.size == 0:
                        # for _ in range(num_paddings):
                        feats = np.zeros((frames, 1024))  # now just object feat may appear the feature is empty
                    else:
                        feats = feats.tolist() + [np.zeros_like(feats[0])
                                                  for _ in range(num_paddings)]
                    # feats = feats.tolist() + [np.zeros_like(feats[0])
                    #                           for _ in range(num_paddings)]
                    feats = np.asarray(feats)
                    sampled_idxs = np.linspace(
                        0, len(feats) - 1, frames, dtype=int)  # return evenly sapced number within the specified
                    feats = feats[sampled_idxs]
                    assert len(feats) == frames
                    if i == 0:
                        self.image_video_feats[vid].append(feats)
                    elif i == 1:
                        self.motion_video_feats[vid].append(feats)
                    elif i == 3:
                        self.rel_feats[vid].append(feats)
                else:
                    if i == 0:
                        self.image_video_feats[vid].append(feats)
                    elif i == 1:
                        self.motion_video_feats[vid].append(feats)
                    elif i == 2:
                        feats = self.load_object_feats(frames=frames, fin_o=fin, fin_b=fin_b, vid=vid)
                        self.object_video_feats[vid].append(feats)
                    elif i == 3:
                        self.rel_feats[vid].append(feats)
            fin.close()
            fin_b.close()

    def load_norel_video_feats(self):
        models = self.C.feat.model.split('_')[1].split('+')
        print('\nEnter the load4 method. data_loader_fusion.py--row279')

        for i, model in enumerate(models):
            frames = self.C.loader.frame_sample_len

            if i == 0:
                continue
            elif i == 1:
                frames = self.C.feat.num_boxes

            fpath = self.C.loader.phase_video_feat_fpath_tpl.format(self.C.corpus,
                                                                    f"{self.C.corpus}_{model}",
                                                                    self.phase, 'hdf5')
            fpath_b = self.C.loader.phase_video_feat_fpath_tpl.format(self.C.corpus,
                                                                    f"{self.C.corpus}_BFeat",
                                                                    self.phase, 'hdf5')

            with (h5py.File(fpath, 'r')) as fin, \
                (h5py.File(fpath_b, 'r')) as fin_b:

                data = fin
                vids = list(data.keys())
                for vid in tqdm(vids, desc=f'Load feature: {model}'):
                    feats = data[vid]
                    if i == 0:
                        continue
                    # Xử lý i == 1 (object_video_feats)
                    elif i == 1:
                        obj_feats = self.load_object_feats(frames=frames, fin_o=data, fin_b=fin_b, vid=vid)
                        self.object_video_feats[vid].append(obj_feats)
                    elif i == 2:
                        zero_arr = torch.zeros([frames - feats.shape[1]], dtype=torch.float32)
                        self.video_mask[vid].append(torch.cat([torch.tensor(feats[0]), zero_arr], dim = 0))

    def load_noobj_video_feats(self):
        models = self.C.feat.model.split('_')[1].split('+')
        print('\nEnter the load4 method. data_loader_fusion.py--row279')

        for i, model in enumerate(models):
            frames = self.C.loader.frame_sample_len

            if i == 0:
                continue
            elif i == 1:
                frames = self.C.feat.three_turple

            fpath = self.C.loader.phase_video_feat_fpath_tpl.format(self.C.corpus,
                                                                    f"{self.C.corpus}_{model}",
                                                                    self.phase, 'hdf5')
            fpath_b = self.C.loader.phase_video_feat_fpath_tpl.format(self.C.corpus,
                                                                    f"{self.C.corpus}_BFeat",
                                                                    self.phase, 'hdf5')

            with (h5py.File(fpath, 'r')) as fin, \
                (h5py.File(fpath_b, 'r')) as fin_b:

                data = fin
                vids = list(data.keys())
                for vid in tqdm(vids, desc=f'Load feature: {model}'):
                    feats = data[vid]
                    if i == 0:
                        continue
                    # Xử lý i == 1 (rel_feats)
                    elif i == 1:
                        if feats.size == 0 or len(feats) < frames:
                            num_paddings = frames - len(feats)
                            feats = np.zeros((frames, 1024)) if feats.size == 0 else np.vstack([feats] + [np.zeros_like(feats[0])] * num_paddings)

                        sampled_idxs = np.linspace(0, len(feats) - 1, frames, dtype=int)
                        self.rel_feats[vid].append(feats[sampled_idxs])
                    elif i == 2:
                        zero_arr = torch.zeros([frames - feats.shape[1]], dtype=torch.float32)
                        self.video_mask[vid].append(torch.cat([torch.tensor(feats[0]), zero_arr], dim = 0))

    def load_video_feats(self):
        models = self.C.feat.model.split('_')[1].split('+')
        print('\nEnter the load4 method. data_loader_fusion.py--row279')

        for i, model in enumerate(models):
            frames = self.C.loader.frame_sample_len

            if i == 0:
                continue
            elif i == 1:
                frames = self.C.feat.num_boxes
            elif i == 2:
                frames = self.C.feat.three_turple

            fpath = self.C.loader.phase_video_feat_fpath_tpl.format(self.C.corpus,
                                                                    f"{self.C.corpus}_{model}",
                                                                    self.phase, 'hdf5')
            fpath_b = self.C.loader.phase_video_feat_fpath_tpl.format(self.C.corpus,
                                                                    f"{self.C.corpus}_BFeat",
                                                                    self.phase, 'hdf5')

            with (h5py.File(fpath, 'r')) as fin, \
                (h5py.File(fpath_b, 'r')) as fin_b:

                data = fin
                vids = list(data.keys())
                for vid in tqdm(vids, desc=f'Load feature: {model}'):
                    feats = data[vid]
                    if i == 0:
                        continue

                    # Xử lý i == 1 (object_video_feats)
                    elif i == 1:
                        obj_feats = self.load_object_feats(frames=frames, fin_o=data, fin_b=fin_b, vid=vid)
                        self.object_video_feats[vid].append(obj_feats)

                    # Xử lý i == 2 (rel_feats) - padding và lấy mẫu
                    elif i == 2:
                        if feats.size == 0 or len(feats) < frames:
                            num_paddings = frames - len(feats)
                            feats = np.zeros((frames, 1024)) if feats.size == 0 else np.vstack([feats] + [np.zeros_like(feats[0])] * num_paddings)

                        sampled_idxs = np.linspace(0, len(feats) - 1, frames, dtype=int)
                        self.rel_feats[vid].append(feats[sampled_idxs])
                    elif i == 3:
                        zero_arr = torch.zeros([frames - feats.shape[1]], dtype=torch.float32)
                        self.video_mask[vid].append(torch.cat([torch.tensor(feats[0]), zero_arr], dim = 0))
                    
    def load_captions(self):
        raise NotImplementedError("You should implement this function.")

    def build_video_caption_pairs(self):
        self.load_captions()
        if self.feature_mode == 'btkg':
            self.load_btkg_video_feats()
            assert self.image_video_feats.keys() == self.motion_video_feats.keys(), "Image feats is not match with " \
                                                                                    "motion feats "
            for vid in self.image_video_feats.keys():
                image_video_feats = self.image_video_feats[vid]
                motion_video_feats = self.motion_video_feats[vid]
                if self.object_video_feats[vid]:
                    object_video_feats = self.object_video_feats[vid]
                else:
                    object_video_feats = list(np.zeros((1, self.C.feat.num_boxes, self.C.msrvtt_dim)))
                if self.rel_feats[vid]:
                    rel_feats = self.rel_feats[vid]
                else:
                    rel_feats = list(np.zeros((1, self.C.feat.num_boxes, self.C.rel_dim)))
                    # self.C.FeatureConfig.size[-1]
                for r2l_caption, l2r_caption in zip(self.r2l_captions[vid], self.l2r_captions[vid]):
                    self.data.append((vid, image_video_feats, motion_video_feats, object_video_feats, rel_feats,
                                      r2l_caption, l2r_caption))
        elif self.feature_mode in ['grid','object']:
            self.load_norel_video_feats()
            for vid in self.object_video_feats.keys():
                video_mask = self.video_mask[vid]
                if self.object_video_feats[vid]:
                    object_video_feats = self.object_video_feats[vid]
                else:
                    object_video_feats = list(np.zeros((1, self.C.feat.num_boxes, self.C.msrvtt_dim)))
                for r2l_caption, l2r_caption in zip(self.r2l_captions[vid], self.l2r_captions[vid]):
                    self.data.append((vid, video_mask, object_video_feats, r2l_caption, l2r_caption))
        elif self.feature_mode in ['grid-rel','object-rel']:
            self.load_video_feats()
            for vid in self.object_video_feats.keys():
                video_mask = self.video_mask[vid]
                if self.object_video_feats[vid]:
                    object_video_feats = self.object_video_feats[vid]
                else:
                    object_video_feats = list(np.zeros((1, self.C.feat.num_boxes, self.C.msrvtt_dim)))
                if self.rel_feats[vid]:
                    rel_feats = self.rel_feats[vid]
                else:
                    rel_feats = list(np.zeros((1, self.C.feat.num_boxes, self.C.rel_dim)))
                for r2l_caption, l2r_caption in zip(self.r2l_captions[vid], self.l2r_captions[vid]):
                    self.data.append((vid, video_mask, object_video_feats, rel_feats, r2l_caption, l2r_caption))
        elif self.feature_mode in ['grid-rel-no_obj']:
            self.load_noobj_video_feats()
            for vid in self.rel_feats.keys():
                video_mask = self.video_mask[vid]
                if self.rel_feats[vid]:
                    rel_feats = self.rel_feats[vid]
                else:
                    rel_feats = list(np.zeros((1, self.C.feat.num_boxes, self.C.rel_dim)))
                for r2l_caption, l2r_caption in zip(self.r2l_captions[vid], self.l2r_captions[vid]):
                    self.data.append((vid, video_mask, rel_feats, r2l_caption, l2r_caption))    
        else:
            raise NotImplementedError("Unknown feature mode: {}".format(self.feature_mode))


class Corpus(object):
    """ Data Loader """

    def __init__(self, C, vocab_cls=CustomVocab, dataset_cls=CustomDataset):
        self.C = C
        self.vocab = None
        self.train_dataset = None
        self.train_data_loader = None
        self.val_dataset = None
        self.val_data_loader = None
        self.test_dataset = None
        self.test_data_loader = None
        self.feature_mode = C.feat.feature_mode
        self.graph_data = {}

        self.CustomVocab = vocab_cls
        self.CustomDataset = dataset_cls

        self.transform_sentence = transforms.Compose([
            TrimExceptAscii(self.C.corpus),
            Lowercase(),
            RemovePunctuation(),
            SplitWithWhiteSpace(),
            Truncate(self.C.loader.max_caption_len),
        ])

        self.build()
        
    def load_graph_data(self):
        if self.feature_mode in ['grid', 'object', 'grid-rel', 'object-rel', 'grid-rel-no_obj']:
            model = self.C.feat.model.split('_')[1].split('+')[0]
            phase_list = ['train', 'val', 'test']
            for phase in phase_list: 
                fpath = self.C.loader.phase_video_feat_fpath_tpl.format(self.C.corpus, self.C.corpus + '_' + model, phase, 'pickle')
                with open(fpath, 'rb') as fs:
                    self.graph_data[phase] = pickle.load(fs)

    def build(self):
        self.build_vocab()
        if self.C.corpus == 'MSR-VTT':
            self.get_category()
            self.get_category_glove()
        self.build_data_loaders()

    def build_vocab(self):
        self.vocab = self.CustomVocab(
            # self.C.loader.total_caption_fpath,
            self.C.loader.train_caption_fpath,
            self.C.vocab.init_word2idx,
            self.C.loader.min_count,
            transform=self.transform_sentence)

    def build_data_loaders(self):
        """ Transformation """
        if self.C.loader.frame_sampling_method == "uniform":
            Sample = UniformSample
        elif self.C.loader.frame_sampling_method == "random":
            Sample = RandomSample
        else:
            raise NotImplementedError("Unknown frame sampling method: {}".format(self.C.loader.frame_sampling_method))

        self.transform_frame = transforms.Compose([
            Sample(self.C.loader.frame_sample_len),
            ToTensor(torch.float),
        ])
        self.transform_caption = transforms.Compose([
            self.transform_sentence,
            ToIndex(self.vocab.word2idx),
            PadFirst(self.vocab.word2idx['<S>']),
            PadLast(self.vocab.word2idx['<S>']),
            PadToLength(self.vocab.word2idx['<PAD>'], self.vocab.max_sentence_len + 2),  # +2 for <SOS> and <EOS>
            ToTensor(torch.long),
        ])
        
        self.train_dataset = self.build_dataset("train", self.C.loader.train_caption_fpath)
        self.val_dataset = self.build_dataset("val", self.C.loader.val_caption_fpath)
        self.test_dataset = self.build_dataset("test", self.C.loader.test_caption_fpath)

        self.train_data_loader = self.build_data_loader(self.train_dataset, phase='train')
        self.val_data_loader = self.build_data_loader(self.val_dataset, phase='val')
        self.test_data_loader = self.build_data_loader(self.test_dataset, phase='test')
        
        self.load_graph_data()

    def build_dataset(self, phase, caption_fpath):
        dataset = self.CustomDataset(
            self.C,
            phase,
            caption_fpath,
            transform_frame=self.transform_frame,
            transform_caption=self.transform_caption,
        )
        return dataset
    
    def build_data_loader(self, dataset, phase):
        if self.feature_mode == 'btkg':
            collate_fn = self.btkg_feature_collate_fn
        elif self.feature_mode in ['grid','object']:
            collate_fn = self.norel_feature_collate_fn
        elif self.feature_mode in ['grid-rel','object-rel']:
            collate_fn = self.feature_collate_fn
        elif self.feature_mode in ['grid-rel-no_obj']:
            collate_fn = self.noobj_feature_collate_fn
        else:
            raise NotImplementedError("Unknown feature mode: {}".format(self.feature_mode))
        if phase == 'test':
            batch_size = 1
        else:
            batch_size = self.C.batch_size
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,  # If sampler is specified, shuffle must be False.
            sampler=RandomSampler(dataset, replacement=False),
            num_workers=self.C.loader.num_workers,
            collate_fn=collate_fn)
        return data_loader
    
    def btkg_feature_collate_fn(self, batch):
        vids, image_video_feats, motion_video_feats, object_video_feats, rel_feats, r2l_captions, l2r_captions = zip(
            *batch)
        image_video_feats_list = zip(*image_video_feats)
        motion_video_feats_list = zip(*motion_video_feats)
        object_video_feats_list = zip(*object_video_feats)
        rel_feats_list = zip(*rel_feats)

        image_video_feats_list = [torch.stack(
            video_feats) for video_feats in image_video_feats_list]
        image_video_feats_list = [video_feats.float()
                                  for video_feats in image_video_feats_list]

        motion_video_feats_list = [torch.stack(
            video_feats) for video_feats in motion_video_feats_list]
        motion_video_feats_list = [video_feats.float()
                                   for video_feats in motion_video_feats_list]

        object_video_feats_list = [torch.stack(
            video_feats) for video_feats in object_video_feats_list]
        object_video_feats_list = [video_feats.float()
                                   for video_feats in object_video_feats_list]

        rel_feats_list = [torch.stack(
            video_feats) for video_feats in rel_feats_list]
        rel_feats_list = [video_feats.float()
                          for video_feats in rel_feats_list]

        if self.C.corpus == 'MSR-VTT':
            cate_vector = []
            for vid in vids:
                # get category
                cate_index = self.video_category[vid]
                # get category glove vector
                cate_vector.append(self.category_vectors[cate_index])
            cate_vector = torch.stack(cate_vector).unsqueeze_(dim=1).repeat(1, self.C.loader.frame_sample_len, 1)

            # cate_vector = [torch.stack(vector) for vector in cate_vector]
            image_video_feats_list = [torch.cat((video_feats, cate_vector), dim=2)
                                      for video_feats in image_video_feats_list]
            motion_video_feats_list = [torch.cat((video_feats, cate_vector), dim=2)
                                       for video_feats in motion_video_feats_list]
            object_video_feats_list = [torch.cat((video_feats, cate_vector), dim=2)
                                       for video_feats in object_video_feats_list]
            rel_feats_list = [torch.cat((video_feats, cate_vector), dim=2)
                              for video_feats in rel_feats_list]

        r2l_captions = torch.stack(r2l_captions)
        l2r_captions = torch.stack(l2r_captions)

        r2l_captions = r2l_captions.float()
        l2r_captions = l2r_captions.float()
        return vids, image_video_feats_list, motion_video_feats_list, object_video_feats_list, rel_feats_list, r2l_captions, l2r_captions

    def norel_feature_collate_fn(self, batch):
        vids, video_masks, object_video_feats, r2l_captions, l2r_captions = zip(*batch)

        video_mask_list = [torch.stack(video_feats) for video_feats in zip(*video_masks)]
        video_mask_list = [video_feats.float() for video_feats in video_mask_list]

        # Xử lý `object_video_feats
        object_video_feats_list = [torch.stack(video_feats) for video_feats in zip(*object_video_feats)]
        object_video_feats_list = [video_feats.float() for video_feats in object_video_feats_list]
        
        if self.C.corpus == 'MSR-VTT':
            cate_vector = []
            for vid in vids:
                # get category
                cate_index = self.video_category[vid]
                # get category glove vector
                cate_vector.append(self.category_vectors[cate_index])
            cate_vector = torch.stack(cate_vector).unsqueeze_(dim=1).repeat(1, self.C.loader.frame_sample_len, 1)

            # cate_vector = [torch.stack(vector) for vector in cate_vector]
            object_video_feats_list = [torch.cat((video_feats, cate_vector), dim=2)
                                       for video_feats in object_video_feats_list]


        # Xử lý caption
        r2l_captions = torch.stack(r2l_captions).float()
        l2r_captions = torch.stack(l2r_captions).float()

        return vids, video_mask_list, object_video_feats_list, r2l_captions, l2r_captions
    def feature_collate_fn(self, batch):
        vids, video_masks, object_video_feats, rel_feats, r2l_captions, l2r_captions = zip(*batch)

        video_mask_list = [torch.stack(video_feats) for video_feats in zip(*video_masks)]
        video_mask_list = [video_feats.float() for video_feats in video_mask_list]

        # Xử lý `object_video_feats` và `rel_feats`
        object_video_feats_list = [torch.stack(video_feats) for video_feats in zip(*object_video_feats)]
        object_video_feats_list = [video_feats.float() for video_feats in object_video_feats_list]

        rel_feats_list = [torch.stack(video_feats) for video_feats in zip(*rel_feats)]
        rel_feats_list = [video_feats.float() for video_feats in rel_feats_list]

        if self.C.corpus == 'MSR-VTT':
            cate_vector = []
            for vid in vids:
                # get category
                cate_index = self.video_category[vid]
                # get category glove vector
                cate_vector.append(self.category_vectors[cate_index])
            cate_vector = torch.stack(cate_vector).unsqueeze_(dim=1).repeat(1, self.C.loader.frame_sample_len, 1)

            # cate_vector = [torch.stack(vector) for vector in cate_vector]
            object_video_feats_list = [torch.cat((video_feats, cate_vector), dim=2)
                                       for video_feats in object_video_feats_list]
            rel_feats_list = [torch.cat((video_feats, cate_vector), dim=2)
                              for video_feats in rel_feats_list]


        # Xử lý caption
        r2l_captions = torch.stack(r2l_captions).float()
        l2r_captions = torch.stack(l2r_captions).float()

        return vids, video_mask_list, object_video_feats_list, rel_feats_list, r2l_captions, l2r_captions

    def noobj_feature_collate_fn(self, batch):
        vids, video_masks, rel_feats, r2l_captions, l2r_captions = zip(*batch)

        video_mask_list = [torch.stack(video_feats) for video_feats in zip(*video_masks)]
        video_mask_list = [video_feats.float() for video_feats in video_mask_list]

        # Xử lý `object_video_feats` và `rel_feats`
        rel_feats_list = [torch.stack(video_feats) for video_feats in zip(*rel_feats)]
        rel_feats_list = [video_feats.float() for video_feats in rel_feats_list]

        if self.C.corpus == 'MSR-VTT':
            cate_vector = []
            for vid in vids:
                # get category
                cate_index = self.video_category[vid]
                # get category glove vector
                cate_vector.append(self.category_vectors[cate_index])
            cate_vector = torch.stack(cate_vector).unsqueeze_(dim=1).repeat(1, self.C.loader.frame_sample_len, 1)

            # cate_vector = [torch.stack(vector) for vector in cate_vector]
            rel_feats_list = [torch.cat((video_feats, cate_vector), dim=2)
                              for video_feats in rel_feats_list]


        # Xử lý caption
        r2l_captions = torch.stack(r2l_captions).float()
        l2r_captions = torch.stack(l2r_captions).float()

        return vids, video_mask_list, rel_feats_list, r2l_captions, l2r_captions
    def get_category(self):
        import json
        with open('./data/MSR-VTT/metadata/category.json') as f:
            self.video_category = json.load(f)

    def get_category_glove(self):
        from loader.Vocab import GloVe
        category = []
        self.category_vectors = []
        glove = GloVe(name='6B', dim=300)
        with open('./data/MSR-VTT/metadata/category.txt') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split()
                category.append(line[0])

        for cate in category:
            vector = None
            cate = cate.split('/')
            if len(cate) == 2:
                vector = glove[cate[0]] + glove[cate[1]]
            elif len(cate) == 3:
                vector = glove[cate[0]] + glove[cate[1]] + glove[cate[2]]
            elif len(cate) == 1:
                vector = glove[cate[0]]
            self.category_vectors.append(vector)
