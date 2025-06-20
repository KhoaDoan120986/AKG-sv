# coding=utf-8
import inspect
import os
import time

import torch
import torch.nn as nn
from tqdm import tqdm
from model.model import pad_mask
from model.label_smoothing import LabelSmoothing
from config import TrainConfig as C
import numpy as np
from torch_geometric.data import Data

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor

class LossChecker:
    def __init__(self, num_losses):
        self.num_losses = num_losses

        self.losses = [[] for _ in range(self.num_losses)]

    def update(self, *loss_vals):
        assert len(loss_vals) == self.num_losses

        for i, loss_val in enumerate(loss_vals):
            self.losses[i].append(loss_val)

    def mean(self, last=0):
        mean_losses = [0. for _ in range(self.num_losses)]
        for i, loss in enumerate(self.losses):
            _loss = loss[-last:]
            mean_losses[i] = sum(_loss) / len(_loss)
        return mean_losses


def parse_batch(batch, feature_mode, graph_data):
    if feature_mode == 'btkg':
        vids, image_feats, motion_feats, object_feats, rel_feats, r2l_captions, l2r_captions = batch
        image_feats = [feat.cuda() for feat in image_feats]
        motion_feats = [feat.cuda() for feat in motion_feats]
        object_feats = [feat.cuda() for feat in object_feats]
        rel_feats = [feat.cuda() for feat in rel_feats]
        image_feats = torch.cat(image_feats, dim=2)
        motion_feats = torch.cat(motion_feats, dim=2)
        object_feats = torch.cat(object_feats, dim=2)
        rel_feats = torch.cat(rel_feats, dim=2)
        feats = (image_feats, motion_feats, object_feats, rel_feats)
        r2l_captions = r2l_captions.long().cuda()
        l2r_captions = l2r_captions.long().cuda()
        return vids, feats, r2l_captions, l2r_captions
    elif feature_mode in ['grid', 'object']:
        vids, video_masks, object_feats, r2l_captions, l2r_captions = batch
        geo_x_feats = []
        geo_edge_index_feats = []
        geo_edge_attr_feats = []
        for video_id in vids:
            stgraph =  graph_data[video_id]
            zero_arr  = torch.zeros([9 * C.loader.frame_sample_len - stgraph['x'].shape[0], stgraph['x'].shape[1]], dtype=torch.float32)
            geo_x_feats.append(torch.cat([stgraph['x'], zero_arr], dim=0).cuda())
            geo_edge_index_feats.append(stgraph['edge_index'].cuda())
            geo_edge_attr_feats.append(torch.tensor(stgraph['edge_attr'].todense(), dtype=torch.float32).cuda())
            
        object_feats = [feat.cuda() for feat in object_feats]
        video_mask_feats = [feat.cuda() for feat in video_masks]

        geo_x_feats = torch.stack(geo_x_feats, dim=0)
        geo_edge_index_feats = torch.stack(geo_edge_index_feats, dim=0)
        geo_edge_attr_feats = torch.stack(geo_edge_attr_feats, dim=0)
        object_feats = torch.cat(object_feats, dim=2)
        video_mask_feats = torch.cat(video_mask_feats, dim=1)

        feats = (geo_x_feats, geo_edge_index_feats, geo_edge_attr_feats, object_feats, video_mask_feats)

        # Chuyển `r2l_captions` và `l2r_captions` sang GPU
        r2l_captions = r2l_captions.long().cuda()
        l2r_captions = l2r_captions.long().cuda()

        return vids, feats, r2l_captions, l2r_captions
    elif feature_mode in ['grid-rel', 'object-rel']:
        vids, video_masks, object_feats, rel_feats, r2l_captions, l2r_captions = batch
        geo_x_feats = []
        geo_edge_index_feats = []
        geo_edge_attr_feats = []
        for video_id in vids:
            stgraph =  graph_data[video_id]
            zero_arr  = torch.zeros([9 * C.loader.frame_sample_len - stgraph['x'].shape[0], stgraph['x'].shape[1]], dtype=torch.float32)
            geo_x_feats.append(torch.cat([stgraph['x'], zero_arr], dim=0).cuda())
            geo_edge_index_feats.append(stgraph['edge_index'].cuda())
            geo_edge_attr_feats.append(torch.tensor(stgraph['edge_attr'].todense(), dtype=torch.float32).cuda())
            
        object_feats = [feat.cuda() for feat in object_feats]
        rel_feats = [feat.cuda() for feat in rel_feats]
        video_mask_feats = [feat.cuda() for feat in video_masks]

        geo_x_feats = torch.stack(geo_x_feats, dim=0)
        geo_edge_index_feats = torch.stack(geo_edge_index_feats, dim=0)
        geo_edge_attr_feats = torch.stack(geo_edge_attr_feats, dim=0)
        object_feats = torch.cat(object_feats, dim=2)
        rel_feats = torch.cat(rel_feats, dim=2)
        video_mask_feats = torch.cat(video_mask_feats, dim=1)

        feats = (geo_x_feats, geo_edge_index_feats, geo_edge_attr_feats, object_feats, rel_feats, video_mask_feats)

        # Chuyển `r2l_captions` và `l2r_captions` sang GPU
        r2l_captions = r2l_captions.long().cuda()
        l2r_captions = l2r_captions.long().cuda()

        return vids, feats, r2l_captions, l2r_captions
    elif feature_mode in ['grid-rel-no_obj']:
        vids, video_masks, rel_feats, r2l_captions, l2r_captions = batch
        geo_x_feats = []
        geo_edge_index_feats = []
        geo_edge_attr_feats = []
        for video_id in vids:
            stgraph =  graph_data[video_id]
            zero_arr  = torch.zeros([9 * C.loader.frame_sample_len - stgraph['x'].shape[0], stgraph['x'].shape[1]], dtype=torch.float32)
            geo_x_feats.append(torch.cat([stgraph['x'], zero_arr], dim=0).cuda())
            geo_edge_index_feats.append(stgraph['edge_index'].cuda())
            geo_edge_attr_feats.append(torch.tensor(stgraph['edge_attr'].todense(), dtype=torch.float32).cuda())
            
        rel_feats = [feat.cuda() for feat in rel_feats]
        video_mask_feats = [feat.cuda() for feat in video_masks]

        geo_x_feats = torch.stack(geo_x_feats, dim=0)
        geo_edge_index_feats = torch.stack(geo_edge_index_feats, dim=0)
        geo_edge_attr_feats = torch.stack(geo_edge_attr_feats, dim=0)
        rel_feats = torch.cat(rel_feats, dim=2)
        video_mask_feats = torch.cat(video_mask_feats, dim=1)

        feats = (geo_x_feats, geo_edge_index_feats, geo_edge_attr_feats, rel_feats, video_mask_feats)

        # Chuyển `r2l_captions` và `l2r_captions` sang GPU
        r2l_captions = r2l_captions.long().cuda()
        l2r_captions = l2r_captions.long().cuda()

        return vids, feats, r2l_captions, l2r_captions


def train(e, model, optimizer, train_iter, graph_data, vocab, reg_lambda, gradient_clip, feature_mode, lr_scheduler):
    gradient_accumulation_steps = 2
    model.train()
    loss_checker = LossChecker(3)
    pad_idx = vocab.word2idx['<PAD>']
    criterion = LabelSmoothing(vocab.n_vocabs, pad_idx, C.label_smoothing)
    t = tqdm(train_iter)
    for step, batch in enumerate(t):
        _, feats, r2l_captions, l2r_captions = parse_batch(batch, feature_mode, graph_data)

        r2l_trg = r2l_captions[:, :-1]
        r2l_trg_y = r2l_captions[:, 1:]
        r2l_norm = (r2l_trg_y != pad_idx).data.sum()
        l2r_trg = l2r_captions[:, :-1]
        l2r_trg_y = l2r_captions[:, 1:]
        l2r_norm = (l2r_trg_y != pad_idx).data.sum()

        if feature_mode in ['grid-rel', 'object-rel']:
            geo_x, geo_edge_index, geo_edge_attr, object_feats, rel_feats, video_mask = feats
            batch_sz = geo_x.shape[0]
            # Chuyển đổi geo_x thành batch format: flatten 2 chiều đầu (batch_sz, n_node) -> (batch_sz*n_node, dim)
            x_batch = geo_x.reshape(geo_x.shape[0] * geo_x.shape[1], geo_x.shape[2])

            # Tính toán offset cho geo_edge_index
            offset = []
            # Duyệt qua từng mẫu trong batch
            for i in range(batch_sz):
                n_edges = geo_edge_index[0].shape[1]
                offset_val = int(np.sqrt(n_edges)) * i
                offset.append(torch.full(geo_edge_index[0].shape, offset_val))
            offset = torch.stack(offset).cuda()
            # Cộng offset vào geo_edge_index để tạo batch offset
            geo_graph_batch_offset = geo_edge_index + offset

            # Tính new_dim và reshape để có edge_index_batch với shape (2, new_dim)
            new_dim = geo_graph_batch_offset.shape[0] * geo_graph_batch_offset.shape[2]
            edge_index_batch = geo_graph_batch_offset.permute(1, 0, 2).reshape(2, new_dim)

            # Reshape geo_edge_attr theo định dạng của code bên dưới
            edge_attr_batch = geo_edge_attr.reshape(geo_edge_attr.shape[0] * geo_edge_attr.shape[1],
                                                    geo_edge_attr.shape[2]).float()

            # Tạo đối tượng Data của PyG (Geometric) với x, edge_index và edge_attr
            data_geo_graph_batch = Data(x=x_batch, edge_index=edge_index_batch, edge_attr=edge_attr_batch)
            feats = (data_geo_graph_batch, object_feats, rel_feats)
            mask = pad_mask(feats, r2l_trg, l2r_trg, pad_idx, video_mask)
        elif feature_mode in ['grid', 'object']: 
            geo_x, geo_edge_index, geo_edge_attr, object_feats, video_mask = feats
            batch_sz = geo_x.shape[0]
            # Chuyển đổi geo_x thành batch format: flatten 2 chiều đầu (batch_sz, n_node) -> (batch_sz*n_node, dim)
            x_batch = geo_x.reshape(geo_x.shape[0] * geo_x.shape[1], geo_x.shape[2])

            # Tính toán offset cho geo_edge_index
            offset = []
            # Duyệt qua từng mẫu trong batch
            for i in range(batch_sz):
                n_edges = geo_edge_index[0].shape[1]
                offset_val = int(np.sqrt(n_edges)) * i
                offset.append(torch.full(geo_edge_index[0].shape, offset_val))
            offset = torch.stack(offset).cuda()
            # Cộng offset vào geo_edge_index để tạo batch offset
            geo_graph_batch_offset = geo_edge_index + offset

            # Tính new_dim và reshape để có edge_index_batch với shape (2, new_dim)
            new_dim = geo_graph_batch_offset.shape[0] * geo_graph_batch_offset.shape[2]
            edge_index_batch = geo_graph_batch_offset.permute(1, 0, 2).reshape(2, new_dim)

            # Reshape geo_edge_attr theo định dạng của code bên dưới
            edge_attr_batch = geo_edge_attr.reshape(geo_edge_attr.shape[0] * geo_edge_attr.shape[1],
                                                    geo_edge_attr.shape[2]).float()

            # Tạo đối tượng Data của PyG (Geometric) với x, edge_index và edge_attr
            data_geo_graph_batch = Data(x=x_batch, edge_index=edge_index_batch, edge_attr=edge_attr_batch)
            feats = (data_geo_graph_batch, object_feats)
            mask = pad_mask(feats, r2l_trg, l2r_trg, pad_idx, video_mask)
        elif feature_mode in ['grid-rel-no_obj']: 
            geo_x, geo_edge_index, geo_edge_attr, rel_feats, video_mask = feats
            batch_sz = geo_x.shape[0]
            # Chuyển đổi geo_x thành batch format: flatten 2 chiều đầu (batch_sz, n_node) -> (batch_sz*n_node, dim)
            x_batch = geo_x.reshape(geo_x.shape[0] * geo_x.shape[1], geo_x.shape[2])

            # Tính toán offset cho geo_edge_index
            offset = []
            # Duyệt qua từng mẫu trong batch
            for i in range(batch_sz):
                n_edges = geo_edge_index[0].shape[1]
                offset_val = int(np.sqrt(n_edges)) * i
                offset.append(torch.full(geo_edge_index[0].shape, offset_val))
            offset = torch.stack(offset).cuda()
            # Cộng offset vào geo_edge_index để tạo batch offset
            geo_graph_batch_offset = geo_edge_index + offset

            # Tính new_dim và reshape để có edge_index_batch với shape (2, new_dim)
            new_dim = geo_graph_batch_offset.shape[0] * geo_graph_batch_offset.shape[2]
            edge_index_batch = geo_graph_batch_offset.permute(1, 0, 2).reshape(2, new_dim)

            # Reshape geo_edge_attr theo định dạng của code bên dưới
            edge_attr_batch = geo_edge_attr.reshape(geo_edge_attr.shape[0] * geo_edge_attr.shape[1],
                                                    geo_edge_attr.shape[2]).float()

            # Tạo đối tượng Data của PyG (Geometric) với x, edge_index và edge_attr
            data_geo_graph_batch = Data(x=x_batch, edge_index=edge_index_batch, edge_attr=edge_attr_batch)
            feats = (data_geo_graph_batch, rel_feats)
            mask = pad_mask(feats, r2l_trg, l2r_trg, pad_idx, video_mask)
        else:
            mask = pad_mask(feats, r2l_trg, l2r_trg, pad_idx, None)

        # optimizer.zero_grad()
        
        r2l_pred, l2r_pred = model(feats, r2l_trg, l2r_trg, mask)

        r2l_loss = criterion(r2l_pred.view(-1, vocab.n_vocabs),
                             r2l_trg_y.contiguous().view(-1)) / r2l_norm
        l2r_loss = criterion(l2r_pred.view(-1, vocab.n_vocabs),
                             l2r_trg_y.contiguous().view(-1)) / l2r_norm

        r2l_loss = r2l_loss /gradient_accumulation_steps
        l2r_loss = l2r_loss/gradient_accumulation_steps
        loss = reg_lambda * l2r_loss + (1 - reg_lambda) * r2l_loss
        loss.backward()
        # if gradient_clip is not None:
        #     torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        # optimizer.step()
        if (step + 1) % gradient_accumulation_steps == 0:
            if gradient_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)

            optimizer.step()
            if lr_scheduler is not None:
                lr_scheduler.step() 
            optimizer.zero_grad()

        loss_checker.update(loss.item(), r2l_loss.item(), l2r_loss.item())
        t.set_description("[Epoch #{0}] loss: {3:.3f} = (reg: {1:.3f} * r2l_loss: {4:.3f} + "
                          "(1 - reg): {2:.3f} * l2r_loss: {5:.3f})"
                          .format(e, 1 - reg_lambda, reg_lambda, *loss_checker.mean(last=10)))
    total_loss, r2l_loss, l2r_loss = loss_checker.mean()
    loss = {
        'total': total_loss,
        'r2l_loss': r2l_loss,
        'l2r_loss': l2r_loss
    }
    return loss


def test(model, val_iter, graph_data, vocab, reg_lambda, feature_mode):
    model.eval()

    loss_checker = LossChecker(3)
    pad_idx = vocab.word2idx['<PAD>']
    criterion = LabelSmoothing(vocab.n_vocabs, pad_idx, C.label_smoothing)
    with torch.no_grad():
        for batch in tqdm(val_iter, desc='Test'):
            _, feats, r2l_captions, l2r_captions = parse_batch(batch, feature_mode, graph_data)
            
            r2l_trg = r2l_captions[:, :-1]
            r2l_trg_y = r2l_captions[:, 1:]
            r2l_norm = (r2l_trg_y != pad_idx).data.sum()
            l2r_trg = l2r_captions[:, :-1]
            l2r_trg_y = l2r_captions[:, 1:]
            l2r_norm = (l2r_trg_y != pad_idx).data.sum()

            if feature_mode in ['grid-rel', 'object-rel']:
                geo_x, geo_edge_index, geo_edge_attr, object_feats, rel_feats, video_mask = feats
                batch_sz = geo_x.shape[0]
                # Chuyển đổi geo_x thành batch format: flatten 2 chiều đầu (batch_sz, n_node) -> (batch_sz*n_node, dim)
                x_batch = geo_x.reshape(geo_x.shape[0] * geo_x.shape[1], geo_x.shape[2])

                # Tính toán offset cho geo_edge_index
                offset = []
                # Duyệt qua từng mẫu trong batch
                for i in range(batch_sz):
                    n_edges = geo_edge_index[0].shape[1]
                    offset_val = int(np.sqrt(n_edges)) * i
                    offset.append(torch.full(geo_edge_index[0].shape, offset_val))
                offset = torch.stack(offset).cuda()
                # Cộng offset vào geo_edge_index để tạo batch offset
                geo_graph_batch_offset = geo_edge_index + offset

                # Tính new_dim và reshape để có edge_index_batch với shape (2, new_dim)
                new_dim = geo_graph_batch_offset.shape[0] * geo_graph_batch_offset.shape[2]
                edge_index_batch = geo_graph_batch_offset.permute(1, 0, 2).reshape(2, new_dim)

                # Reshape geo_edge_attr theo định dạng của code bên dưới
                edge_attr_batch = geo_edge_attr.reshape(geo_edge_attr.shape[0] * geo_edge_attr.shape[1],
                                                        geo_edge_attr.shape[2]).float()

                # Tạo đối tượng Data của PyG (Geometric) với x, edge_index và edge_attr
                data_geo_graph_batch = Data(x=x_batch, edge_index=edge_index_batch, edge_attr=edge_attr_batch)
                feats = (data_geo_graph_batch.cuda(), object_feats, rel_feats)
                mask = pad_mask(feats, r2l_trg, l2r_trg, pad_idx, video_mask)
            elif feature_mode in ['grid', 'object']: 
                geo_x, geo_edge_index, geo_edge_attr, object_feats, video_mask = feats
                batch_sz = geo_x.shape[0]
                # Chuyển đổi geo_x thành batch format: flatten 2 chiều đầu (batch_sz, n_node) -> (batch_sz*n_node, dim)
                x_batch = geo_x.reshape(geo_x.shape[0] * geo_x.shape[1], geo_x.shape[2])

                # Tính toán offset cho geo_edge_index
                offset = []
                # Duyệt qua từng mẫu trong batch
                for i in range(batch_sz):
                    n_edges = geo_edge_index[0].shape[1]
                    offset_val = int(np.sqrt(n_edges)) * i
                    offset.append(torch.full(geo_edge_index[0].shape, offset_val))
                offset = torch.stack(offset).cuda()
                # Cộng offset vào geo_edge_index để tạo batch offset
                geo_graph_batch_offset = geo_edge_index + offset

                # Tính new_dim và reshape để có edge_index_batch với shape (2, new_dim)
                new_dim = geo_graph_batch_offset.shape[0] * geo_graph_batch_offset.shape[2]
                edge_index_batch = geo_graph_batch_offset.permute(1, 0, 2).reshape(2, new_dim)

                # Reshape geo_edge_attr theo định dạng của code bên dưới
                edge_attr_batch = geo_edge_attr.reshape(geo_edge_attr.shape[0] * geo_edge_attr.shape[1],
                                                        geo_edge_attr.shape[2]).float()

                # Tạo đối tượng Data của PyG (Geometric) với x, edge_index và edge_attr
                data_geo_graph_batch = Data(x=x_batch, edge_index=edge_index_batch, edge_attr=edge_attr_batch)
                feats = (data_geo_graph_batch, object_feats)
                mask = pad_mask(feats, r2l_trg, l2r_trg, pad_idx, video_mask)
            elif feature_mode in ['grid-rel-no_obj']: 
                geo_x, geo_edge_index, geo_edge_attr, rel_feats, video_mask = feats
                batch_sz = geo_x.shape[0]
                # Chuyển đổi geo_x thành batch format: flatten 2 chiều đầu (batch_sz, n_node) -> (batch_sz*n_node, dim)
                x_batch = geo_x.reshape(geo_x.shape[0] * geo_x.shape[1], geo_x.shape[2])

                # Tính toán offset cho geo_edge_index
                offset = []
                # Duyệt qua từng mẫu trong batch
                for i in range(batch_sz):
                    n_edges = geo_edge_index[0].shape[1]
                    offset_val = int(np.sqrt(n_edges)) * i
                    offset.append(torch.full(geo_edge_index[0].shape, offset_val))
                offset = torch.stack(offset).cuda()
                # Cộng offset vào geo_edge_index để tạo batch offset
                geo_graph_batch_offset = geo_edge_index + offset

                # Tính new_dim và reshape để có edge_index_batch với shape (2, new_dim)
                new_dim = geo_graph_batch_offset.shape[0] * geo_graph_batch_offset.shape[2]
                edge_index_batch = geo_graph_batch_offset.permute(1, 0, 2).reshape(2, new_dim)

                # Reshape geo_edge_attr theo định dạng của code bên dưới
                edge_attr_batch = geo_edge_attr.reshape(geo_edge_attr.shape[0] * geo_edge_attr.shape[1],
                                                        geo_edge_attr.shape[2]).float()

                # Tạo đối tượng Data của PyG (Geometric) với x, edge_index và edge_attr
                data_geo_graph_batch = Data(x=x_batch, edge_index=edge_index_batch, edge_attr=edge_attr_batch)
                feats = (data_geo_graph_batch, rel_feats)
                mask = pad_mask(feats, r2l_trg, l2r_trg, pad_idx, video_mask)
            else:
                mask = pad_mask(feats, r2l_trg, l2r_trg, pad_idx, None)

            r2l_pred, l2r_pred = model(feats, r2l_trg, l2r_trg, mask)

            r2l_loss = criterion(r2l_pred.view(-1, vocab.n_vocabs),
                                 r2l_trg_y.contiguous().view(-1)) / r2l_norm
            l2r_loss = criterion(l2r_pred.view(-1, vocab.n_vocabs),
                                 l2r_trg_y.contiguous().view(-1)) / l2r_norm
            loss = reg_lambda * l2r_loss + (1 - reg_lambda) * r2l_loss
            loss_checker.update(loss.item(), r2l_loss.item(), l2r_loss.item())

        total_loss, r2l_loss, l2r_loss = loss_checker.mean()
        loss = {
            'total': total_loss,
            'r2l_loss': r2l_loss,
            'l2r_loss': l2r_loss
        }
    return loss


def get_predicted_captions(data_iter, graph_data, model, beam_size, max_len, feature_mode):
    def build_onlyonce_iter(data_iter):
        onlyonce_dataset = {}
        for batch in tqdm(iter(data_iter), desc='build onlyonce_iter'):
            vids, feats, _, _ = parse_batch(batch, feature_mode, graph_data)
            if feature_mode == 'btkg':
                for vid, image_feat, motion_feat, object_feat, rel_feat in zip(vids, feats[0], feats[1], feats[2],
                                                                                feats[3]):
                    if vid not in onlyonce_dataset:
                        onlyonce_dataset[vid] = (image_feat, motion_feat, object_feat, rel_feat)
            elif feature_mode in ['grid','object']:
                for vid, geo_x, geo_edge_indexs, geo_edge_attrs, object_feat, video_mask in zip(vids, feats[0], feats[1], feats[2],
                                                                                feats[3], feats[4]):
                    if vid not in onlyonce_dataset:
                        onlyonce_dataset[vid] = (geo_x, geo_edge_indexs, geo_edge_attrs, object_feat, video_mask)
            elif feature_mode in ['grid-rel-no_obj']:
                for vid, geo_x, geo_edge_indexs, geo_edge_attrs, rel_feat, video_mask in zip(vids, feats[0], feats[1], feats[2],
                                                                                feats[3], feats[4]):
                    if vid not in onlyonce_dataset:
                        onlyonce_dataset[vid] = (geo_x, geo_edge_indexs, geo_edge_attrs, rel_feat, video_mask)
            elif feature_mode in ['grid-rel','object-rel']:
                for vid, geo_x, geo_edge_indexs, geo_edge_attrs, object_feat, rel_feat, video_mask in zip(vids, feats[0], feats[1], feats[2],
                                                                                feats[3], feats[4], feats[5]):
                    if vid not in onlyonce_dataset:
                        onlyonce_dataset[vid] = (geo_x, geo_edge_indexs, geo_edge_attrs, object_feat, rel_feat, video_mask)


        onlyonce_iter = []
        vids = list(onlyonce_dataset.keys())
        feats = list(onlyonce_dataset.values())
        del onlyonce_dataset
        torch.cuda.empty_cache()
        time.sleep(5)
        batch_size = 1
        while len(vids) > 0:
            if feature_mode == 'btkg':
                image_feats = []
                motion_feats = []
                object_feats = []
                rel_feats = []
                for image_feature, motion_feature, object_feat, rel_feat in feats[:batch_size]:
                    image_feats.append(image_feature)
                    motion_feats.append(motion_feature)
                    object_feats.append(object_feat)
                    rel_feats.append(rel_feat)
                onlyonce_iter.append((vids[:batch_size],
                                    (torch.stack(image_feats), torch.stack(motion_feats), torch.stack(object_feats),
                                        torch.stack(rel_feats))))
            elif feature_mode in ['grid','object']:
                geo_x_feats = []
                geo_edge_index_feats = []
                geo_edge_attr_feats = []
                object_feats = []
                video_masks = []
                for geo_x_feat, geo_edge_index_feat, geo_edge_attr_feat, object_feat, video_mask in feats[:batch_size]:
                    geo_x_feats.append(geo_x_feat)
                    geo_edge_index_feats.append(geo_edge_index_feat)
                    geo_edge_attr_feats.append(geo_edge_attr_feat)
                    object_feats.append(object_feat)
                    video_masks.append(video_mask)
                onlyonce_iter.append((vids[:batch_size],
                                    (torch.stack(geo_x_feats), torch.stack(geo_edge_index_feats), torch.stack(geo_edge_attr_feats),
                                    torch.stack(object_feats), torch.stack(video_masks))))
            elif feature_mode in ['grid-rel-no_obj']:
                geo_x_feats = []
                geo_edge_index_feats = []
                geo_edge_attr_feats = []
                rel_feats = []
                video_masks = []
                for geo_x_feat, geo_edge_index_feat, geo_edge_attr_feat, rel_feat, video_mask in feats[:batch_size]:
                    geo_x_feats.append(geo_x_feat)
                    geo_edge_index_feats.append(geo_edge_index_feat)
                    geo_edge_attr_feats.append(geo_edge_attr_feat)
                    rel_feats.append(rel_feat)
                    video_masks.append(video_mask)
                onlyonce_iter.append((vids[:batch_size],
                                    (torch.stack(geo_x_feats), torch.stack(geo_edge_index_feats), torch.stack(geo_edge_attr_feats),
                                    torch.stack(rel_feats), torch.stack(video_masks))))
            elif feature_mode in ['grid-rel','object-rel']:
                geo_x_feats = []
                geo_edge_index_feats = []
                geo_edge_attr_feats = []
                object_feats = []
                rel_feats = []
                video_masks = []
                for geo_x_feat, geo_edge_index_feat, geo_edge_attr_feat, object_feat, rel_feat, video_mask in feats[:batch_size]:
                    geo_x_feats.append(geo_x_feat)
                    geo_edge_index_feats.append(geo_edge_index_feat)
                    geo_edge_attr_feats.append(geo_edge_attr_feat)
                    object_feats.append(object_feat)
                    rel_feats.append(rel_feat)
                    video_masks.append(video_mask)
                onlyonce_iter.append((vids[:batch_size],
                                    (torch.stack(geo_x_feats), torch.stack(geo_edge_index_feats), torch.stack(geo_edge_attr_feats),
                                    torch.stack(object_feats), torch.stack(rel_feats), torch.stack(video_masks))))

            vids = vids[batch_size:]
            feats = feats[batch_size:]
        return onlyonce_iter
    
    model.eval()
    onlyonce_iter = build_onlyonce_iter(data_iter)

    r2l_vid2pred = {}
    l2r_vid2pred = {}

    # BOS_idx = vocab.word2idx['<BOS>']
    with torch.no_grad():
        for vids, feats in tqdm(onlyonce_iter):
            if feature_mode in ['grid-rel','object-rel']:
                geo_x, geo_edge, geo_edge_attr, object_feat, rel_feat, video_mask = feats
                data_geo_graph = Data(x=geo_x[0], edge_index=geo_edge[0], edge_attr=geo_edge_attr[0])
                feats = (data_geo_graph, object_feat, rel_feat)
                r2l_captions, l2r_captions = model.beam_search_decode(feats, beam_size, max_len, video_mask)
            elif feature_mode in ['grid','object']:
                geo_x, geo_edge, geo_edge_attr, object_feat, video_mask = feats
                data_geo_graph = Data(x=geo_x[0], edge_index=geo_edge[0], edge_attr=geo_edge_attr[0])
                feats = (data_geo_graph, object_feat)
                r2l_captions, l2r_captions = model.beam_search_decode(feats, beam_size, max_len, video_mask)
            elif feature_mode in ['grid-rel-no_obj']:
                geo_x, geo_edge, geo_edge_attr, rel_feat, video_mask = feats
                data_geo_graph = Data(x=geo_x[0], edge_index=geo_edge[0], edge_attr=geo_edge_attr[0])
                feats = (data_geo_graph, rel_feat)
                r2l_captions, l2r_captions = model.beam_search_decode(feats, beam_size, max_len, video_mask)
            else: 
                r2l_captions, l2r_captions = model.beam_search_decode(feats, beam_size, max_len, None)
            # r2l_captions = [idxs_to_sentence(caption, vocab.idx2word, BOS_idx) for caption in r2l_captions]
            l2r_captions = [" ".join(caption[0].value) for caption in l2r_captions]
            r2l_captions = [" ".join(caption[0].value) for caption in r2l_captions]
            r2l_vid2pred.update({v: p for v, p in zip(vids, r2l_captions)})
            l2r_vid2pred.update({v: p for v, p in zip(vids, l2r_captions)})
    return r2l_vid2pred, l2r_vid2pred


def get_groundtruth_captions(data_iter, graph_data, vocab, feature_mode):
    r2l_vid2GTs = {}
    l2r_vid2GTs = {}
    S_idx = vocab.word2idx['<S>']
    for batch in tqdm(iter(data_iter), desc='get_groundtruth_captions'):

        vids, _, r2l_captions, l2r_captions = parse_batch(batch, feature_mode, graph_data)

        for vid, r2l_caption, l2r_caption in zip(vids, r2l_captions, l2r_captions):
            if vid not in r2l_vid2GTs:
                r2l_vid2GTs[vid] = []
            if vid not in l2r_vid2GTs:
                l2r_vid2GTs[vid] = []
            r2l_caption = idxs_to_sentence(r2l_caption, vocab.idx2word, S_idx)
            l2r_caption = idxs_to_sentence(l2r_caption, vocab.idx2word, S_idx)
            r2l_vid2GTs[vid].append(r2l_caption)
            l2r_vid2GTs[vid].append(l2r_caption)
    return r2l_vid2GTs, l2r_vid2GTs


def score(vid2pred, vid2GTs):
    assert set(vid2pred.keys()) == set(vid2GTs.keys())
    vid2idx = {v: i for i, v in enumerate(vid2pred.keys())}
    refs = {vid2idx[vid]: GTs for vid, GTs in vid2GTs.items()}
    hypos = {vid2idx[vid]: [pred] for vid, pred in vid2pred.items()}

    scores = calc_scores(refs, hypos)
    return scores


# refers: https://github.com/zhegan27/SCN_for_video_captioning/blob/master/SCN_evaluation.py
def calc_scores(ref, hypo):
    """
    ref, dictionary of reference sentences (id, sentence)
    hypo, dictionary of hypothesis sentences (id, sentence)
    score, dictionary of scores
    """
    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (Meteor(), "METEOR"),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr")
    ]
    final_scores = {}
    for scorer, method in scorers:
        score, scores = scorer.compute_score(ref, hypo)
        if type(score) == list:
            for m, s in zip(method, score):
                final_scores[m] = s
        else:
            final_scores[method] = float(score)
    return final_scores


def evaluate(data_iter, graph_data, model, vocab, beam_size, max_len, feature_mode):
    r2l_vid2pred, l2r_vid2pred = get_predicted_captions(data_iter, graph_data, model, beam_size, max_len, feature_mode)
    r2l_vid2GTs, l2r_vid2GTs = get_groundtruth_captions(data_iter, graph_data, vocab, feature_mode)
    r2l_scores = score(r2l_vid2pred, r2l_vid2GTs)
    l2r_scores = score(l2r_vid2pred, l2r_vid2GTs)
    return r2l_scores, l2r_scores


# refers: https://stackoverflow.com/questions/52660985/pytorch-how-to-get-learning-rate-during-training
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def idxs_to_sentence(idxs, idx2word, EOS_idx):
    words = []
    for idx in idxs[1:]:
        idx = idx.item()
        if idx == EOS_idx:
            break
        word = idx2word[idx]
        words.append(word)
    sentence = ' '.join(words)
    return sentence


def cls_to_dict(cls):
    properties = dir(cls)
    properties = [p for p in properties if not p.startswith("__")]
    d = {}
    for p in properties:
        v = getattr(cls, p)
        if inspect.isclass(v):
            v = cls_to_dict(v)
            v['was_class'] = True
        d[p] = v
    return d


# refers https://stackoverflow.com/questions/1305532/convert-nested-python-dict-to-object
class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)


def dict_to_cls(d):
    cls = Struct(**d)
    properties = dir(cls)
    properties = [p for p in properties if not p.startswith("__")]
    for p in properties:
        v = getattr(cls, p)
        if isinstance(v, dict) and 'was_class' in v and v['was_class']:
            v = dict_to_cls(v)
        setattr(cls, p, v)
    return cls


def load_checkpoint(model, ckpt_fpath):
    checkpoint = torch.load(ckpt_fpath)
    model.load_state_dict(checkpoint['vc_model'])
    return model


def save_checkpoint(e, model, ckpt_fpath, config):
    ckpt_dpath = os.path.dirname(ckpt_fpath)
    if not os.path.exists(ckpt_dpath):
        os.makedirs(ckpt_dpath)

    torch.save({
        'epoch': e,
        'vc_model': model.state_dict(),
        'config': cls_to_dict(config),
    }, ckpt_fpath)


def save_result(vid2pred, vid2GTs, save_fpath):
    assert set(vid2pred.keys()) == set(vid2GTs.keys())

    save_dpath = os.path.dirname(save_fpath)
    if not os.path.exists(save_dpath):
        os.makedirs(save_dpath)

    vids = vid2pred.keys()
    with open(save_fpath, 'w') as fout:
        for vid in vids:
            GTs = ' / '.join(vid2GTs[vid])
            pred = vid2pred[vid]
            line = ', '.join([str(vid), pred, GTs])
            fout.write("{}\n".format(line))