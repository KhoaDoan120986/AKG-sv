import os
import cv2
import numpy as np
import glob
import h5py
from tqdm import tqdm
import torch
import torchvision.transforms as transforms
from PIL import Image
import pathlib

device = torch.device('cuda')

# Arguments
class args:
    msvd = False # for MSR-VTT change this to False
    num_features_logits = 1024
    slice_framepos=2
    root ='./'
    dset ='../' # change based on dataset location
    save_path = '../../extracted'
    max_frames = 20
    
from model.i3d.InceptionI3d import InceptionI3d

# Load I3D model
i3d = InceptionI3d(400, in_channels=3)
# pretrained model is downloaded from here: https://github.com/piergiaj/pytorch-i3d
i3d.load_state_dict(torch.load(os.path.join(args.root, 'pretrained', 'rgb_imagenet.pt')))
i3d = i3d.to(device)
i3d.eval()
print()

# Preprocess video
mean = [0.5, 0.5, 0.5]
std = [0.5, 0.5, 0.5]
data_transform_fasterrcnn = transforms.Compose([transforms.ToTensor()])
data_transform_i3d = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)])

# Preprocess video
if args.msvd:

    dset_path = os.path.join(os.path.join(args.dset,'Dataset'),'MSVD')

    features_path = os.path.join(dset_path,'raw') # video .avi
    name_list = glob.glob(features_path+os.sep+'*')

    url2id = {}
    data_path =os.path.join(os.path.join(dset_path,'captions','youtube-mapping.txt'))
    for line in open(data_path,'r').readlines():
        url2id[line.strip().split(' ')[0]] = line.strip().split(' ')[-1]


    path_to_saved_models = "extracted/msvd_object"
    pathlib.Path(path_to_saved_models).mkdir(parents=True, exist_ok=True)

else:

    dset_path = os.path.join(os.path.join(args.dset,'Dataset'),'MSRVTT')
    features_path = os.path.join(dset_path,'videos')
    features_path = os.path.join(features_path,'all')    

    name_list = glob.glob(features_path+os.sep+'*')
    path_to_saved_models = "extracted/msrvtt"
    pathlib.Path(path_to_saved_models).mkdir(parents=True, exist_ok=True)
    
def video2tensor(video_file,sample_fp=1, start_time=None, end_time=None):
    if start_time is not None or end_time is not None:
        assert isinstance(start_time, int) and isinstance(end_time, int) \
               and start_time > -1 and end_time > start_time
    assert sample_fp > -1
    # Samples a frame sample_fp X frames.
    cap = cv2.VideoCapture(video_file)

    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    total_duration = (frameCount + fps - 1) // fps
    start_sec, end_sec = 0, total_duration



    interval = 1
    if sample_fp > 0:
        interval = fps // sample_fp
    else:
        sample_fp = fps
    if interval == 0: interval = 1

    inds = [ind for ind in np.arange(0, fps, interval)]
    inds_all = [ind for ind in np.arange(0, fps, 1)]
    assert len(inds) >= sample_fp
    inds = inds[:sample_fp]
    inds = set(inds)
    ret = True
    images_fasterrcnn,images_i3d, included = [], [], []
    c = 0
    sampled_indexes = []
    for sec in np.arange(start_sec, end_sec + 1):
        if not ret: break
        # sec_base = int(sec * fps)
        for ia in inds_all:

            ret, frame = cap.read()
            if not ret: break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(frame_rgb).convert("RGB")
            if ia in inds:
                sampled_indexes.append(c)
                images_fasterrcnn.append(data_transform_fasterrcnn(pil))
            images_i3d.append(data_transform_i3d(pil))
            c+=1


    cap.release()
    tensor_fasterrcnn = None
    if len(images_fasterrcnn) > 0:
        video_data_fasterrcnn = torch.tensor(np.stack(images_fasterrcnn))
        #process raw data section
        tensor_size = video_data_fasterrcnn.size()

        tensor_fasterrcnn = video_data_fasterrcnn.view(-1, 1, tensor_size[-3], tensor_size[-2], tensor_size[-1])

        video_data_i3d = torch.tensor(np.stack(images_i3d))
        #process raw data section
        tensor_size = video_data_i3d.size()
        tensor_i3d = video_data_i3d.view(-1, 1, tensor_size[-3], tensor_size[-2], tensor_size[-1])
    else:
        video_data = torch.zeros(1)


    return tensor_fasterrcnn,tensor_i3d, sampled_indexes

# FILE_FO = "<Path to the extracted object node features>"
# save_file = path_to_saved_models + '/<Desired file name>.hdf5'
FILE_FO = "/media02/lnthanh01/phatkhoa/STGraph/extracted/msrvtt/object_node_features.hdf5"
save_file = "/media02/lnthanh01/phatkhoa/STGraph/extracted/msrvtt/object_spatio_graph.hdf5"
NUM_OBJECT = 9
frame_num = 16
with torch.no_grad():
    with h5py.File(FILE_FO, 'r') as fo, h5py.File(save_file, 'w') as f:
        for name in tqdm(name_list):
            tensor_fasterrcnn,tensor_i3d, sampled_indices = video2tensor(name)
            sample_indx =[]
            if args.max_frames < tensor_fasterrcnn.shape[0]:
                if args.slice_framepos == 0:
                    video_slice = raw_video_slice[:args.max_frames, ...]
                elif args.slice_framepos == 1:
                    video_slice = raw_video_slice[-args.max_frames:, ...]
                else:
                    sample_indx = list(np.linspace(0, tensor_fasterrcnn.shape[0] - 1, num=args.max_frames, dtype=int))
                 
            else:
                sample_indx = list(np.arange(tensor_fasterrcnn.shape[0]))

           

            if(len(sample_indx)<args.max_frames):
                additional = 20-len(sample_indx)
                sample_indx += (additional * [-1])
            
            for idx_grp, i in enumerate(sample_indx):
                if args.msvd:
                    ide = url2id[name.split(os.sep)[-1].split('.')[0]]
                else:
                    ide = name.split(os.sep)[-1].split('.')[0]
            
            
                zero = np.array([0.0]*args.num_features_logits)
                Gs_temp = [[zero]* NUM_OBJECT for m in range(NUM_OBJECT)]
                
                if (i==-1):
                    f.create_dataset(ide+'-'+str(idx_grp), data = Gs_temp)# for each frame
                    continue
              
                i_i3d = (sampled_indices[i]//frame_num)*frame_num
                if len(tensor_i3d)-i_i3d < frame_num:
                    i_i3d = len(tensor_i3d)-frame_num
                    
                curr_batch = tensor_i3d[i_i3d:i_i3d+frame_num,...].unsqueeze(0)
                n,video_frame,num,channel,h,w = curr_batch.shape
                curr_batch = curr_batch.view(num,video_frame,channel, h, w)
                curr_batch = curr_batch.permute(0,2,1,3,4)
           

                
                out_logits = i3d.extract_features(curr_batch.to(device))
            
                out_logits= out_logits[:,:,0,0,0]
                out_logits = out_logits.cpu().numpy()
                
                obj = fo[ide+'-'+str(idx_grp)][:]

                
                for k in range(NUM_OBJECT):
                    for l in range(k, NUM_OBJECT):
                        sum_k = np.sum(obj[k])
                        sum_l = np.sum(obj[l])
                        if (sum_k!=0 and sum_l!=0):
                            Gs_temp[k][l] = out_logits.tolist()[0]
                            Gs_temp[l][k] = Gs_temp[k][l]

                          
                if args.msvd:
                    ide = url2id[name.split(os.sep)[-1].split('.')[0]]
                else:
                    ide = name.split(os.sep)[-1].split('.')[0]
                f.create_dataset(ide+'-'+str(idx_grp), data = Gs_temp)