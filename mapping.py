import pickle
import json
from tqdm import tqdm

mapping_path = '/media02/lnthanh01/phatkhoa/Dataset/MSVD/captions/youtube-mapping.txt'

pkl_path = '/media02/lnthanh01/phatkhoa/STGraph/extracted/msvd_20_1fps/stg.pickle'

train_pkl_path = '/media02/lnthanh01/phatkhoa/ZZZ/data/MSVD/features/MSVD_GBased201fps_train.pickle'
test_pkl_path = '/media02/lnthanh01/phatkhoa/ZZZ/data/MSVD/features/MSVD_GBased201fps_test.pickle'
val_pkl_path = '/media02/lnthanh01/phatkhoa/ZZZ/data/MSVD/features/MSVD_GBased201fps_val.pickle'

# Đọc danh sách train, test, val
with open('/media02/lnthanh01/phatkhoa/ZZZ/data/MSVD/metadata/train.list', 'r') as f:
    train_list = set(json.load(f))
with open('/media02/lnthanh01/phatkhoa/ZZZ/data/MSVD/metadata/test.list', 'r') as f:
    test_list = set(json.load(f))
with open('/media02/lnthanh01/phatkhoa/ZZZ/data/MSVD/metadata/valid.list', 'r') as f:
    val_list = set(json.load(f))

# Đọc file mapping
mapping = {}
with open(mapping_path, "r") as f:
    for line in f:
        videoid, videoname = line.strip().split()
        mapping[videoname] = videoid

# Đọc file pickle
with open(pkl_path, "rb") as f:
    data = pickle.load(f)

train_data = {}
test_data = {}
val_data = {}

for videoname, value in tqdm(data.items(), desc="Processing videos"):
    if videoname in mapping:
        videoid = mapping[videoname]
        if videoid in train_list:
            train_data[videoid] = value
        elif videoid in test_list:
            test_data[videoid] = value
        elif videoid in val_list:
            val_data[videoid] = value

for file_path, data, name in zip(
    [train_pkl_path, test_pkl_path, val_pkl_path],
    [train_data, test_data, val_data],
    ["Train", "Test", "Validation"]
):
    with open(file_path, "wb") as f:
        tqdm.write(f"Saving {name} data...")
        pickle.dump(data, f)


