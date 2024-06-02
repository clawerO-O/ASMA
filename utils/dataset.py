import os
import csv
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class DFDCDataset(Dataset):
    def __init__(self, data_csv, required_set, data_root="",
                 ratio=(0.25, 0.05), stable=False, transform=None):
        video_info = []
        data_list = []

        with open(data_csv) as fin:
            reader = csv.DictReader(fin)

            for row in reader:
                if row["set_name"] == required_set:
                    label = int(row["is_fake"])
                    n_frame = int(row["n_frame"])
                    select_frame = round(n_frame * ratio[label])

                    for sample_idx in range(select_frame):
                        data_list.append((len(video_info), sample_idx))

                    video_info.append({
                        "name": row["name"],
                        "label": label,
                        "n_frame": n_frame,
                        "select_frame": select_frame,
                    })
                    video_files = os.listdir(data_root)
                    train_info = []
                    train_list = []
                    for index, video_file in enumerate(video_files):
                        video_file_name = video_file.split('.')[0]
                        for x in video_info:
                            if x["name"] == video_file_name:
                                sel_frame = x["select_frame"]
                                for sa in range(sel_frame):
                                    train_list.append((len(train_info), sa))
                                train_info.append(x)
                                break

                    self.stable = stable
                    self.data_root = data_root
                    self.video_info = train_info
                    self.data_list = train_list
                    self.transform = transform
    def __getitem__(self, index):
        video_idx, sample_idx = self.data_list[index]
        info = self.video_info[video_idx]

        if self.stable:
            frame_idx = info["n_frame"] * sample_idx // info["select_frame"]
        else:
            frame_idx = random.randint(0, info["n_frame"] - 1)

        image_path = os.path.join(self.data_root, info["name"],
                                  "%03d.jpg" % frame_idx)
        try:
            img = Image.open(image_path).convert("RGB")
        except OSError:
            img = Image.fromarray(np.random.randint(0, 255, (320, 320, 3), dtype=np.uint8))
        if self.transform is not None:
            img = self.transform(img)

        return img, info["label"]

    def __len__(self):
        return len(self.data_list)