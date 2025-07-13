# Dataset
import csv
import numpy as np
from torch.utils.data import Dataset
import os


def make_index_dict(label_csv):
    index_lookup = {}
    with open(label_csv, "r") as f:
        csv_reader = csv.DictReader(f)
        for row in csv_reader:
            index_lookup[row["mid"]] = row["index"]
    return index_lookup


class MusicDataset(Dataset):
    def __init__(self, data, logfile):
        """
        Args:
            data: list of dicts, each with keys: 'mel', 'wav', 'music_condition', 'sfx_condition'
            logfile: path to log file
        """
        super().__init__()
        self.data = data
        self.logfile = logfile

    def __getitem__(self, index):
        item = self.data[index]
        data_dict = {}

        # construct dict
        data_dict['fname'] = f"infer_file_{index}"
        data_dict['mel'] = item['mel']            # shape: (T, F)
        data_dict['wav'] = item['wav']            # shape: (T,)
        data_dict['music_condition'] = item['music_condition']  # any type: string, label id, embedding, etc.
        data_dict['sfx_condition'] = item['sfx_condition']

        # logging
        log_line = f"{data_dict['fname']}: music_cond={data_dict['music_condition']}, sfx_cond={data_dict['sfx_condition']}"
        if index == 0:
            with open(os.path.join(self.logfile), 'w') as f:
                f.write(log_line)
        else:
            with open(os.path.join(self.logfile), 'a') as f:
                f.write(f"\n{log_line}")

        return data_dict

    def __len__(self):
        return len(self.data)
