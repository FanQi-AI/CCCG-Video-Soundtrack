import  sys
sys.path.append('/data/HDD1/tjut-nilifeng/makunsheng/mamba-videomuisc/')
import time
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import yaml
import os
# from data.audio.stft import TacotronSTFT
# from data.audio.tools import wav_to_fbank
import json
import torchaudio
import torch
from librosa.filters import mel as librosa_mel_fn
from concurrent.futures import ThreadPoolExecutor, as_completed
# import audioldm_train.utilities.audio as Audio
import warnings
from tqdm import tqdm
import numpy as np
import multiprocessing as mp
# 忽略所有警告
warnings.filterwarnings("ignore")

def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)

def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output

class MusicDataset(Dataset):
    def __init__(
        self,
        config=None,
        split="train",

    ):
        self.config = config
        self.split = split # If none, random choose

        self.dataset_name = self.config["data"][self.split]
        self.build_dataset()
        # self.build_setting_parameters()

        # self.build_dsp()

    def __getitem__(self, index):
        # (
        #     fname,
        #     waveform,
        #     log_mel_spec,
        # ) = self.feature_extraction(index)
        # text = self.get_sample_text_caption(datum, mix_datum, label_vector)
        music_path = "/data/HDD1/tjut_makunsheng/music_effect_movie/music"
        effect_path = "/data/HDD1/tjut_makunsheng/music_effect_movie/effect"
        music_effect_path = "/data/HDD1/tjut_makunsheng/music_effect_movie/effect_npy"
        name, ext = os.path.splitext(os.path.basename(self.data[index]))
        # 更改扩展名为 '.wav'
        new_filename = name + '.wav'
        # log_mel_spec,mel = self.read_audio_mel(self.data[index])
        data = {
            "text": "A melancholic and emotional piano piece, with slow, deep, and expressive melodies, evoking sadness and nostalgia.",  # list
            "fname": os.path.basename(self.data[index]),  # list
            # tensor, [batchsize, class_num]
            # tensor, [batchsize, 1, samples_num]
            "waveform":self.read_audio_file(os.path.join(music_path,new_filename)),
            "effect_waveform":self.read_audio_file(os.path.join(effect_path,new_filename)),
            # tensor, [batchsize, t-steps, mel-bins]
            "fbank": self.read_npy(os.path.join(music_effect_path,os.path.basename(self.data[index]))),
        }

        return data

    def __len__(self):
        return len(self.data)
    
    # def read_audio_mel(self, filename):
    #     waveform, sr = torchaudio.load(filename)  # Faster!!!
    #     waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=16000)
    #     waveform = waveform[0, ...]

    #     # log_mel_spec, stft = self.mel_spectrogram_train(waveform.unsqueeze(0))
    #     fn_STFT = TacotronSTFT(
    #         self.filter_length,
    #         self.hopsize,
    #         self.win_length,
    #         self.melbins,
    #         self.sampling_rate,
    #         self.mel_fmin,
    #         self.mel_fmax,
    #     )
    #     mel, _, _ = wav_to_fbank(
    #         filename,
    #         target_length=self.target_length, fn_STFT=fn_STFT
    #     )
    #     return  mel  
    # def mel_spectrogram_train(self, y):
    #     if torch.min(y) < -1.0:
    #         print("train min value is ", torch.min(y))
    #     if torch.max(y) > 1.0:
    #         print("train max value is ", torch.max(y))

    #     if self.mel_fmax not in self.mel_basis:
    #         mel = librosa_mel_fn(
    #             sr=self.sampling_rate,
    #             n_fft=self.filter_length,
    #             n_mels=self.n_mel,
    #             fmin=self.mel_fmin,
    #             fmax=self.mel_fmax,
    #         )
    #         self.mel_basis[str(self.mel_fmax) + "_" + str(y.device)] = (
    #             torch.from_numpy(mel).float().to(y.device)
    #         )
    #         self.hann_window[str(y.device)] = torch.hann_window(self.win_length).to(
    #             y.device
    #         )

    #     y = torch.nn.functional.pad(
    #         y.unsqueeze(1),
    #         (
    #             int((self.filter_length - self.hop_length) / 2),
    #             int((self.filter_length - self.hop_length) / 2),
    #         ),
    #         mode="reflect",
    #     )

    #     y = y.squeeze(1)

    #     stft_spec = torch.stft(
    #         y,
    #         self.filter_length,
    #         hop_length=self.hop_length,
    #         win_length=self.win_length,
    #         window=self.hann_window[str(y.device)],
    #         center=False,
    #         pad_mode="reflect",
    #         normalized=False,
    #         onesided=True,
    #         return_complex=True,
    #     )

    #     stft_spec = torch.abs(stft_spec)

    #     mel = spectral_normalize_torch(
    #         torch.matmul(
    #             self.mel_basis[str(self.mel_fmax) + "_" + str(y.device)], stft_spec
    #         )
    #     )

    #     return mel[0], stft_spec[0]


    def build_dataset(self):
        with open(self.dataset_name, 'r') as f:
            self.data = json.load(f)
    def read_audio_file(self, filename):
        waveform, sr = torchaudio.load(filename)  # Faster!!!
        waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=16000)
        waveform = waveform[0, ...]
        return waveform 
    
    def read_npy(self,filename):
        data = np.load(filename)
        return data
    # def build_dsp(self):
    #     self.mel_basis = {}
    #     self.hann_window = {}

    #     self.filter_length = self.config["preprocessing"]["stft"]["filter_length"]
    #     self.hop_length = self.config["preprocessing"]["stft"]["hop_length"]
    #     self.win_length = self.config["preprocessing"]["stft"]["win_length"]
    #     self.n_mel = self.config["preprocessing"]["mel"]["n_mel_channels"]
    #     self.sampling_rate = self.config["preprocessing"]["audio"]["sampling_rate"]
    #     self.mel_fmin = self.config["preprocessing"]["mel"]["mel_fmin"]
    #     self.mel_fmax = self.config["preprocessing"]["mel"]["mel_fmax"]

    #     self.STFT = Audio.stft.TacotronSTFT(
    #         self.config["preprocessing"]["stft"]["filter_length"],
    #         self.config["preprocessing"]["stft"]["hop_length"],
    #         self.config["preprocessing"]["stft"]["win_length"],
    #         self.config["preprocessing"]["mel"]["n_mel_channels"],
    #         self.config["preprocessing"]["audio"]["sampling_rate"],
    #         self.config["preprocessing"]["mel"]["mel_fmin"],
    #         self.config["preprocessing"]["mel"]["mel_fmax"],
    #     )
        






    def build_setting_parameters(self):
        # Read from the json config
        self.melbins = self.config["preprocessing"]["mel"]["n_mel_channels"]
        self.mel_fmin = self.config["preprocessing"]["mel"]["mel_fmin"]
        self.mel_fmax = self.config["preprocessing"]["mel"]["mel_fmax"]
        self.sampling_rate = self.config["preprocessing"]["audio"]["sampling_rate"]
        self.hopsize = self.config["preprocessing"]["stft"]["hop_length"]
        self.filter_length= self.config["preprocessing"]["stft"]["filter_length"]
        self.win_length = self.config["preprocessing"]["stft"]["win_length"]
        self.duration = self.config["preprocessing"]["audio"]["duration"]
        self.target_length = int(self.duration * self.sampling_rate / self.hopsize)

        self.mixup = self.config["augmentation"]["mixup"]

        if "train" not in self.split:
            self.mixup = 0.0
            # self.freqm = 0
            # self.timem = 0

def process_wav_wrapper(args):
    return process_wav(*args)

def process_wav(filepath, save_dir,dataset):
    mel = dataset.read_audio_mel(filepath)
    # 保存 mel 为 npy 文件
    mel_save_path = os.path.join(save_dir, f"{os.path.splitext(os.path.basename(filepath))[0]}.npy")
    np.save(mel_save_path, mel)
    return mel_save_path
if __name__ == "__main__":

    # data = np.load("/data/HDD1/tjut_makunsheng/music_effect_movie/music+effect_npy/movie_102_clip_10_4.npy")
    # from tqdm import tqdm
    config_yaml = "/data/HDD1/tjut-nilifeng/makunsheng/mamba-videomuisc/audioldm_train/config/ldm_original_musicbench_mamba.yaml"
    configs = yaml.load(open(config_yaml, "r"), Loader=yaml.FullLoader)
    dataset = MusicDataset(
        config=configs, split="train",
    )
    loader = DataLoader(dataset, batch_size=1, num_workers=0, shuffle=True)

    # for cnt, each in tqdm(enumerate(loader)):
    #     print( each["fname"])
        # print(each['freq_energy_percentile'])
        # pass
    music_dir = "/data/HDD1/tjut_makunsheng/music_effect_movie/music/"
    save_dir = "/data/HDD1/tjut_makunsheng/music_effect_movie/music_npy"  # 替换为保存npy文件的路径
    os.makedirs(save_dir, exist_ok=True)
    all_wav_files = []
    for root, dirs, files in os.walk(music_dir):
        for file in files:
            if file.endswith('.wav'):
                all_wav_files.append(os.path.join(root, file))
    with ThreadPoolExecutor(max_workers=4) as executor:
        # 提交任务
        futures = {executor.submit(process_wav, filepath, save_dir, dataset): filepath for filepath in all_wav_files}
        results = []
        
        # 使用 tqdm 显示进度条
        with tqdm(total=len(all_wav_files)) as progress_bar:
            for future in as_completed(futures):
                results.append(future.result())
                progress_bar.update(1)