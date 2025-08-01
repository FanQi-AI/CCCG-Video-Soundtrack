import torch
import torch.nn as nn
# import clip
from einops import rearrange
# import kornia
from clap.clap_module import create_model
from clap.training.data import get_audio_features
from transformers import CLIPProcessor, CLIPModel
import torchaudio
from transformers import RobertaTokenizer
import torch.nn.functional as F
from vector_quantize_pytorch import ResidualVQ
from einops import rearrange
from PIL import Image
class CLIPVideoEmbdding(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.clip_model = CLIPModel.from_pretrained("/jsj_qf/tjut_makunsheng/MusicLDM-main/openai/clip-vit-base-patch32")
        self.clip_model.eval()
        self.processor = CLIPProcessor.from_pretrained("/jsj_qf/tjut_makunsheng/MusicLDM-main/openai/clip-vit-base-patch32")

        

    def forward(self, batch):
        movie = list(batch['movie'])
        images = [Image.open(image_path) for image_path in movie]
        movie_inputs = self.processor(images=images, return_tensors="pt", padding=True)
        movie_inputs = movie_inputs.to(self.device)

        movie_features = self.clip_model.get_image_features(**movie_inputs)

        return movie_features


class CLAPAudioEmbeddingClassifierFreev2(nn.Module):
    def __init__(
        self,
        pretrained_path,
        sampling_rate=16000,
        embed_mode="audio",
        unconditional_prob=0.1,
        random_mute=False,
        max_random_mute_portion=0.5,
        training_mode=True,
    ):
        super().__init__()
        self.device = "cuda"
        self.precision = "fp32"
        self.amodel = "HTSAT-base"  # or 'PANN-14'
        self.tmodel = "roberta"  # the best text encoder in our training
        self.enable_fusion = False  # False if you do not want to use the fusion model
        self.fusion_type = "aff_2d"
        self.pretrained = pretrained_path
        self.embed_mode = embed_mode
        self.embed_mode_orig = embed_mode
        self.sampling_rate = sampling_rate
        self.unconditional_prob = unconditional_prob
        self.random_mute = random_mute
        self.tokenize = RobertaTokenizer.from_pretrained("roberta-base")
        self.max_random_mute_portion = max_random_mute_portion
        self.training_mode = training_mode
        self.model, self.model_cfg = create_model(
            self.amodel,
            self.tmodel,
            self.pretrained,
            precision=self.precision,
            device=self.device,
            enable_fusion=self.enable_fusion,
            fusion_type=self.fusion_type,
        )
        for p in self.model.parameters():
            p.requires_grad = False

        self.model.eval()

    def get_unconditional_condition(self, batchsize):
        self.unconditional_token = self.model.get_text_embedding(
            self.tokenizer(["", ""])
        )[0:1]
        return torch.cat([self.unconditional_token.unsqueeze(0)] * batchsize, dim=0)

    def batch_to_list(self, batch):
        ret = []
        for i in range(batch.size(0)):
            ret.append(batch[i])
        return ret

    def make_decision(self, probability):
        if float(torch.rand(1)) < probability:
            return True
        else:
            return False

    def random_uniform(self, start, end):
        val = torch.rand(1).item()
        return start + (end - start) * val

    def _random_mute(self, waveform):
        # waveform: [bs, t-steps]
        t_steps = waveform.size(-1)
        for i in range(waveform.size(0)):
            mute_size = int(
                self.random_uniform(0, end=int(t_steps * self.max_random_mute_portion))
            )
            mute_start = int(self.random_uniform(0, t_steps - mute_size))
            waveform[i, mute_start : mute_start + mute_size] = 0
        return waveform

    def cos_similarity(self, waveform, text):
        # waveform: [bs, t_steps]
        with torch.no_grad():
            self.embed_mode = "audio"
            audio_emb = self(waveform.cuda())
            self.embed_mode = "text"
            text_emb = self(text)
            similarity = F.cosine_similarity(audio_emb, text_emb, dim=2)
            return similarity.squeeze()

    def forward(self, batch):
        # If you want this conditioner to be unconditional, set self.unconditional_prob = 1.0
        # If you want this conditioner to be fully conditional, set self.unconditional_prob = 0.0
        if self.model.training == True and not self.training_mode:
            print(
                "The pretrained CLAP model should always be in eval mode. Reloading model just in case you change the parameters."
            )
            self.model, self.model_cfg = create_model(
                self.amodel,
                self.tmodel,
                self.pretrained,
                precision=self.precision,
                device="cuda",
                enable_fusion=self.enable_fusion,
                fusion_type=self.fusion_type,
            )
            for p in self.model.parameters():
                p.requires_grad = False
            self.model.eval()

        # if(self.training_mode):
        #     assert self.model.training == True
        # else:
        #     assert self.model.training == False

        # the 'fusion' truncate mode can be changed to 'rand_trunc' if run in unfusion mode
        if self.embed_mode == "audio":
            with torch.no_grad():
                audio_dict_list = []
                assert (
                    self.sampling_rate == 16000
                ), "We only support 16000 sampling rate"
                if self.random_mute:
                    batch = self._random_mute(batch)
                # batch: [bs, 1, t-samples]
                batch = torchaudio.functional.resample(
                    batch, orig_freq=self.sampling_rate, new_freq=48000
                )
                for waveform in self.batch_to_list(batch):
                    audio_dict = {}
                    audio_dict = get_audio_features(
                        audio_dict,
                        waveform,
                        480000,
                        data_truncating="rand_trunc",
                        data_filling="repeatpad",
                        audio_cfg=self.model_cfg["audio_cfg"],
                    )
                    audio_dict_list.append(audio_dict)
                # [bs, 512]
                embed = self.model.get_audio_embedding(audio_dict_list)
        elif self.embed_mode == "text":
            with torch.no_grad():
                # the 'fusion' truncate mode can be changed to 'rand_trunc' if run in unfusion mode
                text_data = self.tokenizer(batch)
                embed = self.model.get_text_embedding(text_data)
        embed = embed.unsqueeze(1)
        # self.unconditional_token = self.model.get_text_embedding(
        #     self.tokenizer(["", ""])
        # )[0:1]

        # for i in range(embed.size(0)):
        #     if self.make_decision(self.unconditional_prob):
        #         embed[i] = self.unconditional_token
        # [bs, 1, 512]
        return embed.detach()

    def tokenizer(self, text):
        result = self.tokenize(
            text,
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        return {k: v.squeeze(0) for k, v in result.items()}

class CLAPResidualVQ(nn.Module):
    def __init__(self, 
            clap_wrapper: CLAPAudioEmbeddingClassifierFreev2, 
            codebook_size: int = 1024,
            num_quantizers: int = 12,
            ema_decay: float = 0.95,
            ema_dead_threshold: float = 0.0
        ) -> None:
        super().__init__()

        self.clap_wrapper = clap_wrapper
        self.codebook_size = codebook_size

        for param in self.clap_wrapper.parameters():
            param.requires_grad = False

        self.rvq = ResidualVQ(
            dim = self.clap_wrapper.model.joint_embed_shape,
            num_quantizers = num_quantizers,
            codebook_size = codebook_size,
            commitment_weight = 0.,
            decay = ema_decay,
            kmeans_init = True,
            threshold_ema_dead_code = ema_dead_threshold
        )
    
    def forward(self, x, is_text = False):
        '''
        x: either audio input [B, T] or text input [B]
        '''
        with torch.no_grad():
            self.clap_wrapper.eval()
            self.clap_wrapper.embed_mode = 'text' if is_text else 'audio' 
            embedding = self.clap_wrapper(x)
            self.clap_wrapper.embed_mode = self.clap_wrapper.embed_mode_orig

        q, indices, _ = self.rvq(rearrange(embedding, 'n c -> n 1 c'))
        loss = nn.functional.mse_loss(q, rearrange(embedding, 'n c -> n 1 c'))

        indices = rearrange(indices, 'n 1 c -> n c 1')
        return loss, q, indices


           

        
        

