import os

import torch
from matplotlib import pyplot as plt
from torch.nn import functional as F
import numpy as np
from fairseq import utils,tasks
from fairseq import checkpoint_utils
from utils.eval_utils import eval_step
from tasks.mm_tasks.caption import CaptionTask
from models.ofa import OFAModel
from PIL import Image




class ExplanationGenerator:
    def __init__(self):
        # Register caption task
        tasks.register_task('caption', CaptionTask)

        # turn on cuda if GPU is available
        self.use_cuda = False #torch.cuda.is_available()
        # use fp16 only when GPU is available
        self.use_fp16 = False

        # Load pretrained ckpt & config
        overrides = {"bpe_dir": "utils/BPE", "eval_cider": False, "beam": 5, "max_len_b": 16, "no_repeat_ngram_size": 3,
                     "seed": 7}
        self.models, self.cfg, self.task = checkpoint_utils.load_model_ensemble_and_task(
            utils.split_paths('checkpoints/caption.pt'),
            arg_overrides=overrides
        )

        # Move models to GPU
        for model in self.models:
            model.eval()
            if self.use_fp16:
                model.half()
            if self.use_cuda and not self.cfg.distributed_training.pipeline_model_parallel:
                model.cuda()
            model.prepare_for_inference_(self.cfg)

        # Initialize generator
        self.generator = self.task.build_generator(self.models, self.cfg.generation)

        # Image transform
        from torchvision import transforms
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]

        self.patch_resize_transform = transforms.Compose([
            lambda image: image.convert("RGB"),
            transforms.Resize((self.cfg.task.patch_image_size, self.cfg.task.patch_image_size), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

        # Text preprocess
        self.bos_item = torch.LongTensor([self.task.src_dict.bos()])
        self.eos_item = torch.LongTensor([self.task.src_dict.eos()])
        self.pad_idx = self.task.src_dict.pad()


    def encode_text(self, text, length=None, append_bos=False, append_eos=False):
        s = self.task.tgt_dict.encode_line(
            line=self.task.bpe.encode(text),
            add_if_not_exist=False,
            append_eos=False
        ).long()
        if length is not None:
            s = s[:length]
        if append_bos:
            s = torch.cat([self.bos_item, s])
        if append_eos:
            s = torch.cat([s, self.eos_item])
        return s

    # Construct input for caption task
    def construct_sample(self, image: Image, question=None):
        patch_image = self.patch_resize_transform(image).unsqueeze(0)
        patch_mask = torch.tensor([True])
        src_text = self.encode_text(question, append_bos=True, append_eos=True).unsqueeze(0)
        src_length = torch.LongTensor([s.ne(self.pad_idx).long().sum() for s in src_text])
        sample = {
            "id": np.array(['42']),
            "net_input": {
                "src_tokens": src_text,
                "src_lengths": src_length,
                "patch_images": patch_image,
                "patch_masks": patch_mask
            }
        }
        return sample

    # Function to turn FP32 to FP16
    def apply_half(t):
        if t.dtype is torch.float32:
            return t.to(dtype=torch.half)
        return t

    def explain(self, image, question, session=None):
        sample = self.construct_sample(image, question)
        print(sample)
        sample = utils.move_to_cuda(sample) if self.use_cuda else sample
        sample = utils.apply_to_sample(self.apply_half, sample) if self.use_fp16 else sample

        result, scores = eval_step(self.task, self.generator, self.models, sample)
        result_attn = result[0]["attention"]
        unmodified_caption = result[0]["unmodified_caption"]

        output_tokens = result_attn.shape[1]
        fig, axs = plt.subplots(1, output_tokens, figsize=(5 * output_tokens, 5))
        full_caption = ("<cls> " + unmodified_caption + " <eos>").split()
        for i, attn in enumerate(result_attn.T.cpu()):
            num_input_tokens = len(attn)-900
            attn_map = attn
            image_attn = attn_map[:-num_input_tokens]
            image_attn = F.softmax(image_attn)
            heatmap = torch.reshape(image_attn, (1, 1, 30, 30))
            img_sized_heatmap = F.interpolate(heatmap, image.size[::-1], mode='bicubic')
            axs[i].imshow(image)
            axs[i].imshow(img_sized_heatmap.squeeze(0).squeeze(0), zorder=1, alpha=0.8)
            axs[i].axis('off')
            # axs[i].set_title(full_caption[i])

        plt.tight_layout()
        path = os.path.join(os.getcwd(), f"results/{session}/decoder/")
        os.makedirs(path, exist_ok=True)
        plt.savefig(os.path.join(path, "0.png"))
        plt.close()
        print(" ".join(full_caption))
        return result[0]["caption"], [image]*4, [image]