import os
import re

import torch
from matplotlib import pyplot as plt
from torch.nn import functional as F
import numpy as np
from fairseq import utils,tasks
from fairseq import checkpoint_utils
from fairseq import options
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from utils.zero_shot_utils import zero_shot_step
from tasks.mm_tasks.caption import CaptionTask
from PIL import Image




class ExplanationGenerator:
    def __init__(self):

        # Register caption task
        tasks.register_task('caption', CaptionTask)

        # turn on cuda if GPU is available
        self.use_cuda = torch.cuda.is_available()
        # use fp16 only when GPU is available
        self.use_fp16 = False

        parser = options.get_generation_parser()
        input_args = ["", "--task=vqa_gen", "--beam=100", "--unnormalized", "--path=checkpoints/ofa_large_384.pt",
                      "--bpe-dir=utils/BPE"]
        args = options.parse_args_and_arch(parser, input_args)
        self.cfg = convert_namespace_to_omegaconf(args)

        # @TODO lukas: wget https://ofa-silicon.oss-us-west-1.aliyuncs.com/checkpoints/ofa_large_384.pt
        # -> checkpoints/ofa_large_384.pt
        self.task = tasks.setup_task(self.cfg.task)
        self.models, self.cfg = checkpoint_utils.load_model_ensemble(
            utils.split_paths(self.cfg.common_eval.path),
            task=self.task
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

    # Normalize the question
    def pre_question(self, question, max_ques_words):
        question = question.lower().lstrip(",.!?*#:;~").replace('-', ' ').replace('/', ' ')
        question = re.sub(
            r"\s{2,}",
            ' ',
            question,
        )
        question = question.rstrip('\n')
        question = question.strip(' ')
        # truncate question
        question_words = question.split(' ')
        if len(question_words) > max_ques_words:
            question = ' '.join(question_words[:max_ques_words])
        return question

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
    def construct_sample(self, image: Image, question: str):
        patch_image = self.patch_resize_transform(image).unsqueeze(0)
        patch_mask = torch.tensor([True])

        question = self.pre_question(question, self.task.cfg.max_src_length)
        question = question + '?' if not question.endswith('?') else question
        src_text = self.encode_text(' {}'.format(question), append_bos=True, append_eos=True).unsqueeze(0)

        src_length = torch.LongTensor([s.ne(self.pad_idx).long().sum() for s in src_text])
        ref_dict = np.array([{'yes': 1.0}])  # just placeholder
        sample = {
            "id": np.array(['42']),
            "net_input": {
                "src_tokens": src_text,
                "src_lengths": src_length,
                "patch_images": patch_image,
                "patch_masks": patch_mask,
            },
            "ref_dict": ref_dict,
        }
        return sample

    # Function to turn FP32 to FP16
    def apply_half(t):
        if t.dtype is torch.float32:
            return t.to(dtype=torch.half)
        return t

    def explain(self, image, question, encoder_path, decoder_path):
        new_shape = 500 * np.array(image.size()) / np.max(image.size())
        image = image.resize(new_shape, Image.ANTIALIAS)

        sample = self.construct_sample(image, question)
        sample = utils.move_to_cuda(sample) if self.use_cuda else sample
        sample = utils.apply_to_sample(self.apply_half, sample) if self.use_fp16 else sample

        result, scores = zero_shot_step(self.task, self.generator, self.models, sample)
        result_attn = result[0]["attention"]
        answer = result[0]["answer"]

        decoder_idx = list(range(len(result_attn.T.cpu()[:-1])))
        for i, attn_map in enumerate(result_attn.T.cpu()[:-1]):
            num_input_tokens = len(attn_map)-576
            image_attn = attn_map[:-num_input_tokens]
            image_attn = F.softmax(image_attn)
            heatmap = torch.reshape(image_attn, (1, 1, 24, 24))
            heatmap_img = F.interpolate(heatmap, image.size[::-1], mode='bicubic')
            heatmap_img = heatmap_img.squeeze(0).squeeze(0).numpy()
            # heatmap_img -= heatmap_img.min()
            # heatmap_img /= heatmap_img.max()

            plt.imshow(image)
            plt.imshow(heatmap_img, zorder=1, alpha=0.7)
            plt.axis("off")
            path = os.path.join(decoder_path, str(i) + ".png")
            plt.savefig(path, bbox_inches='tight',  pad_inches=0)
            plt.clf()

        model = self.models[0]

        layer = model.encoder.layers[-1]
        head_attn = layer.attention_map
        attn_map = head_attn.mean(axis=0, keepdim=True).squeeze(0).cpu()
        num_tokens = len(attn_map)-576
        encoder_idx = list(range(num_tokens))

        for i in range(num_tokens):
            image_attn = attn_map[576 + i, :-num_tokens]
            # self_attn = attn_map[576 + i, -num_tokens:]
            # self_attn = F.softmax(self_attn)
            # self_attn -= self_attn.min()
            # self_attn /= self_attn.max()

            image_attn = F.softmax(image_attn)
            heatmap = torch.reshape(image_attn, (1, 1, 24, 24))
            img_sized_heatmap = F.interpolate(heatmap, image.size[::-1], mode='bicubic')
            plt.imshow(image)
            plt.imshow(img_sized_heatmap.squeeze(0).squeeze(0), zorder=1, alpha=0.7)
            plt.axis('off')
            path = os.path.join(encoder_path, str(i) + ".png")
            plt.savefig(path, bbox_inches='tight', pad_inches=0)
            plt.clf()

        return answer, encoder_idx, decoder_idx
