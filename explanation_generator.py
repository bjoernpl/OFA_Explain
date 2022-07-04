import os
import re

import cv2
import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F
from torchvision import transforms

from fairseq import checkpoint_utils
from fairseq import options
from fairseq import utils, tasks
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from tasks.mm_tasks.vqa_gen import VqaGenTask
from utils.zero_shot_utils import zero_shot_step


def pre_question(question, max_ques_words):
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


class ExplanationGenerator:
    def __init__(self):

        tasks.register_task('vqa_gen', VqaGenTask)
        self.use_cuda = torch.cuda.is_available()

        parser = options.get_generation_parser()
        input_args = ["", "--task=vqa_gen", "--beam=100", "--unnormalized", "--path=checkpoints/ofa_large_384.pt",
                      "--bpe-dir=utils/BPE"]
        args = options.parse_args_and_arch(parser, input_args)
        self.cfg = convert_namespace_to_omegaconf(args)

        self.task = tasks.setup_task(self.cfg.task)
        self.models, self.cfg = checkpoint_utils.load_model_ensemble(
            utils.split_paths(self.cfg.common_eval.path),
            task=self.task
        )

        # Move models to GPU
        for model in self.models:
            model.eval()
            if self.use_cuda and not self.cfg.distributed_training.pipeline_model_parallel:
                model.cuda()
            model.prepare_for_inference_(self.cfg)

        # Initialize generator
        self.generator = self.task.build_generator(self.models, self.cfg.generation)

        # Image transform
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]

        self.patch_resize_transform = transforms.Compose([
            lambda image: image.convert("RGB"),
            transforms.Resize((self.cfg.task.patch_image_size, self.cfg.task.patch_image_size),
                              interpolation=transforms.InterpolationMode.BICUBIC),
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
    def construct_sample(self, image: Image, question: str):
        patch_image = self.patch_resize_transform(image).unsqueeze(0)
        patch_mask = torch.tensor([True])

        question = pre_question(question, self.task.cfg.max_src_length)
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

    def explain(self, image: Image, question, encoder_path, decoder_path):
        sample = self.construct_sample(image, question)
        sample = utils.move_to_cuda(sample) if self.use_cuda else sample

        result, scores = zero_shot_step(self.task, self.generator, self.models, sample)
        answer = result[0]["answer"]

        encoder_idx = self.encoder_explanation(image, encoder_path)
        decoder_idx = self.decoder_explanation(image, result, decoder_path)

        return answer, encoder_idx, decoder_idx

    def encoder_explanation(self, image, encoder_path):
        model = self.models[0]
        layer = model.encoder.layers[-1]
        head_attn = layer.attention_map
        attn_map = head_attn.mean(axis=0, keepdim=True).squeeze(0).cpu()
        num_tokens = len(attn_map) - 576
        encoder_idx = list(range(num_tokens))

        for i in range(num_tokens):
            image_attn = attn_map[576 + i, :-num_tokens]
            # self_attn = attn_map[576 + i, -num_tokens:]
            # self_attn = F.softmax(self_attn)
            # self_attn -= self_attn.min()
            # self_attn /= self_attn.max()
            image_attn = F.softmax(image_attn)
            heatmap = torch.reshape(image_attn, (1, 1, 24, 24))
            img_sized_heatmap = F.interpolate(heatmap, image.size[::-1], mode='bicubic').squeeze(0).squeeze(0)
            path = os.path.join(encoder_path, str(i) + ".jpg")
            self.generate_heatmap(np.array(image), img_sized_heatmap, opacity=0.7, save_path=path)
        return encoder_idx

    def decoder_explanation(self, image, result, decoder_path):
        result_attn = result[0]["attention"]
        decoder_idx = list(range(len(result_attn.T.cpu()[:-1])))
        for i, attn_map in enumerate(result_attn.T.cpu()[:-1]):
            num_input_tokens = len(attn_map) - 576
            image_attn = attn_map[:-num_input_tokens]
            image_attn = F.softmax(image_attn)
            heatmap = torch.reshape(image_attn, (1, 1, 24, 24))
            heatmap_img = F.interpolate(heatmap, image.size[::-1], mode='bicubic')
            heatmap_img = heatmap_img.squeeze(0).squeeze(0).numpy()
            path = os.path.join(decoder_path, str(i) + ".jpg")
            self.generate_heatmap(np.array(image), heatmap_img, opacity=0.7, save_path=path)
        return decoder_idx


    @staticmethod
    def generate_heatmap(original_image, heat_map, opacity=0.3, cmap=cv2.COLORMAP_VIRIDIS,
                         save_path=None):
        max_value = heat_map.max()
        min_value = heat_map.min()
        heat_map = (heat_map - min_value) / (max_value - min_value)
        heat_map = np.array(heat_map * 255, dtype=np.uint8)
        heat_map = cv2.applyColorMap(heat_map, cmap)
        if original_image.shape[-1] == 4:  # RGBA
            adim = np.ones((heat_map.shape[0], heat_map.shape[1], 1), dtype=np.uint8) * 255
            heat_map = np.concatenate([heat_map, adim], axis=2)

        outImage = cv2.addWeighted(heat_map, opacity, original_image, 1 - opacity, 0)

        if save_path is not None:
            cv2.imwrite(save_path, outImage)
