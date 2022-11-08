# Copyright 2022 The OFA-Sys Team. 
# All rights reserved.
# This source code is licensed under the Apache 2.0 license 
# found in the LICENSE file in the root directory.

from io import BytesIO

import logging
import warnings

import numpy as np
import torch
import base64
from torchvision import transforms

from PIL import Image, ImageFile

from data import data_utils
from data.ofa_dataset import OFADataset

ImageFile.LOAD_TRUNCATED_IMAGES = True
ImageFile.MAX_IMAGE_PIXELS = None
Image.MAX_IMAGE_PIXELS = None

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


def collate(samples, pad_idx, eos_idx):
    if len(samples) == 0:
        return {}

    def merge(key):
        return data_utils.collate_tokens(
            [s[key] for s in samples],
            pad_idx,
            eos_idx=eos_idx,
        )

    id = np.array([s["id"] for s in samples])
    src_tokens = merge("source")
    src_lengths = torch.LongTensor([s["source"].ne(pad_idx).long().sum() for s in samples])

    patch_images = torch.stack([sample['patch_image'] for sample in samples], dim=0)
    patch_masks = torch.cat([sample['patch_mask'] for sample in samples])

    ref_dict = None
    if samples[0].get("ref_dict", None) is not None:
        ref_dict = np.array([s['ref_dict'] for s in samples])

    constraint_masks = None
    if samples[0].get("constraint_mask", None) is not None:
        constraint_masks = merge("constraint_mask")

    decoder_prompts = None
    if samples[0].get("decoder_prompt", None) is not None:
        decoder_prompts = merge("decoder_prompt")

    because_idxs = None
    if samples[0].get("because_idx", None) is not None:
        because_idxs = torch.cat([sample['because_idx'] for sample in samples])

    task_ids = None
    if samples[0].get("task_id", None) is not None:
        task_ids = torch.cat([sample['task_id'] for sample in samples])


    prev_output_tokens = None
    target = None
    if samples[0].get("target", None) is not None:
        target = merge("target")
        tgt_lengths = torch.LongTensor(
            [s["target"].ne(pad_idx).long().sum() for s in samples]
        )
        ntokens = tgt_lengths.sum().item()

        if samples[0].get("prev_output_tokens", None) is not None:
            prev_output_tokens = merge("prev_output_tokens")
    else:
        ntokens = src_lengths.sum().item()

    batch = {
        "id": id,
        "nsentences": len(samples),
        "ntokens": ntokens,
        "net_input": {
            "src_tokens": src_tokens,
            "src_lengths": src_lengths,
            "patch_images": patch_images,
            "patch_masks": patch_masks,
            "prev_output_tokens": prev_output_tokens
        },
        "ref_dict": ref_dict,
        "constraint_masks": constraint_masks,
        "decoder_prompts": decoder_prompts,
        "target": target,
        "because_idxs": because_idxs,
        "task_ids": task_ids
    }

    return batch


class UnifyExplanationDataset(OFADataset):
    def __init__(
        self,
        split,
        dataset,
        bpe,
        src_dict,
        tgt_dict=None,
        max_src_length=80,
        max_tgt_length=30,
        patch_image_size=224,
        add_caption=False,
        constraint_trie=None,
        imagenet_default_mean_and_std=False,
        prompt_type="none",
        **kwargs
    ):
        super().__init__(split, dataset, bpe, src_dict, tgt_dict)
        self.max_src_length = max_src_length
        self.max_tgt_length = max_tgt_length
        self.patch_image_size = patch_image_size

        self.add_caption = add_caption
        self.constraint_trie = constraint_trie
        self.prompt_type = prompt_type

        if imagenet_default_mean_and_std:
            mean = IMAGENET_DEFAULT_MEAN
            std = IMAGENET_DEFAULT_STD
        else:
            mean = [0.5, 0.5, 0.5]
            std = [0.5, 0.5, 0.5]

        self.patch_resize_transform = transforms.Compose([
            lambda image: image.convert("RGB"),
            transforms.Resize((patch_image_size, patch_image_size), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

    def __getitem__(self, index):
        uniq_id, image, hypothesis, explanation, label = self.dataset[index]
        task_ids = {
            "esnlive": torch.tensor([0]),
            "vcr": torch.tensor([1]),
            "vqax": torch.tensor([2]),
        }
        if "#" in uniq_id:
            task = "esnlive"
        elif "-" in uniq_id:
            task = "vcr"
        else:
            task = "vqax"

        image = Image.open(BytesIO(base64.urlsafe_b64decode(image)))
        patch_image = self.patch_resize_transform(image)
        patch_mask = torch.tensor([True])

        hypothesis = self.pre_caption(hypothesis, self.max_src_length)
        explanation = f" because {explanation}"
        expl_target_item = self.encode_text(explanation)
        # enforce max length
        expl_target_item = expl_target_item[:self.max_tgt_length]

        if task == "esnlive":
            if label == 'contradiction':
                label = 'no'
            elif label == 'entailment':
                label = 'yes'
            elif label == 'neutral':
                label = 'maybe'
            else:
                raise NotImplementedError
            src_item = self.encode_text(' does the image describe " {} "?'.format(hypothesis))
            tgt_item = self.encode_text(" {}".format(label))
            ref_dict = {label: 1.0}
        elif task == "vcr":
            src_item = self.encode_text(hypothesis)
            tgt_item = self.encode_text(" {}".format(label))
            ref_dict = {label: 1.0}
        else: # task == "vqax":
            # Process question
            question = self.pre_caption(hypothesis, self.max_src_length)
            question = question + '?' if not question.endswith('?') else question
            src_item = self.encode_text(' {}'.format(question))

            # Process answer
            ref_dict = {item.split('|!+')[1]: float(item.split('|!+')[0]) for item in label.split('&&')}
            answer = max(ref_dict, key=ref_dict.get)
            answer = f" {answer}"
            tgt_item = self.encode_text(answer)


        src_item = torch.cat([self.bos_item, src_item, self.eos_item])
        if self.prompt_type == 'none':
            prev_output_item = torch.cat([self.bos_item, tgt_item])
            target_item = torch.cat([prev_output_item[1:], self.eos_item])
            decoder_prompt = self.bos_item
        elif self.prompt_type == 'src':
            prev_output_item = torch.cat([src_item, tgt_item])
            target_item = torch.cat([prev_output_item[1:], self.eos_item])
            decoder_prompt = src_item
        elif self.prompt_type == 'prev_output':
            prev_output_item = torch.cat([src_item[:-1], tgt_item])
            target_item = torch.cat([prev_output_item[1:], self.eos_item])
            decoder_prompt = src_item[:-1]
        elif self.prompt_type == 'without_decoder_prompt':
            # Prev output item is for teacher forcing
            # includes < bos answer expl>
            prev_output_item = torch.cat([self.bos_item, tgt_item, expl_target_item])

            # Target item includes < question answer expl eos >
            target_item = torch.cat([tgt_item, expl_target_item, self.eos_item])
            decoder_prompt = torch.cat([self.bos_item])
        else:
            raise NotImplementedError

        example = {
            "id": uniq_id,
            "source": src_item,
            "patch_image": patch_image,
            "patch_mask": patch_mask,
            "target": target_item,
            "prev_output_tokens": prev_output_item,
            "decoder_prompt": decoder_prompt,
            "ref_dict": ref_dict,
            "because_idx": torch.tensor([tgt_item.size(0)]),
            "task_id": task_ids[task],
        }
        return example

    def collater(self, samples, pad_to_length=None):
        """Merge a list of samples to form a mini-batch.
        Args:
            samples (List[dict]): samples to collate
        Returns:
            dict: a mini-batch containing the data of the task
        """
        return collate(samples, pad_idx=self.pad, eos_idx=self.eos)
