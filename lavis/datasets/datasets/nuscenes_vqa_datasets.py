"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
import json
import random
import torch
import pandas as pd
from PIL import Image

from lavis.datasets.datasets.vqa_datasets import VQADataset, VQAEvalDataset

from collections import OrderedDict


class __DisplMixin:
    def displ_item(self, index):
        sample, ann = self.__getitem__(index), self.annotation[index]

        return OrderedDict(
            {
                "file": ann["image"],
                "question": ann["question"],
                "question_id": ann["question_id"],
                "answers": "; ".join(ann["answer"]),
                "image": sample["image"],
            }
        )


class NuScenesVQADataset(VQADataset, __DisplMixin):
    def __init__(
            self, vis_processor=None, text_processor=None, vis_root=None, ann_paths=[]
    ):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        self.vis_root = vis_root
        self.annotation = []
        for ann_path in ann_paths:
            with open(ann_path, "r") as f:
                data = json.load(f)
                for scene_token, scene_data in data.items():
                    for frame_token, frame_data in scene_data["key_frames"].items():
                        image_path = frame_data["image_paths"]["CAM_FRONT"]
                        for qa_type, qas in frame_data["QA"].items():
                            for qa in qas:
                                question = qa["Q"]
                                answer = qa["A"]
                                self.annotations.append({
                                    "image_path": image_path,
                                    "question": question,
                                    "answer": answer
                                })

        self.vis_processor = vis_processor
        self.text_processor = text_processor

        self._add_instance_ids()

    def __len__(self):
        return len(self.annotations)
    def _add_instance_ids(self, key="instance_id"):
        for idx, ann in enumerate(self.annotation):
            ann[key] = str(idx)
    def collater(self, samples):
        # Filter out None samples
        samples = [s for s in samples if s is not None]
        # Check if samples is empty after filtering
        if not samples:
            return None
        image_list, question_list, answer_list, weight_list = [], [], [], []

        num_answers = []

        for sample in samples:
            image_list.append(sample["image"])
            question_list.append(sample["text_input"])

            weight_list.extend(sample["weights"])

            answers = sample["answers"]

            answer_list.extend(answers)
            num_answers.append(len(answers))

        return {
            "image": torch.stack(image_list, dim=0),
            "text_input": question_list,
            "answer": answer_list,
            "weight": weight_list,
            "n_answers": torch.LongTensor(num_answers),
        }
    def __getitem__(self, index):
        ann = self.annotation[index]

        image_path = os.path.join(self.vis_root, ann["image_path"])
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)
        question = self.text_processor(ann["question"])

        answers = [ann["answer"]]
        weights = [1.0]  # Equal weight for single answer

        return {
            "image": image,
            "text_input": question,
            "answers": answers,
            "weights": weights,
        }


    

class NuScenesVQAEvalDataset(VQAEvalDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """

        self.vis_root = vis_root

        self.annotation = json.load(open(ann_paths[0]))
        with open(ann_paths[0], "r") as f:
            data = json.load(f)
            for scene_token, scene_data in data.items():
                for frame_token, frame_data in scene_data["key_frames"].items():
                    image_path = frame_data["image_paths"]["CAM_FRONT"]
                    for qa_type, qas in frame_data["QA"].items():
                        for qa in qas:
                            question = qa["Q"]
                            answer = qa["A"]
                            self.annotations.append({
                                "image_path": image_path,
                                "question": question,
                                "answer": answer
                            })
        # Load answer list if provided
        if len(ann_paths) > 1 and os.path.exists(ann_paths[1]):
            self.answer_list = json.load(open(ann_paths[1]))
        else:
            self.answer_list = None

        self.vis_processor = vis_processor
        self.text_processor = text_processor

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        ann = self.annotations[index]

        image_path = os.path.join(self.vis_root, ann["image_path"])
        image = Image.open(image_path).convert("RGB")
        image = self.vis_processor(image)

        question = self.text_processor(ann["question"])

        return {
            "image": image,
            "text_input": question,
        }