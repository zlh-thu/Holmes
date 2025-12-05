import torch
from typing import Dict, List, Optional, Set, Tuple, Union

import torch.nn as nn
# from transformers.models.clip.modeling_clip import CLIPVisionTransformer

from transformers.modeling_outputs import ImageClassifierOutput
from transformers import (
    ResNetForImageClassification
)


class ResNet(nn.Module):
    def __init__(self, config, model):
        super().__init__()
        self.classifier = model.classifier
        self.resnet = model.resnet
        self.config = config
        self.num_labels = config.num_labels


    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
        id=None,
    ) -> Union[tuple, ImageClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """

        outputs = self.resnet(
            pixel_values
        )
        # pooler_output = outputs[1]
        # print(outputs.pooler_output)
        pooler_output = outputs.pooler_output
        logits = self.classifier(pooler_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = nn.MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(
                    logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = nn.BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        '''if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output'''

        return ImageClassifierOutput(
            loss=loss,
            logits=logits,
        )
