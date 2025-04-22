from functools import partial
from typing import Any, Optional
import torch
from torch import nn

from torchvision.transforms._presets import SemanticSegmentation
from torchvision.models._api import register_model, Weights, WeightsEnum
from torchvision.models._meta import _VOC_CATEGORIES
from torchvision.models._utils import _ovewrite_value_param, handle_legacy_interface, IntermediateLayerGetter
from torchvision.models.resnet import ResNet, resnet101, ResNet101_Weights, resnet50, ResNet50_Weights
# from torchvision.models._utils import _SimpleSegmentationModel

# __all__ = ["FCN", "FCN_ResNet50_Weights", "FCN_ResNet101_Weights", "fcn_resnet50", "fcn_resnet101"]


from collections import OrderedDict
from typing import Dict, Optional

from torch import nn, Tensor
from torch.nn import functional as F

# from ...utils import _log_api_usage_once
from torchvision.utils import _log_api_usage_once
from model_s import aug_conv

import random
import numpy as np
seed = 2023
random.seed(seed)
# import numpy as np
np.random.seed(seed)
# import torch
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

class _SimpleSegmentationModel(nn.Module):
    __constants__ = ["aux_classifier"]

    def __init__(self, backbone: nn.Module, classifier: nn.Module, aux_classifier: Optional[nn.Module] = None, model_type='m0') -> None:
        super().__init__()
        _log_api_usage_once(self)
        self.backbone = backbone
        self.classifier = classifier
        self.aux_classifier = aux_classifier
        self.model_type = model_type

        if self.model_type!='m0':
            self.sam_feat = nn.Conv2d(256, 2048, 3, 1, 1)
            self.feat_fusion = nn.Sequential(
                
                # nn.Conv2d(2048*2, 2048*2, 3, 1, 1),
                # nn.BatchNorm2d(2048*2),
                nn.Dropout(),
                nn.Conv2d(2048*2, 2048, 1, 1, 0),
                # aug_conv(2048*2, 2048, 1, False),
                # aug_conv(2048, 2048, 1, False),
            )

            if self.model_type == 'm3':
                base = 64
                self.mask_prompt = nn.Sequential(
                    nn.AvgPool2d(3,2,1),
                    aug_conv(512, base, 2, True),
                    aug_conv(base, base*2, 2, False),
                    aug_conv(base*2, base*4, 2, False),
                    aug_conv(base*4, base*16, 1, False),
                    
                    
                    aug_conv(base*16, base*16, 1, False),
                    aug_conv(base*16, base*16, 1, False),



                    aug_conv(base*16, 512, 1, False),
                    nn.Conv2d(512, 512, 1, 1, 0),
                    




                )
                self.point_prompt = nn.Sequential(
                    nn.Sigmoid(),
                    nn.AvgPool2d(3,2,1),
                    aug_conv(2,     base, 2, True),
                    aug_conv(base, base*2, 2, False),
                    aug_conv(base*2, base*4, 2, False),
                    aug_conv(base*4, base*16, 1, False),
                    
                    
                    aug_conv(base*16, base*16, 1, False),
                    aug_conv(base*16, base*16, 1, False),



                    aug_conv(base*16, 512, 1, False),
                    nn.Conv2d(512, 512, 1, 1, 0),
                )

                


    def forward(self, x: Tensor, input_sam=None) -> Dict[str, Tensor]:
        input_shape = x.shape[-2:]
        # contract: features is a dict of tensors
        features = self.backbone(x)

        result = OrderedDict()
        feat = features["out"]
        if self.model_type != 'm0':
            feat_sam = self.sam_feat(input_sam)
            feat_shape = feat.shape[-2:]
            feat_sam = F.interpolate(feat_sam, size=feat_shape, mode='bilinear', align_corners=False)

            # if self.training:
            #     list_temp_feats = []
            #     list_temp_feats.append(random.choice([feat, feat_sam]))
            #     list_temp_feats.append(random.choice([feat, feat_sam]))
            #     feat_com = torch.cat(list_temp_feats, dim=1)
            # else:
                # list_temp_feats = [feat, feat_sam]
                # feat_com = torch.cat(list_temp_feats, dim=1)
            
            list_temp_feats = [feat, feat_sam]
            feat_com = torch.cat(list_temp_feats, dim=1)
            
            feat_fusion = self.feat_fusion(feat_com)
            feat, out = self.classifier(feat_fusion)
            # out = self.mask_up(out)
            if self.model_type == 'm3':
                temp_feat = F.interpolate(feat, size=input_shape, mode="bilinear", align_corners=False)
                temp_out = F.interpolate(out, size=input_shape, mode="bilinear", align_corners=False)
                result['masks_prompt'] = self.mask_prompt(temp_feat)
                result['points_prompt'] = self.point_prompt(temp_out)
        else:
            out = self.classifier(feat)[-1]
            # out = self.mask_up(out)

        
        out = F.interpolate(out, size=input_shape, mode="bilinear", align_corners=False)
        result["out"] = out
        result['masks'] = out



        return result

class FCN(_SimpleSegmentationModel):
    """
    Implements FCN model from
    `"Fully Convolutional Networks for Semantic Segmentation"
    <https://arxiv.org/abs/1411.4038>`_.

    Args:
        backbone (nn.Module): the network used to compute the features for the model.
            The backbone should return an OrderedDict[Tensor], with the key being
            "out" for the last feature map used, and "aux" if an auxiliary classifier
            is used.
        classifier (nn.Module): module that takes the "out" element returned from
            the backbone and returns a dense prediction.
        aux_classifier (nn.Module, optional): auxiliary classifier used during training
    """

    pass


# class FCNHead(nn.Sequential):
#     def __init__(self, in_channels: int, channels: int) -> None:
#         inter_channels = in_channels // 4
#         layers = [
#             nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
#             nn.BatchNorm2d(inter_channels),
#             nn.ReLU(),
#             nn.Dropout(0.1),
#             nn.Conv2d(inter_channels, channels, 1),
#         ]

#         super().__init__(*layers)

class FCNHead(nn.Module):
    def __init__(self, in_channels: int, channels: int) -> None:
        super().__init__()
        inter_channels = in_channels // 4
        self.feat = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        self.out = nn.Sequential(
            nn.Conv2d(inter_channels, channels, 1),
        )
        
    def forward(self, x):
        feat = self.feat(x)
        out = self.out(feat)
        return feat, out



_COMMON_META = {
    "categories": _VOC_CATEGORIES,
    "min_size": (1, 1),
    "_docs": """
        These weights were trained on a subset of COCO, using only the 20 categories that are present in the Pascal VOC
        dataset.
    """,
}


class FCN_ResNet50_Weights(WeightsEnum):
    COCO_WITH_VOC_LABELS_V1 = Weights(
        url="https://download.pytorch.org/models/fcn_resnet50_coco-1167a1af.pth",
        transforms=partial(SemanticSegmentation, resize_size=520),
        meta={
            **_COMMON_META,
            "num_params": 35322218,
            "recipe": "https://github.com/pytorch/vision/tree/main/references/segmentation#fcn_resnet50",
            "_metrics": {
                "COCO-val2017-VOC-labels": {
                    "miou": 60.5,
                    "pixel_acc": 91.4,
                }
            },
            "_ops": 152.717,
            "_file_size": 135.009,
        },
    )
    DEFAULT = COCO_WITH_VOC_LABELS_V1


class FCN_ResNet101_Weights(WeightsEnum):
    COCO_WITH_VOC_LABELS_V1 = Weights(
        url="https://download.pytorch.org/models/fcn_resnet101_coco-7ecb50ca.pth",
        transforms=partial(SemanticSegmentation, resize_size=520),
        meta={
            **_COMMON_META,
            "num_params": 54314346,
            "recipe": "https://github.com/pytorch/vision/tree/main/references/segmentation#deeplabv3_resnet101",
            "_metrics": {
                "COCO-val2017-VOC-labels": {
                    "miou": 63.7,
                    "pixel_acc": 91.9,
                }
            },
            "_ops": 232.738,
            "_file_size": 207.711,
        },
    )
    DEFAULT = COCO_WITH_VOC_LABELS_V1


def _fcn_resnet(
    backbone: ResNet,
    num_classes: int,
    aux: Optional[bool],
    model_type='m0',
) -> FCN:
    return_layers = {"layer4": "out"}
    if aux:
        return_layers["layer3"] = "aux"
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    aux_classifier = FCNHead(1024, num_classes) if aux else None
    classifier = FCNHead(2048, num_classes)
    return FCN(backbone, classifier, aux_classifier, model_type=model_type)


@register_model()
@handle_legacy_interface(
    weights=("pretrained", FCN_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1),
    weights_backbone=("pretrained_backbone", ResNet50_Weights.IMAGENET1K_V1),
)
def fcn_resnet50_my(
    *,
    weights: Optional[FCN_ResNet50_Weights] = None,
    progress: bool = True,
    num_classes: Optional[int] = None,
    aux_loss: Optional[bool] = None,
    weights_backbone: Optional[ResNet50_Weights] = ResNet50_Weights.IMAGENET1K_V1,
    model_type='m0',
    **kwargs: Any,
) -> FCN:
    """Fully-Convolutional Network model with a ResNet-50 backbone from the `Fully Convolutional
    Networks for Semantic Segmentation <https://arxiv.org/abs/1411.4038>`_ paper.

    .. betastatus:: segmentation module

    Args:
        weights (:class:`~torchvision.models.segmentation.FCN_ResNet50_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.segmentation.FCN_ResNet50_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        num_classes (int, optional): number of output classes of the model (including the background).
        aux_loss (bool, optional): If True, it uses an auxiliary loss.
        weights_backbone (:class:`~torchvision.models.ResNet50_Weights`, optional): The pretrained
            weights for the backbone.
        **kwargs: parameters passed to the ``torchvision.models.segmentation.fcn.FCN``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/segmentation/fcn.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.segmentation.FCN_ResNet50_Weights
        :members:
    """

    weights = FCN_ResNet50_Weights.verify(weights)
    weights_backbone = ResNet50_Weights.verify(weights_backbone)

    if weights is not None:
        weights_backbone = None
        num_classes = _ovewrite_value_param("num_classes", num_classes, len(weights.meta["categories"]))
        aux_loss = _ovewrite_value_param("aux_loss", aux_loss, True)
    elif num_classes is None:
        num_classes = 21

    backbone = resnet50(weights=weights_backbone, replace_stride_with_dilation=[False, True, True])
    model = _fcn_resnet(backbone, num_classes, aux_loss, model_type=model_type)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress, check_hash=True))

    return model
