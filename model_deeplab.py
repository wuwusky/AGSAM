from functools import partial
from typing import Any, List, Optional

import torch
from torch import nn
from torch.nn import functional as F

from torchvision.transforms._presets import SemanticSegmentation
from torchvision.models._api import register_model, Weights, WeightsEnum
from torchvision.models._meta import _VOC_CATEGORIES
from torchvision.models._utils import _ovewrite_value_param, handle_legacy_interface, IntermediateLayerGetter
from torchvision.models.mobilenetv3 import mobilenet_v3_large, MobileNet_V3_Large_Weights, MobileNetV3
from torchvision.models.resnet import ResNet, resnet101, ResNet101_Weights, resnet50, ResNet50_Weights
# from ._utils import _SimpleSegmentationModel
from torchvision.models.segmentation.fcn import FCNHead

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
                    aug_conv(256, base, 2, True),
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

                self.mask_up = nn.Sequential(
                    nn.Conv2d(2, 32, 3, 1, 1),
                    # nn.BatchNorm2d(32),
                    nn.LeakyReLU(0.1),
                    nn.ConvTranspose2d(32, 32, 2, 2),
                    nn.LeakyReLU(0.1),
                    
                    nn.Conv2d(32, 32, 1, 1, 0),
                    # nn.BatchNorm2d(32),
                    nn.LeakyReLU(0.1),
                    nn.ConvTranspose2d(32, 2, 2, 2),
                    nn.LeakyReLU(0.1),

                    # nn.LeakyReLU(0.1),
                )
                self.mask_up_sim = nn.Sequential(
                    nn.Conv2d(2, 32, 3, 1, 1),
                    nn.UpsamplingBilinear2d(scale_factor=2.0),
                    nn.Conv2d(32, 32, 3, 1, 1),
                    nn.UpsamplingBilinear2d(scale_factor=2.0),
                    nn.Conv2d(32, 2, 3, 1, 1),
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



__all__ = [
    "DeepLabV3",
    "DeepLabV3_ResNet50_Weights",
    "DeepLabV3_ResNet101_Weights",
    "DeepLabV3_MobileNet_V3_Large_Weights",
    "deeplabv3_mobilenet_v3_large",
    "deeplabv3_resnet50",
    "deeplabv3_resnet101",
]


class DeepLabV3(_SimpleSegmentationModel):
    """
    Implements DeepLabV3 model from
    `"Rethinking Atrous Convolution for Semantic Image Segmentation"
    <https://arxiv.org/abs/1706.05587>`_.

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


# class DeepLabHead(nn.Sequential):
#     def __init__(self, in_channels: int, num_classes: int) -> None:
#         super().__init__(
#             ASPP(in_channels, [12, 24, 36]),
#             nn.Conv2d(256, 256, 3, padding=1, bias=False),
#             nn.BatchNorm2d(256),
#             nn.ReLU(),
#             nn.Conv2d(256, num_classes, 1),
#         )


class DeepLabHead(nn.Module):
    def __init__(self, in_channels: int, num_classes: int) -> None:
        super().__init__()
        self.feat = nn.Sequential(
            ASPP(in_channels, [12, 24, 36]),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
            
        self.out = nn.Conv2d(256, num_classes, 1)

    def forward(self, x):
        feat = self.feat(x)
        out = self.out(feat)
        return feat, out




class ASPPConv(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, dilation: int) -> None:
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        ]
        super().__init__(*modules)


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


class ASPP(nn.Module):
    def __init__(self, in_channels: int, atrous_rates: List[int], out_channels: int = 256) -> None:
        super().__init__()
        modules = []
        modules.append(
            nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU())
        )

        rates = tuple(atrous_rates)
        for rate in rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))

        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _res = []
        for conv in self.convs:
            _res.append(conv(x))
        res = torch.cat(_res, dim=1)
        return self.project(res)


def _deeplabv3_resnet(
    backbone: ResNet,
    num_classes: int,
    aux: Optional[bool],
    model_type='m0',
) -> DeepLabV3:
    return_layers = {"layer4": "out"}
    if aux:
        return_layers["layer3"] = "aux"
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    aux_classifier = FCNHead(1024, num_classes) if aux else None
    classifier = DeepLabHead(2048, num_classes)
    return DeepLabV3(backbone, classifier, aux_classifier, model_type=model_type)


_COMMON_META = {
    "categories": _VOC_CATEGORIES,
    "min_size": (1, 1),
    "_docs": """
        These weights were trained on a subset of COCO, using only the 20 categories that are present in the Pascal VOC
        dataset.
    """,
}


class DeepLabV3_ResNet50_Weights(WeightsEnum):
    COCO_WITH_VOC_LABELS_V1 = Weights(
        url="https://download.pytorch.org/models/deeplabv3_resnet50_coco-cd0a2569.pth",
        transforms=partial(SemanticSegmentation, resize_size=520),
        meta={
            **_COMMON_META,
            "num_params": 42004074,
            "recipe": "https://github.com/pytorch/vision/tree/main/references/segmentation#deeplabv3_resnet50",
            "_metrics": {
                "COCO-val2017-VOC-labels": {
                    "miou": 66.4,
                    "pixel_acc": 92.4,
                }
            },
            "_ops": 178.722,
            "_file_size": 160.515,
        },
    )
    DEFAULT = COCO_WITH_VOC_LABELS_V1


class DeepLabV3_ResNet101_Weights(WeightsEnum):
    COCO_WITH_VOC_LABELS_V1 = Weights(
        url="https://download.pytorch.org/models/deeplabv3_resnet101_coco-586e9e4e.pth",
        transforms=partial(SemanticSegmentation, resize_size=520),
        meta={
            **_COMMON_META,
            "num_params": 60996202,
            "recipe": "https://github.com/pytorch/vision/tree/main/references/segmentation#fcn_resnet101",
            "_metrics": {
                "COCO-val2017-VOC-labels": {
                    "miou": 67.4,
                    "pixel_acc": 92.4,
                }
            },
            "_ops": 258.743,
            "_file_size": 233.217,
        },
    )
    DEFAULT = COCO_WITH_VOC_LABELS_V1


class DeepLabV3_MobileNet_V3_Large_Weights(WeightsEnum):
    COCO_WITH_VOC_LABELS_V1 = Weights(
        url="https://download.pytorch.org/models/deeplabv3_mobilenet_v3_large-fc3c493d.pth",
        transforms=partial(SemanticSegmentation, resize_size=520),
        meta={
            **_COMMON_META,
            "num_params": 11029328,
            "recipe": "https://github.com/pytorch/vision/tree/main/references/segmentation#deeplabv3_mobilenet_v3_large",
            "_metrics": {
                "COCO-val2017-VOC-labels": {
                    "miou": 60.3,
                    "pixel_acc": 91.2,
                }
            },
            "_ops": 10.452,
            "_file_size": 42.301,
        },
    )
    DEFAULT = COCO_WITH_VOC_LABELS_V1


@register_model()
@handle_legacy_interface(
    weights=("pretrained", DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1),
    weights_backbone=("pretrained_backbone", ResNet50_Weights.IMAGENET1K_V1),
)
def deeplabv3_resnet50_my(
    *,
    weights: Optional[DeepLabV3_ResNet50_Weights] = None,
    progress: bool = True,
    num_classes: Optional[int] = None,
    aux_loss: Optional[bool] = None,
    weights_backbone: Optional[ResNet50_Weights] = ResNet50_Weights.IMAGENET1K_V1,
    model_type='m0',
    **kwargs: Any,
) -> DeepLabV3:
    """Constructs a DeepLabV3 model with a ResNet-50 backbone.

    .. betastatus:: segmentation module

    Reference: `Rethinking Atrous Convolution for Semantic Image Segmentation <https://arxiv.org/abs/1706.05587>`__.

    Args:
        weights (:class:`~torchvision.models.segmentation.DeepLabV3_ResNet50_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.segmentation.DeepLabV3_ResNet50_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        num_classes (int, optional): number of output classes of the model (including the background)
        aux_loss (bool, optional): If True, it uses an auxiliary loss
        weights_backbone (:class:`~torchvision.models.ResNet50_Weights`, optional): The pretrained weights for the
            backbone
        **kwargs: unused

    .. autoclass:: torchvision.models.segmentation.DeepLabV3_ResNet50_Weights
        :members:
    """
    weights = DeepLabV3_ResNet50_Weights.verify(weights)
    weights_backbone = ResNet50_Weights.verify(weights_backbone)

    if weights is not None:
        weights_backbone = None
        num_classes = _ovewrite_value_param("num_classes", num_classes, len(weights.meta["categories"]))
        aux_loss = _ovewrite_value_param("aux_loss", aux_loss, True)
    elif num_classes is None:
        num_classes = 21

    backbone = resnet50(weights=weights_backbone, replace_stride_with_dilation=[False, True, True])
    model = _deeplabv3_resnet(backbone, num_classes, aux_loss, model_type=model_type)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress, check_hash=True))

    return model

