import torch.nn as nn
from Model.EncoderBlock.ViT import ViT
from Model.EncoderBlock.ResNet import ResNet, BasicBlock

class Encoder(nn.Module):
    def __init__(self, resnet_out_dim=512, vit_patch_dim=128):
        super().__init__()
        self.resnet = ResNet(BasicBlock,[2,2,2,2])  # Sử dụng mô hình ResNet bạn đã viết trước đó
        self.vit = ViT(patch_dim=vit_patch_dim, num_patches=resnet_out_dim)

    def forward(self, x):
        feat_map = self.resnet(x)  # [B, 512, 7, 7]
        features = self.vit.forward_features(feat_map)  # [B, vit_patch_dim]
        return features

    def forward_logits(self, x):
        feat_map = self.resnet.forward_features(x)
        logits = self.vit(feat_map)     # [B, num_classes]
        return logits
