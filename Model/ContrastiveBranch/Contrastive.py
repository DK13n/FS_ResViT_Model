import torch
import torch.nn as nn
from Model.EncoderBlock.Encoder import Encoder
from Model.ContrastiveBranch.ContrastiveHead import ContrastiveHead, make_augment, make_prototypes, contrastive_loss


class Contrastive(nn.Module):
    def __init__(self, encoder, head):
        super().__init__()
        self.encoder = encoder
        self.head = head

    def forward(self,x):
        device = next(self.parameters()).device

        augmented_list = []
        group_ids = []

        for i, img in enumerate(x):
            augments = make_augment(img)
            augmented_list.extend(augments)
            group_ids.extend([i]*len(augments))

        augmented_batch = torch.stack(augmented_list).to(device)  # [N_aug, C, H, W]
        group_ids = torch.tensor(group_ids).to(device)

        aug_features = self.head(self.encoder(augmented_batch))

        prototype = make_prototypes(aug_features, group_ids)

        x=x.to(device)
        x = self.head(self.encoder(x))

        loss = contrastive_loss(x, prototype)

        return loss

