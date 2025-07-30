import torch
import torch.nn as nn
import torch.nn.functional as f
import torchvision.transforms.functional as tf

class ContrastiveHead(nn.Module):
    def __init__(self, in_dim, out_dim=128, normalize=True):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, out_dim)
        )
        self.normalize = normalize

    def forward(self, x):
        x = self.projector(x)
        if self.normalize:
            x = f.normalize(x, dim=1)
        return x


def make_augment(image):
    augments = list([])
    for angle in [90, 180, 270]:
        rotated = tf.rotate(image, angle)
        augments.append(rotated)

    # Đảo RGB theo các kiểu
    rgb = [image[i] for i in range(3)]

    augments.append(torch.stack([rgb[1], rgb[0], rgb[2]]))  # GBR
    augments.append(torch.stack([rgb[2], rgb[1], rgb[0]]))  # BGR

    return augments


def make_prototypes(features, group_ids):

    unique_ids = group_ids.unique()

    prototypes = []

    for gid in unique_ids:
        mask = (group_ids == gid)
        proto = features[mask].mean(dim=0)
        prototypes.append(proto)

    return torch.stack(prototypes)


def contrastive_loss(anchor, prototypes, temperature=0.07):
    sim = torch.matmul(anchor, prototypes.T) / temperature  # [B, B]
    labels = torch.arange(sim.size(0)).to(sim.device)  # positive = chính prototype tương ứng
    return f.cross_entropy(sim, labels)
