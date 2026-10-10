"""The training protocols of the paper (Section 3.2, Appendix A.2).

Shared by all runs: random initialisation, 256 x 256 input, AdamW (lr 1e-3,
weight decay 0.03), batch size 8, cosine annealing to 1e-5, exponential moving
average of the weights (decay 0.999) evaluated after every epoch, and the
augmentation schedule below. The protocols differ in:

                         polypgen / polypgen_c4          siim (SIIM-ACR)
  epochs (cosine T_max)  300                             111
  training stream        all images, shuffled            all lesion-bearing images + an equal
                                                         draw of lesion-free images per epoch
  vertical flip          yes                             no
  rotation               one angle in [0, 360) per run   new angle in [-15, 15] per image
  mask resize            bilinear                        nearest neighbour
  Dice term              per image                       over the batch (main runs)
  checkpoint selection   max mIoU + Dice (per image,     max Dice on lesion-bearing images
                         eps 1e-5) on the val split      on the val split
"""
from __future__ import annotations

from dataclasses import dataclass

from . import transforms as T


@dataclass(frozen=True)
class Protocol:
    name: str
    normalisation: str
    epochs: int
    t_max: int
    balanced: bool
    vertical_flip: bool
    rotation: tuple
    rotation_per_image: bool
    nearest_masks: bool
    batch_dice: bool
    selection: str               # "miou_plus_dice" or "dice_pos"
    splits: tuple                # (train, selection, test)
    lr: float = 1e-3
    weight_decay: float = 0.03
    eta_min: float = 1e-5
    batch_size: int = 8
    ema_decay: float = 0.999
    threshold: float = 0.5

    def train_transform(self):
        steps = [T.Normalize(self.normalisation), T.ToTensor(),
                 T.ColorJitter(p=0.5, brightness=0.2, contrast=0.2, saturation=0.2),
                 T.GaussianNoise(p=0.3, std=8.0), T.HorizontalFlip(p=0.5)]
        if self.vertical_flip:
            steps.append(T.VerticalFlip(p=0.5))
        steps += [T.Rotation(p=0.5, degree=self.rotation, per_sample=self.rotation_per_image),
                  T.ElasticDeform(p=0.5, alpha=6.0, control_grid=8),
                  T.ScaleCrop(p=0.3, scale_range=(0.85, 1.15)),
                  T.Resize(256, nearest_masks=self.nearest_masks)]
        return T.Compose(steps)

    def test_transform(self):
        return T.Compose([T.Normalize(self.normalisation), T.ToTensor(),
                          T.Resize(256, nearest_masks=self.nearest_masks)])


_POLYPGEN = dict(epochs=300, t_max=300, balanced=False, vertical_flip=True, rotation=(0, 360),
                 rotation_per_image=False, nearest_masks=False, batch_dice=False,
                 selection="miou_plus_dice")

PROTOCOLS = {
    # mixed partition: checkpoint selection and reporting share the 307 val frames
    "polypgen": Protocol("polypgen", "polypgen", splits=("train", "val", "val"), **_POLYPGEN),
    # centre C4 held out: selection on the stratified val split, reporting on test
    "polypgen_c4": Protocol("polypgen_c4", "polypgen_c4", splits=("train", "val", "test"), **_POLYPGEN),
    "siim": Protocol("siim", "siim", epochs=111, t_max=111, balanced=True, vertical_flip=False,
                     rotation=(-15, 15), rotation_per_image=True, nearest_masks=True, batch_dice=True,
                     selection="dice_pos", splits=("train", "val", "test")),
}

# Which objective family a protocol uses (emc.training.losses.build_objective)
OBJECTIVE_FAMILY = {"polypgen": "polypgen", "polypgen_c4": "polypgen", "siim": "siim"}
