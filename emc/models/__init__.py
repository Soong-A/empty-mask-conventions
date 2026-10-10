"""The eight architectures of the study and how their outputs are read.

``build(key, variant=None)`` constructs a model with the configuration used in the
paper; ``probability(key, output)`` extracts the final probability map from a
forward pass (UNeXt-S returns logits, every other model returns probabilities).
"""
from __future__ import annotations

import importlib
from dataclasses import dataclass, field

import torch

C_LIST = [8, 16, 24, 32, 48, 64]


@dataclass(frozen=True)
class Spec:
    name: str                 # display name used in records and tables
    module: str
    cls: str
    kwargs: dict = field(default_factory=dict)
    logits: bool = False      # forward() returns logits instead of probabilities
    needs_points: bool = False


MODELS = {
    "unetpp": Spec("UNet++", "unetpp", "UNetPlusPlus",
                   dict(num_classes=1, input_channels=3, deep_supervision=True)),
    "mobileunet": Spec("MobileUNet", "mobileunet", "MobileUNet", dict(num_classes=1, input_channels=3)),
    "tinyunet": Spec("UNet-Tiny", "tinyunet", "TinyUNet", dict(in_channels=3, num_classes=1)),
    "prism": Spec("PRISM", "prism", "PRISMNet", dict(num_classes=1, input_channels=3)),
    "unext_s": Spec("UNeXt-S", "unext", "UNext_S",
                    dict(num_classes=1, input_channels=3, deep_supervision=False, img_size=256), logits=True),
    "lbunet": Spec("LB-UNet", "lbunet", "LBUNet", dict(num_classes=1, input_channels=3, c_list=C_LIST),
                   needs_points=True),
    "egeunet": Spec("EGE-UNet", "egeunet", "EGEUNet",
                    dict(num_classes=1, input_channels=3, c_list=C_LIST, bridge=True, gt_ds=True)),
    "malunet": Spec("MALUNet", "malunet", "MALUNet",
                    dict(num_classes=1, input_channels=3, c_list=C_LIST, split_att="fc", bridge=True)),
}
KEY_BY_NAME = {s.name: k for k, s in MODELS.items()}


def key_of(name: str) -> str:
    """Accept a registry key ('unetpp') or a display name ('UNet++')."""
    if name in MODELS:
        return name
    if name in KEY_BY_NAME:
        return KEY_BY_NAME[name]
    raise KeyError(f"unknown model {name!r}; choose from {sorted(MODELS)}")


def build(key: str, variant: str | None = None) -> torch.nn.Module:
    spec = MODELS[key_of(key)]
    kwargs = dict(spec.kwargs)
    if variant is not None:
        if spec.cls != "PRISMNet":
            raise ValueError("variants exist only for PRISM")
        kwargs["variant"] = variant
    module = importlib.import_module(f"emc.models.{spec.module}")
    return getattr(module, spec.cls)(**kwargs)


def probability(key: str, output) -> torch.Tensor:
    """Final [B, 1, H, W] probability map of a forward pass."""
    main = output
    while isinstance(main, (tuple, list)):
        main = main[-1]
    return torch.sigmoid(main) if MODELS[key_of(key)].logits else main


def load_weights(model: torch.nn.Module, path) -> dict:
    """Load a training checkpoint strictly (EMA weights; profiling buffers removed)."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    state = {k[7:] if k.startswith("module.") else k: v for k, v in state.items()
             if not k.endswith(("total_ops", "total_params"))}
    model.load_state_dict(state, strict=True)
    return ckpt if isinstance(ckpt, dict) else {}
