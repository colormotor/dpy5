r"""
 _____              _____ 
|  __ \            | ____|
| |  | |_ __  _   _| |__  
| |  | | '_ \| | | |___ \ 
| |__| | |_) | |_| |___) |
|_____/| .__/ \__, |____/ 
       | |     __/ |      
       |_|    |___/

Processing-like API for DiffVG
© Daniel Berio (@colormotor) 2026 - ...

Some image loss functions
"""

import torch
from torchvision import transforms
import numpy as np
from collections import OrderedDict
import os, numbers
from PIL import Image


cfg = lambda: None
cfg.batch_size = 1  # Hacky, fixme
cfg.clip_models = {}


def default_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        # DiffVG does not work well with ARM
        return torch.device("cpu")


class MultiscaleMSELoss(torch.nn.Module):
    """Multiscale MSE loss for images, adapted from PyDiffvg examples"""

    def __init__(self, sigma=1, rgb=True, device=None, debug=False):
        super(MultiscaleMSELoss, self).__init__()
        self.device = device or default_device()
        self.rgb = rgb
        self.blur = transforms.GaussianBlur(
            kernel_size=int(np.ceil(4 * sigma)) + 1, sigma=(sigma, sigma)
        )
        self.debug = debug

    def forward(self, im, target, mult=1, scale_factor=0.5, num_levels=None):
        im = to_batch(im, self.rgb, self.device)
        target = to_batch(target, self.rgb, self.device).to(im.dtype)
        bs, c, h, w = im.shape

        if num_levels is None:
            num_levels = max(int(np.ceil(np.log2(h))) - 2, 1)

        losses = []
        w = 1.0
        wsum = 0
        for lvl in range(num_levels):
            loss = torch.nn.functional.mse_loss(im, target)
            losses.append(loss * w)
            wsum += w
            w = w * mult

            im = torch.nn.functional.interpolate(
                self.blur(im), scale_factor=scale_factor, mode="nearest"
            )
            target = torch.nn.functional.interpolate(
                self.blur(target), scale_factor=scale_factor, mode="nearest"
            )

        losses = torch.stack(losses)
        return losses.sum()


class CLIPVisualLoss(torch.nn.Module):
    """CLIP visual loss, a-la CLIPAsso"""

    def __init__(
        self,
        input_size=224,
        rgb=True,
        clipag=False,
        clip_model="ViT-B-32",
        semantic_w=0.1,
        geometric_w=1.0,
        crop_scale=0.9,
        distortion_scale=0.5,
        vis_metric="mse",
        blur_sigma=None,
        blur_kernel=21,
        layer_weights=[(2, 1.0), (3, 1.0)],
        device=None,
    ):
        super().__init__()

        if device is None:
            self.device = default_device()
        else:
            self.device = device

        self.semantic_w = semantic_w
        self.geometric_w = geometric_w

        self.blur_sigma = blur_sigma
        self.blur_kernel = blur_kernel
        
        if clipag:
            clip_model = "CLIPAG"
        model, preprocess, tokenizer, self.input_size = load_clip_model(
            clip_model, self.device
        )
        self.crop_scale = crop_scale
        self.distortion_scale = distortion_scale
        self.layer_weights = layer_weights
        self.rgb = rgb
        self.vis_metric = vis_metric
        self.model = model  # .to(device)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        self.normalize = transforms.Compose(
            [
                preprocess.transforms[0],  # Resize
                # preprocess.transforms[1],  # CenterCrop
                preprocess.transforms[-1],  # Normalize
            ]
        )

        self.feature_maps = OrderedDict()

        try:
            for i in range(12):
                model.visual.transformer.resblocks[i].register_forward_hook(
                    self.make_hook(i)
                )
        except AttributeError as e:
            print("Resblocks not present attempting trunk")
            try:
                for i in range(12):
                    model.visual.trunk.blocks[i].register_forward_hook(
                        self.make_hook(i)
                    )
            except AttributeError as e:
                flat_idx = 0
                for stage in model.visual.trunk.stages:
                    for block in stage.blocks:
                        block.register_forward_hook(self.make_hook(flat_idx))
                        flat_idx += 1
                print("ConvNetXt registered ", flat_idx, "hooks")

    def make_hook(self, name):
        def hook(module, input, output):
            if len(output.shape) == 3:
                self.feature_maps[name] = output.permute(1, 0, 2)
            else:
                self.feature_maps[name] = output

        return hook

    def encode_image(self, image):
        self.feature_maps = OrderedDict()

        fc = self.model.encode_image(image)
        feature_maps = self.feature_maps
        return fc, feature_maps

    @property
    def clip_norm_(self):
        return transforms.Normalize(
            (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
        )

    def forward(self, x, y, num_aug=4):
        x = to_batch(x, self.rgb, self.device)
        y = to_batch(y, self.rgb, self.device).to(dtype=x.dtype)
        im_res = self.input_size  # x.shape[-1]
        if isinstance(self.crop_scale, numbers.Number):
            crop_scale = (self.crop_scale, self.crop_scale)
        else:
            crop_scale = self.crop_scale
        # init augmentations
        augment_list = []
        if self.distortion_scale > 0:
            augment_list.append(
                transforms.RandomPerspective(
                    fill=1, p=1.0, distortion_scale=self.distortion_scale
                )  # 0.5)
            )
        augment_list.append(
            transforms.RandomResizedCrop(im_res, scale=crop_scale, ratio=(1.0, 1.0))
        )
        if self.blur_sigma is not None:
            augment_list.append(
                transforms.GaussianBlur(
                    kernel_size=self.blur_kernel,
                    sigma=self.blur_sigma,
                )  # Example: 5x5 kernel
            )
        augment_list.append(self.clip_norm_)  # CLIP Normalize
        # compose augmentations
        augment_compose = transforms.Compose(augment_list)

        # make augmentation pairs
        x_augs, y_augs = [self.normalize(x)], [self.normalize(y)]
        # repeat N times
        for n in range(num_aug):
            augmented_pair = augment_compose(torch.cat([x, y]))
            x_augs.append(augmented_pair[0].unsqueeze(0))
            y_augs.append(augmented_pair[1].unsqueeze(0))

        xs = torch.cat(x_augs, dim=0)
        ys = torch.cat(y_augs, dim=0)
        self.x_augs = [xa.detach().cpu().numpy()[0, 0, :, :] for xa in x_augs]
        self.y_augs = [ya.detach().cpu().numpy()[0, 0, :, :] for ya in y_augs]

        fc_true, fm_true = self.encode_image(ys)
        fc_pred, fm_pred = self.encode_image(xs)

        fc_loss = (1 - torch.cosine_similarity(fc_true, fc_pred, dim=1)).mean()

        fm_loss = 0
        for (
            i,
            w,
        ) in self.layer_weights:  # [3, 5]: #2,3]: #3, 5]: #1,2,3]: #[3, 4]: #2, 3]:
            if self.vis_metric == "mse" or self.vis_metric == "L2":
                fm_loss += w * torch.square(fm_true[i] - fm_pred[i]).mean()
            elif self.vis_metric == "L1":
                fm_loss += w * torch.abs(fm_true[i] - fm_pred[i]).mean()
            else:
                fm_loss += (
                    w
                    * (
                        1 - torch.cosine_similarity(fm_true[i], fm_pred[i], dim=1)
                    ).mean()
                )

        total_loss = self.semantic_w * fc_loss + self.geometric_w * fm_loss
        # print('Clip total', total_loss.item())
        return total_loss


def load_clip_model(model_name, device):
    import open_clip

    if model_name in cfg.clip_models:
        print(model_name, "already loaded")
        return cfg.clip_models[model_name]
    preloaded = False
    if model_name == "CLIPAG":
        print("Downlading CLIPAG")
        url = "https://zenodo.org/records/10446026/files/CLIPAG_ViTB32.pt?download=1"
        path = "./CLIPAG.pt"
        download_file_once(url, path)
        pretrained = path
        model_name = "ViT-B-32"
    elif "FARE4" in model_name:
        model_name = "hf-hub:chs20/FARE4-ViT-B-32-laion2B-s34B-b79K"
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, device=device
        )
        preloaded = True
    elif "TeCoA4" in model_name:
        model_name = "hf-hub:chs20/TeCoA4-ViT-B-32-laion2B-s34B-b79K"
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, device=device
        )
        preloaded = True
    else:
        pretrained_map = {
            "ViT-H/14-quickgelu": "dfn5b",
            "ViT-B-32": "laion2b_s34b_b79k",
            "ViT-B-16-SigLIP-384": "webli",
            "ViT-L-16-SigLIP-256": "webli",
            "ViT-L-16-SigLIP-384": "webli",
            "ViT-SO400M-14-SigLIP-384": "webli",
            "ViT-SO400M-14-SigLIP": "webli",  # Ok
            "ViT-SO400M/14": "webli",
            "ViT-L-14": "laion2b_s32b_b82k",  # Good for sketches?
            "ViT-L-14-quickgelu": "metaclip_fullcc",
            "ViT-g-14": "laion2b_s34b_b88k",
            "ViT-B-16": "datacomp_xl_s13b_b90k",
            "EVA02-L-14": "merged2b_s4b_b131k",
            "ViT-H-14-CLIPA": "datacomp1b",
            "ViT-H-14-378-quickgelu": "dfn5b",  # No mem
            "ViT-L-14-CLIPA-336": "datacomp1b",
            "ViT-H-14-quickgelu": "metaclip_fullcc",  # Good, but slow
            "ViT-H-14-378-quickgelu": "dfn5b",
            "ViT-B-16-SigLIP-384": "webli",
            "ViT-H-14-quickgelu": "metaclip_fullcc",
            "ViT-L-14-CLIPA": "datacomp1b",
            "EVA02-L-14": "merged2b_s4b_b131k",
            "ViT-B-32-256": "datacomp_s34b_b86k",
        }
        pretrained = pretrained_map[model_name]
    if not preloaded:
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained,
            precision="amp",
            weights_only=False,  # Breaks otherwise
            device=device,
        )
    try:
        input_size = preprocess.transforms[0].size[0]
    except TypeError:
        input_size = preprocess.transforms[0].size
    print("CLIP input size is ", input_size)
    tokenizer = open_clip.get_tokenizer(model_name)
    cfg.clip_models[model_name] = (model, preprocess, tokenizer, input_size)
    return model, preprocess, tokenizer, input_size


def download_file_once(url, local_path):
    import requests

    local_path = os.path.expanduser(local_path)

    # Check if the file already exists
    if os.path.exists(local_path):
        print(f"File already exists at {local_path}. Skipping download.")
    else:
        print(f"Downloading file from {url} to {local_path}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an error for bad status codes

        # Save the file in chunks
        with open(local_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        print("Download complete.")

    return local_path


def to_batch(x, rgb, device):
    if isinstance(x, Image.Image):
        if not rgb:
            x = x.convert("L")
        x = torch.tensor(np.array(x) / 255, device=device)
    elif isinstance(x, np.ndarray):
        x = torch.tensor(x, device=device)
    if rgb:
        if len(x.shape) == 3:
            x = x[:, :, :, np.newaxis]
        x = x.permute((3, 2, 0, 1))  # to NCHW
    else:
        if len(x.shape) > 2:
            x = torch.mean(x, axis=-1)
        if len(x.shape) == 2:
            x = x[np.newaxis, np.newaxis, :, :]
        x = x.repeat(1, 3, 1, 1)
    x = x.repeat(cfg.batch_size, 1, 1, 1)
    return x
