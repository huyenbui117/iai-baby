import importlib
import hydra
import pyrootutils
from omegaconf import DictConfig
from pytorch_lightning import LightningModule
import torch
from torchvision.utils import save_image
from datamodules.baby_datamodule import BabyDataModule
from datamodules.baby_segment_localized_datamodule import BabySegmentLocalizedDataModule
from models.baby_segment_module import BabySegmentLitModule

import torchvision.transforms as transforms
# from memory_profiler import profile
from segmentation_models_pytorch import Unet
from models.postprocess.largest_connected_component import MaxAreaProcessor
import os

from hydra import compose, initialize
import json
from collections import defaultdict

root = pyrootutils.setup_root(__file__, dotenv=True, pythonpath=True)
print(root)
read_image = BabySegmentLocalizedDataModule(download=False).read_image
read_label = BabySegmentLocalizedDataModule(download=False).read_label
measure_nt_width = BabySegmentLitModule().measure_nt_width
plot_thickness = BabySegmentLitModule().plot_thickness
calculate_nt_thickness = BabySegmentLitModule().calculate_nt_thickness
transform = transforms.Resize(
                (320, 544),
                interpolation=transforms.InterpolationMode.NEAREST
                )

# @profile
def load_weight_from_ckpt(ckpt_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(ckpt_path, map_location=device)
    return checkpoint["state_dict"]

def infer():
    flipud = False
    fliplr = True
    image_id = "1B_0001_1"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with initialize(version_base="1.2", config_path= "../configs"):
        cfg = compose(config_name="infer.yaml",return_hydra_config=True)
        model: L = hydra.utils.instantiate(cfg.model)
        model.load_state_dict(
            torch.load(cfg.ckpt_path, map_location=device)["state_dict"],
            strict=False
        )
        model.eval()
        print("Model loaded successfully!")


        orig_img = read_image(os.path.dirname(cfg.input_path)+"/source.png", greyscale=False).unsqueeze(0)
        inp = read_image(cfg.input_path, greyscale=True).unsqueeze(0)
        inp_ = inp.clone()
        inp = transform(inp)
        print("Input image loaded successfully!")
        out = model(inp)
        out_mask = torch.argmax(out, dim=1)
        print("Output mask generated successfully!")
        postprocessor = MaxAreaProcessor()
        # if postprocessor is not None:
        out_mask = postprocessor(out_mask)
        out_mask = transforms.Resize(inp_.shape[-2:], interpolation=transforms.InterpolationMode.NEAREST)(out_mask)
        print("Postprocessing done!")
        inp = inp_

        

        # add mask to input image, decrease opacity of mask
        img_mask = inp.repeat(3,1,1,1)
        img_mask[0] = (out_mask == 1) * 0.03 + (inp[0] != 1) * img_mask[1]
        if os.path.exists(cfg.target_path):
            target = read_label(cfg.target_path).unsqueeze(0)
            img_mask[1] = (target[0][0] == 1) * 0.03 + (inp[0] != 1) * img_mask[1]

            gt_keypoints = None
            gt_thickness = None

            imgid_to_keypoints = defaultdict(lambda: [])
            # load gt keypoints
            if cfg.gt_keypoints_path is not None and os.path.exists(cfg.gt_keypoints_path):
                with open(cfg.gt_keypoints_path) as fin:
                                keypoints = json.load(fin)
                                for point in keypoints:
                                    imgid_to_keypoints[point["image_id"]].extend([
                                        point["p1"], point["p2"]
                                ])
                gt_keypoints = torch.tensor(imgid_to_keypoints[image_id]).int()

                gt_keypoints[:,0] = orig_img.shape[-1] - gt_keypoints[:,0] if fliplr else gt_keypoints[:,0]
                gt_keypoints[:,1] = orig_img.shape[-2] - gt_keypoints[:,1] if flipud else gt_keypoints[:,1]

                gt_keypoints = gt_keypoints.unsqueeze(0)
                gt_thickness = calculate_nt_thickness(gt_keypoints)

        # Load bbox
        with open(os.path.dirname(cfg.target_path) + "/bbox.txt") as f:
            bx, by, _, _ = [int(x) for x in f.readline().split()]
        bbox = torch.tensor([[bx, by], [bx, by]]).unsqueeze(0)

        # Predict keypoints and thickness
        keypoints = measure_nt_width(out_mask)+bbox
        thickness = calculate_nt_thickness(keypoints).to(device)

        # plot_keypoints(keypoints, img_mask)
        img_mask=img_mask.swapaxes(1,0)
        img_mask = plot_thickness(img_mask, thickness, gt_thickness,keypoints-bbox, gt_keypoints-bbox, fontScale=0.5)
        orig_img = plot_thickness(orig_img,thickness, gt_thickness,keypoints, gt_keypoints)
        print("Thickness calculated successfully!")

        save_image(img_mask, cfg.output_path_localized)
        save_image(orig_img, cfg.output_path)


if __name__ == "__main__":
    infer()
