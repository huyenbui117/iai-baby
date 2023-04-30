import importlib
import hydra
import pyrootutils
from omegaconf import DictConfig
from pytorch_lightning import LightningModule
import torch
from torchvision.utils import save_image
from datamodules.baby_datamodule import BabyDataModule
from datamodules.baby_segment_localized_datamodule import BabySegmentLocalizedDataModule
import torchvision.transforms as transforms
# from memory_profiler import profile
from segmentation_models_pytorch import Unet

root = pyrootutils.setup_root(__file__, dotenv=True, pythonpath=True)

read_image = BabySegmentLocalizedDataModule().read_image
read_label = BabySegmentLocalizedDataModule().read_label
transform = transforms.Resize(
                (320, 544),
                interpolation=transforms.InterpolationMode.NEAREST
                )
# @profile
def load_weight_from_ckpt(ckpt_path):
    checkpoint = torch.load(ckpt_path)
    return checkpoint["state_dict"]
    

# @profile
@hydra.main(version_base="1.2", config_path=root / "configs", config_name="eval.yaml")
def main(cfg: DictConfig) -> None:
    print("Starting inference...")
    with torch.no_grad():
        model_args = {**cfg.model.net}
        model_args.pop("_target_")
        # model = Unet(
        #     **model_args
        # )
        model: LightningModule = hydra.utils.instantiate(cfg.model)
        print("Loading weights from checkpoint...")
        model.load_state_dict(
            load_weight_from_ckpt(cfg.ckpt_path),
            strict=False
        )
        model.eval()
        print("Model loaded successfully!")
        import IPython; IPython.embed()
        inp = read_image(cfg.input_path, greyscale=True).unsqueeze(0)
        inp = transform(inp)
        print("Input image loaded successfully!")
        out = model(inp)
        out_mask = torch.argmax(out, dim=1)
        print("Output mask generated successfully!")
        postprocessor = hydra.utils.instantiate(cfg.model.postprocessor)
        if postprocessor is not None:
            out_mask = postprocessor(out_mask)
        print("Postprocessing done!")

        target = read_label(cfg.target_path).unsqueeze(0)
        target = transform(target)


        img_mask = inp.repeat(3,1,1,1)
        img_mask[0] = (out_mask == 1) * 0.1 + (inp[0] != 1) * img_mask[1]
        img_mask[1] = (target[0][0] == 1) * 0.1 + (inp[0] != 1) * img_mask[1]
        img_mask=img_mask.swapaxes(1,0)
        # add mask to input image, decrease opacity of mask

        save_image(img_mask, cfg.output_path)
    

if __name__ == "__main__":
    main()
