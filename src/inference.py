import importlib
import hydra
import pyrootutils
from omegaconf import DictConfig
from pytorch_lightning import LightningModule
import torch
from torchvision.utils import save_image
from datamodules.baby_datamodule import BabyDataModule
from memory_profiler import profile
from segmentation_models_pytorch import Unet

root = pyrootutils.setup_root(__file__, dotenv=True, pythonpath=True)

read_image = BabyDataModule().read_image


@profile
def load_weight_from_ckpt(ckpt_path):
    checkpoint = torch.load(ckpt_path)
    return checkpoint["state_dict"]
    

@profile
@hydra.main(version_base="1.2", config_path=root / "configs", config_name="eval.yaml")
def main(cfg: DictConfig) -> None:
    with torch.no_grad():
        model_args = {**cfg.model.net}
        model_args.pop("_target_")
        # model = Unet(
        #     **model_args
        # )
        model: LightningModule = hydra.utils.instantiate(cfg.model)
        model.load_state_dict(
            load_weight_from_ckpt(cfg.ckpt_path),
            strict=False
        )
        model.eval()

        inp = read_image(cfg.input_path, greyscale=True).unsqueeze(0)
        out = model(inp)
        out_mask = torch.argmax(out, dim=1)

        postprocessor = hydra.utils.instantiate(cfg.model.postprocessor)
        if postprocessor is not None:
            out_mask = postprocessor(out_mask)
        save_image(torch.cat((inp[0], out_mask), dim=1), cfg.output_path)
    

if __name__ == "__main__":
    main()
