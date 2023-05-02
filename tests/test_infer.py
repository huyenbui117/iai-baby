import hydra
from hydra import compose, initialize
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, open_dict
import pyrootutils
import torch
from pytorch_lightning import LightningModule as L

root = pyrootutils.setup_root(__file__, dotenv=True, pythonpath=True)

def infer(model_name: str="model-g7p77hh6:v0"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with initialize(version_base="1.2", config_path= "../configs"):
        cfg = compose(config_name="infer.yaml",return_hydra_config=True, overrides=["ckpt_path=${paths.root_dir}/ckpt/"+model_name+"/model.ckpt"])
        model: L = hydra.utils.instantiate(cfg.model)
        model.load_state_dict(
            torch.load(cfg.ckpt_path, map_location=device)["state_dict"],
            strict=False
        )
        model.eval()
        print(cfg.output_path)

if __name__ == "__main__":
    infer()
