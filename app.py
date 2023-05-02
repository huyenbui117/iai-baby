DESCRIPTION = '''# MMDetection
This is a demo for [https://github.com/huyenbui117/iai-baby](https://github.com/huyenbui117/iai-baby).
'''
import numpy as np
import gradio as gr
import torch
from torchvision.utils import save_image
import torchvision.transforms as transforms
import os
import subprocess
import cv2
from hydra import compose, initialize
import hydra
import pyrootutils
from src.datamodules.baby_segment_localized_datamodule import BabySegmentLocalizedDataModule
from src.models.baby_segment_module import BabySegmentLitModule
from src.models.postprocess.largest_connected_component import MaxAreaProcessor
from pytorch_lightning import LightningModule
import json
from collections import defaultdict

# from yolov7.inference import load_model as load_detect_model
root = pyrootutils.setup_root(__file__, dotenv=True, pythonpath=True)
read_image = BabySegmentLocalizedDataModule(download=False).read_image
read_label = BabySegmentLocalizedDataModule(download=False).read_label
measure_nt_width = BabySegmentLitModule().measure_nt_width
plot_thickness = BabySegmentLitModule().plot_thickness
calculate_nt_thickness = BabySegmentLitModule().calculate_nt_thickness

transform = transforms.Resize(
                (320, 544),
                interpolation=transforms.InterpolationMode.NEAREST
                )
MODEL_LIST= {
    "Detect": ["run_19ehq8rw_model:v0","run_3jtlliw3_model:v0","run_365a6tqi_model:v0","run_1jgm77k6_model:v0","run_1bx38yj0_model:v0"],
    "Segment": ["model-g7p77hh6:v0","model-w6da8twl:v0","model-82u0xj8d:v0","model-1gjyuc0n:v0"],
}
seg_model_history = None
seg_model=None
seg_cfg=None

def load_segment_model(model_name: str="model-g7p77hh6:v0"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with initialize(version_base="1.2", config_path= "configs"):
        cfg = compose(config_name="infer.yaml",return_hydra_config=True, overrides=["ckpt_path=${paths.root_dir}/ckpt/"+model_name+"/model.ckpt"])
        model: L = hydra.utils.instantiate(cfg.model)
        model.load_state_dict(
            torch.load(cfg.ckpt_path, map_location=device)["state_dict"],
            strict=False
        )
        return model, cfg
def standardize_image(image):
    image = image.squeeze(0)
    # c x w x h -> w x h x c
    image = image.permute(1,2,0)
    image = image.numpy()
    return image
def infer_seg(input_img, model, cfg, flip, image_id, opacity=0.03):
    gt_keypoints = None
    gt_thickness = None
    flipud = (flip.count("UD") > 0)
    fliplr = (flip.count("LR") > 0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    img_mask[0] = (out_mask == 1) * opacity + (inp[0] != 1) * img_mask[1]
    if os.path.exists(cfg.target_path):
        target = read_label(cfg.target_path).unsqueeze(0)
        img_mask[1] = (target[0][0] == 1) * opacity + (inp[0] != 1) * img_mask[1]



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
    bbox = torch.tensor([[0, 0], [0, 0]]).unsqueeze(0)
    if os.path.exists(os.path.dirname(cfg.target_path) + "/bbox.txt"):
        with open(os.path.dirname(cfg.target_path) + "/bbox.txt") as f:
            bx, by, _, _ = [int(x) for x in f.readline().split()]
        bbox = torch.tensor([[bx, by], [bx, by]]).unsqueeze(0)

    # Predict keypoints and thickness
    keypoints = measure_nt_width(out_mask)+bbox
    thickness = calculate_nt_thickness(keypoints).to(device)

    # plot_keypoints(keypoints, img_mask)
    img_mask=img_mask.swapaxes(1,0)
    
    orig_img = plot_thickness(orig_img,thickness, gt_thickness,keypoints, gt_keypoints)
    print("Thickness calculated successfully!")

    if gt_keypoints is not None:
        gt_keypoints = gt_keypoints - bbox
    keypoints = keypoints - bbox

    img_mask = plot_thickness(img_mask, thickness, gt_thickness,keypoints, gt_keypoints, fontScale=0.5)
    save_image(img_mask, cfg.output_path_localized)
    save_image(orig_img, cfg.output_path)
    
    return orig_img, img_mask

def inference(input_img, flip, task, detect, segment, target_img, file_name=None, opacity=0.03):
    global seg_model_history, seg_model, seg_cfg
    subprocess.run("rm -rf data/inference/*", shell=True)
    input_img.save("data/inference/source.png")
    if target_img is not None:
        target_img.save("data/inference/target.png")
    if task.count("Detect") > 0:
        subprocess.run(f"python yolov7/inference.py --model_name {detect}", shell=True)
        output_img = cv2.imread("data/inference/detect.png"), cv2.imread("data/inference/infer.png")
        print(output_img[0].shape)

    if task.count("Segment") > 0:

        if seg_model_history != segment:
            seg_model, seg_cfg = load_segment_model(segment)
            seg_model_history = segment
        if not os.path.exists(seg_cfg.input_path):
            input_img.save(seg_cfg.input_path)
        output_img = infer_seg(input_img, seg_model, seg_cfg, flip, file_name, opacity)
        output_img = cv2.imread(seg_cfg.output_path), cv2.imread(seg_cfg.output_path_localized)        
        # subprocess.run(f"python src/inference.py", shell=True)
    output_img = cv2.cvtColor(output_img[0], cv2.COLOR_BGR2RGB), cv2.cvtColor(output_img[1], cv2.COLOR_BGR2RGB)
    return output_img


Top_Title="<center>Nuchal Translucency Measurement by <a href='http://github.com/huyenbui117' style='text-decoration: underline' target='_blank'>Khanh Huyen Bui</center></a>"
Custom_description="<center>Custom Training Performed <a href='https://wandb.ai/baby-team/baby' style='text-decoration: underline' target='_blank'>Link</a> </center><br> <center>Automatic Nuchal Translucency Measurement</center> <br> <b>Red</b> mask is for model prediction<br><b>Green</b> mask is for target"

demo = gr.Interface(inference, [gr.Image(type='pil', label='Input image'),
                    gr.CheckboxGroup( value=["LR"],choices=["UD","LR"]),
                    gr.CheckboxGroup( value=["Detect", "Segment"],choices=["Detect", "Segment"]),
                    gr.Dropdown(value="run_19ehq8rw_model:v0",choices=MODEL_LIST["Detect"]),
                    gr.Dropdown(value = "model-g7p77hh6:v0", choices=MODEL_LIST["Segment"]),
                    gr.Image(type='pil', label='Target image'),
                    gr.Textbox(label="File name", value="1B_0001_1", placeholder="1B_0001_1"),
                    gr.Slider(label="Opacity", value=0.03, minimum=0, maximum=0.5, step=0.01),],
                    [gr.Image(label='Output image'),
                    gr.Image(label='Output localized image')],
                    title=Top_Title,
                    description=Custom_description,
                    # live=True,
                    )
if __name__ == "__main__":
    demo.launch()