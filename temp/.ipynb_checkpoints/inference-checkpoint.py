import json
import logging
import sys
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.nets import SwinUNETR
from monai.inferers import sliding_window_inference
from monai import data
import numpy as np
from functools import partial
from monai import transforms
from monai.transforms import (
    AsDiscrete,
    Activations,
)



logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

roi = (128, 128, 128)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SwinUNETR(
        img_size=roi,
        in_channels=4,
        out_channels=3,
        feature_size=48,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        dropout_path_rate=0.0,
        use_checkpoint=True,
    ).to(device)

if torch.cuda.device_count() > 1:
    print(f"Let's use {torch.cuda.device_count()} GPUs!")
    model = torch.nn.DataParallel(model)


# defining model and loading weights to it.
def model_fn(model_dir):
    model.load_state_dict(torch.load(os.path.join(model_dir, "model.pt"))["state_dict"])
    model.to(device).eval()
    return model


# data preprocessing
# def input_fn(request_body, request_content_type):
#     assert request_content_type == "application/json"
#     data = json.loads(request_body)
#     data = torch.tensor(data, dtype=torch.float32, device=device)
#     return data


def input_fn(request_body, request_content_type):
    assert request_content_type == "application/json"
    input_data = json.loads(request_body)
    # DATA PRE PROCESSING STEP
    test_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"]),
            transforms.ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
            transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        ]
    )
    
    test_ds = data.Dataset(data=input_data, transform=test_transform)
    
    test_loader = data.DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )
    return test_loader
    


# inference
def predict_fn(input_object, model):
    # with torch.no_grad():
    #     prediction = model(input_object)
    # return prediction

    # PREDICTION PART - I THINK YOU MIGHT HAVE TO REMOVE THE FOR LOOP
    model_inferer_test = partial(
        sliding_window_inference,
        roi_size=[roi[0], roi[1], roi[2]],
        sw_batch_size=1,
        predictor=model,
        overlap=0.6,
    )

    with torch.no_grad():
        for batch_data in input_object:
            image = batch_data["image"].cuda()
            # image = batch_data["image"].to('cpu')
            prob = torch.sigmoid(model_inferer_test(image))
            return prob
    return



# postprocess
def output_fn(predictions, content_type):
    # assert content_type == "application/json"
    # res = predictions.cpu().numpy().tolist()
    # return json.dumps(res)

    # POST PROCESSING PART
    seg = predictions[0].detach().cpu().numpy()
    seg = (seg > 0.5).astype(np.int8)
    seg_out = np.zeros((seg.shape[1], seg.shape[2], seg.shape[3]))
    seg_out[seg[1] == 1] = 2
    seg_out[seg[0] == 1] = 1
    seg_out[seg[2] == 1] = 4
    return seg_out