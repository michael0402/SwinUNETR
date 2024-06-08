# Copyright 2020 - 2022 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
import sys
import pdb
import shutil
import time
import json
import tempfile
import logging
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Subset
import torch
import torch.nn.parallel
import torch.utils.data.distributed
# from tensorboardX import SummaryWriter
# from torch.cuda.amp import GradScaler, autocast
# from utils.utils import AverageMeter, distributed_all_gather

from monai.data import decollate_batch

# import torch
# import torch.nn.parallel
# import torch.utils.data.distributed
# from optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
# from trainer import run_training
# from utils.data_utils import get_loader

from monai.inferers import sliding_window_inference
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.networks.nets import SwinUNETR
from monai import transforms
from monai import data
from monai.transforms import Activations, AsDiscrete, Compose, LoadImaged
from monai.utils.enums import MetricReduction
from monai.data import decollate_batch
from functools import partial

import logging
import boto3
from botocore.exceptions import ClientError


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = np.where(self.count > 0, self.sum / self.count, self.sum)


def datafold_read(datalist, basedir, fold=0, key="training"):
    # with open(datalist) as f:
    #     json_data = json.load(f)

    # Initialize the S3 client
    s3 = boto3.client('s3')

    # Assuming datalist is the path 's3://swin-unetr-brains/brats21_folds.json'
    bucket_name = 'swin-unetr-brains'  # S3 bucket name
    object_key = 'brats21_folds.json'  # S3 key path for the JSON file

    # Get the object from S3
    response = s3.get_object(Bucket=bucket_name, Key=object_key)

    # Read the file's content into memory
    json_data = json.loads(response['Body'].read().decode('utf-8'))

    json_data = json_data[key]

    for d in json_data:
        for k in d:
            if isinstance(d[k], list):
                d[k] = [os.path.join(basedir, iv) for iv in d[k]]
            elif isinstance(d[k], str):
                d[k] = os.path.join(basedir, d[k]) if len(d[k]) > 0 else d[k]

    tr = []
    val = []
    for d in json_data:
        if "fold" in d and d["fold"] == fold:
            val.append(d)
        else:
            tr.append(d)

    return tr, val


def save_checkpoint(model, epoch, dir_add, filename="model.pt", best_acc=0):
    state_dict = model.state_dict()
    save_dict = {"epoch": epoch, "best_acc": best_acc, "state_dict": state_dict}
    filename = os.path.join(dir_add, filename)
    torch.save(save_dict, filename)
    print("Saving checkpoint", filename)


# class S3Dataset(torch.utils.data.Dataset):
#     def __init__(self, data, transform=None, bucket_name="swin-unetr-brains"):
#         self.data = data
#         self.transform = transform
#         self.s3 = boto3.client('s3')
#         self.bucket_name = bucket_name

#     def __len__(self):
#         return len(self.data)
    
#     def __getitem__(self, idx):
#         item = self.data[idx]
#         image_path, label_path = item['image'], item['label']
        
#         # Handle paths whether they are lists or single strings
#         image_files = self.download_s3_objects(image_path) if isinstance(image_path, list) else [self.download_s3_object(image_path)]
#         label_files = self.download_s3_objects(label_path) if isinstance(label_path, list) else [self.download_s3_object(label_path)]
        
#         # Check if any file failed to download and handle it
#         if None in image_files or None in label_files:
#             logger.error("One or more files could not be downloaded properly.")
#             raise ValueError("Failed to download all necessary files.")

#         # Using the first file as an example, assuming files are required
#         print("image files: ", image_files)
#         image, label = image_files[0], label_files[0]
        
#         data = {'image': image_files, 'label': label}
#         # print("here")
#         # print(image)
#         # nii_img = nib.load(image)
#         # print(nii_img.shape)
#         if self.transform:
#             print("Data before transform:", data)  # Debug print
#             data = self.transform(data)
#         return data

#     def download_s3_object(self, file_key):
#         _, full_extension = os.path.splitext(file_key)
#         if full_extension == ".gz":
#             # Check if the real extension is .nii.gz
#             main_part, pre_extension = os.path.splitext(file_key[:-3])
#             if pre_extension == ".nii":
#                 full_extension = ".nii.gz"
        
#         # local_file_path = tempfile.mktemp()
#         # try:
#         #     logger.info(f"Downloading from {self.bucket_name}: {file_key} to {local_file_path}")
#         #     self.s3.download_file(self.bucket_name, file_key, local_file_path)
#         #     return local_file_path
#         # except Exception as e:
#         #     logger.error(f"Error downloading {file_key}: {e}")
#         #     return None
#         try:
#             with tempfile.NamedTemporaryFile(delete=False, suffix=full_extension) as tmp_file:
#                 self.s3.download_file(self.bucket_name, file_key, tmp_file.name)
#                 logger.info(f"Successfully downloaded {file_key} to {tmp_file.name}")
#                 return tmp_file.name
#         except Exception as e:
#             logger.error(f"Error downloading {file_key}: {e}")
#             return None

#     def download_s3_objects(self, file_keys):
#         local_file_paths = []
#         for file_key in file_keys:
#             file_path = self.download_s3_object(file_key)
#             if file_path is not None:
#                 local_file_paths.append(file_path)
#             else:
#                 local_file_paths.append(None)  # Append None to indicate a failed download
#         return local_file_paths

#     def cleanup(self):
#         # Cleanup any temporary files after use
#         for item in self.data:
#             for path in [item['image'], item['label']]:
#                 if isinstance(path, str) and os.path.exists(path):
#                     os.remove(path)
#                 elif isinstance(path, list):
#                     for sub_path in path:
#                         if os.path.exists(sub_path):
#                             os.remove(sub_path)


def get_loader(batch_size, data_dir, json_list, fold, roi):
    data_dir = data_dir
    datalist_json = json_list
    train_files, validation_files = datafold_read(datalist=datalist_json, basedir=data_dir, fold=fold)
    train_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"]),
            transforms.ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
            transforms.CropForegroundd(
                keys=["image", "label"],
                source_key="image",
                k_divisible=[roi[0], roi[1], roi[2]],
            ),
            transforms.RandSpatialCropd(
                keys=["image", "label"],
                roi_size=[roi[0], roi[1], roi[2]],
                random_size=False,
            ),
            transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
            transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            transforms.RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
            transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"]),
            transforms.ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
            transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        ]
    )

    # # print(train_files)
    # filename='flair_images.txt'
    
    # with open(filename, 'w') as file:
    #     # Iterate over each entry in the training files list
    #     for entry in validation_files:
    #         # Check if 'image' key exists in the dictionary
    #         if 'image' in entry:
    #             # Filter and get only the paths that include 'flair' in their names
    #             flair_images = [img for img in entry['image'] if 'flair' in img]
    #             for img in flair_images:
    #                 # Write each flair image path to the file
    #                 file.write(img + '\n')

    train_ds = data.Dataset(data=train_files, transform=train_transform)
    

    train_loader = data.DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=1,
        pin_memory=True,
    )
    val_ds = data.Dataset(data=validation_files, transform=val_transform)
    val_loader = data.DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
    )


    subset_ratio = 0.1

    # Determine 20% subset size for training data
    train_indices = np.random.choice(len(train_ds), int(len(train_ds) * subset_ratio), replace=False)
    # Create a subset for training data
    train_subset = Subset(train_ds, train_indices)

    # Determine 20% subset size for validation data
    val_indices = np.random.choice(len(val_ds), int(len(val_ds) * subset_ratio), replace=False)
    # Create a subset for validation data
    val_subset = Subset(val_ds, val_indices)

    # Use the subsets to define the data loaders
    train_loader = data.DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=1,
        pin_memory=True,
    )

    val_loader = data.DataLoader(
        val_subset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
    )

    return train_loader, val_loader



def train_epoch(model, loader, optimizer, epoch, max_epochs, loss_func, batch_size, device):
    model.train()
    start_time = time.time()
    run_loss = AverageMeter()
    
    for idx, batch_data in enumerate(loader):
        data, target = batch_data["image"].to(device), batch_data["label"].to(device)
        logits = model(data)
        loss = loss_func(logits, target)
        loss.backward()
        optimizer.step()
        run_loss.update(loss.item(), n=batch_size)
        print(
            "Epoch {}/{} {}/{}".format(epoch, max_epochs, idx, len(loader)),
            "loss: {:.4f}".format(run_loss.avg),
            "time {:.2f}s".format(time.time() - start_time),
        )
        start_time = time.time()

    return run_loss.avg


def val_epoch(
    model,
    loader,
    epoch,
    max_epochs,
    acc_func,
    device,
    model_inferer=None,
    post_sigmoid=None,
    post_pred=None,
):
    model.eval()
    start_time = time.time()
    run_acc = AverageMeter()

    with torch.no_grad():
        for idx, batch_data in enumerate(loader):
            data, target = batch_data["image"].to(device), batch_data["label"].to(device)
            logits = model_inferer(data)
            val_labels_list = decollate_batch(target)
            val_outputs_list = decollate_batch(logits)
            val_output_convert = [post_pred(post_sigmoid(val_pred_tensor)) for val_pred_tensor in val_outputs_list]
            acc_func.reset()
            acc_func(y_pred=val_output_convert, y=val_labels_list)
            acc, not_nans = acc_func.aggregate()
            run_acc.update(acc.cpu().numpy(), n=not_nans.cpu().numpy())
            dice_tc = run_acc.avg[0]
            dice_wt = run_acc.avg[1]
            dice_et = run_acc.avg[2]
            print(
                "Val {}/{} {}/{}".format(epoch, max_epochs, idx, len(loader)),
                ", dice_tc:",
                dice_tc,
                ", dice_wt:",
                dice_wt,
                ", dice_et:",
                dice_et,
                ", time {:.2f}s".format(time.time() - start_time),
            )
            start_time = time.time()

    return run_acc.avg




def trainer(
    model,
    train_loader,
    val_loader,
    optimizer,
    loss_func,
    acc_func,
    scheduler,
    max_epochs,
    root_dir,
    batch_size,
    device,
    val_every,
    model_inferer=None,
    start_epoch=0,
    post_sigmoid=None,
    post_pred=None,
):
    val_acc_max = 0.0
    dices_tc = []
    dices_wt = []
    dices_et = []
    dices_avg = []
    loss_epochs = []
    trains_epoch = []
    for epoch in range(start_epoch, max_epochs):
        print(time.ctime(), "Epoch:", epoch)
        epoch_time = time.time()
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            epoch=epoch,
            max_epochs=max_epochs,
            loss_func=loss_func,
            batch_size=batch_size,
            device=device,
        )
        print(
            "Final training  {}/{}".format(epoch, max_epochs - 1),
            "loss: {:.4f}".format(train_loss),
            "time {:.2f}s".format(time.time() - epoch_time),
        )

        if (epoch + 1) % val_every == 0 or epoch == 0:
            loss_epochs.append(train_loss)
            trains_epoch.append(int(epoch))
            epoch_time = time.time()
            val_acc = val_epoch(
                model,
                val_loader,
                epoch=epoch,
                max_epochs=max_epochs,
                acc_func=acc_func,
                device=device,
                model_inferer=model_inferer,
                post_sigmoid=post_sigmoid,
                post_pred=post_pred,
            )
            dice_tc = val_acc[0]
            dice_wt = val_acc[1]
            dice_et = val_acc[2]
            val_avg_acc = np.mean(val_acc)
            print(
                "Final validation stats {}/{}".format(epoch, max_epochs - 1),
                ", dice_tc:",
                dice_tc,
                ", dice_wt:",
                dice_wt,
                ", dice_et:",
                dice_et,
                ", Dice_Avg:",
                val_avg_acc,
                ", time {:.2f}s".format(time.time() - epoch_time),
            )
            dices_tc.append(dice_tc)
            dices_wt.append(dice_wt)
            dices_et.append(dice_et)
            dices_avg.append(val_avg_acc)
            if val_avg_acc > val_acc_max:
                print("new best ({:.6f} --> {:.6f}). ".format(val_acc_max, val_avg_acc))
                val_acc_max = val_avg_acc
                save_checkpoint(
                    model,
                    epoch,
                    root_dir,
                    best_acc=val_acc_max,
                )
            scheduler.step()
    print("Training Finished !, Best Accuracy: ", val_acc_max)
    return (
        val_acc_max,
        dices_tc,
        dices_wt,
        dices_et,
        dices_avg,
        loss_epochs,
        trains_epoch,
    )



def parse_args():
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1000,
        metavar="N",
        help="input batch size for testing (default: 1000)",
    )
    parser.add_argument(
        "--epochs", type=int, default=1, metavar="N", help="number of epochs to train (default: 1)"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        metavar="LR",
        help="learning rate (default: 0.01)",
    )
    parser.add_argument(
        "--beta_1", type=float, default=0.9, metavar="BETA1", help="beta1 (default: 0.9)"
    )
    parser.add_argument(
        "--beta_2", type=float, default=0.999, metavar="BETA2", help="beta2 (default: 0.999)"
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        metavar="WD",
        help="L2 weight decay (default: 1e-4)",
    )
    parser.add_argument("--seed", type=int, default=1, metavar="S", help="random seed (default: 1)")
    parser.add_argument(
        "--log-interval",
        type=int,
        default=100,
        metavar="N",
        help="how many batches to wait before logging training status",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default=None,
        help="backend for distributed training (tcp, gloo on cpu and gloo, nccl on gpu)",
    )

    # parser.add_argument(
    #         "--model-dir",
    #         type=str,
    #         default="/opt/ml/model",
    #         help="model dir",
    #     )

    # parser.add_argument(
    #         "--train",
    #         type=str,
    #     default="/home/sagemaker-user",
    #         help="training data dir",
    #     )
    

    # Container environment
    parser.add_argument("--hosts", type=list, default=json.loads(os.environ["SM_HOSTS"]))
    parser.add_argument("--current-host", type=str, default=os.environ["SM_CURRENT_HOST"])
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--train", type=str, default=os.environ["SM_CHANNEL_TRAINING"])
    parser.add_argument("--test", type=str, default=os.environ["SM_CHANNEL_TESTING"])
    parser.add_argument("--num-gpus", type=int, default=os.environ["SM_NUM_GPUS"])

    return parser.parse_args()

def download_s3_folder(bucket_name, s3_folder, local_dir=None):
    """
    Download the contents of a folder directory while maintaining the directory structure.
    Args:
    - bucket_name: the name of the s3 bucket
    - s3_folder: the folder path in the s3 bucket
    - local_dir: a relative or absolute directory path in the local file system
    """
    s3 = boto3.client('s3')
    # Ensure the folder path ends with '/' to avoid improper concatenation
    if not s3_folder.endswith('/'):
        s3_folder += '/'
    
    result = s3.list_objects_v2(Bucket=bucket_name, Prefix=s3_folder)
    
    if 'Contents' not in result:
        return  # Exit if the folder is empty or does not exist
    
    contents = result['Contents']
    while result['IsTruncated']:
        result = s3.list_objects_v2(Bucket=bucket_name, Prefix=s3_folder, ContinuationToken=result['NextContinuationToken'])
        contents.extend(result['Contents'])
        
    for content in contents:
        path, filename = os.path.split(content['Key'])
        # Create the directory structure in the local directory if it does not exist
        local_file_path = os.path.join(local_dir, os.path.relpath(path, s3_folder))
        if not os.path.exists(local_file_path):
            os.makedirs(local_file_path)
        local_file_full_path = os.path.join(local_file_path, filename)
        if filename:  # skip directories
            s3.download_file(bucket_name, content['Key'], local_file_full_path)



def main(args):
    bucket_name = 'swin-unetr-brains'
    s3_folder = 'TrainingData/'
    # data = args.train
    # download_s3_folder(bucket_name, s3_folder, data)
    data = args.train
    # data = "/home/sagemaker-user"
    # data = "s3://swin-unetr-brains"
    json_list = "s3://swin-unetr-brains/brats21_folds.json"
    roi = (128, 128, 128)
    fold = 1
    sw_batch_size=1
    infer_overlap = 0.5
    max_epochs = args.epochs
    val_every = 10
    train_loader, val_loader = get_loader(args.batch_size, data, json_list, fold, roi)

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

    torch.backends.cudnn.benchmark = True
    dice_loss = DiceLoss(to_onehot_y=False, sigmoid=True)
    post_sigmoid = Activations(sigmoid=True)
    post_pred = AsDiscrete(argmax=False, threshold=0.5)
    dice_acc = DiceMetric(include_background=True, reduction=MetricReduction.MEAN_BATCH, get_not_nans=True)
    model_inferer = partial(
        sliding_window_inference,
        roi_size=[roi[0], roi[1], roi[2]],
        sw_batch_size=sw_batch_size,
        predictor=model,
        overlap=infer_overlap,
    )
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

    directory = os.environ.get("MONAI_DATA_DIRECTORY")
    root_dir = tempfile.mkdtemp() if directory is None else directory

    start_epoch = 0

    (
        val_acc_max,
        dices_tc,
        dices_wt,
        dices_et,
        dices_avg,
        loss_epochs,
        trains_epoch,
    ) = trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        loss_func=dice_loss,
        acc_func=dice_acc,
        scheduler=scheduler,
        max_epochs=max_epochs,
        root_dir=args.model_dir,
        batch_size=args.batch_size,
        device=device,
        val_every=val_every,
        model_inferer=model_inferer,
        start_epoch=start_epoch,
        post_sigmoid=post_sigmoid,
        post_pred=post_pred,
    )

if __name__ == "__main__":
    args = parse_args()
    main(args)
    # run_training(args)