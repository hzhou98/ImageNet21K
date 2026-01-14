# --------------------------------------------------------
# ImageNet-21K Pretraining for The Masses
# Copyright 2021 Alibaba MIIL (c)
# Licensed under MIT License [see the LICENSE file for details]
# Written by Tal Ridnik
# Modified to include Weights & Biases integration
# --------------------------------------------------------

import argparse
import time
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data.distributed
from torch.optim import lr_scheduler

from src_files.data_loading.data_loader import create_data_loaders
from src_files.helper_functions.distributed import print_at_master, to_ddp, num_distrib, setup_distrib, is_master
from src_files.helper_functions.general_helper_functions import silence_PIL_warnings
from src_files.models import create_model
from torch.cuda.amp import GradScaler, autocast
from src_files.optimizers.create_optimizer import create_optimizer
from src_files.semantic.metrics import AccuracySemanticSoftmaxMet
from src_files.semantic.semantic_loss import SemanticSoftmaxLoss
from src_files.semantic.semantics import ImageNet21kSemanticSoftmax

# Weights & Biases
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not installed. Install with: pip install wandb")

parser = argparse.ArgumentParser(description='PyTorch ImageNet21K Semantic Softmax Training')
parser.add_argument('--data_path', type=str)
parser.add_argument('--lr', default=3e-4, type=float)
parser.add_argument('--model_name', default='tresnet_m')
parser.add_argument('--model_path', default='./tresnet_m.pth', type=str)
parser.add_argument('--num_workers', default=8, type=int)
parser.add_argument('--image_size', default=224, type=int)
parser.add_argument('--num_classes', default=11221, type=int)
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--epochs', default=80, type=int)
parser.add_argument('--weight_decay', default=1e-4, type=float)
parser.add_argument("--local_rank", default=0, type=int)
parser.add_argument("--label_smooth", default=0.2, type=float)
parser.add_argument("--tree_path", default='./resources/imagenet21k_miil_tree.pth', type=str)
# Weights & Biases arguments
parser.add_argument('--use_wandb', action='store_true', help='Enable Weights & Biases logging')
parser.add_argument('--wandb_project', default='imagenet21k', type=str, help='W&B project name')
parser.add_argument('--wandb_entity', default=None, type=str, help='W&B entity (username or team)')
parser.add_argument('--wandb_run_name', default=None, type=str, help='W&B run name')


def main():
    # arguments
    args = parser.parse_args()

    # EXIF warning silent
    silence_PIL_warnings()

    # setup distributed
    setup_distrib(args)

    # Initialize Weights & Biases (only on master process)
    if args.use_wandb and WANDB_AVAILABLE and is_master():
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name,
            config={
                "model_name": args.model_name,
                "learning_rate": args.lr,
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "num_classes": args.num_classes,
                "image_size": args.image_size,
                "weight_decay": args.weight_decay,
                "label_smooth": args.label_smooth,
                "num_gpus": max(num_distrib(), 1),
                "effective_batch_size": args.batch_size * max(num_distrib(), 1),
                "training_mode": "semantic_softmax",
            }
        )
        print_at_master("Weights & Biases initialized: {}".format(wandb.run.url))

    # Setup model
    model = create_model(args).cuda()
    model = to_ddp(model, args)

    # Watch model with wandb (optional - tracks gradients and parameters)
    if args.use_wandb and WANDB_AVAILABLE and is_master():
        wandb.watch(model, log='all', log_freq=100)

    # create optimizer
    optimizer = create_optimizer(model, args)

    # Data loading
    train_loader, val_loader = create_data_loaders(args)

    # semantic
    semantic_softmax_processor = ImageNet21kSemanticSoftmax(args)
    semantic_met = AccuracySemanticSoftmaxMet(semantic_softmax_processor)

    # Actuall Training
    train_21k(model, train_loader, val_loader, optimizer, semantic_softmax_processor, semantic_met, args)

    # Finish wandb run
    if args.use_wandb and WANDB_AVAILABLE and is_master():
        wandb.finish()


def train_21k(model, train_loader, val_loader, optimizer, semantic_softmax_processor, met, args):
    # set loss
    loss_fn = SemanticSoftmaxLoss(semantic_softmax_processor)

    # set scheduler
    scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, steps_per_epoch=len(train_loader),
                                        epochs=args.epochs, pct_start=0.1, cycle_momentum=False, div_factor=20)

    # set scalaer
    scaler = GradScaler()
    
    for epoch in range(args.epochs):
        if num_distrib() > 1:
            train_loader.sampler.set_epoch(epoch)

        # train epoch
        print_at_master("\nEpoch {}".format(epoch))
        epoch_start_time = time.time()
        train_loss = AverageMeter()
        
        for i, (input, target) in enumerate(train_loader):
            with autocast():  # mixed precision
                output = model(input)
                loss = loss_fn(output, target) # note - loss also in fp16
            
            # Track loss
            train_loss.update(loss.item(), input.size(0))
            
            model.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            # Log batch metrics to wandb
            if args.use_wandb and WANDB_AVAILABLE and is_master() and i % 100 == 0:
                wandb.log({
                    "train/batch_loss": loss.item(),
                    "train/learning_rate": scheduler.get_last_lr()[0],
                    "train/epoch": epoch,
                    "train/batch": epoch * len(train_loader) + i,
                })

        epoch_time = time.time() - epoch_start_time
        train_rate = len(train_loader) * args.batch_size / epoch_time * max(num_distrib(), 1)
        
        print_at_master(
            "\nFinished Epoch, Training Rate: {:.1f} [img/sec], Avg Loss: {:.4f}".format(
                train_rate, train_loss.avg))

        # validation epoch
        val_acc = validate_21k(val_loader, model, met)

        # Log epoch metrics to wandb
        if args.use_wandb and WANDB_AVAILABLE and is_master():
            wandb.log({
                "epoch": epoch,
                "train/epoch_loss": train_loss.avg,
                "train/images_per_sec": train_rate,
                "val/semantic_top1_accuracy": val_acc,
                "train/learning_rate_epoch": scheduler.get_last_lr()[0],
            })


def validate_21k(val_loader, model, met):
    print_at_master("starting validation")
    model.eval()
    met.reset()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            # mixed precision
            with autocast():
                logits = model(input).float()

            # measure accuracy and record loss
            met.accumulate(logits, target)

    print_at_master("Validation results:")
    print_at_master('Semantic Acc_Top1 [%] {:.2f} '.format(met.value))
    model.train()
    
    return met.value


if __name__ == '__main__':
    main()
