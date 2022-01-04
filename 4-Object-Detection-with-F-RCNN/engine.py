import math
import sys
import time
import torch
import utils


def train_one_epoch(model, optimizer, data_loader, device, weights, epoch, print_freq):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = '(TRAIN) Epoch: [{}]'.format(epoch)
    total_losses = {'classifier': 0, 'box_reg': 0, 'objectness': 0, 'rpn_box_reg': 0, 'total': 0}
    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    for images, targets, paths in metric_logger.log_every(data_loader, print_freq, header):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        # # Sum of all losses
        # losses = sum(loss for loss in loss_dict.values())
        # Weighted sum
        # weights=[1,1,1,1]#2*[0.3,0.7,0.3,0.7]
        losses = sum(w * loss for loss, w in zip(loss_dict.values(), weights))

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        total_losses['classifier'] = total_losses['classifier'] + loss_dict_reduced['loss_classifier'].item()
        total_losses['box_reg'] = total_losses['box_reg'] + loss_dict_reduced['loss_box_reg'].item()
        total_losses['objectness'] = total_losses['objectness'] + loss_dict_reduced['loss_objectness'].item()
        total_losses['rpn_box_reg'] = total_losses['rpn_box_reg'] + loss_dict_reduced['loss_rpn_box_reg'].item()
        total_losses['total'] = total_losses['total'] + loss_value

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    total_losses['classifier'] = total_losses['classifier'] / (len(data_loader.dataset) / data_loader.batch_size)
    total_losses['box_reg'] = total_losses['box_reg'] / (len(data_loader.dataset) / data_loader.batch_size)
    total_losses['objectness'] = total_losses['objectness'] / (len(data_loader.dataset) / data_loader.batch_size)
    total_losses['rpn_box_reg'] = total_losses['rpn_box_reg'] / (len(data_loader.dataset) / data_loader.batch_size)
    total_losses['total'] = total_losses['total'] / (len(data_loader.dataset) / data_loader.batch_size)
    return total_losses


def eval_one_epoch(model, data_loader, device, epoch, print_freq):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = '(VAL) Epoch: [{}]'.format(epoch)
    total_losses = {'classifier': 0, 'box_reg': 0, 'objectness': 0, 'rpn_box_reg': 0, 'total': 0}

    with torch.no_grad():
        for images, targets, paths in metric_logger.log_every(data_loader, print_freq, header):
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)

            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = utils.reduce_dict(loss_dict)
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())

            loss_value = losses_reduced.item()

            total_losses['classifier'] = total_losses['classifier'] + loss_dict_reduced['loss_classifier'].item()
            total_losses['box_reg'] = total_losses['box_reg'] + loss_dict_reduced['loss_box_reg'].item()
            total_losses['objectness'] = total_losses['objectness'] + loss_dict_reduced['loss_objectness'].item()
            total_losses['rpn_box_reg'] = total_losses['rpn_box_reg'] + loss_dict_reduced['loss_rpn_box_reg'].item()
            total_losses['total'] = total_losses['total'] + loss_value

            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                print(loss_dict_reduced)
                sys.exit(1)

            metric_logger.update(loss=losses_reduced, **loss_dict_reduced)

    total_losses['classifier'] = total_losses['classifier'] / (len(data_loader.dataset) / data_loader.batch_size)
    total_losses['box_reg'] = total_losses['box_reg'] / (len(data_loader.dataset) / data_loader.batch_size)
    total_losses['objectness'] = total_losses['objectness'] / (len(data_loader.dataset) / data_loader.batch_size)
    total_losses['rpn_box_reg'] = total_losses['rpn_box_reg'] / (len(data_loader.dataset) / data_loader.batch_size)
    total_losses['total'] = total_losses['total'] / (len(data_loader.dataset) / data_loader.batch_size)
    return total_losses
