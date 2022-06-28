# Copyright 2020 - 2021 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import time
import shutil
import numpy as np
import torch
from torch.cuda.amp import GradScaler, autocast
from tensorboardX import SummaryWriter
import torch.nn.parallel
from utils.utils import distributed_all_gather
import torch.utils.data.distributed
from monai.data import decollate_batch

def dice(x, y):
    intersect = np.sum(np.sum(np.sum(x * y)))
    y_sum = np.sum(np.sum(np.sum(y)))

    # print("##########################")
    # print(y_sum)
    # print("$$$$$$$$$$$$$$$$$$$$$$$$$$")

    if y_sum == 0:
        return 0.0
    x_sum = np.sum(np.sum(np.sum(x)))
    # print(x_sum)
    # print("*******************************************")
    # print(intersect)
    return 2 * intersect / (x_sum + y_sum)

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
        self.avg = np.where(self.count > 0,
                            self.sum / self.count,
                            self.sum)


def train_epoch(model,
                loader,
                optimizer,
                scaler,
                epoch,
                acc_func,
                loss_func,
                args,
                post_label=None,
                post_pred=None):
    model.train()
    start_time = time.time()
    run_loss = AverageMeter()
    for idx, batch_data in enumerate(loader):

        # print("NNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNN")

        # print(batch_data['label'].shape)

        # print("sssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssss")
        # print(np.unique(batch_data['label']))


        if isinstance(batch_data, list):
            data, target = batch_data
        else:
            data, target = batch_data['image'], batch_data['label']
     
        # print(np.unique(target))

        data, target = data.cuda(args.rank), target.cuda(args.rank)

        for param in model.parameters(): param.grad = None

        # print("QQQQQQQQQQQQQQQQQQQQQQQQQQQQQQ",data.shape)
        # print("NNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNN",target.shape)

        with autocast(enabled=args.amp):
           
            logits = model(data)
        
            # print("aaaaaaaaaaaaaa:", logits.shape)
            # print("bbbbbbbbbbbbbb:", target.unique())
            # print("11111111111111111111111111111")
            print("1111111111")
            print(logits.shape)
            print(target.shape)
            
            loss = loss_func(logits, target)


######training accuracy######
        val_labels_list = decollate_batch(target)
        val_labels_convert = [post_label(val_label_tensor) for val_label_tensor in val_labels_list]
        val_outputs_list = decollate_batch(logits)
        val_output_convert = [post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list]
        acc = acc_func(y_pred=val_output_convert, y=val_labels_convert)
        acc = acc.cuda(args.rank)

        if args.distributed:
            acc_list = distributed_all_gather([acc],
                                                out_numpy=True,
                                                is_valid=idx < loader.sampler.valid_length)
            avg_acc = np.mean([np.nanmean(np.delete(l,0)) for l in acc_list])

        else:
            acc_list = acc.detach().cpu().numpy()
            avg_acc = np.mean([np.nanmean(np.delete(l,0)) for l in acc_list])

######training accuracy######

        if args.amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        if args.distributed:
            loss_list = distributed_all_gather([loss],
                                               out_numpy=True,
                                               is_valid=idx < loader.sampler.valid_length)
            run_loss.update(np.mean(np.mean(np.stack(loss_list, axis=0), axis=0), axis=0),
                            n=args.batch_size * args.world_size)
        else:
            run_loss.update(loss.item(), n=args.batch_size)
        if args.rank == 0:
            print('Epoch {}/{} {}/{}'.format(epoch, args.max_epochs, idx, len(loader)),
                  'loss: {:.4f}'.format(run_loss.avg),
                  'Train_acc',avg_acc,
                  'time {:.2f}s'.format(time.time() - start_time))
        start_time = time.time()
    for param in model.parameters() : param.grad = None

    trainloss_acc = [run_loss.avg,avg_acc]
    return trainloss_acc

def val_epoch(model,
              loader,
              epoch,
              acc_func,
              args,
              model_inferer=None,
              post_label=None,
              post_pred=None):
    model.eval()
    start_time = time.time()
    with torch.no_grad():
        acc_epoch = 0
        for idx, batch_data in enumerate(loader):
            if isinstance(batch_data, list):
                data, target = batch_data
            else:
                data, target = batch_data['image'], batch_data['label']


            # print("LLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLL")
            # print(target.shape)
            # print("BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBs")
            # print(np.unique(target))


            data, target = data.cuda(args.rank), target.cuda(args.rank)

            # t1 = target.cpu().numpy()
            # print("YYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYY")
            # print(np.unique(t1))

            with autocast(enabled=args.amp):
                if model_inferer is not None:
                    logits = model_inferer(data)
                else:
                    logits = model(data)
            if not logits.is_cuda:
                target = target.cpu()
            val_labels_list = decollate_batch(target)
            val_labels_convert = [post_label(val_label_tensor) for val_label_tensor in val_labels_list]
            val_outputs_list = decollate_batch(logits)
            val_output_convert = [post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list]
            acc = acc_func(y_pred=val_output_convert, y=val_labels_convert)
            acc = acc.cuda(args.rank)

            if args.distributed:
                acc_list = distributed_all_gather([acc],
                                                  out_numpy=True,
                                                  is_valid=idx < loader.sampler.valid_length)
                avg_acc = np.mean([np.nanmean(np.delete(l,0)) for l in acc_list])
                acc_epoch += avg_acc

              

            else:
                acc_list = acc.detach().cpu().numpy()
                avg_acc = np.mean([np.nanmean(np.delete(l,0)) for l in acc_list])
                acc_epoch += avg_acc

                # s1 =[np.delete(l,0) for l in acc_list]
                # s2 = [l for l in acc_list]
                # print("111111111111111111111111111111")
                # print(s1)
                # print("22222222222222222222222222222")
                # print(s2)
            # print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@22@2")
            # print(acc_list)
            # print("999999999999999999999999999999999999999999999999999")
            # print(s)

            if args.rank == 0:
                print('Val {}/{} {}/{}'.format(epoch, args.max_epochs, idx, len(loader)),
                      'Validation_acc', avg_acc,
                      'time {:.2f}s'.format(time.time() - start_time))
            start_time = time.time()
            
        avg_acc_epoch = acc_epoch/5

    return avg_acc_epoch

def save_checkpoint(model,
                    epoch,
                    args,
                    filename='model.pt',
                    best_acc=0,
                    optimizer=None,
                    scheduler=None):
    state_dict = model.state_dict() if not args.distributed else model.module.state_dict()
    save_dict = {
            'epoch': epoch,
            'best_acc': best_acc,
            'state_dict': state_dict
            }
    if optimizer is not None:
        save_dict['optimizer'] = optimizer.state_dict()
    if scheduler is not None:
        save_dict['scheduler'] = scheduler.state_dict()
    filename=os.path.join(args.logdir, filename)
    torch.save(save_dict, filename)
    print('Saving checkpoint', filename)

def run_training(model,
                 train_loader,
                 val_loader,
                 optimizer,
                 loss_func,
                 acc_func,
                 args,
                 model_inferer=None,
                 scheduler=None,
                 start_epoch=0,
                 post_label=None,
                 post_pred=None
                 ):
    writer = None
    if args.logdir is not None and args.rank == 0:
        writer = SummaryWriter(log_dir=args.logdir)
        if args.rank == 0: print('Writing Tensorboard logs to ', args.logdir)
    scaler = None
    if args.amp:
        scaler = GradScaler()
    val_acc_max = 0.

    

    for epoch in range(start_epoch, args.max_epochs):
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)
            torch.distributed.barrier()
        print(args.rank, time.ctime(), 'Epoch:', epoch)
        epoch_time = time.time()

        train = train_epoch(model,
                                 train_loader,
                                 optimizer,
                                 scaler=scaler,
                                 epoch=epoch,
                                 acc_func=acc_func,
                                 loss_func=loss_func,
                                 args=args,
                                 post_label=post_label,
                                 post_pred=post_pred
                                 )
    
        train_loss = train[0]
        train_acc = train[1]


        

        if args.rank == 0:
            print('Final training  {}/{}'.format(epoch, args.max_epochs - 1), 'loss: {:.4f}'.format(train_loss),'acc:{:.4f}'.format(train_acc))
                #   'time {:.2f}s'.format(time.time() - epoch_time))
        if args.rank==0 and writer is not None:
            writer.add_scalar('train_loss', train_loss, epoch)
        b_new_best = False
        if (epoch+1) % args.val_every == 0:
            if args.distributed:
                torch.distributed.barrier()
            epoch_time = time.time()
            val_avg_acc = val_epoch(model,
                                    val_loader,
                                    epoch=epoch,
                                    acc_func=acc_func,
                                    model_inferer=model_inferer,
                                    args=args,
                                    post_label=post_label,
                                    post_pred=post_pred)
            if args.rank == 0:
                print('Final validation  {}/{}'.format(epoch, args.max_epochs - 1),
                      'acc', val_avg_acc, 'time {:.2f}s'.format(time.time() - epoch_time))
                if writer is not None:
                    writer.add_scalar('val_acc', val_avg_acc, epoch)
                if val_avg_acc > val_acc_max:
                    print('new best ({:.6f} --> {:.6f}). '.format(val_acc_max, val_avg_acc))
                    val_acc_max = val_avg_acc
                    b_new_best = True
                    if args.rank == 0 and args.logdir is not None and args.save_checkpoint:
                        save_checkpoint(model, epoch, args,
                                        best_acc=val_acc_max,
                                        optimizer=optimizer,
                                        scheduler=scheduler)
            if args.rank == 0 and args.logdir is not None and args.save_checkpoint:
                save_checkpoint(model,
                                epoch,
                                args,
                                best_acc=val_acc_max,
                                filename='model_final.pt')
                if b_new_best:
                    print('Copying to model.pt new best model!!!!')
                    shutil.copyfile(os.path.join(args.logdir, 'model_final.pt'), os.path.join(args.logdir, 'model.pt'))

        if scheduler is not None:
            scheduler.step()

    print('Training Finished !, Best Accuracy: ', val_acc_max)

    return val_acc_max

