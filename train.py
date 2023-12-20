import os
import sys
import argparse
import json
import pathlib
from functools import partial
from tqdm import tqdm
import random
import numpy as np
import math
import warnings

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torch.utils.data import random_split

from vision_transformer import VisionTransformer, DINOHead, MultiCropWrapper
from util import DataAugmentationDINO, DPImageDataset, cosine_scheduler, get_world_size, restart_from_checkpoint, clip_gradients, cancel_gradients_last_layer
from loss import Loss
from evaluation import compute_embedding, compute_knn
warnings.filterwarnings('ignore')

os.environ['CUDA_BLOCK_LAUNCH'] = '0,1,2,3'
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
def train_fn(data_loader, teacher, student, criterion, optimizer, lr_scheduler, momentum_scheduler, wd_scheduler, epochs, fp16_scaler, device):
    student.train()
    criterion.train()
    losses = []
    tk0 = tqdm(data_loader, total=len(data_loader), ncols = 100)
    
    for step, (images, _) in enumerate(tk0):
        
        # Scheduler setting
        it = len(data_loader) * epochs + step
        for i, param_group in enumerate(optimizer.param_groups):
            param_group['lr'] = lr_scheduler[it]
            if i == 0:
                param_group['weight_decay'] = wd_scheduler[it]
                
        images = [img.to(device) for img in images]
        with torch.cuda.amp.autocast(fp16_scaler is not None):
            teacher_output = teacher(images[:2])  # only the 2 global views pass through the teacher
            student_output = student(images)
            loss = criterion(student_output, teacher_output)

        losses.append(loss)
        optimizer.zero_grad()
        param_norm = None
        
        if fp16_scaler is None:
            loss.backward()
            param_norms = clip_gradients(student, 2.0)
            cancel_gradients_last_layer(epochs, student, True)
            optimizer.step()
            tk0.set_postfix(loss = loss.item())
            
        else:
            fp16_scaler.scale(loss).backward()
            fp16_scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
            param_norms = clip_gradients(student, 2.0)
            cancel_gradients_last_layer(epochs, 
                                        student,
                                        True)
            fp16_scaler.step(optimizer)
            fp16_scaler.update()
            tk0.set_postfix(loss = loss.item())
            
        with torch.no_grad():
            m = momentum_scheduler[it]
            for student_ps, teacher_ps in zip(student.parameters(), teacher.parameters()):
                teacher_ps.data.mul_(m)
                teacher_ps.data.add_((1 - m) * student_ps.detach().data)
        
    avg_loss = sum(losses) / len(losses)
    tk0.set_postfix(avg_loss = avg_loss)
            
    return avg_loss

def eval_fn(train_loader, test_loader, student):
    student.eval()
    
    with torch.no_grad():
        # KNN
        current_acc = compute_knn(student.module.backbone,
                                    train_loader,
                                    test_loader)
            
    student.train()
    
    return current_acc
    
def main():
    parser = argparse.ArgumentParser(
        'DINO training CLI',
        formatter_class = argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('-b', '--batch_size', type = int, default = 64)
    parser.add_argument('--device', type = str, default = 'cuda')
    parser.add_argument('--momentum_teacher', type = float, default = 0.998)
    parser.add_argument('-c', '--n_crops', type = int, default = 2)
    parser.add_argument('-e', '--n_epochs', type = int, default = 300)
    parser.add_argument('-o', '--out_dim', type = int, default = 65341)
    parser.add_argument('-t', '--tensorboard_dir', type = str, default = './DINO_knn')
    parser.add_argument('--norm_last_layer', action = 'store_true')
    parser.add_argument('--teacher_temp', type = float, default = 0.04)
    parser.add_argument('--student_temp', type = float, default = 0.1)
    parser.add_argument('--pretrained', action = 'store_true')
    parser.add_argument('-w', '--weight_decay', type = float, default = 0.6)
    parser.add_argument('--use_fp16', type = bool, default = True)
    
    args = parser.parse_args()
    seed_everything(seed=42)
    
    # Parameters
    device = torch.device(args.device)
    num_workers = 4
    
    # Data related
    transform_aug = DataAugmentationDINO(size = 256, global_crops_scale = (0.4, 1), local_crops_scale = (0.05, 0.4), local_crops_number = args.n_crops - 2)
    transforms_plain = transforms.Compose(
        [
            transforms.Resize((256,256)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ]
    )
    
    dataset_pretrain = DPImageDataset('./pretrain_df.csv', transform = transform_aug)
    dataset_train_plain = ImageFolder('./data', transform = transforms_plain)
    print(f"Total {len(dataset_pretrain)} pretrain dataset are ready.")
    
    
    train_size = int(0.8 * len(dataset_train_plain))
    test_size = len(dataset_train_plain) - train_size
    train_dataset, test_dataset= random_split(dataset_train_plain, [train_size, test_size])
    print(f"Total {len(train_dataset)} of train images and {len(test_dataset)} of validation images are ready.")
    
    pretrain_loader = DataLoader(dataset_pretrain, 
                                  batch_size = args.batch_size, 
                                  shuffle = True, 
                                  drop_last = True, 
                                  num_workers = num_workers, 
                                  pin_memory = True
                                  )
    
    train_loader = DataLoader(train_dataset, 
                              batch_size = args.batch_size, 
                              shuffle=True, 
                              drop_last = False,
                              num_workers = num_workers
                              )
    
    test_loader = DataLoader(test_dataset, 
                             batch_size = args.batch_size, 
                             shuffle = False,
                             drop_last = False, 
                             num_workers = num_workers,
                             )
    
    # Neural network related
    patch_size = 8
    img_size = 256
    embed_dim = 384
    num_heads = 6
    depth = 12
    student_vit = VisionTransformer(patch_size=patch_size,
                                    img_size=img_size, 
                                    embed_dim=embed_dim, 
                                    depth=depth, 
                                    num_heads=num_heads, 
                                    mlp_ratio=4, 
                                    qkv_bias=True, 
                                    norm_layer=partial(nn.LayerNorm, 
                                                       eps=1e-6))
    teacher_vit = VisionTransformer(patch_size=patch_size,
                                    img_size=img_size, 
                                    embed_dim=embed_dim, 
                                    depth=depth, 
                                    num_heads=num_heads, 
                                    mlp_ratio=4, 
                                    qkv_bias=True, 
                                    norm_layer=partial(nn.LayerNorm, 
                                                       eps=1e-6))
    
    student = MultiCropWrapper(student_vit, DINOHead(embed_dim, args.out_dim, norm_last_layer = args.norm_last_layer))
    teacher = MultiCropWrapper(teacher_vit, DINOHead(embed_dim, args.out_dim))
    
    student, teacher = student.to(device), teacher.to(device)
    student, teacher = torch.nn.DataParallel(student), torch.nn.DataParallel(teacher)
    
    print('Teacher, Student models are ready.')
    
    # Freezing teacher's parameters
    for p in teacher.parameters():
        p.requires_grad = False
        
    teacher.eval()
    
    fp16_scaler = None
    if args.use_fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()
    
    # Loss related
    dino_loss = Loss(args.out_dim,
                     teacher_temp = args.teacher_temp,
                     student_temp = args.student_temp,
                     ).to(device)
    lr = 0.0001
    min_lr = 1e-8
    optimizer = torch.optim.AdamW(student.parameters(),
                                  lr = lr,
                                  weight_decay = args.weight_decay)
    
    # Scheduler setting
    lr_scheduler = cosine_scheduler(lr * (args.batch_size) / 256.,
                                    min_lr, args.n_epochs, len(pretrain_loader))
    momentum_scheduler = cosine_scheduler(args.momentum_teacher, 1, args.n_epochs, len(pretrain_loader))
    wd_scheduler = cosine_scheduler(0.3, args.weight_decay, args.n_epochs, len(pretrain_loader))
    
    print("Loss, optimizer and schedulers ready.")
    
    # Checkpointer settings.
    to_restore = {"epoch": 0}
    restart_from_checkpoint(
        './DINO_knn/chechpoint/checkpoint.pth',
        run_variables = to_restore,
        student = student,
        teacher = teacher,
        optimizer = optimizer,
        loss = dino_loss,
    )
    start_epoch = to_restore['epoch']
    
    
    # Training loop
    print('Start Training the models!!!')
    best_acc = 0
    best_loss = None
    for e in range(start_epoch, args.n_epochs):
        avg_loss = train_fn(pretrain_loader,teacher, student, dino_loss, optimizer, lr_scheduler, momentum_scheduler, wd_scheduler, e, fp16_scaler, device)
        
        save_dict = {
            'student': student.state_dict(),
            'teacher':teacher.state_dict(),
            'optimizer':optimizer.state_dict(),
            'epoch':e + 1,
            'dino_loss': dino_loss.state_dict()
        }
        if e%5 == 0:
            print(f"Saving checkpoint of Epoch {e + 1}.")
            torch.save(save_dict, f'./DINO_knn/checkpoint/checkpoint.pth')
            
        if best_loss is None:
            print("best_loss start! Make a new best_loss variables.")
            best_loss = avg_loss
            print('First Checkpoint.')
            torch.save(save_dict, f'./DINO_knn/checkpoint/checkpoint.pth')
            
        if avg_loss < best_loss:
            best_loss = avg_loss
            print(f"Training loss is much improved. Saving model.......{e+1}")
            torch.save(student, "./DINO_knn/checkpoint/best_model.pth")
            
        current_acc= eval_fn(train_loader, test_loader, student)
        if current_acc > best_acc:
            best_acc = current_acc
            
        print(f'|Epoch {e+1}| current_acc : {current_acc} , best accuracy : {best_acc}, Current Loss : {avg_loss} , Best Loss : {best_loss}')
        
        if not os.path.exists('./DINO_knn/log.txt'):
            with open('./DINO_knn/log.txt','w', encoding = 'utf-8') as f:
                f.write(f'Epoch : {e + 1}, current_acc : {current_acc} , best accuracy : {best_acc}, Current Loss : {avg_loss} , Best Loss : {best_loss}\n')
                f.close()
        else:
            with open('./DINO_knn/log.txt','a', encoding = 'utf-8') as f:
                f.write(f'Epoch : {e + 1}, current_acc : {current_acc} , best accuracy : {best_acc}, Current Loss : {avg_loss} , Best Loss : {best_loss}\n')
                f.close()
        
        if not math.isfinite(avg_loss):
            print(f"Loss is {avg_loss}, stopping training.", force = True)
            sys.exit(1)
            
if __name__=="__main__":
    main()