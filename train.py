import argparse
import os
import time

import numpy as np
import torch.nn.functional as F
import torch
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from dataset.pet_dataset import PetDataset
from torch.optim import lr_scheduler
from config.config import get_arguments

from model.model import InvISPNet

os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
os.environ['CUDA_VISIBLE_DEVICES'] = str(np.argmax([int(x.split()[2]) for x in open('tmp', 'r').readlines()]))
# os.environ['CUDA_VISIBLE_DEVICES'] = "1"
os.system('rm tmp')

parser = get_arguments()
parser.add_argument("--out_path", type=str, default="./exps/ours_head_sample3_36channel_loss/",help="Path to save checkpoint. ")
parser.add_argument("--root1", type=str, default="./data/fdg_fmz_zubal_head_sample3/train", help="train images. ")
parser.add_argument("--root2", type=str, default="./data/fdg_fmz_zubal_head_sample3/test", help="test images. ")
parser.add_argument("--resume", dest='resume', action='store_true', help="Resume training. ")
parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate")
args = parser.parse_args()
print("Parsed arguments: {}".format(args))

os.makedirs(args.out_path, exist_ok=True)
os.makedirs(args.out_path + "/checkpoint", exist_ok=True)


def main(args):
    # ======================================define the model======================================
    # 创建SummaryWriter对象，将日志记录到指定路径
    writer = SummaryWriter(args.out_path)

    net = InvISPNet(channel_in=36, channel_out=36, block_num=8)

    # initialize_weights_xavier(net, scale=1)  
    net.cuda()

    # load the pretrained weight if there exists one
    if args.resume:
        net.load_state_dict(torch.load(args.out_path + "/checkpoint/latest.pth"))
        print("[INFO] loaded " + args.out_path + "/checkpoint/latest.pth")

    # 优化器
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)  
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[50, 100, 150, 200, 250, 500],
                                         gamma=0.5)
    print("[INFO] Start data loading and preprocessing")

    train_dataset = PetDataset(root_folder=args.root1)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0,
                                  drop_last=True)

    print("[INFO] Start to train")
    step = 0


    for epoch in range(0, 1000):

        for i_batch, (hybrid_batch, fdg_batch, fmz_batch) in enumerate(train_dataloader):
            step_time = time.time()

            hybrid_input = hybrid_batch.permute(0, 3, 1,
                                                2).float().cuda()  
            target_forward_fdg = fdg_batch.permute(0, 3, 1, 2).float().cuda()  
            target_forward_fmz = fmz_batch.permute(0, 3, 1, 2).float().cuda()  
            hybrid_input = hybrid_input.repeat(1, 2, 1, 1)

            reconstruct_for = net(hybrid_input)  

            reconstruct_for_m_fdg = (reconstruct_for[:, 0:18, :, :]).squeeze()
            reconstruct_for_m_fmz = (reconstruct_for[:, 18:36, :, :]).squeeze()

           
            forward_loss_fdg = F.mse_loss(reconstruct_for_m_fdg, target_forward_fdg.squeeze())
            forward_loss_fmz = F.mse_loss(reconstruct_for_m_fmz, target_forward_fmz.squeeze())
            forward_loss = (forward_loss_fdg + forward_loss_fmz) / 2
            writer.add_scalar('forward_loss', forward_loss.item(),
                              global_step=step)
            reconstruct_rev = net(reconstruct_for, rev=True)
            reconstruct_rev = torch.clamp(reconstruct_rev, 0, 1)

            
            rev_loss = F.mse_loss(reconstruct_rev, hybrid_input)
            writer.add_scalar('rev_loss', rev_loss.item(), global_step=step)
            
            loss = args.weight * forward_loss + rev_loss
            writer.add_scalar('loss', loss.item(),
                              global_step=step)  
            
            optimizer.zero_grad()  
            loss.backward()  
            optimizer.step()  

            print("Epoch: %d Step: %d || loss: %.10f rev_loss: %.10f forward_loss: %.10f  || lr: %f time: %f" % (
                epoch, step, loss.detach().cpu().numpy(), rev_loss.detach().cpu().numpy(),
                forward_loss.detach().cpu().numpy(), optimizer.param_groups[0]['lr'], time.time() - step_time
            ))

            step += 1

        if epoch % 1 == 0 or epoch % 999 == 0:
            torch.save(net.state_dict(), args.out_path + "/checkpoint/%04d.pth" % epoch)
            print("[INFO] Successfully saved " + args.out_path + "/checkpoint/%04d.pth" % epoch)
        scheduler.step()


if __name__ == '__main__':
    torch.set_num_threads(4)  
    main(args)
