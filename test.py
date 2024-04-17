import os
import cv2
import tensorflow as tf
import numpy as np
import scipy.io as io
import torch
from skimage.metrics import mean_squared_error as compare_mse
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import normalized_root_mse as compare_nrmse
from config.config import get_arguments

from skimage.metrics import structural_similarity as compare_ssim
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.pet_dataset import PetDataset
from model.model import InvISPNet

os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
os.environ['CUDA_VISIBLE_DEVICES'] = str(np.argmax([int(x.split()[2]) for x in open('tmp', 'r').readlines()]))
# os.environ['CUDA_VISIBLE_DEVICES'] = "1"
os.system('rm tmp')

parser = get_arguments()
parser.add_argument("--ckpt", type=str, help="Checkpoint path.")
parser.add_argument("--out_path", type=str, default="./exps/fdg_fmz_zubal_head_sample3/",
                    help="Path to save results. ")
parser.add_argument("--root1", type=str, default="./data/fdg_fmz_zubal_head_sample3/train", help="train images. ")
parser.add_argument("--root2", type=str, default="./data/fdg_fmz_zubal_head_sample3/test", help="test images. ")
parser.add_argument("--root3", type=str, default="./data/fdg_fmz_zubal_head_sample3/valid", help="valid images. ")
args = parser.parse_args()
print("Parsed arguments: {}".format(args))

ckpt_allname = args.ckpt.split("/")[-1]


def compare_ms_ssim(pred, target):
    pred = tf.convert_to_tensor(pred)
    pred = tf.expand_dims(pred, axis=-1)

    target = tf.convert_to_tensor(target)
    target = tf.expand_dims(target, axis=-1)

    max_val = 1

    result = tf.image.ssim_multiscale(pred, target, max_val, filter_size=1)
    result = result.numpy()
    return result


def compare_psnr_show_save(PSNR, SSIM, MSE, MS_SSIM, NRMSE, show_name, save_path):
    ave_psnr = sum(PSNR) / len(PSNR)
    PSNR_std = np.std(PSNR)

    ave_ssim = sum(SSIM) / len(SSIM)
    SSIM_std = np.std(SSIM)

    ave_ms_ssim = sum(MS_SSIM) / len(MS_SSIM)
    MS_SSIM_std = np.std(MS_SSIM)

    ave_mse = sum(MSE) / len(MSE)
    MSE_std = np.std(MSE)

    ave_nrmse = sum(NRMSE) / len(NRMSE)
    NRMSE_std = np.std(NRMSE)

    print('ave_psnr_' + show_name, ave_psnr)
    print('ave_ssim_' + show_name, ave_ssim)
    print('ave_ms_ssim_' + show_name, ave_ms_ssim)
    print('ave_mse_' + show_name, ave_mse)
    print('ave_nrmse_' + show_name, ave_nrmse)

    file_name = 'results_test.txt'
    file_path = os.path.join(save_path, file_name)
    with open(file_path, 'a+') as f:
        f.write('\n' * 3)
        f.write(ckpt_allname + '_' + show_name + '\n')

        f.write('ave_psnr:' + str(ave_psnr) + ' ' * 3 + 'PSNR_std:' + str(PSNR_std) + '\n')

        f.write('ave_ssim:' + str(ave_ssim) + ' ' * 3 + 'SSIM_std:' + str(SSIM_std) + '\n')

        f.write('ave_ms_ssim:' + str(ave_ms_ssim) + ' ' * 3 + 'SSIM_std:' + str(MS_SSIM_std) + '\n')

        f.write('ave_mse:' + str(ave_mse) + ' ' * 3 + 'MSE_std:' + str(MSE_std) + '\n')

        f.write('ave_nrmse:' + str(ave_nrmse) + ' ' * 3 + 'nrmse_std:' + str(NRMSE_std) + '\n')


def save_img(img, img_path):
    img = np.clip(img * 255, 0,
                  255)  
    inverted_img = 255 - img  

    cv2.imwrite(img_path, inverted_img)


def main(args):
    # ======================================define the model============================================
    net = InvISPNet(channel_in=36, channel_out=36, block_num=8)
    device = torch.device("cuda:0")
    net.to(device) 
    net.eval()
    # load the pretrained weight if there exists one
    if os.path.isfile(args.ckpt):
        net.load_state_dict(torch.load(args.ckpt), strict=False)  # 加载模型参数
        print("[INFO] Loaded checkpoint: {}".format(args.ckpt))

    print("[INFO] Start data load and preprocessing")

    test_dataset = PetDataset(root_folder=args.root2)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0,
                                 drop_last=True)

    PSNR = []
    SSIM = []
    MS_SSIM = []
    MSE = []
    NRMSE = []

    PSNR_FMZ = []
    SSIM_FMZ = []
    MS_SSIM_FMZ = []
    MSE_FMZ = []
    NRMSE_FMZ = []

    save_path = 'exps/fdg_fmz_zubal_head_sample3/test/{}'.format(ckpt_allname)
    
    print("[INFO] Start test...")
    for i_batch, (hybrid_batch, fdg_batch, fmz_batch) in enumerate(
            tqdm(test_dataloader)):

        hybrid_input = hybrid_batch.permute(0, 3, 1,
                                            2).float().cuda()  
        fdg_label = fdg_batch.float().cuda()  
        fmz_label = fmz_batch.float().cuda()
        hybrid_input = hybrid_input.repeat(1, 2, 1, 1)
        with torch.no_grad():  
            reconstruct_for = net(hybrid_input)  

        pre_fdg = (reconstruct_for[:, 0:18, :, :]).squeeze()
        pre_fmz = (reconstruct_for[:, 18:36, :, :]).squeeze()
        pred_for_fdg = pre_fdg.squeeze().permute(1, 2, 0).cpu().numpy()  
        pred_for_fmz = pre_fmz.squeeze().permute(1, 2, 0).cpu().numpy()  

        target_fdg_label = fdg_label.squeeze().cpu().numpy()
        target_fmz_label = fmz_label.squeeze().cpu().numpy()

        for i in range(args.frame):
            psnr = compare_psnr(255 * abs(target_fdg_label[:, :, i]), 255 * abs(pred_for_fdg[:, :, i]),
                                data_range=255)  
            ssim = compare_ssim(abs(target_fdg_label[:, :, i]), abs(pred_for_fdg[:, :, i]), data_range=1)
            ms_ssim = compare_ms_ssim(abs(target_fdg_label[:, :, i]), abs(pred_for_fdg[:, :, i]))
            mse = compare_mse(abs(target_fdg_label[:, :, i]), abs(pred_for_fdg[:, :, i]))
            if not np.all(target_fdg_label[:, :, i] == 0):
                nrmse = compare_nrmse(abs(target_fdg_label[:, :, i]), abs(pred_for_fdg[:, :, i]))
            else:
                nrmse = 0

            if i > 7:  # 前6帧为空
                PSNR.append(psnr)
                SSIM.append(ssim)
                MS_SSIM.append(ms_ssim)
                MSE.append(mse)
                NRMSE.append(nrmse)

            psnr_fmz = compare_psnr(255 * abs(target_fmz_label[:, :, i]), 255 * abs(pred_for_fmz[:, :, i]),
                                    data_range=255)  
            ssim_fmz = compare_ssim(abs(target_fmz_label[:, :, i]), abs(pred_for_fmz[:, :, i]), data_range=1)
            mse_fmz = compare_mse(abs(target_fmz_label[:, :, i]), abs(pred_for_fmz[:, :, i]))
            ms_ssim_fmz = compare_ms_ssim(abs(target_fmz_label[:, :, i]), abs(pred_for_fmz[:, :, i]))
            if not np.all(target_fmz_label[:, :, i] == 0):
                nrmse_fmz = compare_nrmse(abs(target_fmz_label[:, :, i]), abs(pred_for_fmz[:, :, i]))
            else:
                nrmse_fmz = 0

            PSNR_FMZ.append(psnr_fmz)
            SSIM_FMZ.append(ssim_fmz)
            MS_SSIM_FMZ.append(ms_ssim_fmz)
            MSE_FMZ.append(mse_fmz)
            NRMSE_FMZ.append(nrmse_fmz)

           
            os.makedirs(save_path + '/pred_fdg',exist_ok=True)  
            os.makedirs(save_path + '/pred_fdg_mat', exist_ok=True)
            os.makedirs(save_path + '/pred_fmz', exist_ok=True)
            os.makedirs(save_path + '/pred_fmz_mat', exist_ok=True)
            os.makedirs(save_path + '/pred_fdg',exist_ok=True) 
            os.makedirs(save_path + '/pred_hybrid_mat', exist_ok=True)
           
           
            save_img(pred_for_fdg[:, :, i],
                     save_path + '/pred_fdg' + '/pred_fdg_' + str(i_batch + 1) + '_' + str(i + 1) + '.png')
            save_img(pred_for_fmz[:, :, i],
                     save_path + '/pred_fmz' + '/pred_fmz_' + str(i_batch + 1) + '_' + str(i + 1) + '.png')
            save_img(pred_for_fmz[:, :, i] + pred_for_fdg[:, :, i],
                     save_path + '/pred_hybrid' + '/pred_hybrid_' + str(i_batch + 1) + '_' + str(i + 1) + '.png')

            io.savemat(save_path + '/pred_fdg_mat' + '/pred_fdg_' + str(i_batch + 1) + '_' + str(i + 1) + '.mat',
                       {'data': pred_for_fdg[:, :, i]})
            io.savemat(save_path + '/pred_fmz_mat' + '/pred_fmz_' + str(i_batch + 1) + '_' + str(i + 1) + '.mat',
                       {'data': pred_for_fmz[:, :, i]})
            io.savemat(save_path + '/pred_hybrid_mat' + '/pred_hybrid_' + str(i_batch + 1) + '_' + str(i + 1) + '.mat',
                       {'data': pred_for_fmz[:, :, i] + pred_for_fdg[:, :, i]})

          
        del pre_fdg
        del pre_fmz
    compare_psnr_show_save(PSNR, SSIM, MSE, MS_SSIM, NRMSE, "fdg", save_path)
    compare_psnr_show_save(PSNR_FMZ, SSIM_FMZ, MSE_FMZ, MS_SSIM_FMZ, NRMSE_FMZ,
                           "fmz", save_path)


if __name__ == '__main__':
    torch.set_num_threads(4)
    main(args)
