import os
import argparse
import torch
from networks.vnet_sdf_GELU import VNet

from monai.inferers import sliding_window_inference
import h5py
from dataloaders.la_heart import LAHeart, RandomCrop, CenterCrop, RandomRotFlip, ToTensor, TwoStreamBatchSampler
from torch.utils.data import DataLoader
from monai import transforms, data
import numpy as np
import nibabel as nib
from monai.losses import DiceLoss, DiceCELoss


parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='/home/maoyuejingxian/DTC/code/data0/logic', help='Name of Experiment')
parser.add_argument('--model', type=str,
                    default='iter_4000', help='model_name')
parser.add_argument('--gpu', type=str,  default='1', help='GPU to use')
parser.add_argument('--detail', type=int,  default=1,
                    help='print metrics for every samples?')
parser.add_argument('--nms', type=int, default=1,
                    help='apply NMS post-procssing?')


print(torch.cuda.device_count())
FLAGS = parser.parse_args()
train_data_path = FLAGS.root_path

os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
snapshot_path = '/home/maoyuejingxian/DTC/model/LA_gelu_supervised/DTC_with_consis_weight_20labels_beta_0.3/'

num_classes = 4

test_save_path = os.path.join(snapshot_path, "test/")
if not os.path.exists(test_save_path):
    os.makedirs(test_save_path)
print(test_save_path)
with open(FLAGS.root_path + '/test.list', 'r') as f:
    image_list = f.readlines()
image_list = [FLAGS.root_path + "/" + item.replace('\n', '') + "/h5h5.h5" for item in
              image_list]


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





def save_nifti(data,save_path):
        data_ = nib.load("/home/maoyuejingxian/UNETR_semicon/dataset/dataset0/imagesTs/zdb_memory_d1_b2_s1_2_void.nii.gz")
        header = data_.header
        nifti_image = nib.Nifti1Image(data,None,header)
        nib.save(nifti_image,"/home/maoyuejingxian/dtc_visual/logic/"+save_path)
        print('save file sucess')



def test_calculate_metric():
    net = VNet(n_channels=1, n_classes=num_classes,
               normalization='batchnorm', has_dropout=False).cuda()
    
    save_mode_path = os.path.join(
        snapshot_path, 'iter_2000.pth')
    net.load_state_dict(torch.load(save_mode_path))
    print("init weight from {}".format(save_mode_path))
    net.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)
    
    db_test = LAHeart(base_dir=train_data_path,
                       split='test',  # train/val split
                       transform=transforms.Compose([
                        #    RandomRotFlip(),
                        #    RandomCrop(patch_size),
                           ToTensor(),
                       ]))

    testloader = DataLoader(db_test, num_workers=4, pin_memory=True)


    metric_detail=FLAGS.detail
    val_loader = tqdm(image_list) if not metric_detail else image_list


    with torch.no_grad():
        dice_list_case = []
        dice1 = []
        dice2 = []
        dice3 = []
        dice4 = []
        for i, batch in enumerate(testloader):
            # print(type(np.unique(batch["label"])[0]))
            # print(np.unique(batch["label"]))
            val_inputs, val_labels = (batch["images"].cuda(), batch["labels"].cuda())

            # test_val_labels = val_labels.cpu().numpy()
            # test_val_labels2 = batch['label']

            # # print(type(np.unique(test_val_labels)[0]))
            # print("***************************************************")
            # print(np.unique(test_val_labels))


            # img_name = batch['image_meta_dict']['filename_or_obj'][0].split('/')[-1]
            # print("Inference on case {}".format(img_name))


            # val_outputs = sliding_window_inference(val_inputs,
            #                                        (96, 96, 96),
            #                                        4,
            #                                        net,
            #                                        overlap=0.5)

            other, val_outputs = net(val_inputs)
            '''print("22222222222222")
            print(val_outputs.shape)'''

            #val_labels = torch.unsqueeze(val_labels,1)
	
            print(val_outputs.shape)


            val_outputs = torch.softmax(val_outputs, 1).cpu().numpy()
            val_outputs = np.argmax(val_outputs, axis=1).astype(np.uint8)
            # print(val_labels.shape)

            print(val_labels.shape)
            #val_labels = val_labels.cpu().numpy()[:, 0, :, :, :].round().astype(np.uint8)
            val_labels = val_labels.cpu().numpy().round().astype(np.uint8)
            #print(np.unique(val_labels))
            #print(np.unique(val_outputs))
            print(val_labels.shape)
            #print(val_outputs.shape)





            # print(val_outputs.shape)
            # print(val_labels.shape)

            # print("RRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRR")
            # print(np.unique(val_labels))
            # print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
            # print(np.unique(val_outputs))


            # organ_Dice = dice(val_outputs[0] == 1, val_labels[0] == 1)


            image_data = val_outputs.reshape(96,96,96)
            s = str(i)
            # save_nifti(image_data,s)

            dice_list_sub = []
            for i in range(0, 4):

                organ_Dice = dice(val_outputs[0] == i, val_labels[0] == i)
                # print("dice for {}".format(i))
                # print(organ_Dice)
                # dice_list_sub.append(organ_Dice)
            
                if i!=0:
                    print("dice for {}".format(i))
                    print(organ_Dice)
                    dice_list_sub.append(organ_Dice)

                    if i == 1:
                        dice1.append(organ_Dice)

                    if i == 2:
                        dice2.append(organ_Dice)

                    if i == 3:
                        dice3.append(organ_Dice)

                    if i == 4:
                        dice4.append(organ_Dice)

            mean_dice = np.mean(dice_list_sub)
            print("Mean Dice: {}".format(mean_dice))
            dice_list_case.append(mean_dice)

            mean_dice1 = np.mean(dice1)
            mean_dice2 = np.mean(dice2)
            mean_dice3 = np.mean(dice3).round(4)
            mean_dice4 = np.mean(dice4)
        print("Dice1_average: {}".format(mean_dice1))
        print("Dice2_average: {}".format(mean_dice2))
        print("Dice3_average: {}".format(mean_dice3))
        print("Dice4_average: {}".format(mean_dice4))

        print("Overall Mean Dice: {}".format(np.mean(dice_list_case)))
    
    
if __name__ == '__main__':
    test_calculate_metric()
	 

