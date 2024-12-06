import torch
from torch.utils.data import DataLoader
from PIL import Image
import os
from utils.metric import cal_mae, cal_fm, cal_sm, cal_em, cal_wfm, cal_dice, cal_iou, cal_ber, cal_acc
import numpy as np
from utils.test_data import test_dataset
from utils.data import *
from lib.RSAFormer.RSAFormer import RSAFormer
from utils.utils import *
import torch.nn.functional as F

def validation(model, opts):
# Initialization
    model.eval()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    dataset = opts.Validation.Dataset.testsets

    valid_data_dir = "{}/{}".format(opts.Validation.Dataset.root, dataset)

    valid_dataset = eval(opts.Validation.Dataset.type)(valid_data_dir,
                                                transforms=model.transforms)

    valid_loader = DataLoader(dataset=valid_dataset,
                              batch_size=1,
                              shuffle=opts.Validation.DataLoader.shuffle,
                              num_workers=opts.Validation.DataLoader.num_workers)

    save_dir = '{}/results/{}'.format(opts.Model.save_dir, dataset)

    if os.path.exists(save_dir) is False:
        os.makedirs(save_dir)

    mae, fm, sm, em, wfm, m_dice, m_iou, ber, acc = cal_mae(), cal_fm(
        len(valid_dataset)), cal_sm(), cal_em(), cal_wfm(), cal_dice(), cal_iou(), cal_ber(), cal_acc()

# Predict

    with torch.no_grad():
        for i, data_ in enumerate(valid_loader):
            images, masks, shape = data_
            w_, h_ = shape
            images = images.to(device)
            masks = masks.to(device)
            sample = {'images': images, 'masks': masks}
            output = model.forward(sample)
            pre = F.upsample(output['prediction'], size=(h_, w_), mode='bilinear', align_corners=False)
            pre = torch.sigmoid(pre).to('cpu')
            pre = pre.numpy().squeeze()
            pre = (pre - pre.min()) / (pre.max() - pre.min() + 1e-8)
            # pre = (pre > .5)

            mask_name = valid_dataset.name[i]
            mask_path = os.path.join(valid_data_dir, 'masks', mask_name)
            mask = Image.open(mask_path).convert('L')
            mask = np.asarray(mask, np.float32)
            mask /= (mask.max() + 1e-8)
            # mask[mask > 0.5] = 1
            # mask[mask != 1] = 0

            mae.update(pre, mask)
            m_dice.update(pre, mask)
            m_iou.update(pre, mask)


    MAE = mae.show()
    m_dice = m_dice.show()
    m_iou = m_iou.show()
    metrics = {'MAE': MAE, 'm_dice': m_dice, 'm_iou': m_iou}
    print('{}:\nMAE:{:.4f},m_dice:{:.4f},miou:{:.4f}'.format(dataset, MAE, m_dice, m_iou))
    return metrics

if __name__ == '__main__':
    model = RSAFormer().to('cuda')
    checkpoint = torch.load('../files/RSAFormer/RSAFormer.pt',
                            map_location="cuda0")
    model.load_state_dict(checkpoint['net'])
    opts = load_config('configs/RSAFormer.yaml')

    validation(model, opts)
