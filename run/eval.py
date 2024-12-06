import utils.data
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
import torch
import numpy as np
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
from utils.metric import cal_mae, cal_fm, cal_sm, cal_em, cal_wfm, cal_dice, cal_iou, cal_ber, cal_acc
import torch.nn.functional as F
from utils.test_data import test_dataset
import logging

def evaluate(opts, log=False, epoch=None):
    datasets = opts.Test.Dataset.testsets

    for dataset in datasets:
        result_dir = '{}/results/{}'.format(opts.Model.save_dir, dataset)
        mask_dir = "{}/{}/masks".format(opts.Test.Dataset.root, dataset)
        test_loader = test_dataset(result_dir, mask_dir)
        mae, fm, sm, em, wfm, m_dice, m_iou, ber, acc = cal_mae(), cal_fm(
            test_loader.size), cal_sm(), cal_em(), cal_wfm(), cal_dice(), cal_iou(), cal_ber(), cal_acc()

        for i in range(test_loader.size):
            # print('predicting for %d / %d' % (i + 1, test_loader.size))
            sal, gt = test_loader.load_data()

            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)

            # or
            # gt /= 255
            # gt[gt > 0.5] = 1
            # gt[gt != 1] = 0
            res = np.asarray(sal, np.float32)
            res /= 255
            mae.update(res, gt)
            sm.update(res, gt)
            fm.update(res, gt)
            em.update(res, gt)
            wfm.update(res, gt)
            m_dice.update(res, gt)
            m_iou.update(res, gt)
            ber.update(res, gt)
            acc.update(res, gt)
        MAE = mae.show()
        maxf, meanf, _, _ = fm.show()
        sm = sm.show()
        em_, max_em = em.show()
        wfm = wfm.show()
        m_dice = m_dice.show()
        m_iou = m_iou.show()
        ber = ber.show()
        acc = acc.show()
        if log:
            if epoch is not None:
                logging.info('epoch: {}, dataset: {}, MAE: {:.4f}, m_dice: {:.4f}, miou: {:.4f}'.format(epoch, dataset, MAE, m_dice, m_iou))
            else:
                print('epoch is none!')

        # print('{}:\nMAE:{:.4f},m_dice:{:.4f},miou:{:.4f}'.format(dataset, MAE, m_dice, m_iou))
        print('dataset: {}\n {:.3f}   {:.3f}  {:.3f}   {:.3f}  {:.3f} {:.3f} '.format(
            dataset, m_dice, m_iou, wfm, sm, em_, MAE))

def evaluate_train(model, opts, log=False, epoch=None):
# Initialization
    model.eval()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    datasets = opts.Test.Dataset.testsets

# evaluate

    for dataset in datasets:

        image_dir = "{}/{}/images".format(opts.Test.Dataset.root, dataset)
        mask_dir = "{}/{}/masks".format(opts.Test.Dataset.root, dataset)
        test_loader = test_dataset(image_dir, mask_dir, mode='evaluate_train')
        mae, fm, sm, em, wfm, m_dice, m_iou, ber, acc = cal_mae(), cal_fm(
            test_loader.size), cal_sm(), cal_em(), cal_wfm(), cal_dice(), cal_iou(), cal_ber(), cal_acc()

        with torch.no_grad():
            for i in range(test_loader.size):
                # print('predicting for %d / %d' % (i + 1, test_loader.size))
                image, gt, trans_gt = test_loader.load_data()
                w_, h_ = gt.size
                image = image.to(device)
                trans_gt = trans_gt.cuda()
                sample = {'images': image, 'masks': trans_gt}
                output = model.forward(sample)
                pre = F.upsample(output['prediction'], size=(h_, w_), mode='bilinear', align_corners=False)
                pre = torch.sigmoid(pre).to('cpu').numpy().squeeze()
                res = (pre - pre.min()) / (pre.max() - pre.min() + 1e-8)

                gt = np.asarray(gt, np.float32)
                gt /= (gt.max() + 1e-8)

                mae.update(res, gt)
                m_dice.update(res, gt)
                m_iou.update(res, gt)

        MAE = mae.show()
        m_dice = m_dice.show()
        m_iou = m_iou.show()
        if log:
            if epoch is not None:
                logging.info('epoch: {}, dataset: {}, MAE: {:.4f}, m_dice: {:.4f}, miou: {:.4f}'.format(epoch, dataset, MAE, m_dice, m_iou))
            else:
                print('epoch is none!')

        print(f'{dataset}: m_dice: {m_dice:.4f}, miou: {m_iou:.4f}, MAE: {MAE:.4f}')

