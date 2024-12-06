from utils.data import *
import torch
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
import os
from PIL import Image

def test(opts):

# Initialization
    import_module = 'from lib.{}.{} import {}'.format(opts.Model.file_name, opts.Model.file_name, opts.Model.name)
    exec(import_module)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = eval(opts.Model.name)().to(device)
    datasets = opts.Test.Dataset.testsets

# Load Model
    if opts.Test.TestEpochNum is None:
        model_path = '{}/{}.pt'.format(opts.Model.save_dir, opts.Model.name)
    else:
        model_path = '{}/{}_{}.pt'.format(opts.Model.save_dir, opts.Model.name, opts.Test.TestEpochNum)

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['net'])

# Test
    model.eval()
    for dataset in datasets:
        test_data_dir = "{}/{}".format(opts.Test.Dataset.root, dataset)
        test_dataset = eval(opts.Test.Dataset.type)(test_data_dir,
                                       transforms=model.transforms)
        test_loader = DataLoader(dataset=test_dataset,
                                 batch_size=1,
                                 shuffle=opts.Test.DataLoader.shuffle,
                                 num_workers=opts.Test.DataLoader.num_workers)

        save_dir = '{}/results/{}'.format(opts.Model.save_dir, dataset)
        if os.path.exists(save_dir) is False:
            os.makedirs(save_dir)

        with torch.no_grad():
            for i, data_ in enumerate(test_loader):
                images, masks, shape = data_
                w_, h_ = shape
                images = images.to(device)
                masks = masks.to(device)
                sample = {'images': images, 'masks': masks}
                output = model.forward(sample)

                pre = torch.sigmoid(output['prediction']).to('cpu')
                pre = pre.squeeze()
                pre = (pre - pre.min()) / (pre.max() - pre.min() + 1e-8)
                pre = pre * 255

                result_map = torch.stack((pre, pre, pre), 0).permute(1, 2, 0).numpy().astype("uint8")
                result_map = Image.fromarray(result_map)
                result_map = result_map.resize((w_, h_), Image.BILINEAR)
                map_name = test_dataset.name[i]
                save_path = os.path.join(save_dir, map_name)
                result_map.save(save_path)


