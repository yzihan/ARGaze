#!/usr/bin/env python3

import torch
from torch import nn,optim
from torch.autograd import Variable
import torch.utils.data as Data
import numpy as np
import os
import time
import math
import sys
import my_model
import utils
import copy


# machine-specific parameters
#BATCH_SIZE=10000

#BATCH_SIZE=1700 # model 0 for single 12GB GPU
#BATCH_SIZE=3500 # model 0 for double 12GB GPU
#BATCH_SIZE=7000 # model 1 for double 12GB GPU

#BATCH_SIZE=1024 # model 1 for single 12GB GPU
#BATCH_SIZE_TEST=1000
BATCH_SIZE=512
BATCH_SIZE_TEST=32
#BATCH_SIZE=512
#BATCH_SIZE_TEST=512
NO_CUDA=False

# static parameters
CAMERA_COUNT=2
DATASET_SHUFFLE=False
BATCH_DATA_BOUNDARY=True
VISIBLE_CAM=os.environ.get('VISIBLE_CAM')
if VISIBLE_CAM is not None:
    VISIBLE_CAM = list(map(int, VISIBLE_CAM.split(':')))
    CAMERA_COUNT = len(VISIBLE_CAM)
time_threshold = 2

SAVE_FINAL_MODEL=True
SAVE_MODEL_FOR_EVERY_EPOCH=(os.environ.get('SAVE_MODEL_FOR_EVERY_EPOCH') in ['Y', 'y', 'yes', 'true', 'True', '1'])

DATASIZE_LIMIT=int(os.environ.get('DATASIZE_LIMIT')) if 'DATASIZE_LIMIT' in os.environ else None
DATASIZE_LIMIT_DIR=int(os.environ.get('LIMIT_DIR')) if 'LIMIT_DIR' in os.environ else 0

# default parameters
EPOCH_COUNT=1000
PERSON_ID=[1]
DATA_PREFIX='./InvisibleEye'
LOAD_MODEL=None
SAVE_MODEL='./trained.pth'
LOAD_MODEL_DEFAULT='./pretrained.pth'
SAVE_SCRIPT='./trained.pt'

TRAINING_TEST=None

Testing=False

def show_usage():
    print('Usage:')
    print('    Training:   {} <PERSON_ID> [DatasetPrefix] [EPOCH_COUNT] [SAVE_MODEL] [PRETRAINED_MODEL]'.format(sys.argv[0]))
    print('    Testing:    {} <PERSON_ID> <DatasetPrefix> 0 [LOAD_MODEL]'.format(sys.argv[0]))
    print('    Converting: {} <PERSON_ID> <DatasetPrefix> -1 [LOAD_MODEL] [SAVE_SCRIPT]'.format(sys.argv[0]))
    print('')
    print('Note:  PERSON_ID can also be a list of numbers (e.g. 1,3-17), the program will use the data from multiple people')
    print('       also, when PERSON_ID is in the format of XXX::AAA, XXX will be treated as training data and aaa is test data.')
    exit(1)

if len(sys.argv) < 2:
    show_usage()

PERSON_LIST = sys.argv[1].split('::')
if len(PERSON_LIST) == 2:
    TRAINING_TEST = utils.parse_list(PERSON_LIST[1])
elif len(PERSON_LIST) > 2:
    print('PERSON_ID format error!')
    exit(1)
PERSON_ID = utils.parse_list(PERSON_LIST[0])

if len(sys.argv) >= 3:
    DATA_PREFIX = sys.argv[2]
if len(sys.argv) >= 4:
    EPOCH_COUNT = int(sys.argv[3])

if EPOCH_COUNT == 0:
    Testing = True
    Converting = False
    if len(sys.argv) >= 5:
        LOAD_MODEL = sys.argv[4]
    else:
        LOAD_MODEL = LOAD_MODEL_DEFAULT
    if len(sys.argv) >= 6:
        print("Error: too much parameters")
        show_usage()
elif EPOCH_COUNT == -1:
    Testing = False
    Converting = True
    if len(sys.argv) >= 5:
        LOAD_MODEL = sys.argv[4]
    else:
        LOAD_MODEL = LOAD_MODEL_DEFAULT
    if len(sys.argv) >= 6:
        SAVE_SCRIPT = sys.argv[5]
    if len(sys.argv) >= 7:
        print("Error: too much parameters")
        show_usage()
else:
    Testing = False
    Converting = False
    if len(sys.argv) >= 5:
        SAVE_MODEL = sys.argv[4]
    if len(sys.argv) >= 6:
        LOAD_MODEL = sys.argv[5]
    if len(sys.argv) >= 7:
        print("Error: too much parameters")
        show_usage()

# print basic information
print('{} for person #{}'.format('Testing' if Testing else 'Converting' if Converting else 'Training', PERSON_ID))
if TRAINING_TEST is not None:
    print('In-training test will be performed on Person #{}'.format(TRAINING_TEST))
if VISIBLE_CAM is not None:
    print('Visible Cameras: {}'.format(VISIBLE_CAM))
print('Dataset Prefix: {}'.format(DATA_PREFIX))
print('Input Model: {}'.format(LOAD_MODEL))
if not Converting:
    print('Shuffle Dataset: {}'.format(DATASET_SHUFFLE))
    print('Batch Size: {}'.format(BATCH_SIZE))
if not Testing:
    if Converting:
        print('Output TorchScript: {}'.format(SAVE_SCRIPT))
    else:
        print('Output Model: {}'.format(SAVE_MODEL))
        print('Max Epoch: {}'.format(EPOCH_COUNT))

# initialize device-independent context
if not NO_CUDA and torch.cuda.is_available():
    utils.CURRENT_DEVICE = torch.device('cuda:0')
    if torch.cuda.device_count() > 1:
        num = torch.cuda.device_count()
        device_ids=[ x for x in range(num) ]
        utils.wrap_model = lambda model: nn.DataParallel(model.cuda(), device_ids=device_ids)
        utils.wrap_data = lambda data, idx=0: data.cuda()
        utils.wrap_optimizer = lambda opt: nn.DataParallel(opt, device_ids=device_ids)
        utils.run_model = lambda model, inputs: model(inputs)
#        utils.run_model = lambda model, inputs: nn.parallel.data_parallel(model, inputs)
        print("Running on {} GPUs".format(num))
    else:
        utils.wrap_model = lambda model: model.cuda()
        utils.wrap_data = lambda data, idx=0: data.cuda()
        utils.wrap_optimizer = lambda opt: opt
        utils.run_model = lambda model, inputs: model(inputs)
        print("Running on a GPU")
else:
    utils.CURRENT_DEVICE = torch.device('cpu')
    utils.wrap_model = lambda model: model
    utils.wrap_data = lambda data, idx=0: data
    utils.wrap_optimizer = lambda opt: opt
    utils.run_model = lambda model, inputs: model(inputs)
    print("Running on CPU(s)")



class TheDataset(Data.Dataset):
    def __init__(self,data_prefix,person=1,train=True,cam_count=4,restrict_boundary=None,image_suffix='.png',cache_later=False):
        if person is list:
            person = [person]
        self.person = person
        self.cam_count = cam_count
        self.image_suffix = image_suffix
        self.data_prefix = data_prefix
        self.subset_name = 'train' if train else 'test'
        p = utils.read_image(os.path.join(self.data_prefix, 'serial{}'.format(person[0]), '1', '0' + self.image_suffix))
        if len(p.shape) == 2:
            h,w = p.shape
            p = np.concatenate([p,np.zeros((w-h,w),dtype=np.uint8)],axis=0)
        else:
            h,w,c = p.shape
            p = np.concatenate([p,np.zeros((w-h,w,c),dtype=np.uint8)],axis=0)
            
#        p = p[0:h,(w-h)//2:(w+h)//2,:]
        self.dim = p.shape
        self.frame_data = [ np.load(os.path.join(self.data_prefix, 'serial{}'.format(person_id), "target.npy")).astype(np.float) for person_id in person ]
        if DATASIZE_LIMIT is not None:
            if DATASIZE_LIMIT_DIR != 0:
                # tail to head
                limited_frame_data = list(map(lambda v: v[-DATASIZE_LIMIT:], self.frame_data))
                self.limit_offset = [ len(self.frame_data[i]) - len(limited_frame_data[i]) for i in range(len(limited_frame_data)) ]
            else:
                # head to tail
                limited_frame_data = list(map(lambda v: v[:DATASIZE_LIMIT], self.frame_data))
                self.limit_offset = [ 0 for _ in range(len(limited_frame_data)) ]
            self.frame_data = limited_frame_data
#        self.dim = utils.read_image(os.path.join(self.data_prefix, str(person[0]), self.subset_name, '0', '0' + self.image_suffix)).shape
#        self.frame_data = [ np.load(os.path.join(self.data_prefix, str(person_id), self.subset_name, "Y.npy")) for person_id in person ]
        self.restrict_boundary = restrict_boundary
        if restrict_boundary is not None:
            self.data_length = [ x.shape[0] // restrict_boundary * restrict_boundary for x in self.frame_data ]
        else:
            self.data_length = [ x.shape[0] for x in self.frame_data ]
        # cache everything if images are small
        self.cache_all = not Converting and np.product(self.dim) * sum(self.data_length) < 10000000
        print("Dataset Dimension: {} * {}".format(self.dim, sum(self.data_length)))
        if DATASIZE_LIMIT is not None:
            print("Dataset Offset: {}".format(self.limit_offset))
            print("Dataset Length: {}".format(self.data_length))
        if len(self.dim) >= 3:
            print("WARNING: Colored image will be converted into grayscale before processing", flush=True)
        if not cache_later:
            self.flush_cache()

    def flush_cache(self):
        if self.cache_all:
            print("Generating cache", flush=True)
            self.cache_x = []
            self.cache_y = []
            for framid in range(len(self)):
                x, y = self.getitem_raw(framid)
                self.cache_x.append(x)
                self.cache_y.append(y)
            print("Cache generated", flush=True)

    def get_camera_count(self):
        return self.cam_count

    def get_dimensions(self):
        return self.dim

    def get_metrics(self):
        return self.get_dimensions(), self.get_camera_count(), len(self)

    def getindex_raw(self, index):
        person_id = 0
        while index >= self.data_length[person_id]:
            index -= self.data_length[person_id]
            person_id += 1
        return person_id, index

    def is_boundary(self, index):
        _, index = self.getindex_raw(index)
        return index == 0

    def __len__(self):
        return sum(self.data_length)

    def getitem_raw(self, index):
        pics = None
        person_id, index = self.getindex_raw(index)
        pic_index_offset = 0
        if DATASIZE_LIMIT is not None:
            pic_index_offset = self.limit_offset[person_id]
        iterable_camid = range(self.cam_count) if VISIBLE_CAM is None else VISIBLE_CAM
        for camid in iterable_camid:
            arr = utils.read_image(os.path.join(self.data_prefix, 'serial{}'.format(self.person[person_id]), str(camid + 1), str(index + pic_index_offset) + self.image_suffix))
            if len(arr.shape)==2:
                h,w = arr.shape
#            arr = arr[:,(w-h)//2:(w+h)//2,:] # use middle
                arr = np.concatenate([arr, np.zeros((w-h,w), dtype=np.uint8)], axis=0)
            else:
                h,w,c = arr.shape
                arr = np.concatenate([arr, np.zeros((w-h,w,c), dtype=np.uint8)], axis=0)
#            arr = utils.read_image(os.path.join(self.data_prefix, str(self.person[person_id]), self.subset_name, str(camid), str(index) + self.image_suffix))
            t = torch.from_numpy(arr).type(torch.FloatTensor)
            if len(self.dim) >= 3:
                torch.mean(t, dim=2, keepdim=False, out=t)
            t = utils.wrap_data(t, index)
            t.unsqueeze_(0)
            pics = torch.cat([pics, t]) if pics is not None else t
        return pics, utils.wrap_data(torch.from_numpy(self.frame_data[person_id][index]).type(torch.FloatTensor), index)

    def __getitem__(self, index):
        if self.cache_all:
            # use cache if possible
            return self.cache_x[index], self.cache_y[index]
        else:
            return self.getitem_raw(index)

def main():
    st = time.time()
    print('Initializing')

    if BATCH_DATA_BOUNDARY and not Testing and not Converting:
        dataset = TheDataset(DATA_PREFIX, PERSON_ID, not Testing, CAMERA_COUNT, BATCH_SIZE, cache_later=True)
    else:
        dataset = TheDataset(DATA_PREFIX, PERSON_ID, not Testing, CAMERA_COUNT, cache_later=True)
    global TRAINING_TEST
    if TRAINING_TEST is not None:
        dataset.flush_cache()
        testing_dataset = TheDataset(DATA_PREFIX, TRAINING_TEST, Testing, CAMERA_COUNT)
        _, _, testing_len = testing_dataset.get_metrics()
    else:
        dataset.flush_cache()

    # load model according to dataset...
    data_dim, data_camcount, data_len = dataset.get_metrics()

    net, opt = my_model.get_model(data_camcount, data_dim)
    dataloader = Data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=DATASET_SHUFFLE)
    if TRAINING_TEST is not None:
        dataloader_test = Data.DataLoader(testing_dataset, batch_size=BATCH_SIZE_TEST, shuffle=False)

    opt = opt.module if 'module' in dir(opt) else opt
    xnet = net.module if 'module' in dir(net) else net

    utils.print_modelsize(net)

    epochs = EPOCH_COUNT
    loss_func = utils.wrap_data(nn.MSELoss())
    loss_accumu = lambda predict,real: ((predict.sub(real) ** 2).sum(1) ** 0.5).sum()
    #loss_accumu = utils.wrap_data(nn.L1Loss())

    epoch_offset = 0

    # load existed data
    if LOAD_MODEL is not None and not (Testing and SAVE_MODEL_FOR_EVERY_EPOCH):
        model = torch.load(LOAD_MODEL, map_location=utils.CURRENT_DEVICE)
        opt.load_state_dict(model['optimizer'])
        xnet.load_state_dict(model['net'])
        epoch_offset = model['epoch']
        print('Model loaded')

    print('Model Name: {}'.format(xnet.__model_name__))

    if SAVE_MODEL_FOR_EVERY_EPOCH:
        print('SAVE_MODEL_FOR_EVERY_EPOCH: True')

    t = time.time()
    print("Initialization time: {0}".format(t - st), flush=True)

    printed = True

    if 'DEBUG' in os.environ and int(os.environ.get('DEBUG')) != 0:
        print('train', dataset.__getitem__(0))

    if Testing:
        if SAVE_MODEL_FOR_EVERY_EPOCH:
            path = os.path.dirname(LOAD_MODEL)
            if path == '':
                path = '.'
            fn = os.path.basename(LOAD_MODEL)
            epochs = []
            for f in os.listdir(path):
                res = f.split('.epoch')
                if res[0] == fn:
                    if len(res) == 1:
                        continue
                    if len(res) != 2:
                        print('Skip ill-formatted file name: {}'.format(f))
                        continue
                    if res[1] == '':
                        continue
                    try:
                        epochs.append(int(res[1]))
                    except:
                        print('Skip ill-formatted file name: {}'.format(f))
                        continue
            suffixes = [ '.epoch{}'.format(x) for x in sorted(epochs) ]
        else:
            suffixes = ['']
        if suffixes == []:
            print('Training failed, please check the existance of every-step model dump or consider unset SAVE_MODEL_FOR_EVERY_EPOCH')
            exit(1)
        print("Testing started", flush=True)
        for test_epoch, suffix in zip(range(len(suffixes)), suffixes):
            if suffix != '':
                model = torch.load(LOAD_MODEL + suffix, map_location=utils.CURRENT_DEVICE)
                opt.load_state_dict(model['optimizer'])
                xnet.load_state_dict(model['net'])
                epoch_offset = model['epoch']
            net.eval()
            loss_acc = None
            xnet.input_reset()
            with torch.no_grad():
                ct = t
                datalen_acc = 0
                for x,y in dataloader:
                    predict = net(x)
                    loss_acc = loss_accumu(predict, y) if loss_acc is None else loss_acc + loss_accumu(predict, y)
                    datalen_acc += x.shape[0]
                    et = time.time()
                    if et - ct >= time_threshold:
                        print("epoch {5} ({6}/{7}), progress: {0}/{1} = {2:.2%}, time: {3:.1f}, est. time: {4:.1f}".format(datalen_acc, data_len, datalen_acc / data_len, et - t, (et - t) / datalen_acc * (data_len - datalen_acc), epoch_offset + 1, test_epoch + 1, len(suffixes)), end="  \r", flush=True)
                        ct = et
            avgloss = loss_acc / data_len

            tn = time.time()
            if suffix == '':
                print("Done! Epoch {2} ({3}/{4}): Average loss {1}, time cost {0}".format(tn - t, avgloss.cpu().detach().numpy().tolist(), epoch_offset + 1, test_epoch + 1, len(suffixes)), flush=True, end='  \n')
                break
            else:
                print("Epoch {2} ({3}/{4}): Average loss {1}, time cost {0}".format(tn - t, avgloss.cpu().detach().numpy().tolist(), epoch_offset + 1, test_epoch + 1, len(suffixes)), flush=True, end='  \n')
        print('Done!')

    elif Converting:
        print("Converting...", flush=True)
        sampleX, sampleY = dataset.__getitem__(0)
        net.eval()
        script_model = torch.jit.trace(net, sampleX.unsqueeze(0))
        script_model.save(SAVE_SCRIPT)

        tn = time.time()
        print("Done! Saved to {1}, time cost {0}".format(tn - t, SAVE_SCRIPT), flush=True)
    else:
        st = tp = tn = t

        print("Training started from epoch {}".format(epoch_offset), flush=True)
        net.train()

        avgloss = None

        prev_fold = None

        initial_stat = xnet.state_dict(), opt.state_dict()
#        result = None

        min_loss = float('nan')
        min_loss_epoch = None

        for epoch in range(epochs):
#            if epoch == epochs - 1:
#                result = []
            loss = None
            loss_acc = None
            net.zero_grad()
            ct = tn
            datalen_acc = 0
            for x,y in dataloader:
                xnet.input_reset(dataset.is_boundary(datalen_acc))
                predict = net(x)
                loss = loss_func(predict, y)
                opt.zero_grad()
                loss.backward()
                opt.step()
                loss_acc = loss_accumu(predict, y) if loss_acc is None else loss_acc + loss_accumu(predict, y)
                datalen_acc += x.shape[0]
                et = time.time()
                if et - ct >= time_threshold:
                    print("epoch: {0}/{1}, progress: {2}/{3} = {4:.2%}, time: {5:.1f}, est. time: {6:.1f}  ".format(epoch_offset + epoch + 1, epoch_offset + epochs, datalen_acc, data_len, datalen_acc / data_len, et - tn, (et - tn) / datalen_acc * (data_len - datalen_acc)), end="\r", flush=True)
                    ct = et
            avgloss = loss_acc / data_len

            if SAVE_MODEL_FOR_EVERY_EPOCH:
                torch.save({
                    "net": xnet.state_dict(),
                    "optimizer": opt.state_dict(),
                    "epoch": epoch_offset + epoch
                    }, SAVE_MODEL + '.epoch' + str(epoch_offset + epoch))


            t = tn
            tn = time.time()
            if tn - tp >= time_threshold:
                print("epoch: {0}/{1}, loss: {2}, time: {3}, total time: {4}  ".format(epoch_offset + epoch + 1, epoch_offset + epochs, avgloss.cpu().detach().numpy().tolist(), tn - t, tn - st), flush=True)
                if TRAINING_TEST is not None:
                    loss_acc = None
                    xnet.input_reset()
                    with torch.no_grad():
                        ct = tn
                        datalen_acc = 0
                        for x,y in dataloader_test:
                            predict = net(x)
                            loss_acc = loss_accumu(predict, y) if loss_acc is None else loss_acc + loss_accumu(predict, y)
#                            if result is not None:
#                                result.append(predict)
                            datalen_acc += x.shape[0]
                            et = time.time()
                            if et - ct >= time_threshold:
                                print("epoch: {0}/{1}, TEST progress: {2}/{3} = {4:.2%}, time: {5:.1f}, est. time: {6:.1f}  ".format(epoch_offset + epoch + 1, epoch_offset + epochs, datalen_acc, testing_len, datalen_acc / testing_len, et - tn, (et - tn) / datalen_acc * (testing_len - datalen_acc)), end="\r", flush=True)
                                ct = et
                    avgloss = loss_acc / testing_len
                    t = tn
                    tn = time.time()
                    lx = avgloss.cpu().detach().numpy().tolist()
                    print("epoch: {0}/{1}, TEST Loss: {2}, test time: {3}  ".format(epoch_offset + epoch + 1, epoch_offset + epochs, lx, tn - t), flush=True)
                    if not (min_loss <= lx):
                        min_loss = lx
                        min_loss_epoch = epoch_offset + epoch + 1
                tp = tn
                printed = True
            else:
                printed = False

        if not printed:
            print("epoch: {0}/{1}, loss: {2}, time: {3}, total time: {4}  ".format(epoch_offset + epochs, epoch_offset + epochs, avgloss.cpu().detach().numpy().tolist(), tn - t, tn - st), flush=True)
            if TRAINING_TEST is not None:
                t = tn
                loss_acc = None
                xnet.input_reset()
                with torch.no_grad():
                    ct = tn
                    datalen_acc = 0
                    for x,y in dataloader_test:
                        predict = net(x)
                        loss_acc = loss_accumu(predict, y) if loss_acc is None else loss_acc + loss_accumu(predict, y)
#                        if result is not None:
#                            result.append(predict)
                        datalen_acc += x.shape[0]
                        et = time.time()
                        if et - ct >= time_threshold:
                            print("epoch: {0}/{1}, TEST progress: {2}/{3} = {4:.2%}, time: {5:.1f}, est. time: {6:.1f}  ".format(epoch_offset + epochs, epoch_offset + epochs, datalen_acc, testing_len, datalen_acc / testing_len, et - tn, (et - tn) / datalen_acc * (testing_len - datalen_acc)), end="\r", flush=True)
                            ct = et
                avgloss = loss_acc / testing_len
                tn = time.time()
                lx = avgloss.cpu().detach().numpy().tolist()
                print("epoch: {0}/{1}, TEST Loss: {2}, test time: {3}  ".format(epoch_offset + epochs, epoch_offset + epochs, lx, tn - t), flush=True)
                if not (min_loss <= lx):
                    min_loss = lx
                    min_loss_epoch = epoch_offset + epoch + 1

        t = tn

#        np.save('result.npy', torch.cat(result).cpu().numpy())

        # save the model
        epoch_offset += epochs
        print("Model \"{1}\" is at epoch {0}".format(epoch_offset, xnet.__model_name__), flush=True)
        torch.save({
            "net": xnet.state_dict(),
            "optimizer": opt.state_dict(),
            "epoch": epoch_offset
            }, SAVE_MODEL)

        tn = time.time()

        print("Done! Model saved to {0}, cost {1}".format(SAVE_MODEL, tn - t), flush=True)

        if min_loss_epoch is not None:
            print("Min test loss: {} at epoch {}".format(min_loss, min_loss_epoch))

if __name__ == '__main__':
    main()
