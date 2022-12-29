import os
from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau
import torch
from tqdm import tqdm
import torchvision.utils
from torch.autograd import Variable
import time

import models.loss_function
from models.net_model import ResidualFeatureNet, RFNet64, SERFNet64, \
    STNRFNet64, STNResRFNet64, STNResRFNet64v2, STNResRFNet64v3, DeformRFNet64, DilateRFNet64, \
    RFNet64Relu, STNResRFNet64v2Relu, STNResRFNet64v4, STNResRFNet64v5, STNResRFNet32v216, STNResRFNet32v316, STNResRFNet3v316
from models.loss_function import RSIL, ShiftedLoss, MSELoss, HammingDistance, MaskRSIL
from models.pytorch_mssim import SSIM, SSIMGNN, RSSSIM
from torchvision import transforms
import torchvision
from torch.utils.data import DataLoader
from data.data_factory import Factory
from models.EfficientNetV2 import efficientnetv2_s, ConvBNAct, fk_efficientnetv2_s
from collections import OrderedDict
from functools import partial
from torch.nn import functional as F
from loss.stn_ssim import STSSIM


def logging(msg, suc=True):
    if suc:
        print("[*] " + msg)
    else:
        print("[!] " + msg)


model_dict = {
    "RFNet": ResidualFeatureNet().cuda(),
    "FKEfficientNet": fk_efficientnetv2_s().cuda(),
    "RFNet64": RFNet64().cuda(),
    "SERFNet64": SERFNet64().cuda(),
    "STNRFNet64": STNRFNet64().cuda(),
    "STNResRFNet64": STNResRFNet64().cuda(),
    "STNResRFNet64v2": STNResRFNet64v2().cuda(),
    "STNResRFNet64v3": STNResRFNet64v3().cuda(),
    "DilateRFNet64": DilateRFNet64().cuda(),
    "DeformRFNet64": DeformRFNet64().cuda(),
    "RFNet64Relu": RFNet64Relu().cuda(),
    "STNResRFNet64v2Relu": STNResRFNet64v2Relu().cuda(),
    "STNResRFNet64v4": STNResRFNet64v4().cuda(),
    "STNResRFNet64v5": STNResRFNet64v5().cuda(),
    "STNResRFNet32v216": STNResRFNet32v216().cuda(),
    "STNResRFNet32v316": STNResRFNet32v316().cuda(),
    "STNResRFNet3v316": STNResRFNet3v316().cuda()
}


class Model(object):
    def __init__(self, args, writer):
        self.args = args
        self.writer = writer
        self.batch_size = args.batch_size
        self.samples_subject = args.samples_subject
        self.train_loader, self.dataset_size = self._build_dataset_loader(args)
        self.inference, self.loss = self._build_model(args)
        if self.args.loss_type == "stssim":
            self.optimizer1 = torch.optim.Adam(self.inference.parameters(), args.learning_rate1)
            self.optimizer2 = torch.optim.Adam(self.loss.parameters(), args.learning_rate2)
        else:
            self.optimizer1 = torch.optim.Adam(self.inference.parameters(), args.learning_rate1)

    def _build_dataset_loader(self, args):
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        train_dataset = Factory(args.train_path, args.input_size, transform=transform,
                                valid_ext=['.bmp', '.jpg', '.JPG'], train=True, n_tuple=args.n_tuple,
                                if_aug=args.if_augment, if_hsv=args.if_hsv, if_rotation=args.if_rotation,
                                if_translation=args.if_translation, if_scale=args.if_scale)
        logging("Successfully Load {} as training dataset...".format(args.train_path))
        train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)

        if args.n_tuple in ['triplet']:
            examples = iter(train_loader)
            # example_data, example_mask, example_target = examples.next()
            example_data, example_target = examples.next()
            example_anchor = example_data[:, 0:3, :, :]
            example_positive = example_data[:, 3:3 * self.samples_subject, :, :].reshape(-1, 3, example_anchor.size(2),
                                                                                         example_anchor.size(3))
            example_negative = example_data[:, 3 * self.samples_subject:, :, :].reshape(-1, 3, example_anchor.size(2),
                                                                                        example_anchor.size(3))
            anchor_grid = torchvision.utils.make_grid(example_anchor)
            self.writer.add_image(tag="anchor", img_tensor=anchor_grid)
            positive_grid = torchvision.utils.make_grid(example_positive)
            self.writer.add_image(tag="positive", img_tensor=positive_grid)
            negative_grid = torchvision.utils.make_grid(example_negative)
            self.writer.add_image(tag="negative", img_tensor=negative_grid)
        elif args.n_tuple in ['quadruplet']:
            # for showing quadruplet
            examples = iter(train_loader)
            example_data, example_target = examples.next()
            example_anchor = example_data[:, 0:3, :, :]
            example_positive = example_data[:, 3: self.samples_subject * 3, :, :].reshape(-1, 3, example_anchor.size(2),
                                                                                          example_anchor.size(3))
            example_negative = example_data[:, self.samples_subject * 3:2 * 3 * self.samples_subject, :, :].reshape(-1,
                                                                                                                    3,
                                                                                                                    example_anchor.size(
                                                                                                                        2),
                                                                                                                    example_anchor.size(
                                                                                                                        3))
            example_negative2 = example_data[:, 2 * 3 * self.samples_subject:, :, :].reshape(-1, 3,
                                                                                             example_anchor.size(2),
                                                                                             example_anchor.size(3))
            anchor_grid = torchvision.utils.make_grid(example_anchor)
            self.writer.add_image(tag="anchor", img_tensor=anchor_grid)
            positive_grid = torchvision.utils.make_grid(example_positive)
            self.writer.add_image(tag="positive", img_tensor=positive_grid)
            negative_grid = torchvision.utils.make_grid(example_negative)
            self.writer.add_image(tag="negative", img_tensor=negative_grid)
            negative2_grid = torchvision.utils.make_grid(example_negative2)
            self.writer.add_image(tag="negative2", img_tensor=negative2_grid)
        else:
            print('Skip Summary image show')
        return train_loader, len(train_dataset)

    def exp_lr_scheduler(self, epoch, lr_decay=0.1, lr_decay_epoch=100):
        if epoch % lr_decay_epoch == 0:
            for param_group in self.optimizer1.param_groups:
                param_group['lr'] *= lr_decay

    def _build_model(self, args):
        if args.model not in ["RFNet", "FKEfficientNet", "RFNet64", "RFNet64_16", "SERFNet64",
                              "STNRFNet64", "STNResRFNet64", "STNResRFNet64v2", "STNResRFNet64v3",
                              "DilateRFNet64", "DeformRFNet64", "RFNet64Relu", "STNResRFNet64v2Relu",
                              "STNResRFNet64v4", "STNResRFNet64v5", "STNResRFNet32v216", "STNResRFNet32v316",
                              "STNResRFNet3v316"]:
            raise RuntimeError('Model not found')
        inference = model_dict[args.model].cuda()
        if args.model == "RFNet":
            data = torch.randn([1, 128, 128]).unsqueeze(0).cuda()
            data = Variable(data, requires_grad=False)
            self.writer.add_graph(inference, [data])
            inference.cuda()
            inference.train()
        else:
            data = torch.randn([3, 128, 128]).unsqueeze(0).cuda()
            data = Variable(data, requires_grad=False)
            self.writer.add_graph(inference, [data])
            inference.cuda()
            inference.train()

        if args.loss_type == "rsil":
            loss = RSIL(args.vertical_size, args.horizontal_size, args.rotate_angle).cuda()
            logging("Successfully building rsil loss")
        elif args.loss_type == "shiftedloss":
            # loss = MaskRSIL(args.vertical_size, args.horizontal_size, args.rotate_angle).cuda()
            loss = ShiftedLoss(hshift=args.horizontal_size, vshift=args.vertical_size).cuda()
            logging("Successfully building shiftedloss loss")
        else:
            if args.loss_type == "ssim":
                loss = SSIM(data_range=1., size_average=False, channel=3, win_size=7).cuda()
                logging("Successfully building ssim triplet loss")
            elif args.loss_type == "stssim":
                loss = STSSIM(data_range=1., size_average=False, channel=64).cuda()
                logging("Successfully building stssim loss")
            else:
                if args.loss_type == "rsssim":
                    loss = RSSSIM(data_range=1., size_average=False, channel=32, win_size=7, v_shift=args.vertical_size,
                                  h_shift=args.horizontal_size, angle=args.rotate_angle, step=args.step_size).cuda()
                    logging("Successfully building rsssim loss")
                else:
                    raise RuntimeError('Please make sure your loss function!')
        loss.cuda()
        loss.train()
        return inference, loss

    def _triplet_train(self, args):
        epoch_steps = len(self.train_loader)
        train_loss = 0
        start_epoch = ''.join(x for x in os.path.basename(args.start_ckpt) if x.isdigit())
        if start_epoch:
            start_epoch = int(start_epoch) + 1
            self.load(args.start_ckpt)
        else:
            start_epoch = 1

        # 0-100: 0.01; 150-450: 0.001; 450-800:0.0001; 800-：0.00001
        scheduler = MultiStepLR(self.optimizer1, milestones=[10, 500, 1000], gamma=0.1)
        # for freeze spatial transformer network
        freeze_stn = args.freeze_stn

        for e in range(start_epoch, args.epochs + start_epoch):
            # self.exp_lr_scheduler(e, lr_decay_epoch=100)
            self.inference.train()
            agg_loss = 0.
            # for batch_id, (x, _) in enumerate(self.train_loader):
            # for batch_id, (x, _) in tqdm(enumerate(self.train_loader), total=len(self.train_loader)):
            loop = tqdm(enumerate(self.train_loader), total=len(self.train_loader))
            for batch_id, (x, _) in loop:
                if args.model in ["RFNWithSTNet", "STNWithRFNet", "ResidualSTNet"]:
                    if freeze_stn:
                        for name, para in self.inference.named_parameters():
                            if "localization" in name or "fc_loc" in name:
                                para.requires_grad_(False)
                    else:
                        for name, para in self.inference.named_parameters():
                            if "localization" in name or "fc_loc" in name:
                                para.requires_grad_(True)

                # ========================================================= train inference model
                # x.shape :-> [b, 3*3*samples_subject, h, w]
                x = x.cuda()
                x = Variable(x, requires_grad=False)
                fms = self.inference(x.view(-1, 3, x.size(2), x.size(3)))
                bs, ch, he, wi = fms.shape
                # (batch_size, anchor+positive+negative, 32, 32)
                fms = fms.view(x.size(0), -1, fms.size(2), fms.size(3))
                anchor_fm = fms[:, 0:ch, :, :]  # anchor has one sample
                if len(anchor_fm.shape) == 3:
                    anchor_fm.unsqueeze(1)
                pos_fm = fms[:, 1 * ch:self.samples_subject * ch, :, :].contiguous()
                neg_fm = fms[:, self.samples_subject * ch:, :, :].contiguous()
                nneg = int(neg_fm.size(1) / ch)
                neg_fm = neg_fm.view(-1, ch, neg_fm.size(2), neg_fm.size(3))
                an_loss = self.loss(anchor_fm.repeat(1, nneg, 1, 1).view(-1, ch, anchor_fm.size(2), anchor_fm.size(3)),
                                    neg_fm)
                # an_loss.shape:-> (batch_size, 10)
                # min(1) will get min value and the corresponding indices
                # min(1)[0]
                an_loss = an_loss.view((-1, nneg)).min(1)[0]

                npos = int(pos_fm.size(1) / ch)
                pos_fm = pos_fm.view(-1, ch, pos_fm.size(2), pos_fm.size(3))
                ap_loss = self.loss(anchor_fm.repeat(1, npos, 1, 1).view(-1, ch, anchor_fm.size(2), anchor_fm.size(3)),
                                    pos_fm)
                ap_loss = ap_loss.view((-1, npos)).max(1)[0]

                sstl = ap_loss - an_loss + args.alpha
                sstl = torch.clamp(sstl, min=0)

                loss = torch.sum(sstl) / args.batch_size

                loss.backward()
                self.optimizer1.step()
                self.optimizer1.zero_grad()
                agg_loss += loss.item()
                train_loss += loss.item()

                loop.set_description(f'Epoch [{e}/{args.epochs}]')
                loop.set_postfix(loss_inference="{:.6f}".format(agg_loss))

            self.writer.add_scalar("lr", scalar_value=self.optimizer1.state_dict()['param_groups'][0]['lr'],
                                   global_step=(e + 1))
            self.writer.add_scalar("loss_inference", scalar_value=train_loss,
                                   global_step=((e + 1) * epoch_steps))

            if agg_loss <= args.freeze_thre:
                freeze_stn = False

            train_loss = 0

            if args.checkpoint_dir is not None and e % args.checkpoint_interval == 0:
                self.save(args.checkpoint_dir, e)

            scheduler.step()

        self.writer.close()

    def oldtriplet_train(self, args):
        start_epoch = ''.join(x for x in os.path.basename(args.start_ckpt) if x.isdigit())
        if start_epoch:
            start_epoch = int(start_epoch) + 1
            self.load(args.start_ckpt)
        else:
            start_epoch = 1

        for e in range(start_epoch, args.epochs + start_epoch):
            self.exp_lr_scheduler(e, lr_decay_epoch=200)
            self.inference.train()
            agg_loss = 0.
            count = 0
            for batch_id, (x, _) in enumerate(self.train_loader):
                count += len(x)
                self.optimizer1.zero_grad()
                x = x.cuda()

                x = Variable(x, requires_grad=False)
                if "RFN-32" in args.model:
                    fms = self.inference(x.view(-1, 1, x.size(2), x.size(3)).repeat(1, 3, 1, 1))
                else:
                    fms = self.inference(x.view(-1, 1, x.size(2), x.size(3)))
                fms = fms.view(x.size(0), -1, fms.size(2), fms.size(3))

                anchor_fm = fms[:, 0, :, :].unsqueeze(1)
                pos_fm = fms[:, 1, :, :].unsqueeze(1)
                neg_fm = fms[:, 2:, :, :].contiguous()

                nneg = neg_fm.size(1)
                neg_fm = neg_fm.view(-1, 1, neg_fm.size(2), neg_fm.size(3))
                an_loss = self.loss(anchor_fm.repeat(1, nneg, 1, 1).view(-1, 1, anchor_fm.size(2), anchor_fm.size(3)),
                                    neg_fm)
                an_loss = an_loss.view((-1, nneg)).min(1)[0]
                ap_loss = self.loss(anchor_fm, pos_fm)

                sstl = ap_loss - an_loss + args.alpha
                sstl = torch.clamp(sstl, min=0)
                loss = torch.sum(sstl) / args.batch_size

                loss.backward()
                self.optimizer1.step()

                agg_loss += loss.item()

                if e % args.log_interval == 0:
                    mesg = "{}\tEpoch {}:\t[{}/{}]\t {:.6f}".format(
                        time.ctime(), e, count, self.dataset_size, agg_loss / (batch_id + 1)
                    )
                    print(mesg)

            if args.checkpoint_dir is not None and e % args.checkpoint_interval == 0:
                self.save(args.checkpoint_dir, e)

    def quadruplet_loss(self, args):
        epoch_steps = len(self.train_loader)
        train_loss = 0
        start_epoch = ''.join(x for x in os.path.basename(args.start_ckpt) if x.isdigit())
        if start_epoch:
            start_epoch = int(start_epoch) + 1
            self.load(args.start_ckpt)
        else:
            start_epoch = 1

        # 0-100: 0.01; 150-450: 0.001; 450-800:0.0001; 800-：0.00001
        scheduler = MultiStepLR(self.optimizer1, milestones=[10, 500, 1000], gamma=0.1)

        for e in range(start_epoch, args.epochs + start_epoch):
            # self.exp_lr_scheduler(e, lr_decay_epoch=100)
            self.inference.train()
            agg_loss = 0.
            # for batch_id, (x, _) in enumerate(self.train_loader):
            # for batch_id, (x, _) in tqdm(enumerate(self.train_loader), total=len(self.train_loader)):
            loop = tqdm(enumerate(self.train_loader), total=len(self.train_loader))
            for batch_id, (x, _) in loop:
                # ======================================================== train inference model
                # x.shape :-> [b, 3*5*samples_subject, h, w]
                # as for the 3*5*samples_subject, the first 3*samples contains 1 anchor and (samples-1) positive
                # the second 3*2*samples is the fist group negative sample
                # the last 3*2*samples is the second group negative sample
                x = x.cuda()
                x = Variable(x, requires_grad=False)
                fms = self.inference(x.view(-1, 3, x.size(2), x.size(3)))
                # (batch_size, anchor+positive+negative, 32, 32)
                bs, ch, he, wi = fms.shape
                fms = fms.view(x.size(0), -1, fms.size(2), fms.size(3))
                anchor_fm = fms[:, 0:ch, :, :]  # anchor has one sample
                if len(anchor_fm.shape) == 3:
                    anchor_fm.unsqueeze(1)
                pos_fm = fms[:, 1 * ch:self.samples_subject * ch, :, :].contiguous()
                neg_fm = fms[:, self.samples_subject * ch:2 * self.samples_subject * ch, :, :].contiguous()
                neg2_fm = fms[:, 2 * self.samples_subject * ch:, :, :].contiguous()
                # distance anchor negative
                nneg = int(neg_fm.size(1) / ch)
                neg_fm = neg_fm.view(-1, ch, neg_fm.size(2), neg_fm.size(3))
                an_loss = self.loss(anchor_fm.repeat(1, nneg, 1, 1).view(-1, ch, anchor_fm.size(2), anchor_fm.size(3)),
                                    neg_fm)
                # an_loss.shape:-> (batch_size, 10)
                # min(1) will get min value and the corresponding indices
                # min(1)[0]
                an_loss = an_loss.view((-1, nneg)).min(1)[0]
                # distance anchor positive
                npos = int(pos_fm.size(1) / ch)
                pos_fm = pos_fm.view(-1, ch, pos_fm.size(2), pos_fm.size(3))
                ap_loss = self.loss(anchor_fm.repeat(1, npos, 1, 1).view(-1, ch, anchor_fm.size(2), anchor_fm.size(3)),
                                    pos_fm)
                ap_loss = ap_loss.view((-1, npos)).max(1)[0]
                # distance negative negative2
                neg2_fm = neg2_fm.view(-1, ch, neg2_fm.size(2), neg2_fm.size(3))
                nn_loss = self.loss(neg2_fm, neg_fm)
                nn_loss = nn_loss.view((-1, nneg)).min(1)[0]

                sstl = F.relu(ap_loss - an_loss + args.alpha) + F.relu(ap_loss - nn_loss + args.alpha2)
                loss = torch.sum(sstl) / args.batch_size

                loss.backward()
                self.optimizer1.step()
                self.optimizer1.zero_grad()
                agg_loss += loss.item()
                train_loss += loss.item()

                loop.set_description(f'Epoch [{e}/{args.epochs}]')
                loop.set_postfix(loss_inference="{:.6f}".format(agg_loss))

            self.writer.add_scalar("lr", scalar_value=self.optimizer1.state_dict()['param_groups'][0]['lr'],
                                   global_step=(e + 1))
            self.writer.add_scalar("loss_inference", scalar_value=train_loss,
                                   global_step=((e + 1) * epoch_steps))

            train_loss = 0

            if args.checkpoint_dir is not None and e % args.checkpoint_interval == 0:
                self.save(args.checkpoint_dir, e)

            scheduler.step()

        self.writer.close()

    def stssim_train(self, args):
        epoch_steps = len(self.train_loader)
        train_loss = 0
        start_epoch = ''.join(x for x in os.path.basename(args.start_ckpt) if x.isdigit())
        if start_epoch:
            start_epoch = int(start_epoch) + 1
            self.load(args.start_ckpt)
        else:
            start_epoch = 1

        # 0-100: 0.01; 150-450: 0.001; 450-800:0.0001; 800-：0.00001
        scheduler1 = MultiStepLR(self.optimizer1, milestones=[10, 500, 1000], gamma=0.1)
        scheduler2 = MultiStepLR(self.optimizer2, milestones=[10, 500, 1000], gamma=0.1)

        for e in range(start_epoch, args.epochs + start_epoch):
            # self.exp_lr_scheduler(e, lr_decay_epoch=100)
            self.inference.train()
            self.loss.train()
            agg_loss = 0.
            # for batch_id, (x, _) in enumerate(self.train_loader):
            # for batch_id, (x, _) in tqdm(enumerate(self.train_loader), total=len(self.train_loader)):
            loop = tqdm(enumerate(self.train_loader), total=len(self.train_loader))
            for batch_id, (x, _) in loop:

                self.optimizer1.zero_grad()
                self.optimizer2.zero_grad()
                # ======================================================== train inference model
                # x.shape :-> [b, 3*5*samples_subject, h, w]
                # as for the 3*5*samples_subject, the first 3*samples contains 1 anchor and (samples-1) positive
                # the second 3*2*samples is the fist group negative sample
                # the last 3*2*samples is the second group negative sample
                x = x.cuda()
                x = Variable(x, requires_grad=False)
                fms = self.inference(x.view(-1, 3, x.size(2), x.size(3)))
                # (batch_size, anchor+positive+negative, 32, 32)
                bs, ch, he, wi = fms.shape
                fms = fms.view(x.size(0), -1, fms.size(2), fms.size(3))
                anchor_fm = fms[:, 0:ch, :, :]  # anchor has one sample
                if len(anchor_fm.shape) == 3:
                    anchor_fm.unsqueeze(1)
                pos_fm = fms[:, 1 * ch:self.samples_subject * ch, :, :].contiguous()
                neg_fm = fms[:, self.samples_subject * ch:2 * self.samples_subject * ch, :, :].contiguous()
                neg2_fm = fms[:, 2 * self.samples_subject * ch:, :, :].contiguous()
                # distance anchor negative
                nneg = int(neg_fm.size(1) / ch)
                neg_fm = neg_fm.view(-1, ch, neg_fm.size(2), neg_fm.size(3))
                an_loss = self.loss(anchor_fm.repeat(1, nneg, 1, 1).view(-1, ch, anchor_fm.size(2), anchor_fm.size(3)),
                                    neg_fm)
                # an_loss.shape:-> (batch_size, 10)
                # min(1) will get min value and the corresponding indices
                # min(1)[0]
                an_loss = an_loss.view((-1, nneg)).min(1)[0]
                # distance anchor positive
                npos = int(pos_fm.size(1) / ch)
                pos_fm = pos_fm.view(-1, ch, pos_fm.size(2), pos_fm.size(3))
                ap_loss = self.loss(anchor_fm.repeat(1, npos, 1, 1).view(-1, ch, anchor_fm.size(2), anchor_fm.size(3)),
                                    pos_fm)
                ap_loss = ap_loss.view((-1, npos)).max(1)[0]
                # distance negative negative2
                neg2_fm = neg2_fm.view(-1, ch, neg2_fm.size(2), neg2_fm.size(3))
                nn_loss = self.loss(neg2_fm, neg_fm)
                nn_loss = nn_loss.view((-1, nneg)).min(1)[0]

                sstl = F.relu(ap_loss - an_loss + args.alpha) + F.relu(ap_loss - nn_loss + args.alpha2)
                loss = torch.sum(sstl) / args.batch_size

                loss.backward()
                self.optimizer1.step()
                self.optimizer2.step()

                agg_loss += loss.item()
                train_loss += loss.item()

                loop.set_description(f'Epoch [{e}/{args.epochs}]')
                loop.set_postfix(loss_inference="{:.6f}".format(agg_loss))

            self.writer.add_scalar("lr1", scalar_value=self.optimizer1.state_dict()['param_groups'][0]['lr'],
                                   global_step=(e + 1))
            self.writer.add_scalar("lr2", scalar_value=self.optimizer2.state_dict()['param_groups'][0]['lr'],
                                   global_step=(e + 1))
            self.writer.add_scalar("loss_inference", scalar_value=train_loss,
                                   global_step=((e + 1) * epoch_steps))

            train_loss = 0

            if args.checkpoint_dir is not None and e % args.checkpoint_interval == 0:
                self.save(args.checkpoint_dir, e)

            scheduler1.step()
            scheduler2.step()

        self.writer.close()

    def save(self, checkpoint_dir, e):
        if self.args.loss_type == "stssim":
            self.loss.eval()
            self.loss.cpu()
            loss_model_filename = os.path.join(checkpoint_dir, "loss_epoch_" + str(e) + ".pth")
            torch.save(self.loss.state_dict(), loss_model_filename)
            self.loss.cuda()
            self.loss.train()

        self.inference.eval()
        self.inference.cpu()
        ckpt_model_filename = os.path.join(checkpoint_dir, "ckpt_epoch_" + str(e) + ".pth")
        torch.save(self.inference.state_dict(), ckpt_model_filename)
        self.inference.cuda()
        self.inference.train()

    def load(self, checkpoint_dir):
        self.inference.load_state_dict(torch.load(checkpoint_dir))
        self.inference.cuda()
