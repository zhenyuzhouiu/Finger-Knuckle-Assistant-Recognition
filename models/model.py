import os
from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau
import torch
from tqdm import tqdm
import torchvision.utils
from torch.autograd import Variable

import models.loss_function
from models.net_model import ResidualFeatureNet, RFNet64, SERFNet64,\
    STNRFNet64, STNResRFNet64, STNResRFNet64v2, STNResRFNet64v3, DeformRFNet64, DilateRFNet64, RFNet64Relu, STNResRFNet64v2Relu
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
}


class Model(object):
    def __init__(self, args, writer):
        self.args = args
        self.writer = writer
        self.batch_size = args.batch_size
        self.samples_subject = args.samples_subject
        self.train_loader, self.dataset_size = self._build_dataset_loader(args)
        self.inference, self.loss = self._build_model(args)
        if self.args.loss_type == "ssimgnn":
            self.optimizer = torch.optim.Adam([{'params': self.inference.parameters(),
                                                'params': self.loss.parameters()}],
                                              args.learning_rate)
        else:
            self.optimizer = torch.optim.Adam(self.inference.parameters(), args.learning_rate1)

    def _build_dataset_loader(self, args):
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        train_dataset = Factory(args.train_path, args.feature_path, args.input_size, transform=transform,
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
        else:
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

        return train_loader, len(train_dataset)

    def exp_lr_scheduler(self, epoch, lr_decay=0.1, lr_decay_epoch=100):
        if epoch % lr_decay_epoch == 0:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] *= lr_decay

    def _build_model(self, args):
        if args.model not in ["RFNet", "FKEfficientNet", "RFNet64", "RFNet64_16", "SERFNet64",
                              "STNRFNet64", "STNResRFNet64", "STNResRFNet64v2", "STNResRFNet64v3",
                              "DilateRFNet64", "DeformRFNet64", "RFNet64Relu", "STNResRFNet64v2Relu"]:
            raise RuntimeError('Model not found')
        inference = model_dict[args.model].cuda()
        data = torch.randn([3, 128, 128]).unsqueeze(0).cuda()
        data = Variable(data, requires_grad=False)
        self.writer.add_graph(inference, [data])
        inference.cuda()
        inference.train()

        if args.loss_type == "rsil":
            loss = RSIL(args.vertical_size, args.horizontal_size, args.rotate_angle).cuda()
            logging("Successfully building rsil triplet loss")
        elif args.loss_type == "shiftedloss":
            # loss = MaskRSIL(args.vertical_size, args.horizontal_size, args.rotate_angle).cuda()
            loss = ShiftedLoss(hshift=args.horizontal_size, vshift=args.vertical_size).cuda()
            logging("Successfully building shiftedloss triplet loss")
        else:
            if args.loss_type == "ssim":
                loss = SSIM(data_range=1., size_average=False, channel=64).cuda()
                logging("Successfully building ssim triplet loss")
            elif args.loss_type == "ssimgnn":
                loss = SSIMGNN(data_range=1., size_average=False, channel=64, config=args.sglue_conf).cuda()
                logging("Successfully building ssimgnn triplet loss")
            else:
                if args.loss_type == "rsssim":
                    loss = RSSSIM(data_range=1., size_average=False, channel=64, v_shift=args.vertical_size,
                                  h_shift=args.horizontal_size, angle=args.rotate_angle, step=args.step_size).cuda()
                    logging("Successfully building rsssim triplet loss")
                else:
                    raise RuntimeError('Please make sure your loss function!')
        loss.cuda()
        loss.eval()
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
        scheduler = MultiStepLR(self.optimizer, milestones=[10, 500, 1000], gamma=0.1)
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
                self.optimizer.step()
                self.optimizer.zero_grad()
                agg_loss += loss.item()
                train_loss += loss.item()

                loop.set_description(f'Epoch [{e}/{args.epochs}]')
                loop.set_postfix(loss_inference="{:.6f}".format(agg_loss))

            self.writer.add_scalar("lr", scalar_value=self.optimizer.state_dict()['param_groups'][0]['lr'],
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

    def masktriplet_train(self, args):
        epoch_steps = len(self.train_loader)
        train_loss = 0
        start_epoch = ''.join(x for x in os.path.basename(args.start_ckpt) if x.isdigit())
        if start_epoch:
            start_epoch = int(start_epoch) + 1
            self.load(args.start_ckpt)
        else:
            start_epoch = 1

        # 0-100: 0.01; 150-450: 0.001; 450-800:0.0001; 800-：0.00001
        scheduler = MultiStepLR(self.optimizer, milestones=[10, 500, 1000], gamma=0.1)
        # for freeze spatial transformer network
        freeze_stn = args.freeze_stn

        for e in range(start_epoch, args.epochs + start_epoch):
            # self.exp_lr_scheduler(e, lr_decay_epoch=100)
            self.inference.train()
            agg_loss = 0.
            # for batch_id, (x, _) in enumerate(self.train_loader):
            # for batch_id, (x, _) in tqdm(enumerate(self.train_loader), total=len(self.train_loader)):
            loop = tqdm(enumerate(self.train_loader), total=len(self.train_loader))
            for batch_id, (x, mask, _) in loop:
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
                # mask.shape :-> [b, 3*samples_subject, h, w]
                x = x.cuda()
                x = Variable(x, requires_grad=False)
                mask = mask.cuda()
                mask = Variable(mask, requires_grad=False)
                fms, mask = self.inference(x.view(-1, 3, x.size(2), x.size(3)),
                                           mask.view(-1, 1, mask.size(2), mask.size(3)))
                bs, ch, he, wi = fms.shape
                # (batch_size, anchor+positive+negative, 32, 32)
                fms = fms.view(x.size(0), -1, fms.size(2), fms.size(3))
                mask = mask.view(x.size(0), -1, mask.size(2), mask.size(3))

                anchor_fm = fms[:, 0:ch, :, :]  # anchor has one sample
                if len(anchor_fm.shape) == 3:
                    anchor_fm.unsqueeze(1)
                anchor_mask = mask[:, 1, :, :].unsqueeze(1)
                pos_fm = fms[:, 1 * ch:self.samples_subject * ch, :, :].contiguous()
                pos_mask = mask[:, 1:self.samples_subject, :, :].contiguous()
                neg_fm = fms[:, self.samples_subject * ch:, :, :].contiguous()
                neg_mask = mask[:, self.samples_subject:, :, :].contiguous()
                nneg = int(neg_fm.size(1) / ch)
                neg_fm = neg_fm.view(-1, ch, neg_fm.size(2), neg_fm.size(3))
                neg_mask = neg_mask.view(-1, 1, neg_mask.size(2), neg_fm.size(3))
                an_loss = self.loss(anchor_fm.repeat(1, nneg, 1, 1).view(-1, ch, anchor_fm.size(2), anchor_fm.size(3)),
                                    anchor_mask.repeat(1, nneg, 1, 1).view(-1, 1, anchor_mask.size(2),
                                                                           anchor_mask.size(3)),
                                    neg_fm,
                                    neg_mask)
                # an_loss.shape:-> (batch_size, 10)
                # min(1) will get min value and the corresponding indices
                # min(1)[0]
                an_loss = an_loss.view((-1, nneg)).min(1)[0]

                npos = int(pos_fm.size(1) / ch)
                pos_fm = pos_fm.view(-1, ch, pos_fm.size(2), pos_fm.size(3))
                pos_mask = pos_mask.view(-1, 1, pos_mask.size(2), pos_mask.size(3))
                ap_loss = self.loss(anchor_fm.repeat(1, npos, 1, 1).view(-1, ch, anchor_fm.size(2), anchor_fm.size(3)),
                                    anchor_mask.repeat(1, npos, 1, 1).view(-1, 1, anchor_mask.size(2),
                                                                           anchor_mask.size(3)),
                                    pos_fm,
                                    pos_mask)
                ap_loss = ap_loss.view((-1, npos)).max(1)[0]

                sstl = ap_loss - an_loss + args.alpha
                sstl = torch.clamp(sstl, min=0)

                loss = torch.sum(sstl) / args.batch_size

                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                agg_loss += loss.item()
                train_loss += loss.item()

                loop.set_description(f'Epoch [{e}/{args.epochs}]')
                loop.set_postfix(loss_inference="{:.6f}".format(agg_loss))

            self.writer.add_scalar("lr", scalar_value=self.optimizer.state_dict()['param_groups'][0]['lr'],
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
        scheduler = MultiStepLR(self.optimizer, milestones=[10, 500, 1000], gamma=0.1)

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
                self.optimizer.step()
                self.optimizer.zero_grad()
                agg_loss += loss.item()
                train_loss += loss.item()

                loop.set_description(f'Epoch [{e}/{args.epochs}]')
                loop.set_postfix(loss_inference="{:.6f}".format(agg_loss))

            self.writer.add_scalar("lr", scalar_value=self.optimizer.state_dict()['param_groups'][0]['lr'],
                                   global_step=(e + 1))
            self.writer.add_scalar("loss_inference", scalar_value=train_loss,
                                   global_step=((e + 1) * epoch_steps))

            train_loss = 0

            if args.checkpoint_dir is not None and e % args.checkpoint_interval == 0:
                self.save(args.checkpoint_dir, e)

            scheduler.step()

        self.writer.close()

    def assistant_triplet_train(self, args):
        epoch_steps = len(self.train_loader)
        train_loss = 0
        start_epoch = ''.join(x for x in os.path.basename(args.start_ckpt) if x.isdigit())
        if start_epoch:
            start_epoch = int(start_epoch) + 1
            self.load(args.start_ckpt)
        else:
            start_epoch = 1

        # 0-100: 0.01; 150-450: 0.001; 450-800:0.0001; 800-：0.00001
        scheduler = MultiStepLR(self.optimizer, milestones=[10, 1000, 1500], gamma=0.1)

        for e in range(start_epoch, args.epochs + start_epoch):
            # self.exp_lr_scheduler(e, lr_decay_epoch=100)
            self.inference.train()
            agg_loss = 0.
            loop = tqdm(enumerate(self.train_loader), total=len(self.train_loader))
            for batch_id, (x, _, assistant_f8, assistant_f16, assistant_f32) in loop:
                # ============================================================== train assistant model
                # assistant_f8.shape :-> [b, 320*3*samples_subject, 32, 32]
                assistant_f8 = assistant_f8.cuda()
                assistant_f8 = Variable(assistant_f8, requires_grad=False)
                assistant_f16 = assistant_f16.cuda()
                assistant_f16 = Variable(assistant_f16, requires_grad=False)
                assistant_f32 = assistant_f32.cuda()
                assistant_f32 = Variable(assistant_f32, requires_grad=False)

                fms_assistant = self.inference(assistant_f8.view(-1, 320, assistant_f8.size(2), assistant_f8.size(3)),
                                               assistant_f16.view(-1, 640, assistant_f16.size(2),
                                                                  assistant_f16.size(3)),
                                               assistant_f32.view(-1, 1280, assistant_f32.size(2),
                                                                  assistant_f32.size(3)))

                # (batch_size, 12, 32, 32)
                fms_assistant = fms_assistant.view(assistant_f8.size(0), -1, fms_assistant.size(2),
                                                   fms_assistant.size(3))

                anchor_fm = fms_assistant[:, 0, :, :].unsqueeze(1)
                pos_fm = fms_assistant[:, 1:self.samples_subject, :, :].contiguous()
                neg_fm = fms_assistant[:, self.samples_subject:, :, :].contiguous()

                nneg = neg_fm.size(1)
                neg_fm = neg_fm.view(-1, 1, neg_fm.size(2), neg_fm.size(3))
                an_loss = self.loss(anchor_fm.repeat(1, nneg, 1, 1).view(-1, 1, anchor_fm.size(2), anchor_fm.size(3)),
                                    neg_fm)
                an_loss = an_loss.view((-1, nneg)).min(1)[0]

                npos = pos_fm.size(1)
                pos_fm = pos_fm.view(-1, 1, pos_fm.size(2), pos_fm.size(3))
                ap_loss = self.loss(anchor_fm.repeat(1, npos, 1, 1).view(-1, 1, anchor_fm.size(2), anchor_fm.size(3)),
                                    pos_fm)
                ap_loss = ap_loss.view((-1, npos)).max(1)[0]

                sstl = ap_loss - an_loss + args.alpha
                sstl = torch.clamp(sstl, min=0)

                loss = torch.sum(sstl) / args.batch_size
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                agg_loss += loss.item()
                train_loss += loss.item()
                loop.set_description(f'Epoch [{e}/{args.epochs}]')
                loop.set_postfix(loss_inference="{:.6f}".format(agg_loss))

            self.writer.add_scalar("lr", scalar_value=self.optimizer.state_dict()['param_groups'][0]['lr'],
                                   global_step=(e + 1))
            self.writer.add_scalar("loss_inference", scalar_value=train_loss,
                                   global_step=((e + 1) * epoch_steps))

            train_loss = 0

            if args.checkpoint_dir is not None and e % args.checkpoint_interval == 0:
                self.save(args.checkpoint_dir, e)

            scheduler.step()

        self.writer.close()

    def fusion_triplet_train(self, args):
        epoch_steps = len(self.train_loader)
        train_loss = 0
        start_epoch = ''.join(x for x in os.path.basename(args.start_ckpt) if x.isdigit())
        if start_epoch:
            start_epoch = int(start_epoch) + 1
            self.load(args.start_ckpt)
        else:
            start_epoch = 1

        # 0-100: 0.01; 150-450: 0.001; 450-800:0.0001; 800-：0.00001
        scheduler = MultiStepLR(self.optimizer, milestones=[10, 500, 1000], gamma=0.1)

        for e in range(start_epoch, args.epochs + start_epoch):
            # self.exp_lr_scheduler(e, lr_decay_epoch=100)
            self.inference.train()
            agg_loss = 0.
            # for batch_id, (x, _) in enumerate(self.train_loader):
            # for batch_id, (x, _) in tqdm(enumerate(self.train_loader), total=len(self.train_loader)):
            loop = tqdm(enumerate(self.train_loader), total=len(self.train_loader))
            for batch_id, (x, _, assistant_f8, assistant_f16, assistant_f32) in loop:
                # ======================================================== train inference model
                # x.shape :-> [b, 3*3*samples_subject, h, w]
                # y.shape:
                x = x.cuda()
                x = Variable(x, requires_grad=False)
                assistant_f8 = assistant_f8.cuda()
                assistant_f8 = Variable(assistant_f8, requires_grad=False)
                assistant_f16 = assistant_f16.cuda()
                assistant_f16 = Variable(assistant_f16, requires_grad=False)
                assistant_f32 = assistant_f32.cuda()
                assistant_f32 = Variable(assistant_f32, requires_grad=False)

                # def forward(self, x, s8, s16, s32):
                fms = self.inference(x.view(-1, 3, x.size(2), x.size(3)),
                                     assistant_f8.view(-1, 320, assistant_f8.size(2), assistant_f8.size(3)),
                                     assistant_f16.view(-1, 640, assistant_f16.size(2), assistant_f16.size(3)),
                                     assistant_f32.view(-1, 1280, assistant_f32.size(2), assistant_f32.size(3)))
                # (batch_size, anchor+positive+negative, 32, 32)
                fms = fms.view(x.size(0), -1, fms.size(2), fms.size(3))

                anchor_fm = fms[:, 0, :, :].unsqueeze(1)  # anchor has one sample
                pos_fm = fms[:, 1:self.samples_subject, :, :].contiguous()
                neg_fm = fms[:, self.samples_subject:, :, :].contiguous()
                nneg = neg_fm.size(1)
                neg_fm = neg_fm.view(-1, 1, neg_fm.size(2), neg_fm.size(3))
                an_loss = self.loss(anchor_fm.repeat(1, nneg, 1, 1).view(-1, 1, anchor_fm.size(2), anchor_fm.size(3)),
                                    neg_fm)
                an_loss = an_loss.view((-1, nneg)).min(1)[0]

                npos = pos_fm.size(1)
                pos_fm = pos_fm.view(-1, 1, pos_fm.size(2), pos_fm.size(3))
                ap_loss = self.loss(anchor_fm.repeat(1, npos, 1, 1).view(-1, 1, anchor_fm.size(2), anchor_fm.size(3)),
                                    pos_fm)
                ap_loss = ap_loss.view((-1, npos)).max(1)[0]

                sstl = ap_loss - an_loss + args.alpha
                sstl = torch.clamp(sstl, min=0)

                loss = torch.sum(sstl) / args.batch_size

                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                agg_loss += loss.item()
                train_loss += loss.item()

                loop.set_description(f'Epoch [{e}/{args.epochs}]')
                loop.set_postfix(loss_inference="{:.6f}".format(agg_loss))

            self.writer.add_scalar("lr", scalar_value=self.optimizer.state_dict()['param_groups'][0]['lr'],
                                   global_step=(e + 1))
            self.writer.add_scalar("loss_inference", scalar_value=train_loss,
                                   global_step=((e + 1) * epoch_steps))

            train_loss = 0

            if args.checkpoint_dir is not None and e % args.checkpoint_interval == 0:
                self.save(args.checkpoint_dir, e)

            scheduler.step()

        self.writer.close()

    def save(self, checkpoint_dir, e):
        if self.args.loss_type == "ssimgnn":
            self.loss.eval()
            self.loss.cpu()
            ckpt_model_filename = os.path.join(checkpoint_dir, "loss_ckpt_epoch_" + str(e) + ".pth")
            torch.save(self.loss.state_dict(), ckpt_model_filename)
            self.loss.cuda()
            self.loss.train()
        self.inference.eval()
        self.inference.cpu()
        ckpt_model_filename = os.path.join(checkpoint_dir, "ckpt_epoch_" + str(e) + ".pth")
        torch.save(self.inference.state_dict(), ckpt_model_filename)
        self.inference.cuda()
        self.inference.train()

    def load(self, checkpoint_dir):
        if self.args.loss_type == "ssimgnn":
            self.loss.load_state_dict(torch.load(self.args.sglue_conf["weight"]))
            self.loss.cuda()
            self.loss.train()
        self.inference.load_state_dict(torch.load(checkpoint_dir))
        self.inference.cuda()
