import os
from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau
import torch
from tqdm import tqdm
import torchvision.utils
from torch.autograd import Variable
from models.net_model import ResidualFeatureNet, DeConvRFNet, RFNWithSTNet, ConvNet
from models.loss_function import WholeImageRotationAndTranslation, ImageBlockRotationAndTranslation, \
    ShiftedLoss, MSELoss, HammingDistance, AttentionScore
from torchvision import transforms
import torchvision
from torch.utils.data import DataLoader
from data.data_factory import Factory
from models.EfficientNetV2 import efficientnetv2_s, ConvBNAct
from collections import OrderedDict
from functools import partial


def logging(msg, suc=True):
    if suc:
        print("[*] " + msg)
    else:
        print("[!] " + msg)


model_dict = {
    "RFNet": ResidualFeatureNet().cuda(),
    "DeConvRFNet": DeConvRFNet().cuda(),
    "EfficientNetV2-S": efficientnetv2_s().cuda(),
    "RFNWithSTNet": RFNWithSTNet().cuda(),
    "ConvNet": ConvNet().cuda(),
}


class Model(object):
    def __init__(self, args, writer):
        self.writer = writer
        self.batch_size = args.batch_size
        self.train_loader, self.dataset_size = self._build_dataset_loader(args)
        self.inference, self.loss = self._build_model(args)
        self.optimizer = torch.optim.SGD(self.inference.parameters(), args.learning_rate)
        self.samples_subject = args.samples_subject

    def _build_dataset_loader(self, args):
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        train_dataset = Factory(args.train_path, args.feature_path, input_size=args.input_size, transform=transform,
                                valid_ext=['.bmp', '.jpg', '.JPG'], train=True)
        logging("Successfully Load {} as training dataset...".format(args.train_path))
        train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

        examples = iter(train_loader)
        example_data, example_target, example_feature = examples.next()
        example_anchor = example_data[:, 0:3, :, :]
        example_positive = example_data[:, 3:6, :, :]
        example_negative = example_data[:, 6:9, :, :]
        anchor_grid = torchvision.utils.make_grid(example_anchor)
        self.writer.add_image(tag="anchor", img_tensor=anchor_grid)
        positive_grid = torchvision.utils.make_grid(example_positive)
        self.writer.add_image(tag="positive", img_tensor=positive_grid)
        negative_grid = torchvision.utils.make_grid(example_negative)
        self.writer.add_image(tag="negative", img_tensor=negative_grid)

        return train_loader, len(train_dataset)

    def exp_lr_scheduler(self, epoch, lr_decay=0.1, lr_decay_epoch=100):
        if epoch % lr_decay_epoch == 0:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] *= lr_decay

    def _build_model(self, args):
        if args.model not in ["RFNet", "DeConvRFNet", "EfficientNetV2-S", "RFNWithSTNet", "ConvNet"]:
            raise RuntimeError('Model not found')
        inference = model_dict[args.model].cuda().eval()
        if args.model == "EfficientNetV2-S":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            pre_trained = "/home/zhenyuzhou/Desktop/finger-knuckle/deep-learning/" \
                          "Finger-Knuckle-Assistant-Recognition/checkpoint/EfficientNetV2-S/pre_efficientnetv2-s.pth"
            weights_dict = torch.load(pre_trained, map_location=device)
            load_weights_dict = {k: v for k, v in weights_dict.items()
                                 if inference.state_dict()[k].numel() == v.numel()}
            print(inference.load_state_dict(load_weights_dict, strict=False))
            norm_layer = partial(torch.nn.BatchNorm2d, eps=1e-3, momentum=0.1)
            fk_head = OrderedDict()
            # head_input_c of efficientnet_s: 512
            fk_head.update({"conv1": ConvBNAct(256,
                                               64,
                                               kernel_size=3,
                                               norm_layer=norm_layer)})  # 激活函数默认是SiLU

            fk_head.update({"conv2": ConvBNAct(64,
                                               1,
                                               kernel_size=3,
                                               norm_layer=norm_layer)})

            fk_head = torch.nn.Sequential(fk_head)
            inference.head = fk_head
            inference = inference.cuda().eval()
            data = torch.randn([3, 300, 300]).unsqueeze(0).cuda()
        else:
            data = torch.randn([3, 128, 128]).unsqueeze(0).cuda()

        data = Variable(data, requires_grad=False)
        self.writer.add_graph(inference, data)

        if args.shifttype == "wholeimagerotationandtranslation":
            loss = WholeImageRotationAndTranslation(args.vertical_size, args.horizontal_size, args.rotate_angle).cuda()
            logging("Successfully building whole image rotation and translation triplet loss")
            inference.train()
            inference.cuda()
        elif args.shifttype == "imageblockrotationandtranslation":
            loss = ImageBlockRotationAndTranslation(args.block_size, args.vertical_size, args.horizontal_size,
                                                    args.rotate_angle).cuda()
            logging("Successfully building image block rotation and translation triplet loss")
            inference.train()
            inference.cuda
        else:
            if args.shifttype == "shiftedloss":
                loss = ShiftedLoss(hshift=args.vertical_size, vshift=args.horizontal_size).cuda()
                logging("Successfully building shifted triplet loss")
                inference.train()
                inference.cuda
            else:
                if args.shifttype == "attentionscore":
                    loss = AttentionScore().cuda()
                    logging("Successfully building attention score triplet loss")
                    inference.train()
                    inference.cuda()
                else:
                    raise RuntimeError('Model loss not found')

        return inference, loss

    def triplet_train(self, args):
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
            count = 0
            # for batch_id, (x, _) in enumerate(self.train_loader):
            # for batch_id, (x, _) in tqdm(enumerate(self.train_loader), total=len(self.train_loader)):
            loop = tqdm(enumerate(self.train_loader), total=len(self.train_loader))
            for batch_id, (x, _, assistant_f) in loop:

                count += len(x)
                x = x.cuda()
                x = Variable(x, requires_grad=False)
                fms = self.inference(x.view(-1, 3, x.size(2), x.size(3)))
                # (batch_size, 12, 32, 32)
                fms = fms.view(x.size(0), -1, fms.size(2), fms.size(3))

                anchor_fm = fms[:, 0, :, :].unsqueeze(1)
                pos_fm = fms[:, 1:self.samples_subject-1, :, :].unsqueeze(1)
                neg_fm = fms[:, self.samples_subject, :, :].contiguous()

                nneg = neg_fm.size(1)
                neg_fm = neg_fm.view(-1, 1, neg_fm.size(2), neg_fm.size(3))
                an_loss = self.loss(anchor_fm.repeat(1, nneg, 1, 1).view(-1, 1, anchor_fm.size(2), anchor_fm.size(3)),
                                    neg_fm)
                # an_loss.shape:-> (batch_size, 10)
                # min(1) will get min value and the corresponding indices
                # min(1)[0]
                an_loss = an_loss.view((-1, nneg)).min(1)[0]
                ap_loss = self.loss(anchor_fm, pos_fm)

                sstl = ap_loss - an_loss + args.alpha
                sstl = torch.clamp(sstl, min=0)

                loss = torch.sum(sstl) / args.batch_size

                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                agg_loss += loss.item()
                train_loss += loss.item()

                # if e % args.log_interval == 0:
                #     message = "{}\tEpoch {}:\t[{}/{}]\t {:.6f}".\
                #         format(time.ctime(), e, count, self.dataset_size, agg_loss/(batch_id+1))
                # print(message)
                loop.set_description(f'Epoch [{e}/{args.epochs}]')
                loop.set_postfix(cumloss="{:.6f}".format(agg_loss))

            self.writer.add_scalar("lr", scalar_value=self.optimizer.state_dict()['param_groups'][0]['lr'],
                                   global_step=(e + 1))
            self.writer.add_scalar("loss", scalar_value=train_loss,
                                   global_step=((e + 1) * epoch_steps))
            train_loss = 0

            if args.checkpoint_dir is not None and e % args.checkpoint_interval == 0:
                self.save(args.checkpoint_dir, e)

            scheduler.step()

        self.writer.close()

    def save(self, checkpoint_dir, e):
        self.inference.eval()
        self.inference.cpu()
        ckpt_model_filename = os.path.join(checkpoint_dir, "ckpt_epoch_" + str(e) + ".pth")
        torch.save(self.inference.state_dict(), ckpt_model_filename)
        self.inference.cuda()
        self.inference.train()

    def load(self, checkpoint_dir):
        self.inference.load_state_dict(torch.load(checkpoint_dir))
        self.inference.cuda()
