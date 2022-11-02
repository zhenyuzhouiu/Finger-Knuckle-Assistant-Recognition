import os
from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau
import torch
from tqdm import tqdm
import torchvision.utils
from torch.autograd import Variable
from models.net_model import ResidualFeatureNet, DeConvRFNet, RFNWithSTNet, ConvNet, AssistantModel
from models.loss_function import WholeImageRotationAndTranslation, ImageBlockRotationAndTranslation, \
    ShiftedLoss, MSELoss, HammingDistance, AttentionScore
from torchvision import transforms
import torchvision
from torch.utils.data import DataLoader
from data.data_factory import Factory
from models.EfficientNetV2 import efficientnetv2_s, ConvBNAct, fk_efficientnetv2_s
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
    "FKEfficientNet": fk_efficientnetv2_s().cuda(),
    "RFNWithSTNet": RFNWithSTNet().cuda(),
    "ConvNet": ConvNet().cuda(),
}


class Model(object):
    def __init__(self, args, writer):
        self.writer = writer
        self.batch_size = args.batch_size
        self.samples_subject = args.samples_subject
        self.train_loader, self.dataset_size = self._build_dataset_loader(args)
        self.inference, self.assistant, self.loss = self._build_model(args)
        self.optimizer = torch.optim.Adam(self.inference.parameters(), args.learning_rate)
        self.assistant_optimizer = torch.optim.Adam(self.assistant.parameters(), args.learning_rate*0.1)

    def _build_dataset_loader(self, args):
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        train_dataset = Factory(args.train_path, args.feature_path, args.input_size, transform=transform,
                                valid_ext=['.bmp', '.jpg', '.JPG'], train=True)
        logging("Successfully Load {} as training dataset...".format(args.train_path))
        train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

        examples = iter(train_loader)
        example_data, example_target, example_feature8, example_feature16, example_feature32 = examples.next()
        example_anchor = example_data[:, 0:3, :, :]
        example_positive = example_data[:, 3:3 * self.samples_subject, :, :].reshape(-1, 3, example_anchor.size(2), example_anchor.size(3))
        example_negative = example_data[:, 3 * self.samples_subject:, :, :].reshape(-1, 3, example_anchor.size(2), example_anchor.size(3))
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
        if args.model not in ["RFNet", "DeConvRFNet", "FKEfficientNet", "RFNWithSTNet", "ConvNet"]:
            raise RuntimeError('Model not found')
        inference = model_dict[args.model].cuda().eval()
        data = torch.randn([3, 128, 128]).unsqueeze(0).cuda()
        data = Variable(data, requires_grad=False)
        self.writer.add_graph(inference, data)

        assistant = AssistantModel().cuda()

        if args.shifttype == "wholeimagerotationandtranslation":
            loss = WholeImageRotationAndTranslation(args.vertical_size, args.horizontal_size, args.rotate_angle).cuda()
            logging("Successfully building whole image rotation and translation triplet loss")
            inference.train()
            inference.cuda()
            assistant.train()
            assistant.cuda()
        elif args.shifttype == "imageblockrotationandtranslation":
            loss = ImageBlockRotationAndTranslation(args.block_size, args.vertical_size, args.horizontal_size,
                                                    args.rotate_angle).cuda()
            logging("Successfully building image block rotation and translation triplet loss")
            inference.train()
            inference.cuda()
            assistant.train()
            assistant.cuda()
        else:
            if args.shifttype == "shiftedloss":
                loss = ShiftedLoss(hshift=args.vertical_size, vshift=args.horizontal_size).cuda()
                logging("Successfully building shifted triplet loss")
                inference.train()
                inference.cuda()
                assistant.train()
                assistant.cuda()
            else:
                if args.shifttype == "attentionscore":
                    loss = AttentionScore().cuda()
                    logging("Successfully building attention score triplet loss")
                    inference.train()
                    inference.cuda()
                    assistant.train()
                    assistant.cuda()
                else:
                    raise RuntimeError('Model loss not found')

        return inference, assistant, loss

    def triplet_train(self, args):
        epoch_steps = len(self.train_loader)
        total_loss = 0
        train_loss = 0
        train_loss_assistant = 0
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
            self.assistant.train()
            agg_loss_assistant = 0.
            agg_loss_total = 0.
            # for batch_id, (x, _) in enumerate(self.train_loader):
            # for batch_id, (x, _) in tqdm(enumerate(self.train_loader), total=len(self.train_loader)):
            loop = tqdm(enumerate(self.train_loader), total=len(self.train_loader))
            for batch_id, (x, _, assistant_f8, assistant_f16, assistant_f32) in loop:
                # ======================================================== train inference model
                # x.shape :-> [b, 3*3*samples_subject, h, w]
                # y.shape:
                x = x.cuda()
                x = Variable(x, requires_grad=False)
                fms = self.inference(x.view(-1, 3, x.size(2), x.size(3)))
                # (batch_size, anchor+positive+negative, 32, 32)
                fms = fms.view(x.size(0), -1, fms.size(2), fms.size(3))

                anchor_fm = fms[:, 0, :, :].unsqueeze(1)  # anchor has one sample
                pos_fm = fms[:, 1:self.samples_subject, :, :].contiguous()
                neg_fm = fms[:, self.samples_subject:, :, :].contiguous()
                nneg = neg_fm.size(1)
                neg_fm = neg_fm.view(-1, 1, neg_fm.size(2), neg_fm.size(3))
                an_loss = self.loss(anchor_fm.repeat(1, nneg, 1, 1).view(-1, 1, anchor_fm.size(2), anchor_fm.size(3)),
                                    neg_fm)
                # an_loss.shape:-> (batch_size, 10)
                # min(1) will get min value and the corresponding indices
                # min(1)[0]
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

                # ============================================================== train assistant model
                # assistant_f8.shape :-> [b, 320*3*samples_subject, 32, 32]
                assistant_f8 = assistant_f8.cuda()
                assistant_f8 = Variable(assistant_f8, requires_grad=False)
                assistant_f16 = assistant_f16.cuda()
                assistant_f16 = Variable(assistant_f16, requires_grad=False)
                assistant_f32 = assistant_f32.cuda()
                assistant_f32 = Variable(assistant_f32, requires_grad=False)

                fms_assistant = self.assistant(assistant_f8.view(-1, 320, assistant_f8.size(2), assistant_f8.size(3)),
                                               assistant_f16.view(-1, 640, assistant_f16.size(2), assistant_f16.size(3)),
                                               assistant_f32.view(-1, 1280, assistant_f32.size(2), assistant_f32.size(3)))

                # (batch_size, 12, 32, 32)
                fms_assistant = fms_assistant.view(assistant_f8.size(0), -1, fms_assistant.size(2), fms_assistant.size(3))

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
                ap_loss = self.loss(anchor_fm.repeat(1, npos, 1, 1).view(-1, 1, anchor_fm.size(2), anchor_fm.size(3)), pos_fm)
                ap_loss = ap_loss.view((-1, npos)).max(1)[0]

                assistant_sstl = ap_loss - an_loss + args.alpha
                assistant_sstl = torch.clamp(assistant_sstl, min=0)

                assistant_loss = torch.sum(assistant_sstl) / args.batch_size

                assistant_loss.backward()
                self.assistant_optimizer.step()
                self.assistant_optimizer.zero_grad()
                agg_loss_assistant += assistant_loss.item()
                train_loss_assistant += assistant_loss.item()
                # ================================================================= total loss = inference loss + assistant loss
                agg_loss_total = agg_loss + agg_loss_assistant
                total_loss = train_loss + train_loss_assistant
                loop.set_description(f'Epoch [{e}/{args.epochs}]')
                loop.set_postfix(loss_inference="{:.6f}".format(agg_loss),
                                 loss_assistant="{:.6f}".format(agg_loss_assistant),
                                 loss_total="{:.6f}".format(agg_loss_total))

            self.writer.add_scalar("lr", scalar_value=self.optimizer.state_dict()['param_groups'][0]['lr'],
                                   global_step=(e + 1))
            self.writer.add_scalar("loss_inference", scalar_value=train_loss,
                                   global_step=((e + 1) * epoch_steps))
            self.writer.add_scalar("loss_assistant", scalar_value=train_loss_assistant,
                                   global_step=((e + 1) * epoch_steps))
            self.writer.add_scalar("loss_total", scalar_value=total_loss,
                                   global_step=((e + 1) * epoch_steps))

            train_loss = 0
            train_loss_assistant = 0
            total_loss = 0

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
