import os
from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau
import torch
from tqdm import tqdm
import torchvision.utils
from torch.autograd import Variable

from vgg16 import FeatureExtraction, FeatureCorrelation
from models.pytorch_mssim import SSIM, SSIMGNN, RSSSIM
from torchvision import transforms
import torchvision
from torch.utils.data import DataLoader
from data.data_factory import Factory
from torch.nn import functional as F


def logging(msg, suc=True):
    if suc:
        print("[*] " + msg)
    else:
        print("[!] " + msg)


class Model(object):
    def __init__(self, args, writer):
        self.args = args
        self.writer = writer
        self.batch_size = args.batch_size
        self.samples_subject = args.samples_subject
        self.train_loader, self.dataset_size = self._build_dataset_loader(args)
        self.inference, self.loss_t, self.loss_k = self._build_model(args)
        self.optimizer = torch.optim.Adam(self.inference.parameters(), args.learning_rate)

    def _build_dataset_loader(self):
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        train_dataset = Factory(self.args.train_path, self.args.feature_path, self.args.input_size, transform=transform,
                                valid_ext=['.bmp', '.jpg', '.JPG'], train=True, n_tuple=self.args.n_tuple,
                                if_augment=self.args.if_augment)
        logging("Successfully Load {} as training dataset...".format(self.args.train_path))
        train_loader = DataLoader(dataset=train_dataset, batch_size=self.args.batch_size, shuffle=True, num_workers=4)

        if self.args.n_tuple in ['triplet']:
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

    def _build_model(self):
        inference = FeatureExtraction(train_fe=True).cuda()
        data = torch.randn([3, 128, 128]).unsqueeze(0).cuda()
        data = Variable(data, requires_grad=False)
        self.writer.add_graph(inference, [data])
        inference.cuda()
        inference.train()
        logging("Successfully building FeatureExtraction model")

        loss_t = SSIM(data_range=1., size_average=False, channel=256).cuda()
        logging("Successfully building SSIM loss")
        loss_t.cuda()

        loss_k = FeatureCorrelation()
        logging("Successfully building FeatureCorrelation loss")
        loss_k.cuda()

        return inference, loss_t, loss_k

    def quadruplet_train(self):
        epoch_steps = len(self.train_loader)
        train_loss = 0
        start_epoch = ''.join(x for x in os.path.basename(self.args.start_ckpt) if x.isdigit())
        if start_epoch:
            start_epoch = int(start_epoch) + 1
            self.load(self.args.start_ckpt)
        else:
            start_epoch = 1

        # 0-100: 0.01; 150-450: 0.001; 450-800:0.0001; 800-：0.00001
        scheduler = MultiStepLR(self.optimizer, milestones=[10, 500, 1000], gamma=0.1)

        for e in range(start_epoch, self.args.epochs + start_epoch):
            self.inference.train()
            agg_loss = 0.
            loop = tqdm(enumerate(self.train_loader), total=len(self.train_loader))
            for batch_id, (x, _) in loop:
                # ======================================================== train inference model
                x = x.cuda()
                x = Variable(x, requires_grad=False)
                fms32, fms8 = self.inference(x.view(-1, 3, x.size(2), x.size(3)))
                # (batch_size, anchor+positive+negative, 32, 32)
                bs, ch, he, wi = fms32.shape
                # -------------------------------------------------- texture loss
                fms32 = fms32.view(x.size(0), -1, fms32.size(2), fms32.size(3))
                anchor_fm = fms32[:, 0:ch, :, :]  # anchor has one sample
                if len(anchor_fm.shape) == 3:
                    anchor_fm.unsqueeze(1)
                pos_fm = fms32[:, 1 * ch:self.samples_subject * ch, :, :].contiguous()
                neg_fm = fms32[:, self.samples_subject * ch:2 * self.samples_subject * ch, :, :].contiguous()
                neg2_fm = fms32[:, 2 * self.samples_subject * ch:, :, :].contiguous()
                # distance anchor negative
                nneg = int(neg_fm.size(1) / ch)
                neg_fm = neg_fm.view(-1, ch, neg_fm.size(2), neg_fm.size(3))
                an_loss = self.loss_t(
                    anchor_fm.repeat(1, nneg, 1, 1).view(-1, ch, anchor_fm.size(2), anchor_fm.size(3)),
                    neg_fm)
                an_loss = an_loss.view((-1, nneg)).min(1)[0]
                # distance anchor positive
                npos = int(pos_fm.size(1) / ch)
                pos_fm = pos_fm.view(-1, ch, pos_fm.size(2), pos_fm.size(3))
                ap_loss = self.loss_t(
                    anchor_fm.repeat(1, npos, 1, 1).view(-1, ch, anchor_fm.size(2), anchor_fm.size(3)),
                    pos_fm)
                ap_loss = ap_loss.view((-1, npos)).max(1)[0]
                # distance negative negative2
                neg2_fm = neg2_fm.view(-1, ch, neg2_fm.size(2), neg2_fm.size(3))
                nn_loss = self.loss_t(neg2_fm, neg_fm)
                nn_loss = nn_loss.view((-1, nneg)).min(1)[0]
                quadruplet_ssim = F.relu(ap_loss - an_loss + self.args.alpha) + \
                                  F.relu(ap_loss - nn_loss + self.args.alpha2)
                loss_t = torch.sum(quadruplet_ssim) / self.args.batch_size
                # --------------------------------------------------- keypoint loss
                fms8 = fms8.view(x.size(0), -1, fms8.size(2), fms8.size(3))
                anchor_fm = fms8[:, 0:ch, :, :]  # anchor has one sample
                if len(anchor_fm.shape) == 3:
                    anchor_fm.unsqueeze(1)
                pos_fm = fms8[:, 1 * ch:self.samples_subject * ch, :, :].contiguous()
                neg_fm = fms8[:, self.samples_subject * ch:2 * self.samples_subject * ch, :, :].contiguous()
                neg2_fm = fms8[:, 2 * self.samples_subject * ch:, :, :].contiguous()
                # distance anchor negative
                nneg = int(neg_fm.size(1) / ch)
                neg_fm = neg_fm.view(-1, ch, neg_fm.size(2), neg_fm.size(3))
                an_loss = self.loss_k(
                    anchor_fm.repeat(1, nneg, 1, 1).view(-1, ch, anchor_fm.size(2), anchor_fm.size(3)),
                    neg_fm)
                an_loss = an_loss.view((-1, nneg)).min(1)[0]
                # distance anchor positive
                npos = int(pos_fm.size(1) / ch)
                pos_fm = pos_fm.view(-1, ch, pos_fm.size(2), pos_fm.size(3))
                ap_loss = self.loss_k(
                    anchor_fm.repeat(1, npos, 1, 1).view(-1, ch, anchor_fm.size(2), anchor_fm.size(3)),
                    pos_fm)
                ap_loss = ap_loss.view((-1, npos)).max(1)[0]
                # distance negative negative2
                neg2_fm = neg2_fm.view(-1, ch, neg2_fm.size(2), neg2_fm.size(3))
                nn_loss = self.loss_k(neg2_fm, neg_fm)
                nn_loss = nn_loss.view((-1, nneg)).min(1)[0]
                quadruplet_cor = F.relu(ap_loss - an_loss + self.args.alpha) + \
                                 F.relu(ap_loss - nn_loss + self.args.alpha2)
                loss_k = torch.sum(quadruplet_cor) / self.args.batch_size

                loss = loss_t + loss_k
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                agg_loss += loss.item()
                train_loss += loss.item()

                loop.set_description(f'Epoch [{e}/{self.args.epochs}]')
                loop.set_postfix({"total_loss": "{:.6f}".format(agg_loss),
                                  "texture_loss": "{:.6f}".format(loss_t.item()),
                                  "keypoint_loss": "{:.6f}".format(loss_k.item())})

            self.writer.add_scalar("lr", scalar_value=self.optimizer.state_dict()['param_groups'][0]['lr'],
                                   global_step=(e + 1))
            self.writer.add_scalar("loss_inference", scalar_value=train_loss,
                                   global_step=((e + 1) * epoch_steps))

            train_loss = 0

            if self.args.checkpoint_dir is not None and e % self.args.checkpoint_interval == 0:
                self.save(self.args.checkpoint_dir, e)

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
