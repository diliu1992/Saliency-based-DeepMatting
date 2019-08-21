import datetime
from get_dataset_loader import get_train_val_loader
from models.RefineNet import RDFNet
from models.py_encoder_decoder import Encoder_Decoder
from utils.loss import loss_function
import torch
import numpy as np
from torch import optim
import scipy.misc as misc
import torch.nn.functional as F
from utils.metrics import SScore, MScore
from utils.visulization import Visulizer
import time


"""
--------------------------------------------------------------------------
parameters for training
'dataset': dataset for training, in 'Portrait', 'Adobe','HalfHuman'
'stage': training stage, 0 for saliency prediction model,1 for matting model
'img_size': size for training stage 0
'crop_size': size for training stage 1
'pre_trained_model': path for pre-trained model
--------------------------------------------------------------------------
"""
args = {
    'dataset': 'Portrait',
    'stage': 0,
    'batch_size': 5,
    'epoch_num': 100,
    'lr': 1e-5,
    'weight_decay': 1e-4,
    'momentum': 0.9,
    'num_workers': 4,
    'img_size': 320,
    'pre_trained_model': '',
    'crop_size': 320,
}


class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.train_loader, self.valid_loader = get_train_val_loader(dataset_name=args['dataset'],
                                                                    batchsize=args['batch_size'],
                                                                    num_workers=args['num_workers'],
                                                                    img_size=args['img_size'],
                                                                    train_stage=args['stage'])
        self.stage = args['stage']
        base_lr = self.args['lr']
        self.best_mse = 1000000
        if self.stage == 0:
            self.model = RDFNet()
            if self.args['pre_trained_model']:
                self.model.load_state_dict(torch.load(self.args['pretrain_model']))
            self.trainer = optim.Adam(self.model.parameters(), lr=args['lr'], betas=(0.9, 0.99))
        if self.stage == 1:
            self.model = Encoder_Decoder()
            if not self.args['pre_trained_model']:
                self.model.load_vggbn('./checkpoints/vgg16_bn-6c64b313.pth')
            else:
                self.model.load_state_dict(torch.load(self.args['pretrain_model']))
            self.trainer = optim.Adam(self.model.parameters(), lr=args['lr'], betas=(0.9, 0.999), weight_decay=0.0005)
        if torch.cuda.is_available():
            self.model = self.model.cuda()

    def training(self, epoch):
        self.model.train()
        total_loss, prev_loss = 0, 0
        for i, (data, image, alpha, trimap, im_name, size) in enumerate(self.train_loader):
            if torch.cuda.is_available():
                data, image, alpha = data.cuda(), image.cuda(), alpha.cuda()
            self.trainer.zero_grad()
            if self.args['stage'] == 0:
                outputs = self.model(image)

                if torch.cuda.is_available():
                    loss = F.binary_cross_entropy(torch.sigmoid(outputs), alpha).cuda()
                total_loss += loss.item()
                loss.backward()
                self.trainer.step()
                if (i + 1) % 20 == 0:
                    print("Epoch [%d/%d] Loss: %.4f" % (epoch + 1, args['epoch_num'], total_loss / (i + 1)))

            if self.args['stage'] == 1:
                outputs = self.model(data)
                alpha_pre = outputs
                alpha_gt = alpha
                trimap = trimap.data.cpu().numpy()[:, 0, :, :]
                # mask for uncertain area
                mask = np.equal(trimap, 128).astype(np.float32)
                mask = torch.FloatTensor(np.expand_dims(mask, axis=1)).cuda()
                if torch.cuda.is_available():
                    loss = loss_function(image, alpha_pre, alpha_gt, mask).cuda()
                loss.backward()
                self.trainer.step()
                total_loss += loss.item()
                if (i + 1) % 20 == 0:
                    print("Epoch [%d/%d] Loss: %.4f" % (epoch + 1, args['epoch_num'], total_loss / (i + 1)))
                    for j in range(args['batch_size']):
                        visdata = alpha_pre.data.cpu().numpy()[j, 0, :, :]
                        visdata = (visdata - visdata.min()) / (visdata.max() - visdata.min())
                        vis_mask = mask.data.cpu().numpy()[j, 0, :, :]
                        visdata = visdata * vis_mask + np.equal(trimap[j, :, :], 255).astype(np.float32)
                        # vis.img('trimap_{0}'.format(j), trimap[j, :, :])
                        # vis.img('alpha_gt_{0}'.format(j), alpha_gt[j, 0, :, :])
                        # vis.img('alpha_pre_{0}'.format(j), visdata)
        print("Epoch total loss: %.4f" % total_loss)
        # vis.plot("total_loss", total_loss)
        # vis.log("training epoch {0} finished ".format(epoch))

    def validation(self, epoch):
        self.model.eval()
        if self.args['stage'] == 0:
            sal_metrics = SScore()
            for i, (data, image, alpha, trimap, im_name, size) in enumerate(self.valid_loader):
                if torch.cuda.is_available():
                    data, image, alpha = data.cuda(), image.cuda(), alpha.cuda()
                outputs = self.model(image)
                outputs = torch.sigmoid(outputs)
                sal = outputs.data.cpu().numpy()[:, 0, :, :]
                gt = alpha.data.cpu().numpy()[:, 0, :, :]
                sal_metrics.update(sal, gt)
                for j in range(sal.shape[0]):
                    out = misc.imresize(sal[j, :, :], (size[0].numpy()[j], size[1].numpy()[j]))
                    misc.imsave("./outputs/{}/val/{}/{}.png".format(args['dataset'], self.stage, im_name[j]), out)
            score = sal_metrics.get_scores()
            for k, v in score.items():
                print(k, v)
            sal_metrics.reset()
            if score['Mean MSE: \t'] < self.best_mse:
                self.best_mse = score['Mean MSE: \t']
                self.save_model(epoch)
            print('best mse = {}'.format(self.best_mse))

        if self.args['stage'] == 1:
            matte_metrics = MScore()
            for i, (data, image, alpha, trimap, im_name, size) in enumerate(self.valid_loader):
                if torch.cuda.is_available():
                    data, image, alpha = data.cuda(), image.cuda(), alpha.cuda()
                outputs = self.model(data)
                alpha_gt = alpha.data.cpu().numpy()[:, 0, :, :]
                trimap = trimap.data.cpu().numpy()[:, 0, :, :]
                # mask for uncertain area
                mask = np.equal(trimap, 128).astype(np.float32)
                alpha_pre = outputs.data.cpu().numpy()[:, 0, :, :]
                # composed alpha prediction by assigning 1 to the known foreground
                alpha_pre = alpha_pre * mask + np.equal(trimap, 255).astype(np.float32)
                alpha_c = alpha_pre[:, np.newaxis, :, :]
                image = image.data.cpu().numpy()
                # composed image from the predicted alpha
                compose = alpha_c * image + (1 - alpha.data.cpu().numpy()) * image
                # matte evaluation
                matte_metrics.update(alpha_pre, alpha_gt, mask)
                # image saving
                for j in range(alpha.shape[0]):
                    misc.imsave("./outputs/{}/val/{}/alpha/{}.png".format(args['dataset'], self.stage, im_name[j]),
                                alpha_pre[j, :, :])
                    misc.imsave("./outputs/{}/val/{}/alpha/{}.png".format(args['dataset'], self.stage, im_name[j]),
                                compose[j, :, :, :].transpose(1, 2, 0))
                    misc.imsave("./outputs/{}/val/{}/alpha/{}.png".format(args['dataset'], self.stage, im_name[j]),
                                trimap[j, :, :])
                    # self.vis.img('merged_{0}'.format(j), image[j, :, :, :])
                    # self.vis.img('composed_{0}'.format(j), compose[j, :, :, :])
                    # self.vis.img('gt_alpha_{0}'.format(j), alpha[j, 0, :, :])
                    # self.vis.img('trimap_{0}'.format(j), trimap[j, :, :])
                    # self.vis.img('alpha_{0}'.format(j), alpha_pre[j, :, :])
            score = matte_metrics.get_scores()
            for k, v in score.items():
                print(k, v)
            matte_metrics.reset()
            if score['Mean MSE: \t'] < self.best_mse:
                self.best_mse = score['Mean MSE: \t']
                self.save_model(epoch)
            print('best mse = {}'.format(self.best_mse))
            # self.vis.log('the validation of epoch {0}'.format(epoch))

    def save_model(self, epoch):
        file_name = './checkpoints/{0}_stage{1}_epoch{2}_{3}.params'.format(args['dataset'], str(self.stage),
                                                                            str(epoch),
                                                                            str(self.best_mse)[0:6])
        torch.save(self.model.state_dict(), file_name)

    def load_model(self):
        pass


if __name__ == '__main__':
    """the is the main train logic"""
    now = datetime.datetime.now()
    now_str = '{}-{}-{} {}:{}:{}'.format(now.year, now.month, now.day, now.hour, now.minute, now.second)
    trainer = Trainer(args)
    vis = Visulizer(
        env='{0}_{1}'.format('matting', time.strftime('%m_%d')))
    for epoch in range(0, args['epoch_num']):
        trainer.training(epoch)
        trainer.validation(epoch)
    exit(0)
