"""
--------------------------------------------------------------------------
Test on existing dataset
--------------------------------------------------------------------------
"""
import numpy as np
from models.py_encoder_decoder import Encoder_Decoder
from models.RefineNet import *
import scipy.misc as misc
from utils.visulization import Visulizer
import time
from utils.metrics import MScore
import cv2
from get_dataset_loader import *


def trimap_gen(a):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    fg_tr = np.array(np.equal(a, 1).astype(np.float32))
    un_tr = np.array(np.not_equal(a, 0).astype(np.float32))
    un_tr = cv2.dilate(un_tr, kernel,
                       iterations=np.random.randint(1, 20))
    trimap = fg_tr * 255 + (un_tr - fg_tr) * 128
    return trimap


def test():
    matte_metrics = MScore()
    # load t_net for saliency prediction and m_net for matting
    t_net = RDFNet()
    t_net.load_state_dict(torch.load('./checkpoints/Portrait_stage0_epoch31_0.0076.params'))
    t_net.eval()
    m_net = Encoder_Decoder()
    m_net.load_state_dict(torch.load('./checkpoints/encoder_decoder_03_16_34_0.0120002525.params'))
    m_net.eval()
    dataset_name = 'Portrait'
    test_loader = get_test_loader(dataset_name, 1, 10, 320)
    vis = Visulizer(
        env='{0}_{1}'.format('matting', time.strftime('%m_%d')))
    for i, (image, alpha, im_name, size) in enumerate(test_loader):
        print(i)
        image = image.cuda()
        saliency_input = F.upsample(image, size=(320, 320))
        saliency = t_net(saliency_input)
        saliency = torch.sigmoid(saliency)
        saliency = F.upsample(saliency, (image.shape[2], image.shape[3]))
        saliency = saliency.data.cpu().numpy()[0, 0, :, :]
        saliency_in = saliency
        saliency_in[np.where(saliency_in > 0.9)] = 1
        saliency_in[np.where(saliency_in < 0.1)] = 0
        trimap = trimap_gen(saliency_in)
        trimap_in = trimap[np.newaxis, np.newaxis, :, :]
        trimap_in = torch.tensor(trimap_in).div(255).sub_(0.5).div_(0.5)
        data = torch.cat((image.cpu(), trimap_in), dim=1)
        alpha_pre = m_net(data).data.numpy()[:, 0, :, :]
        alpha_gt = alpha.data.cpu().numpy()[:, 0, :, :]
        mask = np.equal(trimap, 128).astype(np.float32)
        alpha_pre = alpha_pre * mask + np.equal(trimap, 255).astype(np.float32)
        alpha_c = alpha_pre[:, np.newaxis, :, :]
        image = image.data.cpu().numpy()
        compose = alpha_c * image + (1 - alpha.data.cpu().numpy()) * image
        matte_metrics.update(alpha_pre, alpha_gt, mask)
        for j in range(alpha_pre.shape[0]):
            misc.imsave("./outputs/{}/test/alpha/{}.png".format(dataset_name, im_name[j]),
                        alpha_pre[j, :, :])
            misc.imsave("./outputs/{}/test/compose/{}.png".format(dataset_name, im_name[j]),
                        compose[j, :, :, :].transpose(1, 2, 0))
            misc.imsave("./outputs/{}/test/trimap/{}.png".format(dataset_name, im_name[j]),
                        trimap)
            misc.imsave("./outputs/{}/test/saliency/{}.png".format(dataset_name, im_name[j]),
                        saliency)
            # vis.img('merged_{0}'.format(j), merged[j, :, :, :])
            # vis.img('alpha_{0}'.format(j), alpha[j, :, :])
            # vis.img('gt_alpha_{0}'.format(j), gt[j, :, :])
            # vis.img('trimap_s_{0}'.format(j), trimap_s[j, :, :])
            # vis.img('trimap_a_{0}'.format(j), trimap_a[j, :, :])
            # vis.img('saliency_{0}'.format(j), saliency[j, :, :])
    score = matte_metrics.get_scores()
    for k, v in score.items():
        print(k, v)
    matte_metrics.reset()


if __name__ == '__main__':
    test()
