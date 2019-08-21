"""
--------------------------------------------------------------------------
Test by selecting image
--------------------------------------------------------------------------
"""

import os
import tkinter.filedialog
import cv2
import numpy as np
from models.py_encoder_decoder import Encoder_Decoder
from models.RefineNet import *

args = {
    'pre_trained_t_net': './checkpoints/Portrait_stage0_epoch59_0.0068.params',
    'pre_trained_m_net': './checkpoints/encoder_decoder_03_16_34_0.0120002525.params'
}


def load_model(args):
    t_net = RDFNet()
    t_net.load_state_dict(torch.load(args['pre_trained_t_net']))
    t_net.eval()
    m_net = Encoder_Decoder()
    m_net.load_state_dict(torch.load(args['pre_trained_m_net']))
    m_net.eval()

    return t_net, m_net


def trimap_gen(a):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    fg_tr = np.array(np.equal(a, 1).astype(np.float32))
    un_tr = np.array(np.not_equal(a, 0).astype(np.float32))
    un_tr = cv2.dilate(un_tr, kernel,
                       iterations=np.random.randint(1, 20))
    trimap = fg_tr * 255 + (un_tr - fg_tr) * 128
    return trimap


def matting(args, image, t_net, m_net):
    image = (image.astype(np.float32) - (114., 121., 134.,)) / 255.0
    image = torch.FloatTensor(image.transpose((2, 0, 1)).astype(float)[np.newaxis, :, :, :])
    t_net_input = F.upsample(image, size=(320, 320))
    saliency = torch.sigmoid(t_net(t_net_input.cuda()))
    saliency = F.upsample(saliency, (image.shape[2], image.shape[3]))
    saliency = saliency.data.cpu().numpy()[0, 0, :, :]
    saliency_in = saliency
    saliency_in[np.where(saliency_in > 0.9)] = 1
    saliency_in[np.where(saliency_in < 0.1)] = 0
    trimap = trimap_gen(saliency_in)
    trimap_in = trimap[np.newaxis, np.newaxis, :, :]
    trimap_in = torch.tensor(trimap_in).div(255).sub_(0.5).div_(0.5)
    m_net_input = torch.cat((image.cpu(), trimap_in), dim=1)
    alpha_pre = m_net(m_net_input).data.numpy()[0, 0, :, :]
    mask = np.equal(trimap, 128).astype(np.float32)
    alpha_pre = alpha_pre * mask + np.equal(trimap, 255).astype(np.float32)

    return alpha_pre, trimap


def main(args):
    # 文件对话框：
    t_net, m_net = load_model(args)
    default_dir = r"文件路径"
    file_path = tkinter.filedialog.askopenfilename(title=u'选择文件', initialdir=(os.path.expanduser(default_dir)))
    img = cv2.imread(file_path)
    b_channel, g_channel, r_channel = cv2.split(img)
    alpha, trimap = matting(args, img, t_net, m_net)
    trimap = trimap.astype(np.uint8)
    alpha = (alpha*255).astype(np.uint8)
    compose = cv2.merge([b_channel, g_channel, r_channel, alpha])
    cv2.namedWindow("Alpha", 0)
    cv2.imshow("Alpha", alpha)
    cv2.namedWindow("Trimap", 0)
    cv2.imshow("Trimap", trimap)
    cv2.namedWindow("Compose", 0)
    cv2.imshow("Compose", compose)
    cv2.imwrite('./test/test.png', compose)
    cv2.waitKey(0)


if __name__ == '__main__':
    main(args)
