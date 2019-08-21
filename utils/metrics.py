import numpy as np
from sklearn import metrics
import math
import cv2


class SScore(object):
    # saliency prediction metrics
    def __init__(self):
        self.MAE = []
        self.MSE = []
        self.maxF = []
        self.meanF = []
        self.meanP = []
        self.meanR = []
        self.AUC = []

    def cal_mae(self, sal, gt):
        return np.mean(np.abs(sal - gt))

    def cal_mse(self, sal, gt):
        return np.mean((sal - gt) ** 2)

    def cal_auc(self, sal, gt):
        fpr, tpr, thresholds = metrics.roc_curve(gt, sal, pos_label=2)
        auc = metrics.auc(fpr, tpr)
        return auc

    def cal_f_score(self, sal, gt):
        t = 2 * np.nanmean(sal)
        beta = 0.3
        sal_map = np.zeros(shape=sal.shape)
        sal_map[sal >= t] = 1
        tp = sum(sal_map[gt > 0.5])
        p = tp / sum(sal_map)
        r = tp / sum(gt)
        f = (1 + beta) * p * r / (beta * p + r)
        return p, r, f

    def update(self, sal, gt):
        for s, g in zip(sal, gt):
            self.MAE.append(self.cal_mae(s.flatten(), g.flatten()))
            self.MSE.append(self.cal_mse(s.flatten(), g.flatten()))
            # mean_p, mean_r, mean_f = self.cal_f_score(s.flatten(), g.flatten())
            # # max_f, mean_f, mean_p, mean_r = self.cal_f_score(s.flatten(), g.flatten())
            # # self.maxF.append(max_f)
            # self.meanF.append(mean_f)
            # self.meanP.append(mean_p)
            # self.meanR.append(mean_r)

    def get_scores(self):
        """Returns accuracy score evaluation result.
            - overall accuracy
            - mean accuracy
            - mean IU
            - fwavacc
        """
        # hist = self.confusion_matrix
        # acc = np.diag(hist).sum() / hist.sum()
        # acc_cls = np.diag(hist) / hist.sum(axis=1)
        # acc_cls = np.nanmean(acc_cls)
        # iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        # mean_iu = np.nanmean(iu)
        # freq = hist.sum(axis=1) / hist.sum()
        # fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        # cls_iu = dict(zip(range(self.n_classes), iu))
        # mean_p = hist[1, 1] / (hist[1, 1] + hist[0, 1])
        # mean_r = hist[1, 1] / (hist[1, 1] + hist[1, 0])
        # f_measure = 1.3 * mean_p * mean_r / (0.3 * mean_p + mean_r)
        mean_mae = np.nanmean(self.MAE)
        mean_mse = np.nanmean(self.MSE)
        # mean_p = np.nanmean(self.meanP)
        # mean_r = np.nanmean(self.meanR)
        #
        # # max_f = np.nanmean(self.maxF)
        # mean_f = np.nanmean(self.meanF)
        return {
            'Mean MAE: \t': mean_mae,
            'Mean MSE: \t': mean_mse}
        # 'Mean Precision: \t': mean_p,
        # 'Mean Recall: \t': mean_r,
        # 'Max F measure: \t': max_f,
        # 'Mean F Measure: \t': mean_f}

    def reset(self):
        self.MAE = []
        self.MSE = []
        self.maxF = []
        self.meanF = []
        self.meanP = []
        self.meanR = []


class MScore(object):
    # matte metrics
    def __init__(self):
        self.SAD = []
        self.MSE = []
        self.Gradient = []
        self.Connectivity = []

    def cal_mse(self, alpha, gt, mask):
        return np.sum((alpha - gt) ** 2 * mask) / np.sum(mask)

    def cal_sad(self, alpha, gt, mask):
        error_map = np.abs(alpha - gt)
        return np.sum(error_map * mask) / 1000

    # def cal_gradient(self, alpha, gt, mask):
    #     # filted_alpha = filter.gaussian_filter(alpha, 1.4, order=1)
    #     # filted_gt = filter.gaussian_filter(gt, 1.4, order=1)
    #     gradient_alpha = np.gradient(alpha)
    #     gradient_gt = np.gradient(gt)
    #     return np.sum(np.sqrt(
    #         (gradient_gt[0] - gradient_alpha[0]) ** 2 + (gradient_gt[1] - gradient_alpha) ** 2) * mask) / np.sum(mask)
    #     # norms_alpha = np.linalg.norm(gradient_alpha, axis=0)
    #     # gradient_alpha = [np.where(norms_alpha == 0, 0, i / norms_alpha) for i in gradient_alpha]
    #     # norms_gt = np.linalg.norm(gradient_gt, axis=0)
    #     # gradient_gt = [np.where(norms_gt == 0, 0, i / norms_gt) for i in gradient_gt]
    #     # return np.sum((gradient_alpha[0].flatten() - gradient_gt[0].flatten()) ** 2) + np.sum(
    #     #     (gradient_alpha[1].flatten() - gradient_gt[1].flatten()) ** 2)

    def cal_gradient(self, alpha, gt, mask):
        gradient_alpha = self.gauss_gradient(alpha, 1.4)
        gradient_gt = self.gauss_gradient(gt, 1.4)
        pred_amp = math.sqrt(gradient_alpha[0] ** 2 + gradient_alpha[1] ** 2)
        gt_amp = math.sqrt(gradient_gt[0] ** 2 + gradient_gt[1] ** 2)
        error_map = (pred_amp - gt_amp) ** 2
        return np.sum(error_map * mask)

    def gauss_gradient(self, img, sigma):
        # determine the the appropriate size of kernel. The smaller epsilon, the larger size.
        epsilon = math.exp(-2)
        half_size = math.ceil(sigma * math.sqrt(-2 * math.log(math.sqrt(2 * math.pi) * sigma * epsilon)))
        size = 2 * half_size + 1
        hx = np.ones(size, size)
        # generate 2D Gaussian kernel along y direction
        for i in range(size):
            for j in range(size):
                hx[i, j] = self.gauss(i - half_size - 1, sigma) * self.dgauss(j - half_size - 1, sigma)
        hx = hx / math.sqrt(sum(abs(hx) * abs(hx)))
        hy = hx.T
        # 2D filtering
        gx = cv2.filter2D(src=img, kernel=hx, ddepth=-1)
        gy = cv2.filter2D(src=img, kernel=hy, ddepth=-1)
        return [gx, gy]

    def gauss(self, x, sigma):
        return math.exp(-x ** 2 / (2 * sigma ** 2)) / (sigma * math.sqrt(2 * math.pi))

    def dgauss(self, x, sigma):
        return -x * self.gauss(x, sigma) / sigma ** 2

    def cal_connectivity_error(self, alpha, gt, mask):
        distance_alpha = self.cal_connectivity((alpha * 255.).astype(np.uint8))
        distance_gt = self.cal_connectivity((gt * 255.0).astype(np.uint8))
        return np.sum(np.sqrt(
            (distance_gt - distance_alpha) ** 2 * mask)) / np.sum(mask)

    def cal_connectivity(self, alpha):
        level = alpha.copy()
        # contours_level = []
        level[:, :] = 0
        for k in range(256):
            ret, gray_fixed = cv2.threshold(alpha.copy(), k - 1, 255, cv2.THRESH_BINARY)
            contours, hierarchy = cv2.findContours(gray_fixed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            area = []
            if len(contours) > 0:
                for i in range(len(contours)):
                    area.append(cv2.contourArea(contours[i]))
                max_idx = np.argmax(area)
                cv2.fillConvexPoly(level, contours[max_idx], k)
            # contours_level.append(contours[max_idx])
        distance = alpha - level
        return distance
        # w = len(range(l, g))
        #     dist = cv2.pointPolygonTest(cnt, (50, 50), True)

    def update(self, alpha, gt, mask):
        for s, g, m in zip(alpha, gt, mask):
            self.MSE.append(self.cal_mse(s, g, m))
            self.SAD.append(self.cal_sad(s, g, m))
            # self.Connectivity.append(self.cal_connectivity_error(s, g, m))
            self.Gradient.append(self.cal_gradient(s, g, m))

    def get_scores(self):
        mean_mse = np.nanmean(self.MSE)
        mean_sad = np.nanmean(self.SAD)
        mean_gradient = np.nanmean(self.Gradient)
        mean_connectivity = np.nanmean(self.Connectivity)
        return {
            'Mean MSE: \t': mean_mse,
            'Mean SAD: \t': mean_sad,
            'Mean Gradient: \t': mean_gradient,
            'Mean Connectivity: \t': mean_connectivity}

    def reset(self):
        self.MSE = []
        self.SAD = []
        self.Gradient = []
        self.Connectivity = []
