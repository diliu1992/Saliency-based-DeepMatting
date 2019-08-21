import torch


def loss_function(img, alpha_pre, alpha_gt, mask):
    # -------------------------------------
    # matting prediction loss loss_matting
    # mask: mask for the unknown area of trimap
    # ------------------------
    eps = 1e-6
    # loss_alpha
    loss_alpha = torch.mul(torch.sqrt(torch.pow(alpha_pre - alpha_gt, 2.) + eps), mask).mean()

    # loss_composition
    fg = torch.cat((alpha_gt, alpha_gt, alpha_gt), 1) * img
    fg_pre = torch.cat((alpha_pre, alpha_pre, alpha_pre), 1) * img
    loss_composition = torch.mul(torch.sqrt(torch.pow(fg - fg_pre, 2.) + eps), mask).mean()

    loss_matting = 0.5 * loss_alpha + 0.5 * loss_composition

    return loss_matting
