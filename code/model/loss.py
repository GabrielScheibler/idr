import torch
from torch import nn
from torch.nn import functional as F

class IDRLoss(nn.Module):
    def __init__(self, eikonal_weight, mask_weight, specular_weight, gen_discrim_weight, alpha):
        super().__init__()
        self.eikonal_weight = eikonal_weight
        self.mask_weight = mask_weight
        self.specular_weight = specular_weight
        self.gen_discrim_weight = gen_discrim_weight
        self.alpha = alpha
        self.l1_loss = nn.L1Loss(reduction='sum')
        self.l2_loss = nn.MSELoss(reduction='sum')
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def get_rgb_loss(self, rgb_values, rgb_gt, network_object_mask, object_mask):
        if (network_object_mask & object_mask).sum() == 0:
            return torch.tensor(0.0).cuda().float()

        rgb_values = rgb_values[network_object_mask & object_mask]
        rgb_gt = rgb_gt.reshape(-1, 3)[network_object_mask & object_mask]
        rgb_loss = self.l1_loss(rgb_values, rgb_gt) / float(object_mask.shape[0])
        return rgb_loss

    def get_specular_loss(self, rgb_specular,network_object_mask, object_mask):
        mask = network_object_mask & object_mask
        if mask.sum() == 0:
            return torch.tensor(0.0).cuda().float()

        rgb = rgb_specular[mask]
        gt = -1 * torch.ones_like(rgb)

        specular_loss = (self.sigmoid(2 * self.l1_loss(rgb, gt) / float(rgb.shape[0])) - 0.5) * 2
        return specular_loss

    def get_eikonal_loss(self, grad_theta):
        if grad_theta.shape[0] == 0:
            return torch.tensor(0.0).cuda().float()

        eikonal_loss = ((grad_theta.norm(2, dim=1) - 1) ** 2).mean()
        return eikonal_loss

    def get_mask_loss(self, sdf_output, network_object_mask, object_mask):
        mask = ~(network_object_mask & object_mask)
        if mask.sum() == 0:
            return torch.tensor(0.0).cuda().float()
        sdf_pred = -self.alpha * sdf_output[mask]
        gt = object_mask[mask].float()
        mask_loss = (1 / self.alpha) * F.binary_cross_entropy_with_logits(sdf_pred.squeeze(), gt, reduction='sum') / float(object_mask.shape[0])
        return mask_loss

    def get_discrim_loss(self, discrim_points, points, network_object_mask, object_mask):
        if (network_object_mask & object_mask).sum() == 0:
            return torch.tensor(0.0).cuda().float(), torch.tensor(0.0).cuda().float()

        discrim_points = discrim_points[network_object_mask & object_mask]
        points = points.reshape(-1, 3)[network_object_mask & object_mask]
        #discrim_loss = (self.sigmoid(2 * self.l2_loss(discrim_points, points) / (float(object_mask.shape[0]))) - 0.5) * 2
        discrim_loss = self.l2_loss(discrim_points, points) / float((network_object_mask & object_mask).shape[0])
        gen_discrim_loss = 1 - discrim_loss
        return discrim_loss, gen_discrim_loss

    def forward(self, model_outputs, ground_truth, use_discrim=False):
        rgb_gt = ground_truth['rgb'].cuda()
        network_object_mask = model_outputs['network_object_mask']
        object_mask = model_outputs['object_mask']

        rgb_loss = self.get_rgb_loss(model_outputs['rgb_values'], rgb_gt, network_object_mask, object_mask)
        mask_loss = self.get_mask_loss(model_outputs['sdf_output'], network_object_mask, object_mask)
        eikonal_loss = self.get_eikonal_loss(model_outputs['grad_theta'])
        specular_loss = self.get_specular_loss(model_outputs["rgb_specular"], network_object_mask, object_mask)
        if use_discrim:
            discrim_loss, gen_discrim_loss = self.get_discrim_loss(model_outputs["discrim_output"], model_outputs["points"], network_object_mask, object_mask)
        else:
            discrim_loss = 0
            gen_discrim_loss = 0

        loss = rgb_loss + \
               self.eikonal_weight * eikonal_loss + \
               self.mask_weight * mask_loss + \
               self.specular_weight * specular_loss +\
               self.gen_discrim_weight * gen_discrim_loss

        return {
            'loss': loss,
            'rgb_loss': rgb_loss,
            'eikonal_loss': eikonal_loss,
            'mask_loss': mask_loss,
            'specular_loss': specular_loss,
            'discrim_loss': discrim_loss,
            'gen_discrim_loss': gen_discrim_loss
        }
