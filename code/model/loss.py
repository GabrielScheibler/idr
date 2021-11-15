import torch
from torch import nn
from torch.nn import functional as F

class IDRLoss(nn.Module):
    def __init__(self, eikonal_weight, mask_weight,specular_weight, dsdf_weight, shading_weight, alpha):
        super().__init__()
        self.eikonal_weight = eikonal_weight
        self.mask_weight = mask_weight
        self.specular_weight = specular_weight
        self.dsdf_weight = dsdf_weight
        self.shading_weight = shading_weight
        self.alpha = alpha
        self.l1_loss = nn.L1Loss(reduction='sum')

    def get_rgb_loss(self, rgb_values, rgb_gt, network_object_mask, object_mask):
        if (network_object_mask & object_mask).sum() == 0:
            return torch.tensor(0.0).cuda().float()

        rgb_values = rgb_values[network_object_mask & object_mask]
        rgb_gt = rgb_gt.reshape(-1, 3)[network_object_mask & object_mask]
        rgb_loss = self.l1_loss(rgb_values, rgb_gt) / float(object_mask.shape[0])
        return rgb_loss

    def get_specular_loss(self, rgb_specular):
        specular_loss = self.l1_loss(rgb_specular, torch.zeros_like(rgb_specular)) / float(rgb_specular.shape[0])
        return specular_loss

    def get_dsdf_loss(self, pointpos_sdf, pointpos_dsdf):
        dsdf_loss = self.l1_loss(pointpos_sdf - pointpos_dsdf, torch.zeros_like(pointpos_sdf)) / float(pointpos_dsdf.shape[0])
        return dsdf_loss

    def get_shading_loss(self, rgb_shading, rgb_gt, network_object_mask, object_mask):
        shading_gt = torch.mean(rgb_gt.reshape(-1,3),dim=1)[network_object_mask & object_mask]
        shading_apx = rgb_shading[:,0]
        shading_apx = shading_apx[network_object_mask & object_mask]
        shading_loss = self.l1_loss(shading_apx, shading_gt) / float(object_mask.shape[0])
        return shading_loss

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

    def forward(self, model_outputs, ground_truth):
        rgb_gt = ground_truth['rgb'].cuda()
        network_object_mask = model_outputs['network_object_mask']
        object_mask = model_outputs['object_mask']

        rgb_loss = self.get_rgb_loss(model_outputs['rgb_values'], rgb_gt, network_object_mask, object_mask)
        mask_loss = self.get_mask_loss(model_outputs['sdf_output'], network_object_mask, object_mask)
        eikonal_loss = self.get_eikonal_loss(model_outputs['grad_theta'])
        specular_loss = self.get_specular_loss(model_outputs["rgb_specular"])
        dsdf_loss = self.get_dsdf_loss(model_outputs["shading_pointpos_sdf"], model_outputs["shading_pointpos_dsdf"])
        shading_loss = self.get_shading_loss(model_outputs["rgb_shading_fake"], rgb_gt, network_object_mask, object_mask)


        loss = rgb_loss + \
               self.eikonal_weight * eikonal_loss + \
               self.mask_weight * mask_loss + \
               self.specular_weight * specular_loss #+ \
               #self.dsdf_weight * dsdf_loss + \
               #self.shading_weight * shading_loss

        return {
            'loss': loss,
            'rgb_loss': rgb_loss,
            'eikonal_loss': eikonal_loss,
            'mask_loss': mask_loss,
            'specular_loss': specular_loss,
            'dsdf_loss': dsdf_loss,
            'shading_loss': shading_loss
        }
