"""Group-centred coordinates for the TWiX baseline."""
import torch
from torch import nn
from twix_adapter import PointTWiX

MODES = {
    'world_pair': dict(group_centering=False, normalization='per_pair_minmax'),
    'group_pair': dict(group_centering=True, normalization='per_pair_minmax'),
    'world_fixed': dict(group_centering=False, normalization='fixed_10_m'),
    'group_fixed': dict(group_centering=True, normalization='fixed_10_m'),
}

def center_observed_group(x):
    """Subtract coordinate-wise observed median at each visible timestamp.

    Uses the lower median for even counts, as torch.nanmedian specifies.
    Completely missing rows retain their NaNs and existing time masks.
    Changing membership can bias this reference: it is not true group motion.
    """
    if x.shape[1] == 0:
        return x
    center = torch.nanmedian(x, dim=1, keepdim=True).values
    center = torch.where(torch.isfinite(center), center, torch.zeros_like(center))
    return x - center

class FixedPhysicalScale(nn.Module):
    """One shared physical unit; no candidate-dependent rescaling/clipping."""
    def forward(self, x):
        return x / 10.

class CoordinateTWiX(PointTWiX):
    def __init__(self, mode='world_pair', layers=1, inter_pair=True, device='cuda'):
        if mode not in MODES:
            raise ValueError(mode)
        super().__init__(layers, inter_pair, device)
        self.mode = mode
        if MODES[mode]['normalization'] == 'fixed_10_m':
            self.norm_coordinates = FixedPhysicalScale()

    def forward(self, coordsP, framesP, coordsF, framesF, fps):
        if MODES[self.mode]['group_centering']:
            coordsP = center_observed_group(coordsP)
            coordsF = center_observed_group(coordsF)
        return super().forward(coordsP, framesP, coordsF, framesF, fps)
