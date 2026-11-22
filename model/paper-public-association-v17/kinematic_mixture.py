"""Registered redevelopment hypothesis: joint observed kinematic densities.

Conditional mixture principle: Bishop (1994), NCRG/94/004. Joint position and
velocity likelihoods and full Gaussian covariance are established statistical
tools, not claimed as new inventions. Their integration with the existing
bidirectional, frame-conditioned association model is an experimental change.

Velocity is estimated from the SAME visible position tracklets. It is not a new
sensor channel. Existing encoders already used velocity; only the scored target
variable changes here. Missing height/vertical velocity are marginalized out.
"""
from pathlib import Path
import sys, math
import torch
from torch import nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'paper-public-association-v10'))
from physical_mixture import PhysicalMixture, inputs, auxiliary
from learners import mlp


def gaussian_log_density(mean, covariance, target):
    """B,N,K,D distributions at B,M,D targets; no identity information."""
    chol = torch.linalg.cholesky(covariance)
    delta = target[:, None, :, None, :] - mean[:, :, None, :, :]
    # B,N,K,D,M right-hand sides avoid an N*M covariance-factor duplication.
    rhs = delta.permute(0, 1, 3, 4, 2)
    whitened = torch.linalg.solve_triangular(chol, rhs, upper=False)
    quad = whitened.square().sum(-2).permute(0, 1, 3, 2)
    logdet = 2 * chol.diagonal(dim1=-2, dim2=-1).log().sum(-1)
    return -.5 * (quad + logdet[:, :, None, :] + mean.shape[-1] * math.log(2 * math.pi))


class KinematicMixture(PhysicalMixture):
    def __init__(self, joint=False, full_covariance=False, global_parameters=False):
        super().__init__(physical=True, attention=False, residual=False)
        self.joint = joint
        self.full_covariance = full_covariance
        self.global_parameters = global_parameters
        self.dimension = 6 if joint else 3
        self.off_count = self.dimension * (self.dimension - 1) // 2 if full_covariance else 0
        self.width = 1 + 2 * self.dimension + self.off_count
        # Position/diagonal retains the exact inherited initializer and weights.
        if joint or full_covariance:
            self.predict = mlp([144, 128, 64, self.width * self.components])
        if global_parameters:
            # Non-neural control: one global mixture and scalar null score,
            # trained on the same supervised association objective and budget.
            # Four known geometric anchors remain; no per-case neural output.
            # Break duplicate-component symmetry BEFORE this control is trained;
            # all-zero initialization otherwise leaves pairs 0/2 and 1/3 tied.
            self.global_raw = nn.Parameter(.01 * torch.randn(self.components, self.width))
            self.global_null = nn.Parameter(torch.tensor(0.))
            del self.gru, self.encode_node, self.predict, self.null
        # Inherited residual branch is disabled, remove its inactive parameters.
        del self.pair, self.mix

    def target_state(self, node):
        return node[..., [0, 1, 2, 6, 7, 8]] if self.joint else node[..., :3]

    def anchor(self, node, context):
        x = self.target_state(node)
        c, s = context[:, None, 6], context[:, None, 7]
        pieces = []
        for offset in range(0, self.dimension, 3):
            v = x[..., offset:offset + 3]
            pieces.append(torch.stack([v[..., 0] * c + v[..., 1] * s,
                                       -v[..., 0] * s + v[..., 1] * c,
                                       v[..., 2]], -1))
        rotated = torch.cat(pieces, -1)
        select = x.new_tensor([0, 1, 0, 1])[None, None, :, None]
        return x[:, :, None, :] + select * (rotated - x)[:, :, None, :]

    def params(self, a, b, mb, node, context):
        if self.global_parameters:
            z = self.global_raw[None, None].expand(node.shape[0], node.shape[1], -1, -1)
        else:
            pool = (b * mb[..., None]).sum(1) / mb.sum(1)[:, None]
            z = self.predict(torch.cat([a, pool[:, None].expand_as(a),
                                        context[:, None].expand(-1, a.shape[1], -1)], -1))
            z = z.reshape(a.shape[0], a.shape[1], self.components, self.width)
        d = self.dimension
        mean = self.anchor(node, context) + z[..., 1:1 + d]
        var = .0025 + F.softplus(z[..., 1 + d:1 + 2 * d]).square()
        weight = F.log_softmax(z[..., 0], -1)
        if not self.full_covariance:
            return mean, var, weight
        chol = torch.diag_embed(var.sqrt())
        ii, jj = torch.tril_indices(d, d, offset=-1, device=z.device)
        # Bounded off-diagonal factors avoid unbounded conditioning at startup.
        chol[..., ii, jj] = .5 * torch.tanh(z[..., 1 + 2 * d:])
        cov = chol @ chol.transpose(-1, -2)
        cov = cov + torch.eye(d, device=z.device, dtype=z.dtype) * 1e-6
        return mean, cov, weight

    def density(self, params, node, coords):
        mean, dispersion, weight = params
        target = self.target_state(node)
        mask = torch.cat([coords, coords], -1) if self.joint else coords
        if not self.full_covariance:
            delta = target[:, None, :, None, :] - mean[:, :, None, :, :]
            lp = -.5 * ((delta.square() / dispersion[:, :, None] +
                         dispersion[:, :, None].log() + math.log(2 * math.pi)) *
                        mask[:, None, None, None]).sum(-1)
            return torch.logsumexp(lp + weight[:, :, None], -1)
        result = mean.new_zeros(mean.shape[0], mean.shape[1], target.shape[1])
        for is_3d in [False, True]:
            rows = torch.nonzero((coords[:, 2] > .5) == is_3d, as_tuple=True)[0]
            if not len(rows):
                continue
            dims = list(range(self.dimension)) if is_3d else ([0, 1, 3, 4] if self.joint else [0, 1])
            # A Gaussian marginal uses the covariance PRINCIPAL SUBMATRIX,
            # not a slice of the precision or Cholesky matrix.
            mm = mean[rows][..., dims]
            cc = dispersion[rows][..., dims, :][..., dims]
            tt = target[rows][..., dims]
            lp = gaussian_log_density(mm, cc, tt)
            result[rows] = torch.logsumexp(lp + weight[rows, :, None, :], -1)
        return result

    def forward(self, s):
        if self.global_parameters:
            a = b = None
        else:
            a = self.encode(s['history_a'], s['a'])
            b = self.encode(s['history_b'], s['b'])
        pa = self.params(a, b, s['mb'], s['a'], s['context_a'])
        pb = self.params(b, a, s['ma'], s['b'], s['context_b'])
        fa = self.density(pa, s['b'], s['coordinates'])
        fb = self.density(pb, s['a'], s['coordinates']).transpose(1, 2)
        logits = ((fa + fb) * .5).masked_fill(~(s['ma'][:, :, None] & s['mb'][:, None, :]), -1e4)
        if self.global_parameters:
            nr = self.global_null.expand(s['ma'].shape)
            nc = self.global_null.expand(s['mb'].shape)
        else:
            nr, nc = self.null(a).squeeze(-1), self.null(b).squeeze(-1)
        return dict(logits=logits, rownull=nr, colnull=nc, forward_density=fa, backward_density=fb)


VARIANTS = {
    'position_diagonal': dict(joint=False, full_covariance=False),
    'position_full': dict(joint=False, full_covariance=True),
    'joint_diagonal': dict(joint=True, full_covariance=False),
    'joint_full': dict(joint=True, full_covariance=True),
    'global_joint_full': dict(joint=True, full_covariance=True, global_parameters=True),
}
