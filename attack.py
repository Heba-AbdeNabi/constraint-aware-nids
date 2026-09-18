"""
attack.py
---------
PGD/FGSM evasion attacks against the victim IDS, in TWO modes:

  1. UNCONSTRAINED  - standard PGD in standardized feature space. Produces
                      adversarial examples that are often INVALID as real flows
                      (fractional packet counts, out-of-range values, perturbed
                      immutable fields). This is what most prior IDS-robustness
                      papers evaluate against -> over-optimistic.

  2. CONSTRAINED    - PGD with a projection step after each iteration that maps
                      the example back to the VALID region defined by the
                      constraint spec: clip to [min,max], round integer features,
                      and zero perturbations on immutable features. Only valid
                      flows count. This is the realistic threat and the paper's
                      contribution.

We also report the VALIDITY RATE of unconstrained attacks (fraction of adv
examples that happen to satisfy all constraints) to quantify how unrealistic
the standard evaluation is.

The attack operates in standardized space (where the model lives) but the
constraints live in raw space, so we convert using scaler mean/scale.
"""
import numpy as np
import torch

def _to_raw(x_std, mean, scale):
    return x_std * scale + mean

def _to_std(x_raw, mean, scale):
    return (x_raw - mean) / scale


class ConstraintSpec:
    def __init__(self, spec, device):
        self.mean = torch.tensor(spec["scaler_mean"], dtype=torch.float32, device=device)
        self.scale = torch.tensor(spec["scaler_scale"], dtype=torch.float32, device=device)
        self.fmin = torch.tensor(spec["min"], dtype=torch.float32, device=device)
        self.fmax = torch.tensor(spec["max"], dtype=torch.float32, device=device)
        self.is_int = torch.tensor(spec["is_integer"], dtype=torch.bool, device=device)
        self.is_immutable = torch.tensor(spec["is_immutable"], dtype=torch.bool, device=device)
        self.device = device

    def project(self, x_std, x_orig_std):
        """Project standardized adv example onto the valid region."""
        x_raw = _to_raw(x_std, self.mean, self.scale)
        orig_raw = _to_raw(x_orig_std, self.mean, self.scale)
        # immutable features revert to original
        x_raw = torch.where(self.is_immutable, orig_raw, x_raw)
        # clip to valid range
        x_raw = torch.max(torch.min(x_raw, self.fmax), self.fmin)
        # round integer features
        x_raw = torch.where(self.is_int, torch.round(x_raw), x_raw)
        return _to_std(x_raw, self.mean, self.scale)

    def validity_mask(self, x_std, x_orig_std, tol=1e-4):
        """Boolean per-sample: does this adv example satisfy all constraints?"""
        x_raw = _to_raw(x_std, self.mean, self.scale)
        orig_raw = _to_raw(x_orig_std, self.mean, self.scale)
        in_range = ((x_raw >= self.fmin - tol) & (x_raw <= self.fmax + tol)).all(1)
        ints_ok = ((~self.is_int) |
                   (torch.abs(x_raw - torch.round(x_raw)) < tol)).all(1)
        imm_ok = ((~self.is_immutable) |
                  (torch.abs(x_raw - orig_raw) < tol)).all(1)
        return in_range & ints_ok & imm_ok


def pgd_attack(model, x, y, spec: ConstraintSpec, eps=0.3, alpha=0.05,
               steps=20, constrained=True, targeted_evasion=True):
    """
    Craft adversarial examples. For IDS evasion we want ATTACK flows (y=attack)
    to be misclassified as BENIGN, so we ascend the loss for the true label
    (untargeted) which pushes them off the correct class.

    Returns adv examples (std space) and, for constrained=False, the validity
    mask so we can measure how many are realizable.
    """
    model.eval()
    x_orig = x.clone().detach()
    # random start within eps ball
    delta = torch.empty_like(x).uniform_(-eps, eps)
    x_adv = (x + delta).detach()
    if constrained:
        x_adv = spec.project(x_adv, x_orig).detach()
    x_adv.requires_grad_(True)

    for _ in range(steps):
        logits = model(x_adv)
        loss = torch.nn.functional.cross_entropy(logits, y)
        grad = torch.autograd.grad(loss, x_adv)[0]
        x_adv = x_adv.detach() + alpha * grad.sign()
        # keep within eps-ball of original (Linf)
        x_adv = torch.max(torch.min(x_adv, x_orig + eps), x_orig - eps)
        if constrained:
            x_adv = spec.project(x_adv, x_orig)
        x_adv = x_adv.detach().requires_grad_(True)

    x_adv = x_adv.detach()
    valid = spec.validity_mask(x_adv, x_orig)
    return x_adv, valid
