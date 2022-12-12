# ------------------- wasserstein distance
# https://github.com/dfdazac/wassdistance

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


# Adapted from https://github.com/gpeyre/SinkhornAutoDiff
class SinkhornDistance(nn.Module):
    r"""
    Given two empirical measures each with :math:`P_1` locations
    :math:`x\in\mathbb{R}^{D_1}` and :math:`P_2` locations :math:`y\in\mathbb{R}^{D_2}`,
    outputs an approximation of the regularized OT cost for point clouds.

    Args:
        eps (float): regularization coefficient
        max_iter (int): maximum number of Sinkhorn iterations
        reduction (string, optional): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
            'mean': the sum of the output will be divided by the number of
            elements in the output, 'sum': the output will be summed. Default: 'none'

    Shape:
        - Input: :math:`(N, P_1, D_1)`, :math:`(N, P_2, D_2)`
        - Output: :math:`(N)` or :math:`()`, depending on `reduction`
    """

    def __init__(self, eps, max_iter, reduction='none'):
        super(SinkhornDistance, self).__init__()
        self.eps = eps
        self.max_iter = max_iter
        self.reduction = reduction

    def forward(self, x, y):
        # The Sinkhorn algorithm takes as input three variables :
        C = self._cost_matrix(x, y)  # Wasserstein cost function
        x_points = x.shape[-2]
        y_points = y.shape[-2]
        if x.dim() == 2:
            batch_size = 1
        else:
            batch_size = x.shape[0]

        # both marginals are fixed with equal weights
        mu = torch.empty(batch_size, x_points, dtype=x.dtype,
                         requires_grad=False).fill_(1.0 / x_points).squeeze()
        nu = torch.empty(batch_size, y_points, dtype=x.dtype,
                         requires_grad=False).fill_(1.0 / y_points).squeeze()

        u = torch.zeros_like(mu)
        v = torch.zeros_like(nu)
        # To check if algorithm terminates because of threshold
        # or max iterations reached
        actual_nits = 0
        # Stopping criterion
        thresh = 1e-1

        # Sinkhorn iterations
        for i in range(self.max_iter):
            u1 = u  # useful to check the update
            u = self.eps * (torch.log(mu + 1e-8) - torch.logsumexp(self.M(C, u, v), dim=-1)) + u
            v = self.eps * (torch.log(nu + 1e-8) - torch.logsumexp(self.M(C, u, v).transpose(-2, -1), dim=-1)) + v
            err = (u - u1).abs().sum(-1).mean()

            actual_nits += 1
            if err.item() < thresh:
                break

        U, V = u, v
        # exp(logA+logB+logC)=ABC
        # Transport plan pi = diag(a)*K*diag(b)
        pi = torch.exp(self.M(C, U, V))
        # Sinkhorn distance
        cost = torch.sum(pi * C, dim=(-2, -1))

        if self.reduction == 'mean':
            cost = cost.mean()
        elif self.reduction == 'sum':
            cost = cost.sum()

        return cost, pi, C

    def M(self, C, u, v):
        """Modified cost for logarithmic updates
        $M_{ij} = (-c_{ij} + u_i + v_j) / \epsilon$ """
        return (-C + u.unsqueeze(-1) + v.unsqueeze(-2)) / self.eps

    @staticmethod
    def _cost_matrix(x, y, p=2):
        r"""
        Returns the matrix of $|x_i-y_j|^p$.

        Shape:
            - x.shape: :-> [b, n_p, xy]
            - y.shape: :-> [b, n_p, xy]
        """
        # addition and subtraction of tensor can automatically expand dims
        x_col = x.unsqueeze(-2)  # [b, n_p, 1, xy]
        y_lin = y.unsqueeze(-3)  # [b, 1, n_p, xy]
        C = torch.sum((torch.abs(x_col - y_lin)) ** p, -1)
        return C

    @staticmethod
    def ave(u, u1, tau):
        """Barycenter subroutine, used by kinetic acceleration through extrapolation."""
        return tau * u + (1 - tau) * u1


def show_assignments(a, b, P):
    norm_P = P / P.max()
    for i in range(a.shape[0]):
        for j in range(b.shape[0]):
            plt.arrow(a[i, 0], a[i, 1], b[j, 0] - a[i, 0], b[j, 1] - a[i, 1],
                      alpha=norm_P[i, j].item())
    plt.title('Assignments')
    plt.scatter(a[:, 0], a[:, 1])
    plt.scatter(b[:, 0], b[:, 1])
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    # n = 5
    # batch_size = 4
    # a = np.array([[[i, 0] for i in range(n)] for b in range(batch_size)])
    # b = np.array([[[i, b + 1] for i in range(n)] for b in range(batch_size)])
    #
    # # Wrap with torch tensors
    # x = torch.tensor(a, dtype=torch.float)
    # y = torch.tensor(b, dtype=torch.float)
    #
    # sinkhorn = SinkhornDistance(eps=0.1, max_iter=100)
    # dist, P, C = sinkhorn(x, y)
    # print("Sinkhorn distances: ", dist)
    #
    # print('P.shape = {}'.format(P.shape))
    # print('C.shape = {}'.format(C.shape))
    from sklearn.datasets import make_moons

    X, Y = make_moons(n_samples=30)
    a = X[Y == 0]
    b = X[Y == 1]

    x = torch.tensor(a, dtype=torch.float)
    y = torch.tensor(b, dtype=torch.float)

    sinkhorn = SinkhornDistance(eps=0.1, max_iter=100)
    dist, P, C = sinkhorn(x, y)
    print("Sinkhorn distance: {:.3f}".format(dist.item()))
    show_assignments(a, b, P)
