import torch


def pairwise_distance(x1, x2, p=2, eps=1e-6):
    r"""
    Computes the batchwise pairwise distance between vectors v1,v2:
        .. math ::
            \Vert x \Vert _p := \left( \sum_{i=1}^n  \vert x_i \vert ^ p \right) ^ {1/p}
        Args:
            x1: first input tensor
            x2: second input tensor
            p: the norm degree. Default: 2
        Shape:
            - Input: :math:`(N, D)` where `D = vector dimension`
            - Output: :math:`(N, 1)`
        >>> input1 = autograd.Variable(torch.randn(100, 128))
        >>> input2 = autograd.Variable(torch.randn(100, 128))
        >>> output = F.pairwise_distance(input1, input2, p=2)
        >>> output.backward()
    """
    assert x1.size() == x2.size(), "Input sizes must be equal."
    assert x1.dim() == 2, "Input must be a 2D matrix."

    return 1 - torch.cosine_similarity(x1, x2, dim=1)
    # diff = torch.abs(x1 - x2)
    # out = torch.sum(torch.pow(diff + eps, p), dim=1)
    #
    # return torch.pow(out, 1. / p)


def triplet_margin_loss_gor_one(anchor, positive, negative, beta=1.0, margin=1.0, p=2, eps=1e-6, swap=False):
    assert anchor.size() == positive.size(), "Input sizes between positive and negative must be equal."
    assert anchor.size() == negative.size(), "Input sizes between anchor and negative must be equal."
    assert positive.size() == negative.size(), "Input sizes between positive and negative must be equal."
    assert anchor.dim() == 2, "Inputd must be a 2D matrix."
    assert margin > 0.0, 'Margin should be positive value.'
    d_p = pairwise_distance(anchor, positive, p, eps)
    d_n = pairwise_distance(anchor, negative, p, eps)

    dist_hinge = torch.clamp(margin + d_p - d_n, min=0.0)

    neg_dis = torch.pow(torch.sum(torch.mul(anchor, negative), 1), 2)
    gor = torch.mean(neg_dis)

    loss = torch.mean(dist_hinge) + beta * (gor)

    return loss


def triplet_margin_loss_gor(anchor, positive, negative1, negative2, beta=1.0, margin=1.0, p=2, eps=1e-6, swap=False):
    assert anchor.size() == positive.size(), "Input sizes between positive and negative must be equal."
    assert anchor.size() == negative1.size(), "Input sizes between anchor and negative must be equal."
    assert positive.size() == negative2.size(), "Input sizes between positive and negative must be equal."
    assert anchor.dim() == 2, "Inputd must be a 2D matrix."
    assert margin > 0.0, 'Margin should be positive value.'

    # loss1 = triplet_margin_loss_gor_one(anchor, positive, negative1)
    # loss2 = triplet_margin_loss_gor_one(anchor, positive, negative2)
    #
    # return 0.5*(loss1+loss2)

    d_p = pairwise_distance(anchor, positive, p, eps)
    d_n1 = pairwise_distance(anchor, negative1, p, eps)
    d_n2 = pairwise_distance(anchor, negative2, p, eps)

    dist_hinge = torch.clamp(margin + d_p - 0.5 * (d_n1 + d_n2), min=0.0)

    neg_dis1 = torch.pow(torch.sum(torch.mul(anchor, negative1), 1), 2)
    gor1 = torch.mean(neg_dis1)
    neg_dis2 = torch.pow(torch.sum(torch.mul(anchor, negative2), 1), 2)
    gor2 = torch.mean(neg_dis2)

    loss = torch.mean(dist_hinge) + beta * (gor1 + gor2)

    return loss


def distance_matrix_vector(anchor, positive):
    """Given batch of anchor descriptors and positive descriptors calculate distance matrix"""
    D = anchor.shape[-1]
    d1_sq = torch.sum(anchor * anchor, dim=1).unsqueeze(-1)
    d2_sq = torch.sum(positive * positive, dim=1).unsqueeze(-1)

    eps = 1e-3
    return torch.sqrt((d1_sq.repeat(1, positive.size(0)) + torch.t(d2_sq.repeat(1, anchor.size(0)))
                       - 2.0 * torch.bmm(anchor.unsqueeze(0), torch.t(positive).unsqueeze(0)).squeeze(0))+eps)

    # anchor = anchor.permute(1, 0).view(D, -1, 1)
    # positive = positive.permute(1, 0).view(D, 1, -1)
    # return torch.norm(anchor - positive, dim=0)


def percentile(t, q):
    """
    Return the ``q``-th percentile of the flattened input tensor's data.

    CAUTION:
     * Needs PyTorch >= 1.1.0, as ``torch.kthvalue()`` is used.
     * Values are not interpolated, which corresponds to
       ``numpy.percentile(..., interpolation="nearest")``.

    :param t: Input tensor.
    :param q: Percentile to compute, which must be between 0 and 100 inclusive.
    :return: Resulting value (scalar).
    """
    # Note that ``kthvalue()`` works one-based, i.e. the first sorted value
    # indeed corresponds to k=1, not k=0! Use float(q) instead of q directly,
    # so that ``round()`` returns an integer, even if q is a np.float32.
    k = 1 + round(.01 * float(q) * (t.numel() - 1))
    result = t.view(-1).kthvalue(int(k)).values.item()
    return result


""" Triplet loss usd in SOSNet """
def sos_reg(anchor, positive, KNN=True, k=1, eps=1e-8):
    dist_matrix_a = distance_matrix_vector(anchor, anchor) + eps
    dist_matrix_b = distance_matrix_vector(positive, positive) + eps
    if KNN:
        k_max = percentile(dist_matrix_b, k)
        #print("k_max:", k_max)
        mask = dist_matrix_b.lt(k_max)
        dist_matrix_a = dist_matrix_a*mask.int().float()
        dist_matrix_b = dist_matrix_b*mask.int().float()
    SOS_temp = torch.sqrt(torch.sum(torch.pow(dist_matrix_a-dist_matrix_b, 2)))
    return torch.mean(SOS_temp)
