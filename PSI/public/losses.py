import torch
import torch.nn as nn
import torch.nn.functional as F
from  torch.nn.modules.loss import _Loss

class EntropyLoss(_Loss):
    def __init__(self, size_average=None, reduce=None, reduction='mean'):
        super(EntropyLoss, self).__init__(size_average, reduce, reduction)

    # input is probability distribution of output classes
    def forward(self, input):
        probs = torch.softmax(input, dim=1)
        loss = -torch.sum(probs * torch.log(probs + 1e-8), dim=1).mean()

        return loss

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self,contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        # self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature


    def forward(self, features, labels=None, temperature=0.07, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """

        device = (torch.device('cuda:0')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T), temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        mask_sum = mask.sum(1)
        mask_sum[mask_sum == 0] += 1

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_sum

        # loss
        loss = - (temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss

class KLLoss(nn.Module):
    def __init__(self):
        super(KLLoss, self).__init__()

    def forward(self, x, mu, logvar):
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)

class HSICLoss(nn.Module):
    def __init__(self, sigma_z=1.0, sigma_s=1.0):
        super(HSICLoss, self).__init__()
        self.sigma_z = sigma_z
        self.sigma_s = sigma_s

    def rbf_kernel(self, X, sigma):
        # Compute the Gaussian RBF kernel matrix K
        X_flat = X.view(X.size(0), -1)
        dist_sq = torch.cdist(X_flat, X_flat, p=2) ** 2
        K = torch.exp(-dist_sq / (2 * sigma ** 2))
        return K

    def forward(self, Z, S):
        """
        Z: Extracted intermediate features, such as shape (Batch, 64, H, W) or (Batch, 64)
        S: Sensitive tags, such as shape (Batch)
        """
        n = Z.size(0)

        S_float = S.float().view(n, -1)

        K = self.rbf_kernel(Z, self.sigma_z)
        L = self.rbf_kernel(S_float, self.sigma_s)

        H = torch.eye(n, device=Z.device) - (1.0 / n) * torch.ones((n, n), device=Z.device)

        Kc = torch.mm(K, H)
        Lc = torch.mm(L, H)

        hsic_value = torch.trace(torch.mm(Kc, Lc)) / ((n - 1) ** 2)

        return hsic_value

class CLUBLoss(nn.Module):
    def __init__(self):
        super(CLUBLoss, self).__init__()

    def forward(self, logits, labels):
        """
        Calculating the upper bound of vCLUB mutual information (for classification tasks with discrete labels)
        logits: shape [Batch, NumClasses], i.e. the output of adversary_model
        labels: shape [Batch], i.e. protected_labels (e.g. 0, 1, 2)
        """
        batch_size = logits.size(0)

        # 1. Convert logits to log probabilities: log q(y|x)
        log_probs = F.log_softmax(logits, dim=1) # 形状: [Batch, NumClasses]

        # 2. Compute the log-likelihood of the ‘true positive samples’: log q(y_i | x_i)
        # Use torch.arange to obtain the probability corresponding to the true label of each sample
        pos_log_probs = log_probs[torch.arange(batch_size), labels] # Shape: [Batch]

        # 3. Calculate the mean of the logarithmic probabilities of negative samples for the ‘full permutation cross-validation’: 1/N \sum_j log q(y_j | x_i)
        # The element in row i, column j of the matrix is precisely log q(y_j | x_i)
        neg_log_probs = log_probs[:, labels].mean(dim=1) # 形状: [Batch]

        # 4. Calculate the upper bound of vCLUB ((mean of positive samples – mean of negative samples))
        club_bound = (pos_log_probs - neg_log_probs).mean()

        return club_bound

class KernelGuidedCLUBLoss(nn.Module):
    def __init__(self, is_continuous=False):
        super(KernelGuidedCLUBLoss, self).__init__()
        self.is_continuous = is_continuous
    def forward(self, adversary_out, target_S, task_labels, features):
        """
        adversary_out: [Batch, NumClasses]/[Batch, c, h, w]
        target_S: [Batch]/[Batch, c, h, w]
        task_labels: [Batch]
        features: [Batch, ...]
        """
        batch_size = adversary_out.size(0)
        flat_features = features.view(batch_size, -1).detach()

        x_norm = (flat_features ** 2).sum(1).view(-1, 1)
        dist = x_norm + x_norm.view(1, -1) - 2.0 * torch.mm(flat_features, flat_features.t())
        dist = torch.clamp(dist, min=0.0)
        median_dist = torch.median(dist[dist > 0])
        sigma = median_dist.item() if median_dist.item() > 1e-8 else 1.0
        K_Z = torch.exp(-dist / (2.0 * sigma))

        task_labels = task_labels.view(-1)
        mask = torch.eq(task_labels.unsqueeze(1), task_labels.unsqueeze(0)).float()
        weighted_mask = mask * K_Z

        if self.is_continuous:
            adv_flat = adversary_out.view(batch_size, -1)
            tgt_flat = target_S.view(batch_size, -1)

            D = adv_flat.size(1)

            pos_log_probs = - torch.sum((adv_flat - tgt_flat) ** 2, dim=1) / D
            adv_norm_sq = (adv_flat ** 2).sum(1).view(-1, 1)
            tgt_norm_sq = (tgt_flat ** 2).sum(1).view(1, -1)
            cross_log_probs = -(adv_norm_sq + tgt_norm_sq - 2.0 * torch.mm(adv_flat, tgt_flat.t())) / D

        else:
            log_probs = F.log_softmax(adversary_out, dim=1)
            pos_log_probs = log_probs[torch.arange(batch_size), target_S]
            cross_log_probs = log_probs[:, target_S]

        neg_log_probs = (cross_log_probs * weighted_mask).sum(dim=1) / (weighted_mask.sum(dim=1) + 1e-8)

        k_club_bound = (pos_log_probs - neg_log_probs).mean()
        k_club_bound = torch.clamp(k_club_bound, min=0.0)

        return k_club_bound