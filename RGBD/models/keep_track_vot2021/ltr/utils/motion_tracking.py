import torch
from ltr.data.processing_utils import gauss_2d
import random


class DiMPScoreJittering():
    def __init__(self, p_zero=0.0, distractor_ratio=1.0, p_distractor=0, max_distractor_enhance_factor=1,
                 min_distractor_enhance_factor=0.75):
        """ Jitters predicted score map by randomly enhancing distractor peaks and masking out target peaks"""
        self.p_zero = p_zero
        self.distractor_ratio = distractor_ratio
        self.p_distractor = p_distractor
        self.max_distractor_enhance_factor = max_distractor_enhance_factor
        self.min_distractor_enhance_factor = min_distractor_enhance_factor

    def rand(self, sz, min_val, max_val):
        return torch.rand(sz, device=min_val.device) * (max_val - min_val) + min_val

    def __call__(self, score, label):
        score_shape = score.shape

        score = score.view(-1, score_shape[-2]*score_shape[-1])
        num_score_maps = score.shape[0]

        label = label.view(score.shape)

        dist_roll_value = torch.rand(num_score_maps).to(score.device)

        score_c = score.clone().detach()
        score_neg = score_c * (label < 1e-4).float()
        score_pos = score_c * (label > 0.2).float()

        target_max_val, _ = torch.max(score_pos, dim=1)
        dist_max_val, dist_id = torch.max(score_neg, dim=1)

        jitter_score = (dist_roll_value < self.p_distractor) & ((dist_max_val / target_max_val) > self.distractor_ratio)

        for i in range(num_score_maps):
            score_c[i, dist_id[i]] = self.rand(1, target_max_val[i]*self.min_distractor_enhance_factor,
                                               target_max_val[i]*self.max_distractor_enhance_factor)

        zero_roll_value = torch.rand(num_score_maps).to(score.device)
        zero_score = (zero_roll_value < self.p_zero) & ~jitter_score

        score_c[zero_score, :] = 0

        score_jittered = score*(1.0 - (jitter_score | zero_score).float()).view(num_score_maps, 1).float() + \
                         score_c*(jitter_score | zero_score).float().view(num_score_maps, 1).float()

        return score_jittered.view(score_shape)


class LabelJittering():
    def __init__(self, label_sigma, p_random=0, p_prev_distractor=0):
        self.p_random = p_random
        self.p_prev_distractor = p_prev_distractor
        self.label_sigma = label_sigma

    def rand(self, sz, min_val, max_val):
        return torch.rand(sz, device=min_val.device) * (max_val - min_val) + min_val

    def __call__(self, label, score):
        input_label = label.clone()
        prev_score = score.clone()
        label_shape = label.shape

        label = label.view(-1, label_shape[-2], label_shape[-1])
        prev_score = prev_score.view(prev_score.shape[0], prev_score.shape[-2], prev_score.shape[-1])
        prev_score = prev_score[:, :-1, :-1]

        num_labels = label.shape[0]

        prev_score_neg = (prev_score * (label < 0.05).float()).view(num_labels, -1)
        dist_max_val, dist_id = torch.max(prev_score_neg, dim=1)

        dist_y = dist_id // label_shape[-1]
        dist_x = dist_id % label_shape[-1]

        feat_sz = torch.Tensor([label_shape[-2], label_shape[-1]])
        feat_center = (feat_sz - 1) / 2

        for i in range(num_labels):
            if torch.rand(1) < self.p_random:
                new_center = self.rand((1,2), min_val=torch.Tensor([1]), max_val=torch.Tensor([label_shape[-2]-1])) - feat_center
                label[i, :, :] = gauss_2d(feat_sz, self.label_sigma, new_center)
            elif torch.rand(1) < self.p_prev_distractor and dist_max_val[i] > 0.1:
                new_center = torch.Tensor([dist_x[i], dist_y[i]]).view(1, 2) - feat_center
                label[i, :, :] = gauss_2d(feat_sz, self.label_sigma, new_center)
            else:
                pass

        return label.view(label_shape)


class DiMPScoreJitteringState:
    def __init__(self, num_sequence, p_distractor_lo=0, p_distractor_hi=0, p_dimp_fail=0.4):
        self.num_sequence = num_sequence
        self.p_distractor_lo = p_distractor_lo
        self.p_distractor_hi = p_distractor_hi

        self.p_dimp_fail = p_dimp_fail

        self.remaining_distractor_frames = torch.zeros(num_sequence).long()
        self.dimp_fail_seq = torch.zeros(num_sequence).long()

    def __call__(self, score, label):
        assert score.dim() == 4

        score_shape = score.shape

        score = score.view(-1, score.shape[-2], score.shape[-1])

        # If target occluded, do nothing
        score_re = score.view(-1, label.shape[-2]*label.shape[-1])
        max_score = score_re.max(-1)[0]

        label_re = label.view(-1, label.shape[-2]*label.shape[-1])
        max_label_value, max_pos = label_re.max(-1)

        max_col = max_pos % label.shape[-1]
        max_row = max_pos // label.shape[-1]

        occluded = (max_label_value < 0.25).cpu()

        # Else, determine whether to jitter
        roll = torch.rand(self.num_sequence)

        jitter_seq_1 = (roll < self.p_distractor_hi) * (self.remaining_distractor_frames > 0)
        self.remaining_distractor_frames -= 1

        jitter_seq_2 = (1 - jitter_seq_1) * (roll < self.p_distractor_lo)
        self.remaining_distractor_frames[jitter_seq_2] = torch.randint(0, 10, (jitter_seq_2.sum(),))
        self.dimp_fail_seq[jitter_seq_2] = (torch.rand(jitter_seq_2.sum()) < self.p_dimp_fail).long()

        jittered_seq = ((jitter_seq_1 + jitter_seq_2) > 0) * (1 - occluded)

        jitter_info = []
        for i_ in range(self.num_sequence):
            if jittered_seq[i_]:
                if self.dimp_fail_seq[i_]:
                    score[i_, max_row[i_], max_col[i_]] = random.uniform(0.05, 0.2)

                    if max_row[i_] > 0:
                        if max_col[i_] > 0:
                            score[i_, max_row[i_] - 1, max_col[i_] - 1] = score[i_, max_row[i_], max_col[i_]] * random.uniform(0.05, 0.3)
                        if max_col[i_] < score.shape[-1] - 1:
                            score[i_, max_row[i_] - 1, max_col[i_] + 1] = score[i_, max_row[i_], max_col[i_]] * random.uniform(0.05, 0.3)
                        score[i_, max_row[i_] - 1, max_col[i_]] = score[i_, max_row[i_], max_col[i_]] * random.uniform(0.2, 0.55)

                    if max_row[i_] < score.shape[-2] - 1:
                        if max_col[i_] > 0:
                            score[i_, max_row[i_] + 1, max_col[i_] - 1] = score[i_, max_row[i_], max_col[i_]] * random.uniform(0.05, 0.3)
                        if max_col[i_] < score.shape[-1] - 1:
                            score[i_, max_row[i_] + 1, max_col[i_] + 1] = score[i_, max_row[i_], max_col[i_]] * random.uniform(0.05, 0.3)
                        score[i_, max_row[i_] + 1, max_col[i_]] = score[i_, max_row[i_], max_col[i_]] * random.uniform(0.2, 0.55)

                    if max_col[i_] > 0:
                        score[i_, max_row[i_], max_col[i_] - 1] = score[i_, max_row[i_], max_col[i_]] * random.uniform(0.05, 0.3)
                    if max_col[i_] < score.shape[-1] - 1:
                        score[i_, max_row[i_], max_col[i_] + 1] = score[i_, max_row[i_], max_col[i_]] * random.uniform(0.05, 0.3)

                    peak_score = 0.35
                else:
                    peak_score = max_score[i_]

                # jitter
                jittered_row_del = random.randint(3, 8) * random.choice([-1, 1])
                jittered_col_del = random.randint(3, 8) * random.choice([-1, 1])

                row = (max_row[i_] + jittered_row_del) % label.shape[-2]
                col = (max_col[i_] + jittered_col_del) % label.shape[-1]

                score[i_, row, col] = peak_score * random.uniform(0.7, 1.3)

                if row > 0:
                    if col > 0:
                        score[i_, row-1, col-1] = peak_score * random.uniform(0.05, 0.3)
                    if col < (score.shape[-1] - 1):
                        score[i_, row-1, col+1] = peak_score * random.uniform(0.05, 0.3)
                    score[i_, row - 1, col] = peak_score * random.uniform(0.2, 0.55)

                if row < (score.shape[-2] - 1):
                    if col > 0:
                        score[i_, row+1, col-1] = peak_score * random.uniform(0.05, 0.3)
                    if col < (score.shape[-1] - 1):
                        score[i_, row+1, col+1] = peak_score * random.uniform(0.05, 0.3)
                    score[i_, row + 1, col] = peak_score * random.uniform(0.2, 0.55)

                if col > 0:
                    score[i_, row, col - 1] = peak_score * random.uniform(0.05, 0.3)
                if col < (score.shape[-1] - 1):
                    score[i_, row, col + 1] = peak_score * random.uniform(0.05, 0.3)

                jitter_info.append({'id': i_, 'dist_row': row, 'dist_col': col, 'dimp_fail': self.dimp_fail_seq[i_]})

        score = score.view(score_shape)
        return score, jitter_info
