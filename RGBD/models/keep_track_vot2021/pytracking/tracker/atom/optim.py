import torch
import torch.nn.functional as F
from pytracking import optimization, TensorList, operation
import math


class FactorizedConvProblem(optimization.L2Problem):
    def __init__(self, training_samples: TensorList, y:TensorList, filter_reg: torch.Tensor, projection_reg, params, sample_weights: TensorList,
                 projection_activation, response_activation):
        self.training_samples = training_samples
        self.y = y
        self.filter_reg = filter_reg
        self.sample_weights = sample_weights
        self.params = params
        self.projection_reg = projection_reg
        self.projection_activation = projection_activation
        self.response_activation = response_activation

        self.diag_M = self.filter_reg.concat(projection_reg)

    def __call__(self, x: TensorList):
        """
        Compute residuals
        :param x: [filters, projection_matrices]
        :return: [data_terms, filter_regularizations, proj_mat_regularizations]
        """
        filter = x[:len(x)//2]  # w2 in paper
        P = x[len(x)//2:]       # w1 in paper

        # Do first convolution
        compressed_samples = operation.conv1x1(self.training_samples, P).apply(self.projection_activation)

        # Do second convolution
        residuals = operation.conv2d(compressed_samples, filter, mode='same').apply(self.response_activation)

        # Compute data residuals
        residuals = residuals - self.y

        residuals = self.sample_weights.sqrt().view(-1, 1, 1, 1) * residuals

        # Add regularization for projection matrix
        residuals.extend(self.filter_reg.apply(math.sqrt) * filter)

        # Add regularization for projection matrix
        residuals.extend(self.projection_reg.apply(math.sqrt) * P)

        return residuals


    def ip_input(self, a: TensorList, b: TensorList):
        num = len(a) // 2       # Number of filters
        a_filter = a[:num]
        b_filter = b[:num]
        a_P = a[num:]
        b_P = b[num:]

        # Filter inner product
        # ip_out = a_filter.reshape(-1) @ b_filter.reshape(-1)
        ip_out = operation.conv2d(a_filter, b_filter).view(-1)

        # Add projection matrix part
        # ip_out += a_P.reshape(-1) @ b_P.reshape(-1)
        ip_out += operation.conv2d(a_P.view(1,-1,1,1), b_P.view(1,-1,1,1)).view(-1)

        # Have independent inner products for each filter
        return ip_out.concat(ip_out.clone())

    def M1(self, x: TensorList):
        return x / self.diag_M



class FactorizedConvProblemHinge(optimization.L2Problem):
    def __init__(self, training_samples: TensorList, y:TensorList, filter_reg: torch.Tensor, projection_reg, params, sample_weights: TensorList,
                 projection_activation, response_activation, mask):
        self.training_samples = training_samples
        self.y = y
        self.mask = mask
        self.filter_reg = filter_reg
        self.sample_weights = sample_weights
        self.params = params
        self.projection_reg = projection_reg
        self.projection_activation = projection_activation
        self.response_activation = response_activation

        self.diag_M = self.filter_reg.concat(projection_reg)

    def __call__(self, x: TensorList):
        """
        Compute residuals
        :param x: [filters, projection_matrices]
        :return: [data_terms, filter_regularizations, proj_mat_regularizations]
        """
        filter = x[:len(x)//2]
        P = x[len(x)//2:]

        compressed_samples = operation.conv1x1(self.training_samples, P).apply(self.projection_activation)
        residuals = operation.conv2d(compressed_samples, filter, mode='same').apply(self.response_activation)
        residuals = self.mask * residuals + (1.0 - self.mask) * residuals.apply(F.relu) - self.mask * self.y

        residuals = self.sample_weights.sqrt().view(-1, 1, 1, 1) * residuals

        # Add regularization for projection matrix
        residuals.extend(self.filter_reg.apply(math.sqrt) * filter)

        # Add regularization for projection matrix
        residuals.extend(self.projection_reg.apply(math.sqrt) * P)

        return residuals


    def ip_input(self, a: TensorList, b: TensorList):
        num = len(a) // 2       # Number of filters
        a_filter = a[:num]
        b_filter = b[:num]
        a_P = a[num:]
        b_P = b[num:]

        # Filter inner product
        # ip_out = a_filter.reshape(-1) @ b_filter.reshape(-1)
        ip_out = operation.conv2d(a_filter, b_filter).view(-1)

        # Add projection matrix part
        # ip_out += a_P.reshape(-1) @ b_P.reshape(-1)
        ip_out += operation.conv2d(a_P.view(1,-1,1,1), b_P.view(1,-1,1,1)).view(-1)

        # Have independent inner products for each filter
        return ip_out.concat(ip_out.clone())

    def M1(self, x: TensorList):
        return x / self.diag_M


class FactorizedConvProblemJoint(optimization.L2Problem):
    def __init__(self, training_samples: TensorList, y: torch.Tensor, filter_reg: torch.Tensor, projection_reg, params, sample_weights: torch.Tensor,
                 projection_activation, response_activation):
        self.training_samples = training_samples
        self.y = y
        self.filter_reg = filter_reg
        self.sample_weights = sample_weights
        self.params = params
        self.projection_reg = projection_reg
        self.projection_activation = projection_activation
        self.response_activation = response_activation

        self.diag_M = self.filter_reg.concat(projection_reg)

    def __call__(self, x: TensorList):
        """
        Compute residuals
        :param x: [filters, projection_matrices]
        :return: [data_terms, filter_regularizations, proj_mat_regularizations]
        """
        filter = x[:len(x)//2]
        P = x[len(x)//2:]

        compressed_samples = operation.conv2d(self.training_samples, P).apply(self.projection_activation)
        scores = self.response_activation(operation.sum_interpolated(operation.conv2d(compressed_samples, filter, mode='same')))
        residuals = TensorList([scores - self.y])

        if self.sample_weights is not None:
            residuals = residuals * self.sample_weights.sqrt().view(-1, 1, 1, 1)

        # Add regularization for projection matrix
        residuals.extend(self.filter_reg.apply(math.sqrt) * filter)

        # Add regularization for projection matrix
        residuals.extend(self.projection_reg.apply(math.sqrt) * P)

        return residuals

    def M1(self, x: TensorList):
        return x / self.diag_M



class FactorizedConvProblemSoftmaxCE(optimization.SoftmaxCEProblem):
    def __init__(self, training_samples: TensorList, y: TensorList, filter_reg: torch.Tensor, projection_reg, params, sample_weights: TensorList,
                 projection_activation, response_activation, softmax_reg=None, label_shrink=0, label_threshold=0):
        super().__init__(sample_weights[0], softmax_reg)
        self.training_samples = training_samples
        self.y = y
        self.filter_reg = filter_reg
        self.params = params
        self.projection_reg = projection_reg
        self.projection_activation = projection_activation
        self.response_activation = response_activation
        self.label_shrink = label_shrink
        self.label_threshold = label_threshold

        self.diag_M = self.filter_reg.concat(projection_reg)

    def get_label_density(self):
        label_prob = self.y[0] / self.y[0].sum(dim=(-2,-1), keepdim=True)
        label_prob *= (label_prob > self.label_threshold).float()
        label_prob *= 1 - self.label_shrink
        return label_prob


    def get_scores(self, x: TensorList):
        filter = x[:len(x)//2]  # w2 in paper
        P = x[len(x)//2:]       # w1 in paper

        # Do first convolution
        compressed_samples = operation.conv1x1(self.training_samples, P).apply(self.projection_activation)

        # Do second convolution
        scores = operation.conv2d(compressed_samples, filter, mode='same').apply(self.response_activation)
        return scores[0]

    def get_residuals(self, x: TensorList):
        filter = x[:len(x)//2]  # w2 in paper
        P = x[len(x)//2:]       # w1 in paper

        # Add regularization for projection matrix
        residuals = self.filter_reg.apply(math.sqrt) * filter

        # Add regularization for projection matrix
        residuals.extend(self.projection_reg.apply(math.sqrt) * P)

        return residuals


    def ip_input(self, a: TensorList, b: TensorList):
        num = len(a) // 2       # Number of filters
        a_filter = a[:num]
        b_filter = b[:num]
        a_P = a[num:]
        b_P = b[num:]

        # Filter inner product
        # ip_out = a_filter.reshape(-1) @ b_filter.reshape(-1)
        ip_out = operation.conv2d(a_filter, b_filter).view(-1)

        # Add projection matrix part
        # ip_out += a_P.reshape(-1) @ b_P.reshape(-1)
        ip_out += operation.conv2d(a_P.view(1,-1,1,1), b_P.view(1,-1,1,1)).view(-1)

        # Have independent inner products for each filter
        return ip_out.concat(ip_out.clone())

    def M1(self, x: TensorList):
        return x / self.diag_M


class FactorizedConvProblemLogistic(optimization.MinimizationProblem):
    def __init__(self, training_samples: TensorList, y:TensorList, filter_reg: torch.Tensor, projection_reg, params, sample_weights: TensorList,
                 projection_activation):
        self.training_samples = training_samples
        self.y = y
        self.filter_reg = filter_reg
        self.sample_weights = sample_weights
        self.params = params
        self.projection_reg = projection_reg
        self.projection_activation = projection_activation

        self.diag_M = self.filter_reg.concat(projection_reg)

    def __call__(self, x: TensorList):
        """
        Compute residuals
        :param x: [filters, projection_matrices]
        :return: [data_terms, filter_regularizations, proj_mat_regularizations]
        """
        filter = x[:len(x)//2]
        P = x[len(x)//2:]

        compressed_samples = operation.conv1x1(self.training_samples, P).apply(self.projection_activation)
        scores = operation.conv2d(compressed_samples, filter, mode='same')

        loss = TensorList([F.binary_cross_entropy_with_logits(s, y, reduction='none') for s, y in zip(scores, self.y)])
        loss = TensorList([l.view(l.shape[0],-1).mean(dim=1) for l in loss])
        loss = (self.sample_weights * loss).sum()

        # Add regularization for filter
        loss = loss + self.filter_reg * (filter.view(-1) @ filter.view(-1))

        # Add regularization for projection matrix
        loss = loss + self.projection_reg * (P.view(-1) @ P.view(-1))

        if len(loss) == 1:
            return loss[0]
        return sum(loss)


    def ip_input(self, a: TensorList, b: TensorList):
        num = len(a) // 2       # Number of filters
        a_filter = a[:num]
        b_filter = b[:num]
        a_P = a[num:]
        b_P = b[num:]

        # Filter inner product
        ip_out = operation.conv2d(a_filter, b_filter).view(-1)

        # Add projection matrix part
        ip_out += operation.conv2d(a_P.view(1,-1,1,1), b_P.view(1,-1,1,1)).view(-1)

        # Have independent inner products for each filter
        return ip_out.concat(ip_out.clone())

    # def M1(self, x: TensorList):
    #     return x / self.diag_M


class FactorizedConvProblemCE(optimization.MinimizationProblem):
    def __init__(self, training_samples: TensorList, y:TensorList, filter_reg: torch.Tensor, projection_reg, params, sample_weights: TensorList,
                 projection_activation):
        self.training_samples = training_samples
        self.y = y
        self.filter_reg = filter_reg
        self.sample_weights = sample_weights
        self.params = params
        self.projection_reg = projection_reg
        self.projection_activation = projection_activation

        self.diag_M = self.filter_reg.concat(projection_reg)

    def __call__(self, x: TensorList):
        """
        Compute residuals
        :param x: [filters, projection_matrices]
        :return: [data_terms, filter_regularizations, proj_mat_regularizations]
        """
        filter = x[:len(x)//2]
        P = x[len(x)//2:]

        compressed_samples = operation.conv1x1(self.training_samples, P).apply(self.projection_activation)
        scores = operation.conv2d(compressed_samples, filter, mode='same')

        # Reshape
        label_prob = TensorList([y.view(y.shape[0], -1) for y in self.y])
        scores = TensorList([s.reshape(s.shape[0], -1) for s in scores])

        # Normalize
        label_prob = label_prob / label_prob.sum(dim=1, keepdim=True)

        loss = scores.exp().sum(dim=1).log() - (label_prob * scores).sum(dim=1)
        loss = (self.sample_weights * loss).sum()

        # Add regularization for filter
        loss = loss + self.filter_reg * (filter.view(-1) @ filter.view(-1))

        # Add regularization for projection matrix
        loss = loss + self.projection_reg * (P.view(-1) @ P.view(-1))

        if len(loss) == 1:
            return loss[0]
        return sum(loss)


    def ip_input(self, a: TensorList, b: TensorList):
        num = len(a) // 2       # Number of filters
        a_filter = a[:num]
        b_filter = b[:num]
        a_P = a[num:]
        b_P = b[num:]

        # Filter inner product
        ip_out = operation.conv2d(a_filter, b_filter).view(-1)

        # Add projection matrix part
        ip_out += operation.conv2d(a_P.view(1,-1,1,1), b_P.view(1,-1,1,1)).view(-1)

        # Have independent inner products for each filter
        return ip_out.concat(ip_out.clone())

    # def M1(self, x: TensorList):
    #     return x / self.diag_M


class ConvProblem(optimization.L2Problem):
    def __init__(self, training_samples: TensorList, y:TensorList, filter_reg: torch.Tensor, sample_weights: TensorList, response_activation):
        self.training_samples = training_samples
        self.y = y
        self.filter_reg = filter_reg
        self.sample_weights = sample_weights
        self.response_activation = response_activation

    def __call__(self, x: TensorList):
        """
        Compute residuals
        :param x: [filters]
        :return: [data_terms, filter_regularizations]
        """
        # Do convolution and compute residuals
        residuals = operation.conv2d(self.training_samples, x, mode='same').apply(self.response_activation)
        residuals = residuals - self.y

        residuals = self.sample_weights.sqrt().view(-1, 1, 1, 1) * residuals

        # Add regularization for projection matrix
        residuals.extend(self.filter_reg.apply(math.sqrt) * x)

        return residuals

    def ip_input(self, a: TensorList, b: TensorList):
        # return a.reshape(-1) @ b.reshape(-1)
        # return (a * b).sum()
        return operation.conv2d(a, b).view(-1)



class ConvProblemHinge(optimization.L2Problem):
    def __init__(self, training_samples: TensorList, y: TensorList, filter_reg: torch.Tensor, sample_weights: TensorList,
                 response_activation, mask: TensorList):
        self.training_samples = training_samples
        self.y = y
        self.mask = mask
        self.filter_reg = filter_reg
        self.sample_weights = sample_weights
        self.response_activation = response_activation

    def __call__(self, x: TensorList):
        """
        Compute residuals
        :param x: [filters]
        :return: [data_terms, filter_regularizations]
        """
        residuals = operation.conv2d(self.training_samples, x, mode='same').apply(self.response_activation)
        residuals = self.mask * residuals + (1.0 - self.mask) * residuals.apply(F.relu) - self.mask * self.y

        residuals = self.sample_weights.sqrt().view(-1, 1, 1, 1) * residuals

        # Add regularization for projection matrix
        residuals.extend(self.filter_reg.apply(math.sqrt) * x)

        return residuals

    def ip_input(self, a: TensorList, b: TensorList):
        # return a.reshape(-1) @ b.reshape(-1)
        # return (a * b).sum()
        return operation.conv2d(a, b).view(-1)


class ConvProblemSoftmaxCE(optimization.SoftmaxCEProblem):
    def __init__(self, training_samples: TensorList, y: TensorList, filter_reg: torch.Tensor, sample_weights: TensorList,
                 response_activation, softmax_reg=None, label_shrink=0, label_threshold=0):
        super().__init__(sample_weights[0], softmax_reg)
        self.training_samples = training_samples
        self.y = y
        self.filter_reg = filter_reg
        self.response_activation = response_activation
        self.label_shrink = label_shrink
        self.label_threshold = label_threshold

    def get_label_density(self):
        label_prob = self.y[0] / (self.y[0] + 1e-6).sum(dim=(-2, -1), keepdim=True)
        label_prob *= (label_prob > self.label_threshold).float()
        label_prob *= 1 - self.label_shrink
        return label_prob

    def get_scores(self, x: TensorList):
        # Do second convolution
        scores = operation.conv2d(self.training_samples, x, mode='same').apply(self.response_activation)
        return scores[0]

    def get_residuals(self, x: TensorList):
        # Add regularization for projection matrix
        residuals = self.filter_reg.apply(math.sqrt) * x
        return residuals

    def ip_input(self, a: TensorList, b: TensorList):
        # return a.reshape(-1) @ b.reshape(-1)
        # return (a * b).sum()
        return operation.conv2d(a, b).view(-1)


class ConvProblemLogistic(optimization.MinimizationProblem):
    def __init__(self, training_samples: TensorList, y:TensorList, filter_reg: torch.Tensor, sample_weights: TensorList):
        self.training_samples = training_samples
        self.y = y
        self.filter_reg = filter_reg
        self.sample_weights = sample_weights

    def __call__(self, x: TensorList):

        scores = operation.conv2d(self.training_samples, x, mode='same')

        loss = TensorList([F.binary_cross_entropy_with_logits(s, y, reduction='none') for s, y in zip(scores, self.y)])
        loss = TensorList([l.view(l.shape[0],-1).mean(dim=1) for l in loss])
        loss = (self.sample_weights * loss).sum()

        # Add regularization for filter
        loss = loss + self.filter_reg * (x.view(-1) @ x.view(-1))

        if len(loss) == 1:
            return loss[0]
        return sum(loss)

    def ip_input(self, a: TensorList, b: TensorList):
        return operation.conv2d(a, b).view(-1)


class ConvProblemCE(optimization.MinimizationProblem):
    def __init__(self, training_samples: TensorList, y:TensorList, filter_reg: torch.Tensor, sample_weights: TensorList):
        self.training_samples = training_samples
        self.y = y
        self.filter_reg = filter_reg
        self.sample_weights = sample_weights

    def __call__(self, x: TensorList):

        scores = operation.conv2d(self.training_samples, x, mode='same')

        # Reshape
        label_prob = TensorList([y.view(y.shape[0], -1) for y in self.y])
        scores = TensorList([s.reshape(s.shape[0], -1) for s in scores])

        # Normalize
        label_prob = label_prob / (label_prob.sum(dim=1, keepdim=True) + 1e-8)

        loss = scores.exp().sum(dim=1).log() - (label_prob * scores).sum(dim=1)
        loss = (self.sample_weights * loss).sum()

        # Add regularization for filter
        loss = loss + self.filter_reg * (x.view(-1) @ x.view(-1))

        if len(loss) == 1:
            return loss[0]
        return sum(loss)

    def ip_input(self, a: TensorList, b: TensorList):
        return operation.conv2d(a, b).view(-1)


class ConvProblemJoint(optimization.L2Problem):
    def __init__(self, training_samples: TensorList, y: torch.Tensor, filter_reg: torch.Tensor, sample_weights: torch.Tensor, response_activation):
        self.training_samples = training_samples
        self.y = y
        self.filter_reg = filter_reg
        self.sample_weights = sample_weights
        self.response_activation = response_activation

    def __call__(self, x: TensorList):
        """
        Compute residuals
        :param x: [filters]
        :return: [data_terms, filter_regularizations]
        """
        scores = self.response_activation(operation.sum_interpolated(operation.conv2d(self.training_samples, x, mode='same')))
        residuals = TensorList([scores - self.y])

        residuals = residuals * self.sample_weights.sqrt().view(-1, 1, 1, 1)

        # Add regularization for projection matrix
        residuals.extend(self.filter_reg.apply(math.sqrt) * x)

        return residuals

    def ip_input(self, a: TensorList, b: TensorList):
        return sum(a.reshape(-1) @ b.reshape(-1))

