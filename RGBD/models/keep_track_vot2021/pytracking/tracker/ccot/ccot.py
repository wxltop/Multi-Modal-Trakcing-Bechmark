from pytracking.tracker.base import BaseTracker
import torch
import math
from pytracking.libs import complex, dcf, fourier, TensorList
from pytracking.features.preprocessing import numpy_to_torch
from pytracking.utils.plotting import show_tensor
from .optim import FilterOptim



class CCOT(BaseTracker):

    multiobj_mode = 'parallel'
    def initialize(self, image, info: dict) -> dict:
        state = info['init_bbox']

        # Get position and size
        self.pos = torch.Tensor([state[1] + (state[3] - 1)/2, state[0] + (state[2] - 1)/2])
        self.target_sz = torch.Tensor([state[3], state[2]])

        # Set search area
        self.target_scale = 1.0
        search_area = torch.prod(self.target_sz * self.params.search_area_scale).item()
        if search_area > self.params.max_image_sample_size:
            self.target_scale =  math.sqrt(search_area / self.params.max_image_sample_size)
        elif search_area < self.params.min_image_sample_size:
            self.target_scale =  math.sqrt(search_area / self.params.min_image_sample_size)

        # Chack if image is color
        self.params.features.set_is_color(image.shape[2] == 3)

        # Target size in base scale
        self.base_target_sz = self.target_sz / self.target_scale

        # Use odd square search area and set sizes
        feat_max_stride = max(self.params.features.stride())
        self.img_sample_sz = torch.round(torch.sqrt(torch.prod(self.base_target_sz * self.params.search_area_scale))) * torch.ones(2)
        self.img_sample_sz += feat_max_stride - self.img_sample_sz % (2 * feat_max_stride)

        # Set other sizes (corresponds to ECO code)
        self.img_support_sz = self.img_sample_sz
        self.feature_sz = self.params.features.size(self.img_sample_sz)
        self.filter_sz = self.feature_sz
        self.output_sz = self.img_support_sz    # Interpolated size of the output

        # Number of filters
        self.num_filters = len(self.filter_sz)

        # Get window function
        self.window = TensorList([dcf.hann2d(sz) for sz in self.filter_sz])

        # Get interpolation function
        self.interp_fs = TensorList([dcf.get_interp_fourier(sz, self.params.interpolation_method,
                                                self.params.interpolation_bicubic_a, self.params.interpolation_centering,
                                                self.params.interpolation_windowing) for sz in self.filter_sz])

        # Get regularization filter
        self.reg_filter = dcf.get_reg_filter(self.img_support_sz, self.base_target_sz, self.params)
        self.reg_energy = self.reg_filter.view(-1) @ self.reg_filter.view(-1)

        # Get label function
        sigma = (self.filter_sz / self.img_support_sz) * torch.sqrt(self.base_target_sz.prod()) * self.params.output_sigma_factor
        self.yf = TensorList([dcf.label_function(sz, sig) for sz, sig in zip(self.filter_sz, sigma)])

        # Optimization options
        if self.params.CG_forgetting_rate is None or self.params.learning_rate >= 1:
            self.params.direction_forget_factor = 0
        else:
            self.params.direction_forget_factor = (1 - self.params.learning_rate)**self.params.CG_forgetting_rate

        # Convert image
        im = numpy_to_torch(image)

        # Extract and transform sample
        train_xf = self.extract_fourier_sample(im, self.pos, self.target_scale, self.img_sample_sz)

        # Shift sample
        shift_samp = 2*math.pi * (self.pos - self.pos.round()) / (self.target_scale * self.img_support_sz)
        train_xf = fourier.shift_fs(train_xf, shift=shift_samp)

        # Initialize memory
        self.num_stored_samples = 0
        self.sample_weights = torch.zeros(self.params.sample_memory_size)
        self.training_samples = TensorList([torch.zeros(xf.shape[2], xf.shape[3], self.params.sample_memory_size, xf.shape[1], 2) for xf in train_xf])

        # Update memory
        self.update_memory(train_xf)

        # Initialize filter
        self.filter = TensorList([torch.zeros_like(xf) for xf in train_xf])

        # Initialize optimizer
        self.filter_optimizer = FilterOptim(self.params, self.reg_energy)
        self.filter_optimizer.register(self.filter, self.training_samples, self.yf, self.sample_weights, self.reg_filter)

        # Optimize
        self.filter_optimizer.run(self.params.init_CG_iter, train_xf)


    def track(self, image, info: dict = None) -> dict:
        # Convert image
        im = numpy_to_torch(image)

        # ------- LOCALIZATION ------- #

        # Get sample
        sample_pos = self.pos.round()
        sample_scales = self.target_scale * self.params.scale_factors
        test_xf = self.extract_fourier_sample(im, self.pos, sample_scales, self.img_sample_sz)

        # Compute scores
        sf = self.apply_filter(test_xf)
        s = fourier.sample_fs(sf, self.output_sz)

        # Get maximum
        max_score, max_disp = dcf.max2d(s)
        _, scale_ind = torch.max(max_score, dim=0)

        # Convert to displacements in the base scale
        disp = (max_disp.float() + self.output_sz/2) % self.output_sz - self.output_sz/2

        # Compute translation vector and scale change factor
        translation_vec = disp[scale_ind,...].view(-1) * (self.img_support_sz / self.output_sz) * sample_scales[scale_ind]
        scale_change_factor = self.params.scale_factors[scale_ind]

        # Update position and scale
        self.pos = sample_pos + translation_vec
        self.target_scale *= scale_change_factor

        if self.params.debug >= 1:
            show_tensor(s[scale_ind,...], 5)
            show_tensor(fourier.cifft2(self.filter[0]).abs().sum(1), 6)


        # ------- UPDATE ------- #

        # Extract and transform sample
        # train_xf = self.extract_fourier_sample(im, self.pos, self.target_scale, self.img_sample_sz)

        # Get train sample
        train_xf = TensorList([xf[scale_ind:scale_ind+1, ...] for xf in test_xf])

        # Shift the sample
        shift_samp = 2*math.pi * (self.pos - sample_pos) / (sample_scales[scale_ind] * self.img_support_sz)
        train_xf = fourier.shift_fs(train_xf, shift=shift_samp)

        # Update memory and filter
        self.update_memory(train_xf)
        self.filter_optimizer.run(self.params.CG_iter, train_xf)

        # Return new state
        self.target_sz = self.base_target_sz * self.target_scale
        new_state = torch.cat((self.pos[[1,0]] - (self.target_sz[[1,0]]-1)/2, self.target_sz[[1,0]]))

        out = {'target_bbox': new_state.tolist()}
        return out


    def apply_filter(self, sample_xf: TensorList) -> torch.Tensor:
        return fourier.sum_fs(complex.mult(self.filter, sample_xf).sum(1, keepdim=True))


    def extract_fourier_sample(self, im: torch.Tensor, pos: torch.Tensor, scales, sz: torch.Tensor) -> TensorList:
        x = self.params.features.extract(im, pos, scales, sz)[0]
        x *= self.window
        sample_xf = fourier.cfft2(x)
        return TensorList([dcf.interpolate_dft(xf, bf) for xf, bf in zip(sample_xf, self.interp_fs)])


    def update_memory(self, sample_xf: TensorList):
        # Update weights and get index to replace
        replace_ind = self.update_sample_weights()
        for train_samp, xf in zip(self.training_samples, sample_xf):
            train_samp[:,:,replace_ind:replace_ind+1,:,:] = xf.permute(2, 3, 0, 1, 4)
        self.num_stored_samples += 1


    def update_sample_weights(self):
        if self.num_stored_samples == 0:
            replace_ind = 0
            self.sample_weights[0] = 1
        else:
            # Get index to replace
            _, replace_ind = torch.min(self.sample_weights, 0)
            replace_ind = replace_ind.item()

            # Update weights
            if self.num_stored_samples == 1 or self.params.learning_rate == 1:
                self.sample_weights[self.previous_replace_ind] = 1 - self.params.learning_rate
                self.sample_weights[replace_ind] = self.params.learning_rate
            else:
                self.sample_weights[replace_ind] = self.sample_weights[self.previous_replace_ind] / (1 - self.params.learning_rate)

        self.sample_weights /= self.sample_weights.sum()
        self.previous_replace_ind = replace_ind
        return replace_ind