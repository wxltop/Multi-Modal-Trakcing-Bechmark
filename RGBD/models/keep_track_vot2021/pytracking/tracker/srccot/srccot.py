from tracker.base import BaseTracker
import torch
import math
from libs import complex, dcf, fourier
from features.preprocessing import numpy_to_torch
from utils.plotting import show_tensor
from .optim import FilterOptim



class SRCCOT(BaseTracker):

    multiobj_mode = 'parallel'
    """Single resolution CCOT."""

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

        # Target size in base scale
        self.base_target_sz = self.target_sz / self.target_scale

        # Use odd square search area and set sizes
        self.img_sample_sz = torch.round(torch.sqrt(torch.prod(self.base_target_sz * self.params.search_area_scale))) * torch.ones(2)
        self.img_sample_sz += self.params.features.stride() - self.img_sample_sz % (2 * self.params.features.stride())

        # Set other sizes (corresponds to ECO code)
        self.img_support_sz = self.img_sample_sz
        self.filter_sz = self.img_support_sz / self.params.features.stride()
        self.output_sz = self.img_support_sz    # Interpolated size of the output

        # Get window function
        self.window = dcf.hann2d(self.filter_sz)

        # Get interpolation function
        self.interp_fs = dcf.get_interp_fourier(self.filter_sz, self.params.interpolation_method,
                                                self.params.interpolation_bicubic_a, self.params.interpolation_centering,
                                                self.params.interpolation_windowing)

        # Get label function
        sigma = torch.sqrt(self.base_target_sz.prod()) * self.params.output_sigma_factor * (self.filter_sz / self.img_support_sz)
        self.yf = dcf.label_function(self.filter_sz, sigma)

        # Optimization options
        if self.params.CG_forgetting_rate is None or self.params.learning_rate >= 1:
            self.params.direction_forget_factor = 0
        else:
            self.params.direction_forget_factor = (1 - self.params.learning_rate)**self.params.CG_forgetting_rate

        # Convert image
        im = numpy_to_torch(image)

        # Extract and transform sample
        xf = self.extract_fourier_sample(im, self.pos, self.target_scale, self.img_sample_sz)

        # Initialize memory
        self.num_stored_samples = 0
        self.sample_weights = torch.zeros(self.params.sample_memory_size)
        self.training_samples = torch.zeros(xf.shape[2], xf.shape[3], self.params.sample_memory_size, xf.shape[1], 2)

        # Update memory
        self.update_memory(xf)

        # Initialize filter
        self.filter = torch.zeros_like(xf)

        # Initialize optimizer
        self.filter_optimizer = FilterOptim(self.params, self.params.reg_factor)
        self.filter_optimizer.register(self.filter, self.training_samples, self.yf, self.sample_weights)

        # Optimize
        self.filter_optimizer.run(self.params.init_CG_iter, xf)



    def track(self, image, info: dict = None) -> dict:
        # Convert image
        im = numpy_to_torch(image)

        # ------- LOCALIZATION ------- #

        # Get sample
        sample_pos = self.pos.round()
        xf = self.extract_fourier_sample(im, self.pos, self.target_scale * self.params.scale_factors, self.img_sample_sz)

        # Compute scores
        sf = self.apply_filter(xf)
        s = fourier.sample_fs(sf, self.output_sz)

        # Get maximum
        max_score, max_disp = dcf.max2d(s)
        _, scale_ind = torch.max(max_score, dim=0)

        # Convert to displacements in the base scale
        disp = (max_disp.float() + self.output_sz/2) % self.output_sz - self.output_sz/2

        # Compute translation vector and scale change factor
        translation_vec = disp[scale_ind,...].view(-1) * (self.img_support_sz / self.output_sz) * \
                          self.target_scale * self.params.scale_factors[scale_ind]
        scale_change_factor = self.params.scale_factors[scale_ind]

        # Update position and scale
        self.pos = sample_pos + translation_vec
        self.target_scale *= scale_change_factor

        if self.params.debug >= 1:
            show_tensor(s[scale_ind,...], 5)


        # ------- UPDATE ------- #

        # Extract and transform sample
        xf = self.extract_fourier_sample(im, self.pos, self.target_scale, self.img_sample_sz)

        # Update memory and filter
        self.update_memory(xf)
        self.filter_optimizer.run(self.params.CG_iter, xf)

        # Return new state
        self.target_sz = self.base_target_sz * self.target_scale
        new_state = torch.cat((self.pos[[1,0]] - (self.target_sz[[1,0]]-1)/2, self.target_sz[[1,0]]))
        out = {'target_bbox': new_state.tolist()}
        return out


    def apply_filter(self, xf: torch.Tensor) -> torch.Tensor:
        return complex.mult(self.filter, xf).sum(1, keepdim=True)


    def extract_fourier_sample(self, im: torch.Tensor, pos: torch.Tensor, scales, sz: torch.Tensor) -> torch.Tensor:
        x = self.params.features.extract(im, pos, scales, sz)[0]
        x = self.window * x
        xf = fourier.cfft2(x)
        return dcf.interpolate_dft(xf, self.interp_fs)


    def update_memory(self, xf: torch.Tensor):
        # Update weights and get index to replace
        replace_ind = self.update_sample_weights()
        self.training_samples[:,:,replace_ind:replace_ind+1,:,:] = xf.permute(2, 3, 0, 1, 4)
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
            if self.num_stored_samples == 1:
                self.sample_weights[self.previous_replace_ind] = 1 - self.params.learning_rate
                self.sample_weights[replace_ind] = self.params.learning_rate
            else:
                self.sample_weights[replace_ind] = self.sample_weights[self.previous_replace_ind] / (1 - self.params.learning_rate)

        self.sample_weights /= self.sample_weights.sum()
        self.previous_replace_ind = replace_ind
        return replace_ind