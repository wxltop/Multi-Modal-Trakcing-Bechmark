from pytracking.tracker.base import BaseTracker
import torch
import math
from pytracking import complex, dcf, fourier
from pytracking.features.preprocessing import numpy_to_torch
from pytracking.utils.plotting import show_tensor



class MOSSE(BaseTracker):

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

        # Initialize filters
        self.filter_num = None
        self.filter_den = None

        # Convert image
        im = numpy_to_torch(image)

        # Extract and transform sample
        xf = self.extract_fourier_sample(im, self.pos, self.target_scale, self.img_sample_sz)

        # Update filter
        self.update_filter(xf, self.yf)


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

        # Update filter
        self.update_filter(xf, self.yf)

        # Return new state
        self.target_sz = self.base_target_sz * self.target_scale
        new_state = torch.cat((self.pos[[1,0]] - (self.target_sz[[1,0]]-1)/2, self.target_sz[[1,0]]))
        out = {'target_bbox': new_state.tolist()}
        return out


    def update_filter(self, xf: torch.Tensor, yf: torch.Tensor) -> None:
        # Filter updates
        filter_num_update = complex.mult_conj(yf, xf)
        filter_den_update = torch.sum(complex.abs_sqr(xf), 1)

        if self.filter_num is None:
            self.filter_num = filter_num_update
        else:
            self.filter_num = (1 - self.params.learning_rate) * self.filter_num + self.params.learning_rate * filter_num_update

        if self.filter_den is None:
            self.filter_den = filter_den_update
        else:
            self.filter_den = (1 - self.params.learning_rate) * self.filter_den + self.params.learning_rate * filter_den_update


    def apply_filter(self, xf: torch.Tensor) -> torch.Tensor:
        return complex.div(complex.mult(self.filter_num, xf).sum(1, keepdim=True), self.filter_den + self.params.reg_factor)


    def extract_fourier_sample(self, im: torch.Tensor, pos: torch.Tensor, scales, sz: torch.Tensor) -> torch.Tensor:
        x = self.params.features.extract(im, pos, scales, sz)[0]
        x = self.window * x
        xf = fourier.cfft2(x)
        return dcf.interpolate_dft(xf, self.interp_fs)
