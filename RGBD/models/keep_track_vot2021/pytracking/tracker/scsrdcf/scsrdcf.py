from pytracking.tracker.base import BaseTracker
import torch
import torch.nn.functional as F
import math
from pytracking import complex, dcf, fourier
from pytracking.features.preprocessing import numpy_to_torch
from pytracking.utils.plotting import show_tensor, plot_graph



class SCSRDCF(BaseTracker):

    multiobj_mode = 'parallel'
    def initialize(self, image, info: dict) -> dict:
        state = info['init_bbox']

        if not self.params.has('device'):
            self.params.device = 'cuda' if self.params.use_gpu else 'cpu'

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
        self.window = dcf.hann2d(self.filter_sz).to(self.params.device)

        # Get interpolation function
        self.interp_fs = dcf.get_interp_fourier(self.filter_sz, self.params.interpolation_method,
                                                self.params.interpolation_bicubic_a, self.params.interpolation_centering,
                                                self.params.interpolation_windowing, self.params.device)

        # self.reg_factor = self.params.reg_window_min
        # self.params.reg_window_min = 0

        # Get regularization filter
        self.reg_filter = dcf.get_reg_filter(self.img_support_sz, self.base_target_sz, self.params).to(self.params.device)
        #self.reg_energy = self.reg_filter.view(-1) @ self.reg_filter.view(-1)

        self.reg_factor = self.params.reg_factor + self.reg_filter[0,0,self.reg_filter.shape[2]//2, self.reg_filter.shape[3]//2]
        self.reg_filter[0, 0, self.reg_filter.shape[2] // 2, self.reg_filter.shape[3] // 2] = 0

        # Get label function
        sigma = torch.sqrt(self.base_target_sz.prod()) * self.params.output_sigma_factor * (self.filter_sz / self.img_support_sz)
        self.yf = dcf.label_function(self.filter_sz, sigma).to(self.params.device)

        # Initialize data and filters
        self.rhs = None
        self.lhs_data = None
        self.lhs = None
        self.f = None
        self.g = None
        self.a = None
        self.f_bias = None
        self.g_bias = None
        self.a_bias = None

        self.lossvec = []
        self.resvec = []

        # Convert image
        im = numpy_to_torch(image)

        # Extract and transform sample
        xf = self.extract_fourier_sample(im, self.pos, self.target_scale, self.img_sample_sz)

        # Update filter
        self.update_memory(xf, self.yf)
        self.optimize_filter()


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
        disp = (max_disp.float().cpu() + self.output_sz/2) % self.output_sz - self.output_sz/2

        # Compute translation vector and scale change factor
        translation_vec = disp[scale_ind,...].view(-1) * (self.img_support_sz / self.output_sz) * \
                          self.target_scale * self.params.scale_factors[scale_ind]
        scale_change_factor = self.params.scale_factors[scale_ind]

        # Update position and scale
        self.pos = sample_pos + translation_vec
        self.target_scale *= scale_change_factor

        if self.params.debug >= 2:
            show_tensor(s[scale_ind,...], 5)
            show_tensor(fourier.sample_fs(self.f).sum(dim=1), 6)


        # ------- UPDATE ------- #

        # Extract and transform sample
        xf = self.extract_fourier_sample(im, self.pos, self.target_scale, self.img_sample_sz)

        # Update filter
        if self.params.learning_rate > 0:
            self.update_memory(xf, self.yf)
            self.optimize_filter()

        # Return new state
        self.target_sz = self.base_target_sz * self.target_scale
        new_state = torch.cat((self.pos[[1,0]] - (self.target_sz[[1,0]]-1)/2, self.target_sz[[1,0]]))
        out = {'target_bbox': new_state.tolist()}
        return out


    def update_memory(self, xf: torch.Tensor, yf: torch.Tensor) -> None:
        # Filter updates
        rhs_update = complex.mult_conj(yf, xf)
        lhs_update = complex.abs_sqr(xf)

        # Filter "numerator"
        if self.rhs is None:
            self.rhs = rhs_update
        else:
            self.rhs = (1 - self.params.learning_rate) * self.rhs + self.params.learning_rate * rhs_update

        # Filter "denominator"
        if self.lhs_data is None:
            self.lhs_data = lhs_update
        else:
            self.lhs_data = (1 - self.params.learning_rate) * self.lhs_data + self.params.learning_rate * lhs_update

        self.lhs = self.lhs_data + self.reg_factor**2


    def apply_filter(self, xf: torch.Tensor) -> torch.Tensor:
        return complex.mult(self.f, xf).sum(1, keepdim=True)


    def extract_fourier_sample(self, im: torch.Tensor, pos: torch.Tensor, scales, sz: torch.Tensor) -> torch.Tensor:
        x = self.params.features.extract(im, pos, scales, sz)[0]
        x = self.window * x
        xf = fourier.cfft2(x)
        return dcf.interpolate_dft(xf, self.interp_fs)


    def optimize_filter(self):
        self.update_GS_bias()

        if self.f is None:
            self.f = self.f_bias.clone()
            # self.g = self.g_bias
            # self.a = self.a_bias

        if self.params.debug >= 3:
            self.log_train_loss()

        for i in range(self.params.num_GS_iter):
            # Dinv_g = self.Dinv(self.reg_factor * self.g)
            # W_Dinv_g = self.W(Dinv_g)
            # Dinv_Wctp_a = self.Dinv(self.Wctp(self.a))
            # W_Dinv_Wctp_a = self.W(Dinv_Wctp_a)
            #
            # self.f = self.f_bias  - Dinv_g  - Dinv_Wctp_a
            # self.g = self.g_bias  - W_Dinv_g  - W_Dinv_Wctp_a
            # self.a = self.a_bias  - self.reg_factor * Dinv_g - self.reg_factor * W_Dinv_g  - self.reg_factor * Dinv_Wctp_a - W_Dinv_Wctp_a

            tau = self.params.SOR_weight

            # self.f = self.f_bias  + (1 - tau) * self.f - tau * Dinv_Wctp_g
            # self.g = self.g_bias  + (tau - tau**2) * self.W(self.f)  - tau**2 * self.W(Dinv_Wctp_g) + (1 - tau) * self.g

            # Update formula
            self.g = self.W(self.f)
            Dinv_Wctp_g = self.Dinv(self.Wctp(self.g) + 2 * self.reg_factor * self.g)
            self.f = self.f_bias  + (1 - tau) * self.f - tau * Dinv_Wctp_g

            if self.params.debug >= 3:
                self.log_train_loss()


    def update_GS_bias(self):
        # self.f_bias = self.Dinv(self.rhs)
        # self.g_bias = self.W(self.f_bias)
        # self.a_bias = self.reg_factor * self.f_bias + self.g_bias

        self.f_bias = self.params.SOR_weight * self.Dinv(self.rhs)
        # self.g_bias = self.params.SOR_weight**2 * self.W(self.f_bias)


    def Dinv(self, h):
        return complex.div(h, self.lhs)

    def W(self, h):
        reg_pad1 = min(self.reg_filter.shape[-2] - 1, h.shape[-3] - 1) // 2
        reg_pad2 = min(self.reg_filter.shape[-1] - 1, h.shape[-2] - 1) // 2

        # Add part needed for convolution
        if reg_pad2 > 0:
            hfe_left_padd = complex.conj(h[..., 1:reg_pad2 + 1, :].clone().detach().flip((2,3)))
            hfe_conv = torch.cat([hfe_left_padd, h], -2)
        else:
            hfe_conv = h.clone()

        # Shift data to batch dimension
        hfe_conv = hfe_conv.permute(0, 1, 4, 2, 3).reshape(-1, 1, hfe_conv.shape[-3], hfe_conv.shape[-2])

        # Do first convolution
        hfe_conv = F.conv2d(hfe_conv, self.reg_filter, padding=(reg_pad1, reg_pad2))

        return hfe_conv[..., reg_pad2:].reshape(h.shape[0], h.shape[1], 2, h.shape[2], h.shape[3]).permute(0, 1, 3, 4, 2)

    def Wctp(self, h):
        return self.W(h)

    def log_train_loss(self):
        Wf = self.W(self.f) + self.reg_factor * self.f
        Xf = complex.mult(self.lhs_data, self.f)

        loss = 0.5 * self.ip(self.f, Xf)
        loss -= self.ip(self.f, self.rhs)
        loss += 0.5 * self.ip(self.yf, self.yf)
        loss += 0.5 * self.ip(Wf, Wf)

        grad_f = Xf + self.reg_factor * Wf + self.Wctp(Wf) - self.rhs
        res = self.ip(grad_f, grad_f)

        # Wf = self.W(self.f)
        # Xf = complex.mult(self.lhs_data, self.f)
        #
        # loss = 0.5 * self.ip(self.f, Xf)
        # loss -= self.ip(self.f, self.rhs)
        # loss += 0.5 * self.ip(self.yf, self.yf)
        # loss += 0.5 * self.ip(Wf, Wf)
        # loss += 0.5 * self.reg_factor**2 * self.ip(self.f, self.f)
        #
        # grad_f = Xf + self.reg_factor**2 * self.f + self.Wctp(Wf)
        # res = self.ip(grad_f, grad_f)

        self.lossvec.append(loss)
        self.resvec.append(res)

        plot_graph(torch.Tensor(self.lossvec), 8)
        plot_graph(torch.Tensor(self.resvec), 9)


    def ip(self, a: torch.Tensor, b: torch.Tensor):
        return 2 * (a.reshape(-1) @ b.reshape(-1)) - a[:, :, :, 0].reshape(-1) @ b[:, :, :, 0].reshape(-1)