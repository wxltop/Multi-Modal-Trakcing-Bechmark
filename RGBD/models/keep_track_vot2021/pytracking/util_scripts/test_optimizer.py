import torch
from libs import TensorList
from libs.optimization import GaussNewtonCG


class res_func():
    def __init__(self):
        self.num_dim, self.num_var = 300, 100
        self.A = torch.rand(self.num_dim, self.num_var) - 0.5
        self.y = torch.rand(self.num_dim) - 0.5
        self.lam = 0.1

    def __call__(self, x):
        return TensorList([(self.A @ x[0]) - self.y, self.lam * x[0]])


class res_func2():
    def __init__(self):
        self.num_dim = 300
        self.num_var = (100, 20)
        self.A = torch.rand(self.num_dim, self.num_var[0]) - 0.5
        self.y = torch.rand(self.num_dim) - 0.5
        self.lam = 0.01

    def __call__(self, x):
        return TensorList([(self.A @ x[0]) @ x[1] - self.y, self.lam * x[0], self.lam * x[1]])


def main():
    f = res_func()
    x = TensorList([torch.zeros(f.num_var)])

    opt = GaussNewtonCG(debug=True)
    opt.register_var(x)
    opt.register_residual_func(f)

    x_opt = opt.run(5, 4)[0]

    x_true = torch.inverse(f.A.t() @ f.A + f.lam ** 2 * torch.eye(f.num_var)) @ (f.A.t() @ f.y)

    print(x_true)
    print(x_opt)
    print('Error:  {}'.format(torch.mean((x_true - x_opt).abs())))


if __name__ == '__main__':
    main()