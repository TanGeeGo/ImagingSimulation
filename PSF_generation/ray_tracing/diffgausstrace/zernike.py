import torch
import numpy as np


class Zernike_polynomial:
    """https://en.wikipedia.org/wiki/Zernike_polynomials"""
    def __init__(self, coefficient=None, device=torch.device('cpu')):
        if coefficient is not None:
            self.c = coefficient
        else:
            self.c = torch.zeros(15)
        self.device = device
    def get_coefficient(self):
        return self.c

    def evaluate(self, R, P, require_matrix=False):
        # 计算各项数值，在外面乘系数c
        matrix = []
        # value = []
        if not torch.is_tensor(R):
            R = torch.tensor(R)
        if not torch.is_tensor(P):
            P = torch.tensor(P)
        for r, p in zip(R, P):
            zernike_terms = [torch.tensor(1),
                             2 * r * torch.sin(p),
                             2 * r * torch.cos(p),
                             np.sqrt(6) * torch.pow(r, 2) * torch.sin(2 * p),
                             np.sqrt(3) * (2 * torch.pow(r, 2) - 1),
                             np.sqrt(6) * torch.pow(r, 2) * torch.cos(2 * p),
                             np.sqrt(8) * torch.pow(r, 3) * torch.sin(3 * p),
                             np.sqrt(8) * (3 * torch.pow(r, 3) - 2 * r) * torch.sin(p),
                             np.sqrt(8) * (3 * torch.pow(r, 3) - 2 * r) * torch.cos(p),
                             np.sqrt(8) * torch.pow(r, 3) * torch.cos(3 * p),
                             np.sqrt(10) * torch.pow(r, 4) * torch.sin(4 * p),
                             np.sqrt(10) * (4 * torch.pow(r, 4) - 3 * torch.pow(r, 2)) * torch.sin(2 * p),
                             np.sqrt(5) * (6 * torch.pow(r, 4) - 6 * torch.pow(r, 2) + 1),
                             np.sqrt(10) * (4 * torch.pow(r, 4) - 3 * torch.pow(r, 2)) * torch.cos(2 * p),
                             np.sqrt(10) * torch.pow(r, 4) * torch.cos(4 * p)]
            matrix.append(zernike_terms)
            # value.append(sum(map(lambda x,y:x*y,zernike_terms, self.c)))
        if require_matrix:
            return torch.tensor(matrix).to(self.device)
        else:
            return torch.einsum('ij,j->i', (torch.tensor(matrix).to(self.device), self.c))

    def polynomial_fit(self, R, P, V):
        A = self.evaluate(R, P, require_matrix=True)
        B = V.reshape(-1)
        X, residuals, rank, singular_values = torch.linalg.lstsq(A, B, driver='gels')
        print("fitted coefficients:{}".format(X))
        print("lstsq residuals:{}".format(residuals))
        self.c = X

    def diff(self, R, P):
        """return dz/dx,dz/dy"""
        dzdx = []
        dzdy = []
        if not torch.is_tensor(R):
            R = torch.tensor(R)
        if not torch.is_tensor(P):
            P = torch.tensor(P)
        for r, p in zip(R, P):
            zernike_diff_p = [torch.tensor(0),
                             2 * r * torch.cos(p),
                             2 * r * (-1) * torch.sin(p),
                             np.sqrt(6) * torch.pow(r, 2) * 2 * torch.cos(2 * p),
                             torch.tensor(0),
                             np.sqrt(6) * torch.pow(r, 2) * (-2) * torch.sin(2 * p),
                             np.sqrt(8) * torch.pow(r, 3) * 3 * torch.cos(3 * p),
                             np.sqrt(8) * (3 * torch.pow(r, 3) - 2 * r) * torch.cos(p),
                             np.sqrt(8) * (3 * torch.pow(r, 3) - 2 * r) * (-1) * torch.sin(p),
                             np.sqrt(8) * torch.pow(r, 3) * (-3) * torch.sin(3 * p),
                             np.sqrt(10) * torch.pow(r, 4) * 4 * torch.cos(4 * p),
                             np.sqrt(10) * (4 * torch.pow(r, 4) - 3 * torch.pow(r, 2)) * 2 * torch.cos(2 * p),
                             torch.tensor(0),
                             np.sqrt(10) * (4 * torch.pow(r, 4) - 3 * torch.pow(r, 2)) * (-2) * torch.sin(2 * p),
                             np.sqrt(10) * torch.pow(r, 4) * (-4) * torch.sin(4 * p)]
            dzdp = torch.einsum('i,i', torch.tensor(zernike_diff_p).to(self.device), self.c)
            zernike_diff_r = [torch.tensor(0),
                             2 * torch.sin(p),
                             2 * torch.cos(p),
                             np.sqrt(6) * 2 * r * torch.sin(2 * p),
                             np.sqrt(3) * (4 * r),
                             np.sqrt(6) * 2 * r * torch.cos(2 * p),
                             np.sqrt(8) * 3 * torch.pow(r, 2) * torch.sin(3 * p),
                             np.sqrt(8) * (9 * torch.pow(r, 2) - 2) * torch.sin(p),
                             np.sqrt(8) * (9 * torch.pow(r, 2) - 2) * torch.cos(p),
                             np.sqrt(8) * 3 * torch.pow(r, 2) * torch.cos(3 * p),
                             np.sqrt(10) * 4 * torch.pow(r, 3) * torch.sin(4 * p),
                             np.sqrt(10) * (16 * torch.pow(r, 3) - 6 * r) * torch.sin(2 * p),
                             np.sqrt(5) * (24 * torch.pow(r, 3) - 12 * r),
                             np.sqrt(10) * (16 * torch.pow(r, 3) - 6 * r) * torch.cos(2 * p),
                             np.sqrt(10) * 4 * torch.pow(r, 3) * torch.cos(4 * p)]
            dzdr = torch.einsum('i,i', torch.tensor(zernike_diff_r).to(self.device), self.c)
            # jacobian
            if r == 0:
                # divide 0 will cause nan grad
                dzdx.append(torch.tensor(0))
                dzdy.append(torch.tensor(0))
            else:
                dzdx.append((r * torch.cos(p) * dzdr - torch.sin(p) * dzdp) / r)
                dzdy.append((r * torch.sin(p) * dzdr + torch.cos(p) * dzdp) / r)
        return torch.tensor(dzdx).to(self.device), torch.tensor(dzdy).to(self.device)




if __name__ == "__main__":
    rho = torch.linspace(-1,1,20)
    phi = torch.linspace(-2,2,20)
    # value = torch.rand(20)
    # test_poly = Zernike_polynomial()
    # test_poly.polynomial_fit(rho, phi, value)
    # test_poly.evaluate([1,0.5],[0,1],require_value=True)
    # print(test_poly.evaluate([1,0.5],[0,1]))
    # def cartesian2polar(X, Y):
    #     """coordinates conversion 好吧 直接生成complex然后取abs和angle就好了"""
    #     # r = torch.tensor(list(map(lambda x, y: torch.sqrt(x**2+y**2), X, Y)))
    #     # p = torch.tensor(list(map(lambda x, y: torch.atan2(y, x), X, Y)))
    #     # return torch.polar(r, p)
    #     return torch.complex(X, Y)
    print(cartesian2polar(rho,phi))
    print(rho.dtype)