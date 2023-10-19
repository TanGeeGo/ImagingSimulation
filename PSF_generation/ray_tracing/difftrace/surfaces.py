import torch
import numpy as np
from .utils import *


class TransformMixin(PrettyPrinter):
    """
    Used for object transform, e.g., rotation, shift
    """

    def __init__(self, transformation, device=torch.device('cpu')):
        self.to_world = transformation.inverse()  # ! to_world R and t is here
        self.to_object = transformation  # ! to_object R and t is here
        self.to(device)
        self.device = device

    def intersect(self, ray):
        raise NotImplementedError()

    def sample_ray(self, position_sample=None):
        raise NotImplementedError()

    def draw_points(self, ax, options, seq=range(3)):
        raise NotImplementedError()

    def update_Rt(self, R, t):
        self.to_object = Transformation(R, t)
        self.to_world = self.to_object.inverse()
        self.to(self.device)


class Surface(TransformMixin):

    # ======================================================================================
    # Initialization （初始化）
    # ======================================================================================

    def __init__(self,
                 index=None,
                 radius=None,
                 distance=None,
                 distance_prev=None,
                 distance_after=None,
                 material=None,
                 material_prev=None,
                 #  origin = np.zeros(3),
                 shift=np.zeros(3),
                 theta_x=0.,
                 theta_y=0.,
                 theta_z=0.,
                 is_square=False,
                 dtype=torch.float64,
                 device=torch.device('cpu')
                 ):

        #
        self.index = index
        self.dtype = dtype
        self.device = device

        #
        self.r = radius if torch.is_tensor(radius) else \
            torch.tensor(np.asarray(radius), dtype=self.dtype).to(self.device)
        self.d = distance if torch.is_tensor(distance) else \
            torch.tensor(np.asarray(distance),
                         dtype=self.dtype).to(self.device)
        self.d_prev = distance_prev if torch.is_tensor(distance_prev) else \
            torch.tensor(np.asarray(distance_prev),
                         dtype=self.dtype).to(self.device)
        self.d_after = distance_after if torch.is_tensor(distance_after) else \
            torch.tensor(np.asarray(distance_after),
                         dtype=self.dtype).to(self.device)

        #
        self.Material = Material(
            str(material), dtype=self.dtype, device=self.device)
        self.Material_Prev = Material(
            str(material_prev), dtype=self.dtype, device=self.device)

        #
        # self.origin = origin if torch.is_tensor(origin) else \
        #     torch.tensor(np.asarray(origin), dtype=self.dtype).to(self.device)
        self.shift = shift if torch.is_tensor(shift) else \
            torch.tensor(np.asarray(shift), dtype=self.dtype).to(self.device)
        self.theta_x = theta_x if torch.is_tensor(theta_x) else \
            torch.tensor(np.asarray(theta_x), dtype=self.dtype).to(self.device)
        self.theta_y = theta_y if torch.is_tensor(theta_y) else \
            torch.tensor(np.asarray(theta_y), dtype=self.dtype).to(self.device)
        self.theta_z = theta_z if torch.is_tensor(theta_z) else \
            torch.tensor(np.asarray(theta_z), dtype=self.dtype).to(self.device)
        TransformMixin.__init__(
            self, self._compute_transformation(), self.device)

        #
        self.is_square = is_square

        # There are the parameters controlling the accuracy of ray tracing.
        self.NEWTONS_MAXITER = 200
        self.NEWTONS_TOLERANCE_TIGHT = 1e-11  # in [mm], i.e. 0.1 [nm] here (up to <10 [nm])
        self.NEWTONS_TOLERANCE_LOOSE = 1e-9  # in [mm], i.e. 1 [nm] here (up to <10 [nm])
        self.SURFACE_SAMPLING = 257

    def _compute_transformation(self, _x=0.0, _y=0.0, _z=0.0):
        # we compute to_world transformation given the input positional parameters (angles)
        R = (rodrigues_rotation_matrix(torch.tensor([1., 0., 0.], dtype=self.dtype).to(self.device),
                                       torch.deg2rad(self.theta_x + _x)) @
             rodrigues_rotation_matrix(torch.tensor([0., 1., 0.], dtype=self.dtype).to(self.device),
                                       torch.deg2rad(self.theta_y + _y)) @
             rodrigues_rotation_matrix(torch.tensor([0., 0., 1.], dtype=self.dtype).to(self.device),
                                       torch.deg2rad(self.theta_z + _z)))
        # t = self.origin + R @ self.shift
        t = self.shift
        return Transformation(R, t)

    # ======================================================================================
    # Common methods (must not be overridden in child class)
    # ======================================================================================

    def refractive_index(self, wavelength):
        """
        Return the refractive index of the material after this surface, i.e., n1
        """
        return self.Material.refractive_index(wavelength)

    def refractive_index_prev(self, wavelength):
        """
        Return the refractive index of the material before this surface, i.e., n0
        """
        return self.Material_Prev.refractive_index(wavelength)

    @property
    def abbe_number(self):
        """
        Return the abbe number of the material after this surface, i.e., vd
        """
        return self.Material.abbe_number()

    @property
    def abbe_number_prev(self):
        """
        Return the abbe number of the material before this surface, i.e., vd_
        """
        return self.Material_Prev.abbe_number()

    def surface(self, x, y):
        """
        Solve z from h(z) = -g(x,y).
        """
        raise NotImplementedError()

    def surface_with_offset(self, x, y):
        """
        Return the global z coordinates of the point on surface
        """
        return self.surface(x, y) + self.d

    def surface_normal(self, x, y):
        """
        The normal of surface in the position (x, y), i.e., [NORX, NORY, NORZ] in Zemax
        Output format: [NORX, NORY, NORZ] (normalized to 1)
        """
        ds_dxyz = self.surface_derivatives(x, y)
        return normalize(torch.stack(ds_dxyz, dim=-1))

    @property
    def mesh(self):
        """
        Generate a meshgrid mesh for the current surface.
        """
        x, y = torch.meshgrid(
            torch.linspace(-self.r, self.r,
                           self.SURFACE_SAMPLING, device=self.device),
            torch.linspace(-self.r, self.r,
                           self.SURFACE_SAMPLING, device=self.device),
            indexing='ij'
        )
        valid_map = self.is_valid(torch.stack((x, y), dim=-1))
        return self.surface(x, y) * valid_map

    def sdf_approx(self, xy):  # approximated SDF
        """
        This function is more computationally efficient than `sdf`.
        大致上是点中心的距离 减去 孔径
        - < 0: valid
        """
        if self.is_square:
            return torch.max(torch.abs(xy) - self.r, dim=-1)[0]
        else:  # is round
            return length2(xy) - self.r ** 2

    def is_valid(self, xy):
        return (self.sdf_approx(xy) <= 0.0).bool()

    # ====================================================================================== 
    # Key methods for propagation 
    # ======================================================================================

    def propagate(self, ray, active, record_propagate_loss=False):
        # save the global coor of o for optical path calculation
        o_glb = ray.o.clone()

        # free propagation to the initial plane of the last surface
        # inplace o_tmp
        if self.d:
            o_tmpp = torch.zeros_like(ray.o, dtype=self.dtype, device=self.device)
            t0 = (self.d + self.sag_min - ray.o[active, 2]) / ray.d[active, 2]
            o_tmpp[active, :2] = ray.o[active, :2] + t0[..., None] * ray.d[active, :2]
            o_tmpp[active, 2] = self.sag_min
            ray = Ray(o_tmpp, ray.d, ray.wavelength, device=self.device)

        # to local coordinates of the surface
        ray = self.to_object.transform_ray(ray)

        # intersection
        # local coor according to the origin of this surface

        valid_o, o_tmp, loss_no_intersection = self._ray_surface_intersection(ray, active)
        valid_d, d_tmp, loss_total_reflection = self._refract(o_tmp, ray, valid_o)
        valid_a = length2(o_tmp[..., 0:2]) < self.r ** 2
        valid_a_keepShape = valid_o.clone()
        valid_a_keepShape[valid_o] &= valid_a
        loss_outRange = (length(o_tmp[~valid_a, 0:2]) - self.r).sum()
        loss = loss_no_intersection + loss_total_reflection + loss_outRange

        valid = active & valid_o & valid_d & valid_a_keepShape
        # update the information after tracing
        ray.o[valid_o] = o_tmp  # * the coordinates of o_tmp are based on the current surface as the origin
        ray.d[valid_d] = d_tmp

        # to world coordinates
        ray = self.to_world.transform_ray(ray)

        # assign distance to match the global coordinates
        ray.o[valid, 2] += self.d

        # path length (for optical path calculation)
        ray.t = length(torch.clamp(torch.abs(ray.o - o_glb), min=1e-16)) * torch.sign(ray.o[..., 2] - o_glb[..., 2])

        # saving tracing data
        self.ray = ray
        # for debug
        self.ry = torch.sqrt(torch.sum(ray.o[..., 0:2] ** 2, dim=-1)).clone().detach().requires_grad_(True)

        self.valid_map = valid

        if record_propagate_loss:
            return valid, ray, loss
        else:
            return valid, ray

    def _refract(self, o_tmp, ray, active=None, gv=0):
        """
        Snell's law (surface normal n defined along the positive z axis)
        https://en.wikipedia.org/wiki/Snell%27s_law
        todo 验证一下return的wt需不需要normlize
        """
        # rayd = ray.d[active]
        # # get the n_in / n_out
        # mu = self.refractive_index_prev(ray.wavelength) / self.refractive_index(ray.wavelength)
        # # norm of each position
        # n = self.surface_normal(o_tmp[..., 0], o_tmp[..., 1])
        # # incident cosine, i.e., RAIN in zemax
        # # neg before norm to let angle is 0~90 degree
        # cosi = torch.sum(rayd * -n, dim=-1) / (length(rayd)*length(-n))
        # coso_squared = 1. - (1. - torch.square(cosi)) * torch.square(mu)
        # # validate the direction
        # # 1. get valid map; 2. zero out invalid points; 3. add eps to avoid NaN grad at cost2==0.
        # valid_d = coso_squared > 0.
        # coso = torch.sqrt(coso_squared[valid_d]) # RAEN in zemax
        # d = mu * rayd[valid_d] + (mu * cosi[valid_d].unsqueeze(-1) - coso.unsqueeze(-1)) * n[valid_d]  # REAA, REAB, REAC
        #
        # if valid_d.sum() != len(valid_d):
        #     print("total refract")
        # loss_total_reflection = - coso_squared[~valid_d].sum()
        # valid_out = active.clone()
        # valid_out[active] &= valid_d
        # return valid_out, normalize(d), loss_total_reflection

        n1 = self.refractive_index_prev(ray.wavelength)
        n2 = self.refractive_index(ray.wavelength)
        u = (n1 * ray.d[active] + gv) / n2
        normal = - self.surface_normal(o_tmp[..., 0], o_tmp[..., 1])
        cost2 = 1. - length2(u) + torch.einsum('...k,...k', normal, u) ** 2
        # 1. get valid map; 2. zero out invalid points; 3. add eps to avoid
        # NaN grad at cost2==0.
        valid_d = cost2 > 0.
        tmp = torch.sqrt(cost2[valid_d])
        # here we do not have to do normalization because if both wi and n are normalized,
        # then output is also normalized.
        wt = u[valid_d] + normal[valid_d] * \
             (tmp - torch.einsum('...k,...k', normal, u)[valid_d])[..., None]

        # if valid_d.sum() != len(valid_d):
        #     print("total refract")
        loss_total_reflection = - cost2[~valid_d].sum()
        valid_out = active.clone()
        valid_out[active] &= valid_d

        return valid_out, normalize(wt), loss_total_reflection

    def gausslets_ray_surface_intersection(self, ray):
        """
        Returns:
        - p: intersection point
        - g: explicit funciton
        """
        valid_s, local, t = self.newtons_method(ray.maxt, ray.o, ray.d)
        valid_a = self.is_valid(local[..., 0:2])
        return valid_s, valid_a, local, t

    def propagate_paraxial(self, ray, active=None):
        # to surface coordinates
        ray = self.to_object.transform_ray(ray)

        o_tmp = torch.zeros_like(ray.o)
        d_tmp = torch.zeros_like(ray.d)

        m = self.paraxial_matrix(ray.wavelength)
        # nu is represent as n*tan(u) (the same as Zemax)
        od = torch.vstack((ray.o[..., 0:2].unsqueeze(0),
                           torch.tan(torch.arcsin(ray.d[..., 0:2].unsqueeze(0))) *
                           self.refractive_index_prev(ray.wavelength)))
        od_tmp = torch.einsum('ij,jlp->ilp', m, od)

        # update the information after paraxial tracing
        o_tmp[..., 0:2], d_tmp[..., 0:2] = torch.vsplit(od_tmp, 2)
        # check if the position of ray is valid
        valid_o = self.is_valid(o_tmp[..., 0:2])
        if active is not None:
            valid = active & valid_o
        else:
            valid = valid_o
        # ATTENTION: no normalizing d here (same as Zemax)
        u = torch.arctan(d_tmp / self.refractive_index(ray.wavelength))
        d_tmp = torch.concatenate((torch.sin(u)[..., 0].unsqueeze(-1),
                                   torch.sin(u)[..., 1].unsqueeze(-1),
                                   torch.cos(u)[..., 1].unsqueeze(-1)), dim=-1)

        ray.o = o_tmp
        ray.d = d_tmp

        # to world coordinates
        ray = self.to_world.transform_ray(ray)

        # assign distance to match the global coordinates
        ray.o[..., 2] += self.d
        return valid, ray

    def paraxial_matrix(self, wavelength):
        """
        2x2 block matrix, M = [[A, B], [C, D]]
        """
        m = torch.eye(2, dtype=self.dtype, device=self.device)
        # here distance is the propagate distance
        m[0, 1] = self.d_prev / self.refractive_index_prev(wavelength)
        return m

    # === Virtual methods (must be overridden)
    def sag_min_calc(self, sampling):
        raise NotImplementedError()

    def dgd(self, x, y):
        """
        Derivatives of g: (g'x, g'y).
        """
        raise NotImplementedError()

    def _dgd(self, r2):
        """
        calcuelate the dirivation
        """
        raise NotImplementedError()

    def _dgd2(self, r2):
        """
        calcuelate the second dirivation
        """
        raise NotImplementedError()

    def h(self, z):
        raise NotImplementedError()

    def dhd(self, z):
        """
        Derivative of h.
        """
        raise NotImplementedError()

    def ray_surface_intersection(self, ray, active=None):
        """
        Get the intersections of ray and surface
        ray length to intersection with element
        only reference plane, overridden in subclasses
        solution for z=0
        """
        # free propagation to the initial plane of the last surface
        # if self.d:
        #     o_tmp = torch.zeros_like(ray.o, dtype=self.dtype).to(self.device)
        #     t = (self.d - ray.o[..., 2]) / ray.d[..., 2]
        #     o_tmp[..., :2] = ray.o[..., :2] + t[..., None] * ray.d[..., :2]
        # else:
        #     o_tmp = ray.o

        valid_o = self.is_valid(ray.o[..., 0:2])
        if active is not None:
            valid_o = active & valid_o
        return valid_o, ray.o

    def _ray_surface_intersection(self, ray, active):
        """
        Get the intersections of ray and surface
        ray length to intersection with element
        only reference plane, overridden in subclasses
        solution for z=0
        """
        # free propagation to the initial plane of the last surface
        # if self.d:
        #     o_tmp = torch.zeros_like(ray.o, dtype=self.dtype).to(self.device)
        #     t = (self.d - ray.o[..., 2]) / ray.d[..., 2]
        #     o_tmp[..., :2] = ray.o[..., :2] + t[..., None] * ray.d[..., :2]
        # else:
        #     o_tmp = ray.o

        valid_o = self.is_valid(ray.o[..., 0:2])
        valid_o = active & valid_o
        loss_no_intersection = (length(ray.o[..., 0:2]) - self.r).sum()

        return valid_o, ray.o, loss_no_intersection

    def reverse(self):
        raise NotImplementedError()

    def surface_derivatives(self, x, y):
        """
        Returns \nabla f = \nabla (g(x,y) + h(z)) = (dg/dx, dg/dy, dh/dz).
        (Note: this default implementation is not efficient)
        """
        gx, gy = self.dgd(x, y)
        z = self.surface(x, y)
        return gx, gy, self.dhd(z)

    # for newton iteration
    def surface_and_derivatives_dot_D(self, t, dx, dy, dz, ox, oy, z):
        """
        Returns g(x,y)+h(z) and dot((g'x,g'y,h'), (dx,dy,dz)).
        (Note: this default implementation is not efficient)
        """
        x = ox + t * dx
        y = oy + t * dy
        s = self.surface(x, y) + self.h(z) - self.sag_min
        sx, sy = self.dgd(x, y)
        sz = self.dhd(z)
        return s, sx * dx + sy * dy + sz * dz

    # for tolerance analysis
    def fr(self, dr, du, s_prev, valid_map):
        """refract func"""
        n_prev = self.refractive_index_prev(self.ray.wavelength)
        cosu_prev = s_prev.ray.d[valid_map, 2]
        sinu_prev = torch.sqrt(1 - cosu_prev ** 2)
        n = self.refractive_index(self.ray.wavelength)
        cosu = self.ray.d[valid_map, 2]
        sinu = torch.sqrt(1 - cosu ** 2)
        ds_dr = self._dgd(self.ry[valid_map] ** 2) * 2 * self.ry[valid_map]
        ds_dr2 = self._dgd2(self.ry[valid_map] ** 2) * 2 * self.ry[valid_map]
        D = n_prev * cosu_prev - ds_dr * n_prev * sinu_prev
        E = -n * cosu + ds_dr * n * sinu
        F = (n_prev * cosu_prev - n * cosu) * ds_dr2
        du_prev = -(F * dr + E * du) / D
        return du_prev

    def fp(self, dr, du_prev, s_prev, valid_map):
        """prop func"""
        u_prev = torch.arccos(s_prev.ray.d[valid_map, 2])
        ds_dr = self._dgd(self.ry[valid_map] ** 2) * 2 * self.ry[valid_map]
        ds_dr_prev = s_prev._dgd(s_prev.ry[valid_map] ** 2) * 2 * s_prev.ry[valid_map]
        A = torch.tan(u_prev) * ds_dr - 1
        B = (self.ray.o[valid_map, 2] - s_prev.ray.o[valid_map, 2]) / torch.cos(u_prev) ** 2
        C = torch.tan(u_prev) * ds_dr_prev + 1
        dr_prev = -(A * dr + B * du_prev) / C
        return dr_prev


class Spheric(Surface):
    """
    Spheric surface
    """

    def __init__(self, roc, conic, **kwargs):
        super().__init__(**kwargs)
        if roc is not None and roc != 0.:
            self.c = 1 / roc if torch.is_tensor(roc) else \
                torch.tensor(np.asarray(1 / roc),
                             dtype=self.dtype).to(self.device)
        elif roc == 0.:
            self.c = torch.tensor(np.asarray(
                0.), dtype=self.dtype).to(self.device)

        if conic is not None:
            self.k = conic if torch.is_tensor(conic) else \
                torch.tensor(np.asarray(conic),
                             dtype=self.dtype).to(self.device)
        else:
            self.k = torch.tensor(np.asarray(
                0.), dtype=self.dtype).to(self.device)

        self.sag_min = self.sag_min_calc()

    def sag_min_calc(self, sampling=101):
        r_samp = torch.linspace(-self.r.item(), self.r.item(), sampling, dtype=self.dtype, device=self.device)
        return self.surface(x=torch.zeros_like(r_samp), y=r_samp).min()

    def ray_surface_intersection(self, ray, active=None):
        """
        solve the quadric equation to obtain the intersection of ray and spheric
        """
        if active is not None:
            rayo = ray.o[active].clone()
            rayd = ray.d[active].clone()
        else:
            rayo = ray.o.clone()
            rayd = ray.d.clone()
        if self.c == 0:  # degrade to plane propagation
            o_tmp = rayo
        else:
            if self.k is None:
                do = (rayo * rayd).sum(-1)
                dd = 1.
                oo = torch.square(rayo).sum(-1)
            else:
                k = torch.tensor([1., 1., 1 + self.k],
                                 dtype=self.dtype).to(self.device)
                do = (rayo * rayd * k).sum(-1)
                dd = (torch.square(rayd) * k).sum(-1)
                oo = (torch.square(rayo) * k).sum(-1)

            d = self.c * do - rayd[..., -1]
            e = self.c * dd
            f = self.c * oo - 2 * rayo[..., -1]
            g = torch.sqrt(torch.square(d) - e * f)
            t = -(d + g) / e
            o_tmp = rayo + t[..., None] * rayd

        valid_o = self.is_valid(o_tmp[..., 0:2])
        if active is not None:
            valid_out = active.clone()
            valid_out[active] &= valid_o
        else:
            valid_out = valid_o
        # o_tmp 仅是有效的光束
        return valid_out, o_tmp[valid_o]
    
    def _ray_surface_intersection(self, ray, active):
        """
        solve the quadric equation to obtain the intersection of ray and spheric
        active: valid map for rays bundle
        """

        rayo = ray.o[active].clone()
        rayd = ray.d[active].clone()

        if rayo[..., 2].any() != 0:
            t0 = (self.sag_min - rayo[..., 2]) / rayd[..., 2]
            rayo = rayo + t0[..., None] * rayd

        if self.c == 0:  # degrade to plane propagation
            o_tmp = rayo
            valid_s = active[active] if active is not None else torch.ones(ray.o.shape[0]).bool()
            loss_no_intersection = 0
        else:
            if self.k is None:
                do = (rayo * rayd).sum(-1)
                dd = 1.
                oo = torch.square(rayo).sum(-1)
            else:
                k = torch.tensor([1., 1., 1+self.k],
                                 dtype=self.dtype).to(self.device)
                do = (rayo * rayd * k).sum(-1)
                dd = (torch.square(rayd) * k).sum(-1)
                oo = (torch.square(rayo) * k).sum(-1)

            d = self.c * do - rayd[..., -1]
            e = self.c * dd
            f = self.c * oo - 2 * rayo[..., -1]
            valid_s = torch.square(d) - e * f > 0
            g = torch.sqrt(torch.square(d[valid_s]) - e[valid_s]*f[valid_s])
            t = -(d[valid_s] + g) / e[valid_s]
            o_tmp = rayo[valid_s] + t[..., None] * rayd[valid_s]
            loss_no_intersection = (e[~valid_s] * f[~valid_s] - torch.square(d)[~valid_s]).sum()

        valid_out = active.clone()
        valid_out[active] &= valid_s

        # if valid_s is not None and valid_s.sum() != len(valid_s):
        #     print("intersection loss")
        return valid_out, o_tmp, loss_no_intersection

    # ========================
    # paraxial
    # ========================
    def paraxial_matrix(self, wavelength):
        mp = super().paraxial_matrix(wavelength)  # matrix of plane propagate
        m = torch.eye(2, dtype=self.dtype, device=self.device)
        if 'mirror' in self.Material.material.keys():  # reflection
            if self.Material.material['mirror']:
                m[1, 0] = 2 * self.c
        else:
            # mu = self.refractive_index_prev(wavelength) / self.refractive_index(wavelength)
            # m[1, 0] = self.refractive_index_prev(wavelength)*self.c*(mu - 1)
            m[1, 0] = -(self.refractive_index(wavelength) -
                        self.refractive_index_prev(wavelength)) * self.c

        m = torch.matmul(m, mp)
        return m

    # ========================
    def surface(self, x, y):
        return self._g(x ** 2 + y ** 2)

    def dgd(self, x, y):
        """
        Derivatives of g: (g'x, g'y).
        """
        dsdr2 = 2 * self._dgd(x ** 2 + y ** 2)
        return dsdr2 * x, dsdr2 * y

    def dhd(self, z):
        """
        Derivatives of z.
        """
        return -torch.ones_like(z)

    # === Private methods
    def _g(self, r2):
        """
        return the surface value of this position.
        """
        # 返回了非球面参数
        tmp = r2 * self.c
        # zrr修改nan!!

        sq = torch.clamp((1 - (1 + self.k) * tmp * self.c), min=1e-8)
        total_surface = tmp / (1 + torch.sqrt(sq))
        return total_surface

    def _dgd(self, r2):
        """
        calcuelate the dirivation
        """
        alpha_r2 = (1 + self.k) * self.c ** 2 * r2
        # tmp = torch.sqrt(1 - alpha_r2)  # TODO: potential NaN grad
        # zrr nan!!!
        tmp = torch.sqrt(torch.clamp(1. - alpha_r2, min=1e-8))
        total_derivative = self.c * \
                           (1. + tmp - 0.5 * alpha_r2) / (tmp * (1. + tmp) ** 2)

        return total_derivative

    def _dgd2(self, r2):
        """
        calculate the dirivation of _dgd
        """
        alpha_r2 = (1 + self.k) * self.c ** 2 * r2
        tmp = torch.sqrt(torch.clamp(1. - alpha_r2, min=1e-8))

        A = self.c ** 3 * (1 + self.k)
        B = alpha_r2 ** 2 - 8 * alpha_r2 - 4 * alpha_r2 * tmp + 8 * tmp + 8
        C = 4 * (1 + tmp) ** 4 * tmp ** 3

        return A * B / C

    # def _dgd2(self, r2):
    #     """
    #     calcuelate the second dirivation
    #     """
    #     alpha_r2 = (1 + self.k) * self.c ** 2 * r2
    #     t = torch.sqrt(torch.clamp(1. - alpha_r2, min=1e-8))
    #     total_derivative = self.c ** 3 * (1 + self.k) / 4 \
    #         * (t**3 + 7 * t**2 + 6 * t + 2)/((t**2 + t)**3)

    #     return total_derivative


class Aspheric(Spheric):
    """
    Aspheric surface: https://en.wikipedia.org/wiki/Aspheric_lens.
    偶次非球面, 计算式基于r2
    """

    def __init__(self, ai=None, **kwargs):
        self.ai = torch.tensor(np.array(ai), dtype=kwargs['dtype']).to(kwargs['device']) if ai is not None else None
        super().__init__(**kwargs)
        self.sag_min = self.sag_min_calc()

    # === Common methods
    def surface(self, x, y):
        return self._g(x ** 2 + y ** 2)

    def h(self, z):
        return -z

    def reverse(self):
        self.c = -self.c
        if self.ai is not None:
            self.ai = -self.ai

    # === Private methods
    def _g(self, r2):
        """
        return the surface value of this position.
        """
        total_surface = super()._g(r2)
        higher_surface = 0
        # 从r4开始的
        if self.ai is not None:
            for i in np.flip(range(len(self.ai))):
                higher_surface = r2 * higher_surface + self.ai[i]
            higher_surface = higher_surface * r2 ** 2
        return total_surface + higher_surface

    def _dgd(self, r2):
        """
        calculate the dierivation of this position
        """
        total_derivative = super()._dgd(r2)

        higher_derivative = 0
        if self.ai is not None:
            for i in np.flip(range(len(self.ai))):
                higher_derivative = r2 * \
                                    higher_derivative + (i + 2) * self.ai[i]
        return total_derivative + higher_derivative * r2

    def _dgd2(self, r2):

        total_dgd2 = super()._dgd2(r2)

        higher_d2dr = 0
        if self.ai is not None:
            for i in np.flip(range(len(self.ai))):
                higher_d2dr = r2 * higher_d2dr + (i + 2) * (i + 1) * self.ai[i]

        return total_dgd2 + higher_d2dr

    # ==================================================================================

    def ray_surface_intersection(self, ray, active=None):
        """
        Get the intersections of ray and surface
        Returns:
        - p: intersection point
        - g: explicit funciton
        """
        if active is not None:
            valid_s, local, loss_no_intersection = self.newtons_method(ray.maxt, ray.o[active], ray.d[active])
            valid_out = active.clone()
            valid_out[active] &= valid_s
        else:
            valid_s, local, loss_no_intersection = self.newtons_method(ray.maxt, ray.o, ray.d)
            valid_out = valid_s

        return valid_out, local

    def newtons_method(self, maxt, o, D, option='implicit'):
        # Newton's method to find the root of the ray-surface intersection point.
        # Two modes are supported here:
        #
        # 1. 'explicit": This implements the loop using autodiff, and gradients will be
        # accurate for o, D, and self.parameters. Slow and memory-consuming.
        #
        # 2. 'implicit": This implements the loop using implicit-layer theory, find the
        # solution without autodiff, then hook up the gradient. Less memory-consuming.
        # pre-compute constants
        ox, oy, oz = (o[..., i].clone() for i in range(3))
        dx, dy, dz = (D[..., i].clone() for i in range(3))

        # initial guess of t
        # t0 = (self.d - oz) / dz
        if option == 'explicit':
            t, t_delta, valid = self.newtons_method_impl(
                maxt, dx, dy, dz, ox, oy, oz
            )
        elif option == 'implicit':
            with torch.no_grad():
                t_last, t_delta, valid = self.newtons_method_impl(
                    maxt, dx, dy, dz, ox, oy, oz
                )
                s_derivatives_dot_D = self.surface_and_derivatives_dot_D(
                    t_last, dx, dy, dz, ox, oy, t_last * dz
                )[1]
            t = t_last[valid]
            t = t - (self.surface(ox[valid] + t * dx[valid], oy[valid] + t * dy[valid]) + self.h(oz[valid] + t * dz[valid])) / s_derivatives_dot_D[valid]
        else:
            raise Exception('option={} is not available!'.format(option))
        p = o[valid] + t.unsqueeze(-1) * D[valid]
        _local_x, _local_y = ox[~valid] + t_last[~valid] * dx[~valid], oy[~valid] + t_last[~valid] * dy[~valid]
        loss_no_intersection = ((1 + self.k) * self.c ** 2 * (_local_x ** 2 + _local_y ** 2) - 1).sum()
        # 返回的是有效光束中的有交点部分
        return valid, p, loss_no_intersection

    def newtons_method_impl(self, maxt, dx, dy, dz, ox, oy, oz):
        t_delta = torch.zeros_like(oz, dtype=self.dtype, device=self.device)
        t_last = t_delta
        residual = maxt * torch.ones_like(oz, dtype=self.dtype, device=self.device)
        it = 0
        while (torch.abs(residual) > self.NEWTONS_TOLERANCE_TIGHT).any() and (it < self.NEWTONS_MAXITER):
            it += 1
            t_last = t_delta
            residual, s_derivatives_dot_D = self.surface_and_derivatives_dot_D(
                t_last, dx, dy, dz, ox, oy, t_last * dz
            )
            # t_delta = torch.where(torch.isnan(residual), t_last, t_last - residual / s_derivatives_dot_D)
            t_delta = t_last - residual / s_derivatives_dot_D
            residual[residual.isnan()] = 0
        valid = (torch.abs(residual) < self.NEWTONS_TOLERANCE_LOOSE) & (t_delta <= maxt)
        return t_last, t_delta, valid

    def _ray_surface_intersection(self, ray, active):
        """
        Get the intersections of ray and surface
        Returns:
        - p: intersection point
        - g: explicit funciton
        """
        valid_s, local, loss_no_intersection = self.newtons_method(ray.maxt, ray.o[active], ray.d[active])
        valid_out = active.clone()
        valid_out[active] &= valid_s
        # if len(valid_s) != valid_s.sum():
        #     print("intersection loss")

        return valid_out, local, loss_no_intersection

    # =========================
    # =========================
    def paraxial_matrix(self, wavelength):
        mp = torch.eye(2, dtype=self.dtype, device=self.device)
        # here distance is the propagate distance
        mp[0, 1] = self.d_prev / self.refractive_index_prev(wavelength)

        if self.ai is not None:
            c = self.c + 2 * self.ai[0]

        m = torch.eye(2, dtype=self.dtype, device=self.device)
        if 'mirror' in self.Material.material.keys():  # reflection
            if self.Material.material['mirror']:
                m[1, 0] = 2 * self.c
        else:
            # mu = self.refractive_index_prev(wavelength) / self.refractive_index(wavelength)
            # m[1, 0] = self.refractive_index_prev(wavelength)*self.c*(mu - 1)
            m[1, 0] = -(self.refractive_index(wavelength) - self.refractive_index_prev(wavelength)) * self.c

        m = torch.matmul(m, mp)
        return m

    def _dgd2(self, r2):

        total_d2dr = super()._dgd2(r2)
        higher_d2dr = 0
        if self.ai is not None:
            for i in np.flip(range(len(self.ai))):
                higher_d2dr = r2 * higher_d2dr + (i + 2) * (i + 1) * self.ai[i]

        return total_d2dr + higher_d2dr


class ExtendAspheric(Aspheric):
    """原本是为了写拓展非球面的,后面发现其公式就是包含10阶后的，因此这里是错误的"""

    def __int__(self, r, d, c=0., k=0., ai=None, is_square=False, device=torch.device('cpu')):
        Surface.__init__(self, r, d, is_square, device)
        self.c, self.k = (torch.Tensor(np.array(v)) for v in [c, k])
        self.ai = None
        if ai is not None:
            self.ai = torch.Tensor(np.array(a) / (r ** (2 * i + 2))
                                   for i, a in enumerate(ai))


class Doe(PrettyPrinter):
    """用于计算位相表达式以及加工出来的local opd，衍射效率暂时对比理想情况即全波段均为100%"""

    def __init__(self, wavelength=500e-6, m=0, di=None, device=torch.device('cpu')):
        if torch.is_tensor(wavelength):
            self.wavelength = wavelength
        else:
            self.wavelength = torch.tensor(wavelength)
        if torch.is_tensor(m):
            self.m = m
        else:
            self.m = torch.tensor(m)
        if di is not None:
            self.di = torch.Tensor(np.array(di)).to(device)
        else:
            self.di = None
        self.device = device

    def phase(self, x, y):
        r2 = x ** 2 + y ** 2
        tmp = 0.
        if self.di is not None:
            for i in np.flip(range(len(self.di))):
                tmp = r2 * tmp + self.di[i]
            tmp = tmp * r2
        return tmp

    def grating_vector(self, x, y):
        if self.di is None:
            return torch.tensor([0, 0, 0]).to(self.device)
        r2 = x ** 2 + y ** 2
        derivative = 0
        for i in np.flip(range(len(self.di))):
            derivative = r2 * derivative + (i + 1) * self.di[i]
        dx = derivative * 2 * x
        dy = derivative * 2 * y
        return torch.stack((dx, dy, torch.zeros_like(dx)), dim=-1)

    def local_opd(self, x, y):
        """理论上应当再乘上(n'-1)/(n0-1),但是基本不变"""
        if self.m == 0:
            return torch.tensor(0).to(self.device)
        else:
            return self.phase(x, y) % (self.wavelength * self.m)
        # return self.phase(x, y) % (self.wavelength * self.m) * 1.073812966
        # 基本不变的原因在于每个子束之间的相位差乘上很小的系数，但是量化周期变了，每个子束迹点处的相位有显著区别
