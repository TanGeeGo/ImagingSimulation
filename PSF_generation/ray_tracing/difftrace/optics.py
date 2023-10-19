import os
import json
import torch
import torch.nn.functional as F
from itertools import groupby

from .surfaces import *
from .conjugates import *
from .utils import lambda_d, lambda_F, lambda_C  # 587nm, 486nm, 656nm


class System(list):
    """
    The Lensgroup (consisted of multiple optical surfaces) is mounted on a rod, whose
    origin is `origin`. The Lensgroup has full degree-of-freedom to rotate around the
    x/y axes, with the rotation angles defined as `theta_x`, `theta_y`, and `theta_z` (in degree).

    In Lensgroup's coordinate (i.e. object frame coordinate), surfaces are allocated
    starting from `z = 0`. There is an additional, comparatively small 3D origin shift
    (`shift`) between the surface center (0,0,0) and the origin of the mount, i.e.
    shift + origin = lensgroup_origin.

    In forward mode,rays start from `d = 0` surface and propagate along the +z axis;
    todo self.r_last self.d_sensor 原本在load——file中读出 一般没什么用 (用于draw)
    """

    def __init__(self,
                 config_file,
                 dtype=torch.float64,
                 device=torch.device('cpu')
                 ):
        self.dtype = dtype
        self.device = device
        self.config = config_file
        if os.path.splitext(config_file)[-1] == '.txt':
            self.load_file_from_txt(config_file)
        elif os.path.splitext(config_file)[-1] == '.json':
            self.load_file_from_json(config_file)
        super().__init__(self.surfaces)

        self.APERTURE_TOLERENCE = 1e-5  # 1e-5 mm
        self.APERTURE_OPT_STEP = 5e-2
        self.APERTURE_ITER_MAX = 1000

    # ====================================================================================
    # Initialization
    # ====================================================================================

    def load_file_from_txt(self, file_path):
        self.surfaces = []
        ds = []
        with open(file_path, encoding="utf-8") as file:
            line_no = 0
            d_global = 0.
            for line in file:
                if line_no == 0:  # first two lines are comments
                    self.LensName = str(line)
                    line_no += 1
                    continue
                elif line_no == 1:  # first two lines are comments
                    line_no += 1
                    continue
                else:
                    ls = line.split()
                    surface_type, roc, distance, material, radius, conic = \
                        ls[0], np.float64(ls[1]), np.float64(ls[2]), ls[3], \
                            np.float64(ls[4]), np.float64(ls[5])
                    ds.append(distance)
                    if surface_type == 'O':  # object plane
                        self.surfaces.append(InfiniteConjugate())
                        material_prev = material
                    elif surface_type == 'S' or surface_type == 'A':
                        # aperture, spherical and aspherical surface
                        if len(ls) < 7:  # spherical (standard)
                            self.surfaces.append(Spheric(index=line_no-1,
                                                         roc=roc, 
                                                         distance=d_global, 
                                                         distance_prev=d_prev,
                                                         distance_after= distance,
                                                         material=material, 
                                                         material_prev=material_prev, 
                                                         radius=radius, 
                                                         conic=conic, 
                                                         dtype=self.dtype, 
                                                         device=self.device))
                        else:  # aspherical
                            ai = []
                            for ac in range(6, len(ls)):
                                ai.append(np.float64(ls[ac]))
                            self.surfaces.append(
                                Aspheric(index=line_no-1,
                                         roc=roc, 
                                         distance=d_global, 
                                         distance_prev=d_prev,
                                         distance_after= distance,
                                         material=material, 
                                         material_prev=material_prev, 
                                         radius=radius, 
                                         conic=conic, 
                                         ai=ai,
                                         dtype=self.dtype, 
                                         device=self.device
                                         ))

                        if surface_type == 'A':
                            # with the object plane
                            self.aperture_ind = len(self.surfaces)
                            self.aperture = self.surfaces[-1]
                    elif surface_type == 'I':
                        self.surfaces.append(Spheric(index=line_no-1,
                                                     roc=roc, 
                                                     distance=d_global, 
                                                     distance_prev=d_prev,
                                                     distance_after= distance,
                                                     material=material, 
                                                     material_prev=material_prev, 
                                                     radius=radius, 
                                                     conic=conic, 
                                                     dtype=self.dtype, 
                                                     device=self.device))

                    line_no += 1

                material_prev = material
                d_prev = distance
                d_global += distance
        
        file.close()
        self.d_sensor = torch.tensor(ds[-2], dtype=self.dtype)  # distance from last surface to sensor plane

    def load_file_from_json(self, file_path):
        self.surfaces = []
        d_global = 0.
        with open(file_path) as file:
            lens_dict = json.load(file)
        file.close()

        for item in lens_dict:
            # split the string and the numbers
            itemize = [''.join(list(g)) for k, g in groupby(item, key=lambda x: x.isdigit())]
            
            if itemize[0] == 'Description':
                self.LensName = lens_dict['Description']
                continue
            elif itemize[0] == 'OBJECT':
                self.surfaces.append(InfiniteConjugate())
                material_prev = lens_dict[item]['material']
            elif itemize[0] == 'Standard' or itemize[0] == 'STOP':
                if lens_dict[item]['ai-terms'] is not None:
                    self.surfaces.append(Aspheric(index=lens_dict[item]['index'],
                                                  roc=lens_dict[item]['roc'], 
                                                  distance=d_global, 
                                                  distance_prev=d_prev,
                                                  distance_after= lens_dict[item]['distance'],
                                                  material=lens_dict[item]['material'], 
                                                  material_prev=material_prev, 
                                                  radius=lens_dict[item]['radius'], 
                                                  conic=lens_dict[item]['conic'], 
                                                  ai=lens_dict[item]['ai-terms'],
                                                  shift=lens_dict[item]['shift'],
                                                  dtype=self.dtype, 
                                                  device=self.device
                                                  ))
                else:
                    self.surfaces.append(Spheric(index=lens_dict[item]['index'],
                                                 roc=lens_dict[item]['roc'], 
                                                 distance=d_global, 
                                                 distance_prev=d_prev,
                                                 distance_after= lens_dict[item]['distance'],
                                                 material=lens_dict[item]['material'], 
                                                 material_prev=material_prev, 
                                                 radius=lens_dict[item]['radius'], 
                                                 conic=lens_dict[item]['conic'], 
                                                 dtype=self.dtype, 
                                                 device=self.device
                                                 ))
            
                if itemize[0] == 'STOP':
                    self.aperture_ind = len(self.surfaces)
                    self.aperture = self.surfaces[-1]

            elif itemize[0] == 'IMAGE':
                self.surfaces.append(Spheric(index=lens_dict[item]['index'],
                                             roc=lens_dict[item]['roc'], 
                                             distance=d_global, 
                                             distance_prev=d_prev,
                                             distance_after= lens_dict[item]['distance'],
                                             material=lens_dict[item]['material'], 
                                             material_prev=material_prev, 
                                             radius=lens_dict[item]['radius'], 
                                             conic=lens_dict[item]['conic'], 
                                             dtype=self.dtype, 
                                             device=self.device
                                             ))

            material_prev = lens_dict[item]['material']
            d_prev = lens_dict[item]['distance']
            d_global += lens_dict[item]['distance']

    def load_doe(self, doe):
        # 弃用，init后外部声明即可
        self.doe = doe

    def set_aperture_index(self, index):
        # init后外部声明即可
        self.aperture_ind = index

    def _sync(self):
        for i in range(len(self.surfaces)):
            self.surfaces[i].to(self.device)
        self.aperture_ind = self._find_aperture()

    # ====================================================================================
    # Rays aiming
    # ====================================================================================
    def _aim_chief(self, wavelength=lambda_d):
        """
        Aiming the position of the chief ray 
        to make the ray of this field could pass through the center of stop
        return the position of chief ray on the first surface's reference plane
        """
        # generate the initial chief ray
        x_init = torch.tensor(0., requires_grad=True)
        if self[0]._type == 'infinite':
            # infinite case: fixed angle
            o = torch.stack([x_init, torch.tensor(0.), torch.tensor(
                0.)], dim=0).unsqueeze(0).unsqueeze(0)
            d = self[0].map().unsqueeze(0).unsqueeze(0)
        elif self[0]._type == 'finite':
            # finite case: fixed object position
            o = torch.stack([x_init, torch.tensor(0.), torch.tensor(
                0.)], dim=0).unsqueeze(0).unsqueeze(0)
            d = normalize(torch.Tensor([0., 0., 0.]) -
                          self[0].o).unsqueeze(0).unsqueeze(0)

        # form the ray
        ray_init = Ray(o, d, wavelength=wavelength)
        # optimize the ray's path to let it pass through the center of stop
        it = 1
        momentum = torch.zeros_like(x_init)  # for Adam optimizer
        velocity = torch.zeros_like(x_init)  # for Adam optimizer

        # calcuate the initial residual
        ray_stop = self.propagate(ray_init, start=1, stop=self.aperture_ind)[1]
        residual = length(ray_stop.o[..., 0:2])

        while (torch.abs(residual) > self.APERTURE_TOLERENCE) and (it < self.APERTURE_ITER_MAX):
            residual.backward()
            # ==============================================
            # update initial position of chief ray with Adam
            lr = self.APERTURE_OPT_STEP * \
                (1 - 0.999 ** it) ** 0.5 / (1 - 0.9 ** it)
            momentum = 0.9 * momentum + (1-0.9)*x_init.grad.data
            velocity = 0.999 * velocity + (1-0.999)*(x_init.grad.data**2)
            x_init.data = x_init.data - lr * \
                momentum / (velocity ** 0.5 + 1e-8)
            # ==============================================
            x_init.grad.data.zero_()
            # form the ray
            o = torch.stack([x_init, torch.tensor(0.), torch.tensor(
                0.)], dim=0).unsqueeze(0).unsqueeze(0)
            ray_init = Ray(o, d, wavelength=wavelength)

            ray_stop = self.propagate(
                ray_init, start=1, stop=self.aperture_ind)[1]
            residual = length(ray_stop.o[..., 0:2])
            # print('residual:', residual.item(), 'x_init:', x_init.item())
            it += 1
        return o.detach()  # chief ray o on the first surface's reference plane

    def _aim_marginal(self, chief_ray_o, wavelength=lambda_d):
        """
        Aiming the position of the marginal ray 
        to make the ray of this field could pass through the margin of stop
        return the position of marginal ray on the first surface's reference plane
        Two directions are optimized: Top, Bot, Left, Right, margin of the aperture
        """
        marginal_pos = torch.zeros(2, 1, 3)
        for dim in range(2):
            # generate the initial chief ray
            x_init = torch.tensor(chief_ray_o[..., 0].item(), requires_grad=True) if dim == 0 else \
                torch.tensor(0., requires_grad=True)
            o = torch.stack([x_init, torch.tensor(0.), torch.tensor(0.)], dim=0).unsqueeze(0).unsqueeze(0) if dim == 0 else \
                torch.stack([torch.tensor(chief_ray_o[..., 0].item()), x_init, torch.tensor(
                    0.)], dim=0).unsqueeze(0).unsqueeze(0)
            if self[0]._type == 'infinite':
                # infinite case: fixed angle
                d = self[0].map().unsqueeze(0).unsqueeze(0)
            elif self[0]._type == 'finite':
                # finite case: fixed object position
                d = normalize(torch.Tensor(
                    [0., 0., 0.]) - self[0].o).unsqueeze(0).unsqueeze(0)

            # form the ray
            ray_init = Ray(o, d, wavelength=wavelength)
            # optimize the ray's path to let it pass through the margin of stop
            it = 1
            momentum = torch.zeros_like(x_init)  # for Adam optimizer
            velocity = torch.zeros_like(x_init)  # for Adam optimizer

            # calcuate the initial residual
            ray_stop = self.propagate(
                ray_init, start=1, stop=self.aperture_ind)[1]
            residual = F.l1_loss(
                ray_stop.o[..., dim], self[self.aperture_ind-1].r)

            while (torch.abs(residual) > self.APERTURE_TOLERENCE) and (it < self.APERTURE_ITER_MAX):
                residual.backward()
                # ==============================================
                # update initial position of chief ray with Adam
                lr = self.APERTURE_OPT_STEP * \
                    (1 - 0.999 ** it) ** 0.5 / (1 - 0.9 ** it)
                momentum = 0.9 * momentum + (1-0.9)*x_init.grad.data
                velocity = 0.999 * velocity + (1-0.999)*(x_init.grad.data**2)
                x_init.data = x_init.data - lr * \
                    momentum / (velocity ** 0.5 + 1e-8)
                # ==============================================
                x_init.grad.data.zero_()
                # form the ray
                o = torch.stack([x_init, torch.tensor(0.), torch.tensor(0.)], dim=0).unsqueeze(0).unsqueeze(0) if dim == 0 else \
                    torch.stack([torch.tensor(chief_ray_o[..., 0].item()), x_init, torch.tensor(
                        0.)], dim=0).unsqueeze(0).unsqueeze(0)
                ray_init = Ray(o, d, wavelength=wavelength)
                ray_stop = self.propagate(
                    ray_init, start=1, stop=self.aperture_ind)[1]
                residual = F.l1_loss(
                    ray_stop.o[..., dim], self[self.aperture_ind-1].r)
                it += 1
            # store the marginal positions of the initial
            marginal_pos[dim, ...] = o.detach()

        return marginal_pos

    def _aim_ray(self, Px, Py, view, wavelength=lambda_d, paraxial=False):
        """
        Aiming the position of the normalized position [Px, Py] on pupil plane
        using [geometric] propagate or [paraxial] propagate
        return the cls Ray on the first surface's reference plane
        """
        # update the field-of-view of system
        self[0].update(view)
        # generate the initial ray
        x_init = torch.tensor(0., dtype=self.dtype, requires_grad=True)
        y_init = torch.tensor(0., dtype=self.dtype, requires_grad=True)
        if self[0]._type == 'infinite':
            # infinite case: fixed angle
            o = torch.stack([x_init, y_init, torch.tensor(0., dtype=self.dtype)], dim=0).unsqueeze(0)
            d = self[0].map().unsqueeze(0)
        elif self[0]._type == 'finite':
            # finite case: fixed object position
            o = torch.stack([x_init, y_init, torch.tensor(0., dtype=self.dtype)], dim=0).unsqueeze(0)
            d = normalize(torch.tensor([0., 0., 0.], dtype=self.dtype) - self[0].o).unsqueeze(0)

        # form the ray
        ray_init = Ray(o, d, wavelength=wavelength, device=self.device)
        # optimize the ray's path to let it pass through the center of stop
        it = 1
        momentum_x = torch.zeros_like(x_init)  # for Adam optimizer
        velocity_x = torch.zeros_like(x_init)  # for Adam optimizer
        momentum_y = torch.zeros_like(y_init)  # for Adam optimizer
        velocity_y = torch.zeros_like(y_init)  # for Adam optimizer

        # calcuate the initial residual
        if paraxial:
            ray_stop = self.propagate_paraxial(ray_init, start=1, stop=self.aperture_ind)[1]
        else:
            ray_stop = self.propagate(ray_init, start=1, stop=self.aperture_ind)[1]

        pupil_aim_pos = torch.stack((Px*self[self.aperture_ind-1].r,
                                     Py*self[self.aperture_ind-1].r)).unsqueeze(0) # TODO: recalc the entrance pupil
        residual = F.l1_loss(ray_stop.o[..., 0:2], pupil_aim_pos)

        while (torch.abs(residual) > 2*self.APERTURE_TOLERENCE) and (it < self.APERTURE_ITER_MAX):
            residual.backward()
            # ==============================================
            # update initial position of chief ray with Adam
            lr = self.APERTURE_OPT_STEP * \
                (1 - 0.999 ** it) ** 0.5 / (1 - 0.9 ** it)
            momentum_x = 0.9 * momentum_x + (1-0.9)*x_init.grad.data
            velocity_x = 0.999 * velocity_x + (1-0.999)*(x_init.grad.data**2)
            momentum_y = 0.9 * momentum_y + (1-0.9)*y_init.grad.data
            velocity_y = 0.999 * velocity_y + (1-0.999)*(y_init.grad.data**2)
            x_init.data = x_init.data - lr * \
                momentum_x / (velocity_x ** 0.5 + 1e-8)
            y_init.data = y_init.data - lr * \
                momentum_y / (velocity_y ** 0.5 + 1e-8)
            # ==============================================
            x_init.grad.data.zero_()
            y_init.grad.data.zero_()
            # form the ray
            o = torch.stack([x_init, y_init, torch.tensor(0., dtype=self.dtype)], dim=0).unsqueeze(0)
            ray_init = Ray(o, d, wavelength=wavelength, device=self.device)

            if paraxial:
                ray_stop = self.propagate_paraxial(ray_init, start=1, stop=self.aperture_ind)[1]
            else:
                ray_stop = self.propagate(ray_init, start=1, stop=self.aperture_ind)[1]
            residual = F.l1_loss(ray_stop.o[..., 0:2], pupil_aim_pos)
            # print('residual:', residual.item(), 'x_init:', x_init.item())
            it += 1

        # form the ray not required grad
        # chief ray o on the first surface's reference plane
        return Ray(o.detach(), d, wavelength=wavelength, device=self.device)

    def rays_aiming(self, chief_pos, marginal_pos, wavelength=lambda_d, rays_h=101, rays_w=101, sampling_distribution="hexapolar"):
        """
        Aiming the ray to fill the pupil
        Return the Ray sampled on the first surface of system
        """
        # generate the normalized entrace pupil sampling
        o_p = pupil_distribution(
            rays_h, rays_w, sampling_distribution)  # o_p is [h, w, 2]
        # sample the real coordinates according to the chief and marginal position
        o = torch.stack([o_p[..., 0] * torch.abs((marginal_pos[0, 0, 0] - marginal_pos[1, 0, 0])) + chief_pos[..., 0],
                         o_p[..., 1] * torch.abs((marginal_pos[0, 0, 1] -
                                                 marginal_pos[1, 0, 1])) + chief_pos[..., 1],
                         torch.zeros_like(o_p[..., 0])], dim=2)  # o is [h, w, 3]
        # judge the d of this field
        if self[0]._type == 'infinite':
            # infinite case: fixed angle
            d = self[0].map().unsqueeze(0).unsqueeze(0)
        elif self[0]._type == 'finite':
            # finite case: fixed object position
            d = normalize(torch.tensor([0., 0., 0.], dtype=self.dtype) -
                          self[0].o).unsqueeze(0).unsqueeze(0)
        d = d.repeat(o.shape[0], o.shape[1], 1)  # repeat to the same size as o
        return Ray(o, d, wavelength=wavelength)

    def _aim_ray_enumerate(self, Px, Py, view, wavelength=lambda_d, paraxial=False):
        # update the field-of-view of system
        self[0].update(view)
        # calculate the sample radius on the first surface
        R = np.tan(self[0].angle_fov) * self[1].surface(self[1].r, 0.) + self[1].r

        APERTURE_SAMPLING = 501 # enumerate sampling density
        x_e, y_e = torch.meshgrid(torch.linspace(-R, R, APERTURE_SAMPLING, dtype=self.dtype),
                                  torch.linspace(-R, R, APERTURE_SAMPLING, dtype=self.dtype),
                                  indexing='ij')
        x, y = x_e.reshape(APERTURE_SAMPLING**2), y_e.reshape(APERTURE_SAMPLING**2)

        pupil_aim_pos = torch.stack((Px*self[self.aperture_ind-1].r,
                                     Py*self[self.aperture_ind-1].r)).unsqueeze(0)
        # generate rays, flatten the hw dimension , and find valid map
        o = torch.stack((x, y, torch.zeros_like(x)), axis=1)
        d = self[0].map().unsqueeze(0).repeat(APERTURE_SAMPLING**2, 1)
        ray = Ray(o, d, wavelength=wavelength)
        # check the validation of the ray
        if paraxial:
            valid, ray_stop = self.propagate_paraxial(ray, start=1, stop=self.aperture_ind)
            valid, _ = self.propagate_paraxial(ray_stop.clone(), start=self.aperture_ind, 
                                                       stop=None, valid=valid_stop)
        else:
            valid_stop, ray_stop = self.propagate(ray, start=1, stop=self.aperture_ind)
            valid, _ = self.propagate(ray_stop.clone(), start=self.aperture_ind, 
                                              stop=None, valid=valid_stop)

        # find the valid position nearest to the Position [Px, Py]
        o_valid = o[valid]
        residual = length(ray_stop.o[valid][..., 0:2] - pupil_aim_pos)
        o_pos = torch.squeeze(o_valid[residual==residual.min()])
        
        # form the one ray
        o = o_pos.unsqueeze(0)
        d = self[0].map().unsqueeze(0)
        return Ray(o, d, wavelength=wavelength)

    # ====================================================================================
    # Propagation
    # ====================================================================================

    def propagate(self, ray, start=1, stop=None, record=False, valid=None):
        """
        oss: points
        dss: directions
        tss: lenghts
        shape: rays,surfaces,data
        """
        dim = ray.o[..., 2].shape
        if valid is None:
            valid = torch.ones(dim, device=ray.device).bool()

        oss = ray.o.unsqueeze(-2)
        dss = ray.d.unsqueeze(-2)
        tss = ray.t.unsqueeze(-2)
        valids = valid.unsqueeze(-1)
        for s in self[start:stop]:
            valid, ray = s.propagate(ray, valid)
            oss = torch.cat((oss, ray.o.unsqueeze(-2)), -2)
            dss = torch.cat((dss, ray.d.unsqueeze(-2)), -2)
            tss = torch.cat((tss, ray.t.unsqueeze(-2)), -2)
            valids = torch.cat((valids, valid.unsqueeze(-1)), -1)

        if record:
            return oss, dss, tss, valids
        else:
            return valid, ray

    def propagate_paraxial(self, ray, start=1, stop=None, record=False):
        """
        paraxiaracing with parl taxial matrix
        """
        valid = None
        dim = ray.o[..., 2].shape

        if record:
            oss = []
            dss = []
            for i in range(dim[0]):
                oss.append([ray.o[i, :].cpu().detach().numpy()])
                dss.append([ray.d[i, :].cpu().detach().numpy()])

        for s in self[start:stop]:
            valid, ray = s.propagate_paraxial(ray)
            
            if record:
                if dim[0] == 1:
                    # if valid.cpu().detach().numpy():
                    oss[0].append(ray.o[0, :].cpu().detach().numpy())
                    dss[0].append(normalize(ray.d[0, :]).cpu().detach().numpy())
                else:
                    for os, ds, v, op, dp in zip(oss, dss,
                                                 valid.cpu().detach().numpy(),
                                                 ray.o.cpu().detach().numpy(),
                                                 ray.d.cpu().detach().numpy()):
                        # if v.any():
                        os.append(op)
                        ds.append(normalize(dp))

        if record:
            return valid, ray, oss, dss
        else:
            return valid, ray

    def propagate_gausslets(self, gausslets, start=1, stop=None, record=False):
        valid = None
        if record:
            oss = []
            dss = []
            tss = []
            # arranged as [surface, h, w, xyz]
            oss.append(gausslets.base_ray.o)
            dss.append(gausslets.base_ray.d)
            tss.append(gausslets.base_ray.t)

        for s in self[start:stop]:

            valid_base, gausslets.base_ray = s.propagate(gausslets.base_ray, valid)
            # valid_wx, gausslets.waist_rayx = s.propagate(gausslets.waist_rayx, valid)
            # valid_wy, gausslets.waist_rayy = s.propagate(gausslets.waist_rayy, valid)
            # valid_dx, gausslets.div_rayx = s.propagate(gausslets.div_rayx, valid)
            # valid_dy, gausslets.div_rayy = s.propagate(gausslets.div_rayy, valid)
            gausslets.opd += gausslets.base_ray.t
            # valid = valid_base & valid_dy & valid_dx & valid_wy & valid_wx
            valid = valid_base
            if record:
                oss.append(gausslets.base_ray.o)
                dss.append(gausslets.base_ray.d)
                tss.append(gausslets.base_ray.t)

        if record:
            return valid, gausslets, oss, dss, tss
        else:
            return valid, gausslets
    # ====================================================================================
    # Paraxial Analysis
    # ====================================================================================

    def _paraxial_info(self, view_list):
        """
        locate the first paraxial and the second paraxial rays
        tracing the rays with paraxial propagate
        store the tracing for system data calculation
        """
        # first paraxial ray pass through the border of the pupil with 0 degree rays
        first_paraxial_ray = self._aim_ray(Px=0., Py=1., view=view_list[0], 
                                                     wavelength=lambda_d, paraxial=True)
        # second paraxial ray pass through the border of the pupil with the max field-of-view
        second_paraxial_ray = self._aim_ray(Px=0., Py=0., view=view_list[-1], 
                                                      wavelength=lambda_d, paraxial=True) 
        # tracing the two rays
        _, _, oss_fir, dss_fir = self.propagate_paraxial(first_paraxial_ray, record=True)
        _, _, oss_scd, dss_scd = self.propagate_paraxial(second_paraxial_ray, record=True)
        self.oss_fir = np.asarray(oss_fir[0]) # arranged as [surfaces, xyz]
        self.dss_fir = np.asarray(dss_fir[0]) # arranged as [surfaces, xyz]
        self.oss_scd = np.asarray(oss_scd[0]) # arranged as [surfaces, xyz]
        self.dss_scd = np.asarray(dss_scd[0]) # arranged as [surfaces, xyz]
        

    @property
    def Effective_Focal_Length(self):
        return self.oss_fir[1][1] / np.tan(np.acrsin(self.dss_fir[-2][1]))
    
    @property
    def Total_Track(self):
        return self.oss_fir[-1][2]
    
    @property
    def Entrance_Pupil_Position(self):
        # position related to the first plane of system
        return - self.oss_scd[1][1] / np.tan(np.arcsin(self.dss_scd[0][1]))
    
    @property
    def Exit_Pupil_Position(self):
        # position related to the image plane of system
        return - self.oss_scd[-2][1] / np.tan(np.arcsin(self.dss_scd[-2][1])) - self[-1].d

    @property
    def Exit_Pupil_Diameter(self):
        return 2 * (- self.oss_scd[-2][1] / np.tan(np.arcsin(self.dss_scd[-2][1])) - self.d_sensor) * np.tan(np.arcsin(self.dss_fir[-2][1]))
        
    # ====================================================================================
    # Geometric validation
    # ====================================================================================
    def geometric_val(self):
        """
        traverse the surface and check their validation in geometric
        """
        VALIDATION_SAMPLEING = 101
        valid = True
        for s_idx in range(len(self) - 1):
            if s_idx == 0:
                continue # object plane, jump out the iteration

            if s_idx == self.aperture_ind-1:
                # if the surface is a plane, do not check its validation
                if (self[s_idx].c is None) or (self[s_idx].c == 0.):
                    continue # jump to the next surface
                
            # compare the radius of this surface and the next surface for radius checking
            R = torch.min(torch.tensor([self[s_idx].r, self[s_idx+1].r], 
                                       dtype=self.dtype, device=self.device))
            r_samp = torch.linspace(-R, R, VALIDATION_SAMPLEING,
                                    dtype=self.dtype, device=self.device)
            # global z-coordinates of the surface
            surf_val = self[s_idx].d + self[s_idx].surface(x=torch.zeros_like(r_samp), y=r_samp)
            surf_next_val = self[s_idx+1].d + self[s_idx+1].surface(x=torch.zeros_like(r_samp), y=r_samp)
            surf_diff = surf_next_val - surf_val
            
            if (surf_diff < 0.).any():
                valid = False

            # check the validation of surface
            if not valid:
                break

        return valid
    
    @torch.no_grad()
    def update_image(self, view, wavelengths):
        """
        Update the radius of the image plane 
        Ensuring the rays of all wavelengths in the largest fov could pass
        """
        # update the field-of-view of system
        self[0].update(view)
        R = torch.tan(self.system[0].angle_fov) * \
            self.system[1].surface(self.system[1].r, 0.) + \
            self.system[1].r

        APERTURE_SAMPLING = 201

        x_e, y_e = torch.meshgrid(torch.linspace(-R, R, APERTURE_SAMPLING, dtype=self.dtype),
                                  torch.linspace(-R, R, APERTURE_SAMPLING, dtype=self.dtype),
                                  indexing='ij')
        # generate rays and find valid map
        o = torch.stack((x_e, y_e, torch.zeros_like(x_e)), dim=2)
        d = self[0].map().unsqueeze(0).unsqueeze(0).\
            repeat(APERTURE_SAMPLING, APERTURE_SAMPLING, 1)
        r_max = []
        for wavelength in wavelengths:
            ray = Ray(o, d, wavelength=wavelength)
            # propagate the ray to the last surface before the image plane
            valid, ray_after = self.propagate(ray, stop=-1)

            # free propagate with the valid rays
            o_after_valid = ray_after.o[valid]
            d_after_valid = ray_after.d[valid]
            t_after_valid = (self[-1].d - o_after_valid[..., 2]) / d_after_valid[..., 2]
            o_image_valid = o_after_valid + d_after_valid * t_after_valid[..., None]
            r_image_valid = torch.sqrt(torch.sum(torch.square(o_image_valid[..., 0:2]), dim=1))
            r_max.append(torch.max(r_image_valid))

        # set the max radius as the image plane
        setattr(self[-1], 'r', max(r_max))

    # ------------------------------------------------------------------------------------

    # ------------------------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------------------------
    def save_to_json(self, json_path=None):
        """
        Save the geometry of systems to the path of {json_path}
        Input Args:
            json_path: None or string, optional
                path to save the systems
        """
        if json_path is None:
            json_path = self.config
        elif isinstance(json_path, str):
            # check the existance of the directory 
            assert os.path.exists(os.path.split(json_path)[0]), \
                ValueError('save directory not exists')
        json_file = open(json_path, mode='w')

        json_content = {}
        json_content["Description"] = self.LensName.split('\n')[0]
        for surf_idx, surf in enumerate(self):
            if isinstance(surf, Conjugate):
                # object plane
                json_content["OBJECT"] = {
                    "index"   : int(surf_idx),
                    "roc"     : 0.000000000000000E+000,
                    "distance": 0.000000000000000E+000,
                    "material": "vacuum",
                    "radius"  : 0.000000000000000E+000,
                    "conic"   : 0.000000000000000E+000,
                    "ai-terms": None,
                }
            elif isinstance(surf, Surface):
                if surf_idx == (self.aperture_ind-1):
                    json_content["STOP"] = {
                        "index"   : int(surf_idx),
                        "roc"     : 0.0 if surf.c == 0. else (1/surf.c).item(),
                        "distance": surf.d_after.item(),
                        "material": surf.Material.name,
                        "radius"  : surf.r.item(),
                        "conic"   : surf.k.item(),
                        "ai-terms": None if not isinstance(surf, Aspheric) else surf.ai.tolist(),
                    }
                elif surf_idx == (len(self) - 1):
                    json_content["IMAGE"] = {
                        "index"   : int(surf_idx),
                        "roc"     : 0.0 if surf.c == 0. else (1/surf.c).item(),
                        "distance": 0.,
                        "material": surf.Material.name,
                        "radius"  : surf.r.item(),
                        "conic"   : surf.k.item(),
                        "ai-terms": None if not isinstance(surf, Aspheric) else surf.ai.tolist(),
                    }
                else:
                    json_content["Standard"+str(surf_idx)] = {
                        "index"   : int(surf_idx),
                        "roc"     : 0.0 if surf.c == 0. else (1/surf.c).item(),
                        "distance": self[surf_idx+1].d.item() - surf.d.item(),
                        "material": surf.Material.name,
                        "radius"  : surf.r.item(),
                        "conic"   : surf.k.item(),
                        "ai-terms": None if not isinstance(surf, Aspheric) else surf.ai.tolist(),
                    }
                
        
        json.dump(json_content, json_file, indent=4)

        json_file.close()

    def find_ray_2D(self, view=0.0, y=0.0):
        """
        This function finds chief and marginal rays at a specific view.
        """
        wavelength = torch.Tensor([589.3]).to(self.device)
        R_aperture = self.surfaces[self.aperture_ind].r
        angle = np.radians(view)
        d = torch.Tensor(np.stack((
            np.sin(angle),
            0,
            np.cos(angle)), axis=-1
        )).to(self.device)

        def find_x(alpha=1.0):  # TODO: does not work for wide-angle lenses!
            x = - np.tan(angle) * \
                self.surfaces[self.aperture_ind].d.cpu().detach().numpy()
            is_converge = False
            for k in range(30):
                o = torch.Tensor([x, y, 0.0])
                ray = Ray(o, d, wavelength, device=self.device)
                ray_final, valid = self.trace(
                    ray, stop_ind=self.aperture_ind)[:2]
                x_aperture = ray_final.o[0].cpu().detach().numpy()
                diff = 0.0 - x_aperture
                if np.abs(diff) < 0.001:
                    print('`find_x` converges!')
                    is_converge = True
                    break
                if valid:
                    x_last = x
                    if diff > 0.0:
                        x += alpha * diff
                    else:
                        x -= alpha * diff
                else:
                    x = (x + x_last) / 2
            return x, is_converge

        def find_bx(x_center, R_aperture, alpha=1.0):
            x = x_center
            x_last = 0.0  # temp
            for k in range(100):
                o = torch.Tensor([x, y, 0.0])
                ray = Ray(o, d, wavelength, device=self.device)
                ray_final, valid = self.trace(
                    ray, stop_ind=self.aperture_ind)[:2]
                x_aperture = ray_final.o[0].cpu().detach().numpy()
                diff = R_aperture - x_aperture
                if np.abs(diff) < 0.01:
                    print('`find_x` converges!')
                    break
                if valid:
                    x_last = x
                    if diff > 0.0:
                        x += alpha * diff
                    else:
                        x -= alpha * diff
                else:
                    x = (x + x_last) / 2
            return x_last

        x_center, is_converge = find_x(alpha=-np.sign(view) * 1.0)
        if not is_converge:
            x_center, is_converge = find_x(alpha=np.sign(view) * 1.0)

        x_up = find_bx(x_center, R_aperture, alpha=1)
        x_down = find_bx(x_center, -R_aperture, alpha=-1)
        return x_up, x_down, x_center

    # ------------------------------------------------------------------------------------
    # diff optics

    # ------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------
    # ours

    def gausslet_trace(self, gausslets, stop_ind=None):
        # update transformation when doing pose estimation
        # if (
        #         self.origin.requires_grad
        #         or
        #         self.shift.requires_grad
        #         or
        #         self.theta_x.requires_grad
        #         or
        #         self.theta_y.requires_grad
        #         or
        #         self.theta_z.requires_grad
        # ):
        #     self.update()
        # 拆分成五条光线，然后送去_trace;区别在于中心线需要一个单独的写法，计入光程以及能量损失(打算和衍射一起)
        return self._gausslets_trace(gausslets, stop_ind=stop_ind)

    # ------------------------------------------------------------------------------------

    def _gausslets_refract(self, wi, n1, n2, normal, gv):
        """
        Snell's law (surface normal n defined along the positive z axis)
        https://physics.stackexchange.com/a/436252/104805
        """
        u = (n1 * wi + gv) / n2
        # print("u shape: ", u.shape)
        cost2 = 1. - length2(u) + torch.einsum('...k,...k', normal, u) ** 2
        # 1. get valid map; 2. zero out invalid points; 3. add eps to avoid
        # NaN grad at cost2==0.
        valid = cost2 > 0.
        cost2 = torch.clamp(cost2, min=1e-8)
        # cost2 = torch.clamp(cost2, min=0)
        tmp = torch.sqrt(cost2)[..., None]
        # here we do not have to do normalization because if both wi and n are normalized,
        # then output is also normalized.
        # todo 大视场求解的交点在球面外会引入nan
        wt = u + normal * \
            (tmp - torch.einsum('...k,...k', normal, u)[..., None])
        # return valid, normalize(wt)
        return valid, wt

    def _gausslets_trace(self, gausslets, stop_ind=None):
        if stop_ind is None:
            stop_ind = len(self.surfaces) - 1  # last index to stop
        # 先不考虑lens group的整体坐标变换
        # ray_in = self.to_object.transform_ray(ray)
        # ray_final = self.to_world.transform_ray(ray_out)
        for i in range(stop_ind + 1):
            # 每个面计算光线并裁切，随后用于下一个面
            valid_base, opd, pt = self._gausslets_forward_tracing(
                gausslets.base_ray, i, base_r=True)
            valid_waistx = self._gausslets_forward_tracing(
                gausslets.waist_rayx, i)
            valid_waisty = self._gausslets_forward_tracing(
                gausslets.waist_rayy, i)
            valid_divx = self._gausslets_forward_tracing(gausslets.div_rayx, i)
            valid_divy = self._gausslets_forward_tracing(gausslets.div_rayy, i)
            gausslets.axial_amp *= pt
            gausslets.opd += opd
            valid = valid_base & valid_divy & valid_divx & valid_waisty & valid_waistx
            # print("left {} gausslet after trace surface{}".format(valid.cpu().sum(), i+1))
            # valid_base 可以感知大像差可能带来的仿真不准确问题
            gausslets = gausslets.clip(valid)
        return gausslets

    def _gausslets_forward_tracing(self, ray, index, base_r=False):
        wavelength = ray.wavelength
        dim = ray.o[..., 2].shape
        t_sum = torch.zeros(dim).to(self.device)
        n1 = self.surfaces[index].material.refractive_index(wavelength)
        n2 = self.surfaces[index + 1].material.refractive_index(wavelength)
        # ray intersecting surface
        valid_s, valid_a, p, t = self.surfaces[index].gausslets_ray_surface_intersection(
            ray)
        # get surface normal and refract
        r2 = p[..., 0]**2+p[..., 1]**2
        c = torch.clamp(self.surfaces[index].c, min=1e-8)
        nan = torch.where(
            r2 > 0.95/((1+self.surfaces[index].k)*c**2), True, False)
        p[nan] = torch.tensor([0., 0., self.surfaces[index].d]).to(self.device)
        # 注意！！一些光线与面是无解的，即nan
        grating_v = self.doe[index].grating_vector(
            p[..., 0], p[..., 1]) * self.doe[index].m / self.doe[index].wavelength * wavelength
        t_sum += self.doe[index].local_opd(p[..., 0], p[..., 1]) / \
            self.doe[index].wavelength * wavelength
        n = self.surfaces[index].normal(p[..., 0], p[..., 1])
        valid_d, d = self._gausslets_refract(ray.d, n1, n2, -n, grating_v)

        # fresnel equation: power transmit
        cosi = torch.sum(ray.d * -n, dim=-1)
        cost = torch.sum(d * -n, dim=-1)
        rs = ((n1 * cosi - n2 * cost) / (n1 * cosi + n2 * cost)) ** 2
        rp = ((n1 * cost - n2 * cosi) / (n1 * cost + n2 * cosi)) ** 2
        pt = torch.sqrt(torch.clamp((2 - rs - rp) / 2, min=1e-8))
        # pt = torch.tensor(1).to(self.device)

        ray.o = p
        ray.d = d
        # 计入光程
        t_sum += t * \
            self.surfaces[index].materials.refractive_index(wavelength)

        # check validity
        if base_r:
            # 有交点、孔径内、不是全反射
            valid = valid_s & valid_a & valid_d
            return valid, t_sum, pt
        else:
            valid = valid_s & valid_d
            return valid

    def _generate_points(self, surface, with_boundary=False):
        R = surface.r
        x = y = torch.linspace(-R,
                               R,
                               surface.APERTURE_SAMPLING,
                               device=self.device)
        X, Y = torch.meshgrid(x, y, indexing='ij')
        Z = surface.surface_with_offset(X, Y)
        valid = surface.is_valid(torch.stack((x, y), axis=-1))

        if with_boundary:
            from scipy import ndimage
            tmp = ndimage.convolve(valid.cpu().numpy().astype(
                'float'), np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]))
            boundary = valid.cpu().numpy() & (tmp != 4)
            boundary = boundary[valid.cpu().numpy()].flatten()
        points_local = torch.stack(
            tuple(
                v[valid].flatten() for v in [
                    X, Y, Z]), axis=-1)
        points_world = self.to_world.transform_point(
            points_local).T.cpu().detach().numpy()
        if with_boundary:
            return points_world, boundary
        else:
            return points_world
