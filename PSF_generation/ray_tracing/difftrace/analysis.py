import matplotlib.pyplot as plt
import torch
import numpy as np

from .surfaces import *
from .utils import lambda_d, pupil_distribution, fraunhofer
from tqdm import tqdm

class Analysis:

    def __init__(self, system, views, wavelengths,
                 dtype=torch.float64, device=torch.device('cpu')):
        self.system = system
        self.views = views
        self.wavelengths = wavelengths
        self.dtype = dtype
        self.device = device
        
    # ====================================================================================
    # System Viewers
    # ====================================================================================

    def plot_setup_2d(self, ax=None, fig=None, show=True, color='k'):
        """
        Plot elements in 2D.
        """
        if ax is None and fig is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        else:
            show = False

        # to world coordinate
        def plot(ax, z, x, color):
            p = torch.stack((x, torch.zeros_like(x), z.squeeze(0)), dim=-1).cpu().detach().numpy()
            ax.plot(p[..., 2], p[..., 0], color)

        def draw_aperture(ax, surface, color):
            N = 3
            d = surface.d.cpu()
            R = surface.r
            APERTURE_WEDGE_LENGTH = 0.05 * R
            APERTURE_WEDGE_HEIGHT = 0.15 * R

            # wedge length
            z = torch.linspace(
                d - APERTURE_WEDGE_LENGTH,
                d + APERTURE_WEDGE_LENGTH,
                N,
                device=self.system.device)
            x = -R * torch.ones(N, device=self.system.device)
            plot(ax, z, x, color)
            x = R * torch.ones(N, device=self.system.device)
            plot(ax, z, x, color)

            # wedge height
            z = d * torch.ones(N, device=self.system.device)
            x = torch.linspace(R, R + APERTURE_WEDGE_HEIGHT, N, device=self.system.device)
            plot(ax, z, x, color)
            x = torch.linspace(-R - APERTURE_WEDGE_HEIGHT, -R, N, device=self.system.device)
            plot(ax, z, x, color)

        # if there is only one surface, then it has to be the aperture
        if len(self.system) == 1:
            draw_aperture(ax, self.system[0], color)
        else:
            # draw surface
            for i, s in enumerate(self.system):
                if i == 0:
                    continue
                if i == self.system.aperture_ind - 1:
                    draw_aperture(ax, self.system.aperture, color)
                    continue
                # surface sampling
                r = torch.linspace(-s.r, s.r, s.SURFACE_SAMPLING, device=self.system.device)
                z = s.surface_with_offset(r, torch.zeros(len(r), device=self.system.device))
                plot(ax, z, r, color)

            # draw boundary
            for i, s in enumerate(self.system.surfaces):
                if i == 0:
                    continue
                if s.Material.nd > 1.0003:
                    s_post = self.system.surfaces[i + 1]
                    r_post = s_post.r
                    r = s.r

                    sag = s.surface_with_offset(r, 0.0).squeeze()
                    sag_post = s_post.surface_with_offset(r_post, 0.0).squeeze()

                    z = torch.stack((sag, sag_post))
                    x = torch.stack([r, r_post])

                    plot(ax, z, x, color)
                    plot(ax, z, -x, color)

        plt.gca().set_aspect('equal', adjustable='box')
        plt.xlabel('z [mm]')
        plt.ylabel('r [mm]')
        plt.title("Layout 2D")
        if show:
            plt.show()
        return ax, fig

    def plot_setup_2d_with_trace(
            self, views=None, wavelength=None, M=2, R=None, entrance_pupil=True, show=True):
        """
        Plot elements and rays in different views.
        M: number of rays in yz plane
        R: radius of rays bundle
        """
        if views is None:
            views = self.views
        if wavelength is None:
            wavelength = self.wavelengths[0]

        if R is None:
            R = self.system.surfaces[1].r
        colors_list = 'bgrymck'
        ax, fig = self.plot_setup_2d(show=False)

        # plot rays
        for i, view in enumerate(views):
            ray = self.sample_ray_2d(R, wavelength, view=view, M=M, entrance_pupil=entrance_pupil)
            oss, _, _, valids = self.system.propagate(ray, record=True)
            ax, fig = self.plot_raytraces(oss, valids, ax=ax, fig=fig, color=colors_list[i])
        if show:
            plt.show()
        return ax, fig

    def sample_ray_2d(self, R, wavelength, view=0.0, M=15,
                      shift_y=0., entrance_pupil=False):
        """
        sample rays in yz plane, use to plot ray trace
        """
        if entrance_pupil:
            ys = self.calc_entrance_pupil(view=view, wavelength=wavelength)[2]
            y_up = ys.min()
            y_down = ys.max()
            y_center = ys.mean()

            y = torch.hstack((torch.linspace(y_down, y_center, M + 1, device=self.system.device, dtype=self.dtype)[:M],
                              torch.linspace(y_center, y_up, M + 1, device=self.system.device, dtype=self.dtype)))
        else:
            y = torch.linspace(-R, R, M, device=self.device, dtype=self.dtype)
        p = 2 * R / M
        y = y + p * shift_y

        o = torch.stack((torch.zeros_like(y), y, torch.zeros_like(y)), dim=1)
        self.system[0].update(view)
        d = self.system[0].map().unsqueeze(0).repeat(2*M+1, 1)
        return Ray(o, d, wavelength, device=self.device)

    def calc_entrance_pupil(self, view=0.0, wavelength=lambda_d, R=None):
        """
        prepares valid rays that successfully propagate through the system
        """
        # update the field-of-view of system
        self.system[0].update(view)
        # maximum radius input
        if R is None:
            with torch.no_grad():
                R = torch.tan(self.system[0].angle_fov) * \
                    self.system[1].surface(self.system[1].r, 0.) + \
                    self.system[1].r

        APERTURE_SAMPLING = 201
        
        x_e, y_e = torch.meshgrid(torch.linspace(-R, R, APERTURE_SAMPLING, dtype=self.dtype, device=self.device),
                                  torch.linspace(-R, R, APERTURE_SAMPLING, dtype=self.dtype, device=self.device),
                                  indexing='ij')
        x_e, y_e = x_e.reshape(-1), y_e.reshape(-1)
        # generate rays and find valid map
        o = torch.stack((x_e, y_e, torch.zeros_like(x_e)), dim=-1)
        d = self.system[0].map().unsqueeze(0).unsqueeze(0).\
            repeat(APERTURE_SAMPLING, APERTURE_SAMPLING, 1)
        ray = Ray(o, d.reshape(-1, 3), wavelength=wavelength, device=self.device)
        valid_map, _ = self.system.propagate(ray)

        # find bounding box
        xs, ys = x_e[valid_map], y_e[valid_map]

        return valid_map, xs, ys

    def plot_raytraces(self, oss, valids, ax=None, fig=None, color='b-'):
        """
        Plot all ray traces (oss).
        """
        if ax is None and fig is None:
            ax, fig = self.plot_setup_2d(show=False)
        for os, v in zip(oss, valids):
            y = os[v, 1].cpu().detach().numpy()
            z = os[v, 2].cpu().detach().numpy()
            ax.plot(z, y, color, linewidth=1.0)
        return ax, fig

    # ====================================================================================
    # Image Quality
    # ====================================================================================
    def sample_ray(self,
                   R=None,
                   wavelength=lambda_d,
                   view=0.0,
                   M=15,
                   shift_x=0.,
                   shift_y=0.,
                   sampling='grid',
                   entrance_pupil=False):
        """
        sample rays from view with different sampling: grid,radial
        rays fulfill the first surface
        """
        angle = np.radians(np.asarray(view))
        if R is None:
            sag = self.system.surfaces[1].surface(self.system.surfaces[1].r, 0.0)
            R = np.tan(angle) * sag + self.system.surfaces[1].r  # [mm]
            sag = self.system.surfaces[1].surface(
                self.system.surfaces[1].r, 0.0)
            R = torch.tan(angle) * sag + self.system.surfaces[1].r  # [mm]
            R = R.item()

        if entrance_pupil:
            xs, ys = self.calc_entrance_pupil(view, wavelength, R)[1:]
            if sampling == 'grid':
                x, y = torch.meshgrid(
                    torch.linspace(xs.min(), xs.max(), M, device=self.device, dtype=self.dtype),
                    torch.linspace(ys.min(), ys.max(), M, device=self.device, dtype=self.dtype),
                    indexing='ij'
                )
            elif sampling == 'radial':
                R = np.minimum(xs.max() - xs.min(), ys.max() - ys.min())
                r = torch.linspace(0, R, M, device=self.device, dtype=self.dtype)
                theta = torch.linspace(0, 2 * np.pi, M + 1, device=self.device, dtype=self.dtype)[0:M]
                x = xs.mean() + r[None, ...] * torch.cos(theta[..., None])
                y = ys.mean() + r[None, ...] * torch.sin(theta[..., None])
        else:
            if sampling == 'grid':
                x, y = torch.meshgrid(
                    torch.linspace(-R, R, M, device=self.device, dtype=self.dtype),
                    torch.linspace(-R, R, M, device=self.device, dtype=self.dtype),
                    indexing='ij'
                )
            elif sampling == 'radial':
                r = torch.linspace(0, R, M, device=self.device, dtype=self.dtype)
                theta = torch.linspace(0, 2 * np.pi, M + 1, device=self.device, dtype=self.dtype)[0:M]
                x = r[None, ...] * torch.cos(theta[..., None])
                y = r[None, ...] * torch.sin(theta[..., None])

        p = 2 * R / M
        x = x.reshape(-1) + p * shift_x
        y = y.reshape(-1) + p * shift_y
        o = torch.stack((x, y, torch.zeros_like(x, dtype=self.dtype)), dim=-1)
        d = torch.stack((
            torch.zeros_like(x, dtype=self.dtype),
            np.sin(angle) * torch.ones_like(x),
            np.cos(angle) * torch.ones_like(x)), dim=-1
        )
        return Ray(o, d, wavelength, device=self.device)

    def rms(self, ps, squared=False):
        """
        rms of spot points
        return rms and centered ps
        """
        ps = ps[..., :2]
        ps_mean = torch.mean(ps, dim=0)
        ps = ps - ps_mean[None, ...]  # we now use normalized ps
        if squared:
            return torch.mean(torch.sum(ps ** 2, dim=-1)), ps
        else:
            return torch.sqrt(torch.mean(torch.sum(ps ** 2, dim=-1))), ps

    def spot_diagram(self, views=None, wavelengths=None, M=6,
                     R=None, sampling='radial', entrance_pupil=True, show=True):
        """
        plot spot diagram
        return mean wavelengths spots for different views
        """
        if R is None:
            R = self.system.surfaces[1].r.item()

        if views is None:
            views = self.views

        if wavelengths is None:
            wavelengths = self.wavelengths

        ps_dic = {}
        spot_rmss = torch.zeros(len(views))
        lim = 0

        for j, view in enumerate(views):
            pss = torch.tensor([[0, 0]], device=self.device)
            for i, wavelength in enumerate(wavelengths):
                ray = self.sample_ray(R, wavelength, view=view, M=M, sampling=sampling, entrance_pupil=entrance_pupil)
                oss, _, _, valids = self.system.propagate(ray, record=True)
                ps = oss[valids[:, -1], -1, :2]
                pss = torch.cat((pss, ps), dim=0)

                if show:
                    ps = ps.cpu().detach().numpy()[..., :2]
                    ps_mean = np.mean(ps, axis=0)  # centroid
                    ps = ps - ps_mean[None, ...]  # we now use normalized ps
                    lim_ = np.abs(ps).max() * 2
                    if lim_ > lim:
                        lim = lim_
                    ps_dic[j, i] = ps

            pss = pss[1:, :]
            spot_rms, _ = self.rms(pss)
            spot_rmss[j] = spot_rms

        if show:
            colors_list = 'bgrymck'
            for j, view in enumerate(views):
                fig = plt.figure()
                ax = plt.axes()
                for i, wavelength in enumerate(wavelengths):
                    ax.scatter(ps_dic[j, i][..., 0], ps_dic[j, i]
                               [..., 1], color=colors_list[i], s=0.1)

                plt.gca().set_aspect('equal', adjustable='box')
                xlims = [-lim, lim]
                ylims = [-lim, lim]
                plt.xlim(*xlims)
                plt.ylim(*ylims)
                ax.set_aspect(1. / ax.get_data_ratio())

                units_str = '[mm]'
                plt.xlabel('x ' + units_str)
                plt.ylabel('y ' + units_str)

                ax.set_title('view: {} degree, RMS: {:.4f}'.format(view, float(spot_rmss[j])))
            plt.show()
        return spot_rmss

    # =================================
    # Rays and Spots
    # =================================
    def single_ray_trace(self, Px=0, Py=1, wavelength=None, view=None):
        """
        Tracing a single ray with
        the normalized pupil coordinates [Px, Py],
        the [wavelength],
        the [view] (in degree),
        and use the [global coordinates] or [local coordinates]
        """
        if wavelength is None:
            wavelength = self.wavelengths[len(self.wavelengths)//2] 
        if view is None:
            view = self.views[-1]
        # update the field of view of this system
        self.system[0].update(view)
        ray = self.system._aim_ray_enumerate(Px, Py, view=view, wavelength=wavelength)
        # propagate the ray and get its intersection on each surface
        oss, dss, tss, _ = self.system.propagate(ray, start=1, stop=None, record=True)
        ###############################
        # print the trace of single ray
        ###############################
        # the head
        print('Ray Trace Data \n')
        print('Lens Title: {}'.format(self.system.LensName))

        print('Units         :   Millimeters')
        print('Wavelength    :   {:0.6f}  um'.format(wavelength * 1e3))
        print('Coordinates   :   Global coordinates relative to surface 1 \n')

        print(
            'Field-of-View :   {:f} degree (represent in angle) '.format(view))
        print('Normalized X Pupil Coord (Px) :      {:0.10f}'.format(Px))
        print('Normalized Y Pupil Coord (Py) :      {:0.10f} \n'.format(Py))

        # rays data
        print('Real Ray Trace Data: \n')
        itv = 4  # interval between rows
        print('Surf' + ' ' * itv + ' ' * 4 + 'X-coordinate' +
                       ' ' * itv + ' ' * 5 + 'Y-coordinate' +
                       ' ' * itv + ' ' * 5 + 'Z-coordinate' +
                       ' ' * itv + ' ' * 5 + 'X-cosine' +
                       ' ' * itv + ' ' * 5 + 'Y-cosine' +
                       ' ' * itv + ' ' * 5 + 'Z-cosine' +
                       ' ' * itv + ' ' * 5 + 'Path length')
        surf_no = 0
        # print(tss)
        # for os, ds, ts in zip(oss[0], dss[0], tss[0]):
        oss = torch.where(oss < torch.finfo(self.dtype).eps, 
                          torch.tensor(0., dtype=self.dtype, device=self.device),
                          oss)
        for idx in range(oss.shape[1]): # for oss and dss the 1 dimension is the surface index
            os, ds, ts = oss[0, idx, :], dss[0, idx, :], tss[idx, :]
            msg = 'OBJ' if surf_no == 0 else '{:>3d}'.format(surf_no)
            if surf_no == 0:
                if self.system[0]._type == 'infinite':
                    msg += ' ' * itv + ' ' * 9 + 'Infinity' + ' ' * itv + ' ' * 9 + 'Infinity' + \
                           ' ' * itv + ' ' * 9 + 'Infinity'
                elif self.system[0]._type == 'finite':
                    msg += ' ' * itv + '{:>17.10E}'.format(os[0].item()) + \
                           ' ' * itv + '{:>17.10E}'.format(os[1].item()) + \
                           ' ' * itv + '{:>17.10E}'.format(os[2].item())

                msg += ' ' * itv + '{:>13.10f}'.format(ds[0].item()) + ' ' * itv + \
                       '{:>13.10f}'.format(ds[1].item()) + \
                       ' ' * itv + '{:>13.10f}'.format(ds[2].item())  # direction cosines
                msg += ' ' * itv + ' ' * 15 + '-'  # path length
            else:
                msg += ' ' * itv + '{:>17.10E}'.format(os[0].item()) + ' ' * itv + \
                       '{:>17.10E}'.format(os[1].item()) + \
                       ' ' * itv + '{:>17.10E}'.format(os[2].item())  # xyz coordinates
                msg += ' ' * itv + '{:>13.10f}'.format(ds[0].item()) + ' ' * itv + \
                       '{:>13.10f}'.format(ds[1].item()) + \
                       ' ' * itv + '{:>13.10f}'.format(ds[2].item())  # direction cosines
                msg += ' ' * itv + '{:>13.10E}'.format(ts.item())  # path length

            print(msg)
            surf_no += 1

    def optical_path_difference_fan(self, rays_num, wavelength, view, surface):
        """
        calculate the optical path difference of the rays with
        rays sampling on x and y direction [rays_num]
        wavelength [wavelength]
        field-of-view [view]
        on the surface index [surface]
        return the curve on x and y direction
        """
        raise NotImplementedError

    def wavefront_map(self, sampling=201, wavelength=None,
                      view=None, surface=-1, show=True):
        """
        calculate the wavefront map of system with
        rays sampling [rays_num]
        wavelength [wavelength]
        field-of-view [view]
        on the surface index [surface]
        return the 2D wavefront map
        """
        if wavelength is None:
            wavelength = self.wavelengths[len(self.wavelengths)//2] 
        if view is None:
            view = self.views[-1]
        ray, ray_chief = self.image_cache(
            sampling, view, wavelength, 'hexapolar')

        valid = torch.ones(ray.o[..., 2].shape, device=ray.device).bool()
        valid_chief = torch.ones(ray_chief.o[..., 2].shape, device=ray_chief.device).bool()
        # optical path calculation with propagate
        op = torch.einsum('ij,ij->i', ray.o - ray_chief.o, ray_chief.d)
        op_chief = torch.einsum(
            'ij,ij->i',
            ray_chief.o -
            ray_chief.o,
            ray_chief.d)

        # propagate to the last surface of lens (before image plane)
        for s in self.system[1:None]:
            valid, ray = s.propagate(ray, valid)
            op += ray.t * s.refractive_index_prev(wavelength)
            valid_chief, ray_chief = s.propagate(ray_chief, valid_chief)
            op_chief += ray_chief.t * s.refractive_index_prev(wavelength)

        # calculate the intersection of ray and optical axis to judge the exit pupil
        # NOTE: Only y direction cosine calculation is supported
        if view == 0:
            self.system._paraxial_info(self.views)
            ep_roc = - self.system.Exit_Pupil_Position
        else:
            ep_roc = ray_chief.o[..., 1] / ray_chief.d[..., 1]
            ep_roc = ep_roc.squeeze()

        # origin_exit_pupil = ray_chief.o - t_ep * ray_chief.d
        ep_distance = self.system[-1].d - ep_roc
        ep_distance_prev = -ep_roc
        ep_radius = 2. * ep_roc
        ep_surf = Spheric(roc=ep_roc,
                          conic=None,
                          radius=ep_radius,
                          distance=ep_distance,
                          distance_prev=ep_distance_prev,
                          distance_after=-ep_distance_prev,
                          material='vacuum',
                          material_prev='vacuum'
                          )  # exit pupil surface related to the center of sensor
        # shift rays on image plane to the center of sensor
        op = op[valid]
        o_final = torch.zeros_like(
            ray.o[valid],
            dtype=self.dtype,
            device=self.device)
        o_final[..., 0:2] = ray.o[valid][..., 0:2] - ray_chief.o[..., 0:2]
        o_final[..., 2] = ray.o[valid][..., 2]
        # form the ray backpropagate to the sensor plane
        ray_shift = Ray(
            o=o_final,
            d=ray.d[valid],
            wavelength=ray.wavelength,
            dtype=self.dtype,
            device=self.device)
        valid_ep = torch.ones(ray_shift.o[..., 2].shape, device=ray.device).bool()
        ray_ep = ep_surf.propagate(ray_shift, valid_ep)[1]
        ray_ep.o[..., 0:2] += ray_chief.o[..., 0:2]
        op += ray_ep.t

        # shift chief rays on image plane to the center of sensor
        o_chief_shift = torch.tensor([0, 0, ray_chief.o[0, 2]], 
                                     dtype=self.dtype, device=ray_chief.device).unsqueeze(0)
        ray_chief_shift = Ray(o=o_chief_shift,
                              d=ray_chief.d,
                              wavelength=ray_chief.wavelength,
                              dtype=self.dtype,
                              device=self.device)
        valid_chief_ep = torch.ones(ray_chief_shift.o[..., 2].shape, device=ray.device).bool()
        ray_chief_ep = ep_surf.propagate(ray_chief_shift, valid_chief_ep)[1]
        op_chief += ray_chief_ep.t

        op_diff = op_chief - op

        ps = torch.Tensor(np.asarray(ray_ep.o)
                          [..., :2]).cpu().detach().numpy()
        x = ps[..., 1]
        y = ps[..., 0]

        from scipy.interpolate import griddata
        xs, ys = np.mgrid[x.min():x.max():1j * sampling,
                          y.min():y.max():1j * sampling]
        points = np.asarray([x, y]).T
        op_diff = griddata(points, op_diff, (xs, ys), method='linear')

        if show:
            plt.imshow(op_diff / wavelength, cmap='jet')
            plt.colorbar()

        return op_diff / wavelength

    def pupil_cache(self, pupil_sampling, view=0.0,
                    wavelength=lambda_d, distrubution='hexapolar'):
        """
        quickly get the ray cache in entrace pupil
        chief ray and a bundle of rays are gathered
        """
        # chacing a large number of ray to quickly judge the incident rays
        # range
        self.system[0].update(view)

        R = torch.tan(self.system[0].angle_fov) * self.system[1].surface(self.system[1].r, 0.) + \
            self.system[1].r
        APERTURE_SAMPLING = 1001  # enumerate sampling density
        x_e, y_e = torch.meshgrid(torch.linspace(-R.item(), R.item(), APERTURE_SAMPLING, dtype=self.dtype, device=self.device),
                                  torch.linspace(-R.item(), R.item(), APERTURE_SAMPLING, dtype=self.dtype, device=self.device),
                                  indexing='ij')
        x_e, y_e = x_e.reshape(APERTURE_SAMPLING * APERTURE_SAMPLING, -
                               1), y_e.reshape(APERTURE_SAMPLING * APERTURE_SAMPLING, -1)
        o = torch.stack((x_e, y_e, torch.zeros_like(x_e)), dim=2)
        o = o.reshape(APERTURE_SAMPLING * APERTURE_SAMPLING, -1)
        d = self.system[0].map().unsqueeze(0).\
            repeat(APERTURE_SAMPLING, APERTURE_SAMPLING, 1).reshape(
                APERTURE_SAMPLING * APERTURE_SAMPLING, -1)
        ray_to_pupil = Ray(o, d, wavelength=wavelength, device=self.device)
        valid, ray_stop = self.system.propagate(
            ray_to_pupil, start=1, stop=self.system.aperture_ind)

        # find the valid position nearest to the aiming position [Px, Py]
        # position: [chief, top, bot, lft, rit]
        pupil_aims = torch.tensor(
            ([[0., 0.], [0., 1.], [0., -1.], [1., 0.], [-1., 0.]]), device=self.device)
        pupil_cache = []
        x_valid, y_valid = x_e[valid], y_e[valid]
        for i, aim in enumerate(pupil_aims):
            pupil_aim_pos = aim * \
                self.system[self.system.aperture_ind - 1].r.unsqueeze(0)
            residual = length(ray_stop.o[valid][..., 0:2] - pupil_aim_pos)
            x = torch.squeeze(x_valid[residual == residual.min()])
            y = torch.squeeze(y_valid[residual == residual.min()])
            pupil_cache.append(torch.tensor((x, y)))

        # sample rays from the pupil cache
        # n*n sampling point, each point has x and y coordinate
        o_p = pupil_distribution(pupil_sampling, pupil_sampling, distrubution, device=self.system.device)

        o = torch.zeros((pupil_sampling, pupil_sampling, 3),
                        dtype=self.dtype, device=self.device)
        o[0:pupil_sampling // 2, :, 1] = o_p[0:pupil_sampling // 2, :, 0] * \
            torch.abs(pupil_cache[1][1] - pupil_cache[0]
                      [1]) + pupil_cache[0][1]
        o[pupil_sampling // 2:, :, 1] = o_p[pupil_sampling // 2:, :, 0] * \
            torch.abs(pupil_cache[2][1] - pupil_cache[0]
                      [1]) + pupil_cache[0][1]
        o[:, 0:pupil_sampling // 2, 0] = o_p[:, 0:pupil_sampling // 2, 1] * \
            torch.abs(pupil_cache[3][0] - pupil_cache[0]
                      [0]) + pupil_cache[0][0]
        o[:, pupil_sampling // 2:, 0] = o_p[:, pupil_sampling // 2:, 1] * \
            torch.abs(pupil_cache[4][0] - pupil_cache[0]
                      [0]) + pupil_cache[0][0]
        o = o.reshape(pupil_sampling * pupil_sampling, -1)
        d = self.system[0].map().unsqueeze(0).repeat(
            pupil_sampling * pupil_sampling, 1).reshape(pupil_sampling * pupil_sampling, -1)
        ray = Ray(o, d, wavelength=wavelength, device=self.device)
        ray_chief = Ray(o=torch.tensor((pupil_cache[0][0], pupil_cache[0][1], 0.))[None, :],
                        d=self.system[0].map().unsqueeze(0),
                        wavelength=wavelength,
                        device=self.device)
        return ray, ray_chief
    
    def image_cache(self, pupil_sampling, view=0.0,
                    wavelength=lambda_d, distrubution='hexapolar'):
        """
        quickly get the ray cache in the image plane
        chief ray and a bundle of rays are gathered
        """
        # chacing a large number of ray to quickly judge the incident rays
        # range
        self.system[0].update(view)

        R = torch.tan(self.system[0].angle_fov) * self.system[1].surface(self.system[1].r, 0.) + \
            self.system[1].r
        APERTURE_SAMPLING = 101  # enumerate sampling density
        x_e, y_e = torch.meshgrid(torch.linspace(-R.item(), R.item(), APERTURE_SAMPLING, dtype=self.dtype, device=self.device),
                                  torch.linspace(-R.item(), R.item(), APERTURE_SAMPLING, dtype=self.dtype, device=self.device),
                                  indexing='ij')
        x_e, y_e = x_e.reshape(APERTURE_SAMPLING * APERTURE_SAMPLING, -
                               1), y_e.reshape(APERTURE_SAMPLING * APERTURE_SAMPLING, -1)
        o = torch.stack((x_e, y_e, torch.zeros_like(x_e)), dim=2)
        o = o.reshape(APERTURE_SAMPLING * APERTURE_SAMPLING, -1)
        d = self.system[0].map().unsqueeze(0).\
            repeat(APERTURE_SAMPLING, APERTURE_SAMPLING, 1).reshape(
                APERTURE_SAMPLING * APERTURE_SAMPLING, -1)
        ray_to_image = Ray(o, d, wavelength=wavelength, device=self.device)
        valid, _ = self.system.propagate(ray_to_image, start=1, stop=None)  # to the final plane of the optical system

        # find the valid position on image plane
        # position: [chief, top, bot, lft, rit]
        x_valid, y_valid = x_e[valid], y_e[valid]
        image_cache = []
        image_cache.append(torch.tensor((x_valid.mean(), y_valid.mean())))  # chief ray
        image_cache.append(torch.tensor((x_valid.mean(), y_valid.max())))  # top ray
        image_cache.append(torch.tensor((x_valid.mean(), y_valid.min())))  # bot ray
        image_cache.append(torch.tensor((x_valid.max(), y_valid.mean())))  # lft ray
        image_cache.append(torch.tensor((x_valid.min(), y_valid.mean())))  # rit ray

        # sample rays from the pupil cache
        # n*n sampling point, each point has x and y coordinate
        o_p = pupil_distribution(pupil_sampling, pupil_sampling, distrubution, device=self.system.device)

        if distrubution == 'fibonacci' or distrubution == 'ring':
            o = torch.zeros((o_p.shape[0], 3), dtype=self.dtype, device=self.device)
            o[o_p[..., 0]<0, 1] = o_p[o_p[..., 0]<0, 0] * \
                torch.abs(image_cache[1][1] - image_cache[0][1]) + image_cache[0][1]
            o[o_p[..., 0]>=0, 1] = o_p[o_p[..., 0]>=0, 0] * \
                torch.abs(image_cache[2][1] - image_cache[0][1]) + image_cache[0][1]
            o[o_p[..., 1]<0, 0] = o_p[o_p[..., 0]<0, 1] * \
                torch.abs(image_cache[3][0] - image_cache[0][0]) + image_cache[0][0]
            o[o_p[..., 1]>=0, 0] = o_p[o_p[..., 0]>=0, 1] * \
                torch.abs(image_cache[4][0] - image_cache[0][0]) + image_cache[0][0]
        else:
            o = torch.zeros((pupil_sampling, pupil_sampling, 3), dtype=self.dtype, device=self.device)
            o[0:pupil_sampling // 2, :, 1] = o_p[0:pupil_sampling // 2, :, 0] * \
                torch.abs(image_cache[1][1] - image_cache[0][1]) + image_cache[0][1]
            o[pupil_sampling // 2:, :, 1] = o_p[pupil_sampling // 2:, :, 0] * \
                torch.abs(image_cache[2][1] - image_cache[0][1]) + image_cache[0][1]
            o[:, 0:pupil_sampling // 2, 0] = o_p[:, 0:pupil_sampling // 2, 1] * \
                torch.abs(image_cache[3][0] - image_cache[0][0]) + image_cache[0][0]
            o[:, pupil_sampling // 2:, 0] = o_p[:, pupil_sampling // 2:, 1] * \
                torch.abs(image_cache[4][0] - image_cache[0][0]) + image_cache[0][0]
            o = o.reshape(pupil_sampling * pupil_sampling, -1)

        d = self.system[0].map().unsqueeze(0).repeat(
            pupil_sampling * pupil_sampling, 1).reshape(pupil_sampling * pupil_sampling, -1)
        ray = Ray(o, d, wavelength=wavelength, device=self.device)
        ray_chief = Ray(o=torch.tensor((image_cache[0][0], image_cache[0][1], 0.))[None, :],
                        d=self.system[0].map().unsqueeze(0), wavelength=wavelength, device=self.device)
        return ray, ray_chief

    def psf_spot(self, pupil_sampling, image_sampling,
                 image_delta, wavelength=None, view=None, sample_distribution='hexapolar'):
        """
        counting rays number!
        """
        if wavelength is None:
            wavelength = self.wavelengths[len(self.wavelengths)//2] 
        if view is None:
            view = self.views[-1]
        ray, ray_chief = self.image_cache(
            pupil_sampling, view, wavelength, sample_distribution)

        valid = torch.ones(ray.o[..., 2].shape, device=ray.device).bool()
        valid_chief = torch.ones(ray_chief.o[..., 2].shape, device=ray_chief.device).bool()
        for s in self.system[1:]:
            valid, ray = s.propagate(ray, valid)
            valid_chief, ray_chief = s.propagate(ray_chief, valid_chief)

        ray_o_valid = ray.o[valid]
        # initialize the sample rays number mat on image plane
        intensity_map = torch.zeros(
            (image_sampling,
             image_sampling),
            dtype=self.dtype,
            device=self.device)
        # calculate the rays number of each sampled image position
        for h_img in range(intensity_map.shape[0]):
            for w_img in range(intensity_map.shape[1]):
                # real image position on the image plane, center of the
                # sampling region
                h_img_corr = h_img - int((intensity_map.shape[0] - 1) / 2)
                w_img_corr = w_img - int((intensity_map.shape[1] - 1) / 2)
                # here only two demensions needed
                rel_coor = ray_chief.o[..., 0:2] + \
                    torch.tensor([h_img_corr * image_delta, w_img_corr * image_delta],
                                 dtype=self.dtype, device=self.device)
                # sampling boundary
                h_img_lower, h_img_upper = (rel_coor[..., 1] - image_delta / 2), \
                                           (rel_coor[..., 1] + image_delta / 2)
                w_img_lower, w_img_upper = (rel_coor[..., 0] - image_delta / 2), \
                                           (rel_coor[..., 0] + image_delta / 2)

                # calculate the number of rays within the sampling boundary
                yhl = torch.where((ray_o_valid[..., 1] > h_img_lower)[
                                  :, None], ray_o_valid, torch.nan)
                yhu = torch.where((ray_o_valid[..., 1] < h_img_upper)[
                                  :, None], yhl, torch.nan)
                ywl = torch.where((ray_o_valid[..., 0] > w_img_lower)[
                                  :, None], yhu, torch.nan)
                ywu = torch.where((ray_o_valid[..., 0] < w_img_upper)[
                                  :, None], ywl, torch.nan)
                ywu = ywu[torch.all(torch.isfinite(ywu), dim=1), :]
                rays_num = ywu.shape[0]
                intensity_map[w_img, h_img] = rays_num

        # center point position (ref ray position)
        # chief_pos = self.y[-1, self.ref, 0:2]  # x and y coor
        psf = (intensity_map/intensity_map.sum())
        return psf

    def psf_coherent(self, pupil_sampling, image_sampling,
                     image_delta, wavelength=None, view=None, sample_distribution='hexapolar'):
        """
        coherent superposition with complex amplitude!
        """
        if wavelength is None:
            wavelength = self.wavelengths[len(self.wavelengths)//2] 
        if view is None:
            view = self.views[-1]
        # rays sampling
        ray, ray_chief = self.image_cache(pupil_sampling, view, wavelength, sample_distribution)  # square

        # optical path calculation with propagate
        valid, valid_chief = torch.ones_like(ray.t).bool(), torch.ones_like(ray_chief.t).bool()
        ## if input object plane is infinite and tilt       ##
        ## we assume the optical path of the chief ray is 0 ##
        op = torch.einsum('ij,ij->i', ray.o - ray_chief.o, ray_chief.d)

        for s in self.system[1:-1]:  # propagate to the last surface of lens (before image plane)
            valid, ray = s.propagate(ray, valid)
            op += ray.t * s.refractive_index_prev(wavelength)
            valid_chief, ray_chief = s.propagate(ray_chief, valid_chief)

        valid_chief, ray_chief_final = self.system[-1].propagate(ray_chief, valid_chief)
        # calculate the optical path difference of the sampled point on image
        # initialize the sample complex amplitude mat on image plane
        intensity_map = torch.zeros((image_sampling, image_sampling))
        wave_num = torch.tensor(
            2 * np.pi / wavelength.item(),
            dtype=self.dtype,
            device=self.device)  # 10000
        # calculate the complex amplitude of each sampled image position

        line_sample = torch.linspace(- int((intensity_map.shape[0] - 1) / 2), int(
            (intensity_map.shape[0] - 1) / 2), image_sampling, device=self.device) * image_delta
        x, y = torch.meshgrid(line_sample, line_sample, indexing='ij')
        rel_coor = ray_chief_final.o + torch.stack([x, y, torch.zeros_like(x)], dim=-1)
        r = rel_coor[:, :, None, :] - ray.o[None, None, ...]
        # inner production for the final length
        dr = torch.einsum('ijkl,ijkl->ijk', ray.d[None, None, ...], r)
        # complex amplitude
        amp = torch.einsum(
            'ijk->ij', torch.exp(((op[None, None, :] + dr) * wave_num) * (0 + 1j)))
        psf = torch.abs(amp).T ** 2
        psf = psf / psf.sum()
        return psf

    def psf_huygens(self, pupil_sampling, image_sampling,
                    image_delta, wavelength=None, view=None, sample_distribution='hexapolar'):
        """
        coherent superposition with complex amplitude back to the exit pupil plane
        """
        if wavelength is None:
            wavelength = self.wavelengths[len(self.wavelengths)//2] 
        if view is None:
            view = self.views[-1]
        # rays sampling
        ray, ray_chief = self.image_cache(pupil_sampling, view, wavelength, sample_distribution)
        # ray, ray_chief = self.pupil_cache(pupil_sampling, view, wavelength, 'hexapolar')
        # optical path calculation with propagate
        valid, valid_chief = torch.ones_like(ray.t).bool(), torch.ones_like(ray_chief.t).bool()
        ## if input object plane is infinite and tilt       ##
        ## we assume the optical path of the chief ray is 0 ##
        op = torch.einsum('ij,ij->i', ray.o - ray_chief.o, ray_chief.d)
        op_chief = torch.einsum('ij,ij->i', ray_chief.o - ray_chief.o, ray_chief.d)

        # propagate to the last surface of lens (before image plane)
        for s in self.system[1:None]:
            valid, ray = s.propagate(ray, valid)
            op += ray.t * s.refractive_index_prev(wavelength)
            valid_chief, ray_chief = s.propagate(ray_chief, valid_chief)
            op_chief += ray_chief.t * s.refractive_index_prev(wavelength)

        # calculate the intersection of ray and optical axis to judge the exit pupil
        # NOTE: Only y direction cosine calculation is supported
        if view == 0:
            self.system._paraxial_info(self.views)
            ep_roc = - self.system.Exit_Pupil_Position
        else:
            ep_roc = ray_chief.o[..., 1] / ray_chief.d[..., 1]
            ep_roc = ep_roc.squeeze()

        # origin_exit_pupil = ray_chief.o - t_ep * ray_chief.d
        ep_distance = self.system[-1].d - ep_roc
        ep_distance_prev = -ep_roc
        ep_radius = 2. * ep_roc

        ep_surf = Spheric(roc=ep_roc,
                          conic=None,
                          radius=ep_radius,
                          distance=ep_distance,
                          distance_prev=ep_distance_prev,
                          distance_after=-ep_distance_prev,
                          material='vacuum',
                          material_prev='vacuum',
                          device=self.device
                          )  # exit pupil surface related to the center of sensor
        # shift rays on image plane to the center of sensor
        op = op[valid]
        o_final = torch.zeros_like(
            ray.o[valid],
            dtype=self.dtype,
            device=self.device)
        o_final[..., 0:2] = ray.o[valid][..., 0:2] - ray_chief.o[..., 0:2]
        o_final[..., 2] = ray.o[valid][..., 2]
        # form the ray backpropagate to the sensor plane
        ray_shift = Ray(
            o=o_final,
            d=ray.d[valid],
            wavelength=ray.wavelength,
            dtype=self.dtype,
            device=self.device)
        ray_ep = ep_surf.propagate(ray_shift, torch.ones_like(ray_shift.t).bool())[1]
        ray_ep.o[..., 0:2] += ray_chief.o[..., 0:2]
        op += ray_ep.t
        # calculate the optical path difference of the sampled point on image
        # initialize the sample complex amplitude mat on image plane
        intensity_map = torch.zeros((image_sampling, image_sampling))
        wave_num = torch.tensor(
            2 * np.pi / wavelength.item(),
            dtype=self.dtype,
            device=self.device)  # 10000

        # calculate the complex amplitude of each sampled image position
        line_sample = torch.linspace(- int((intensity_map.shape[0] - 1) / 2), int(
            (intensity_map.shape[0] - 1) / 2), image_sampling, device=self.device) * image_delta
        x, y = torch.meshgrid(line_sample, line_sample, indexing='ij')
        rel_coor = ray_chief.o + torch.stack([x, y, torch.zeros_like(x, dtype=self.dtype, device=self.device)], dim=-1)
        r = rel_coor[:, :, None, :] - ray_ep.o[None, None, ...]
        # inner production for the final length
        dr = torch.einsum('ijkl,ijkl->ijk', ray_ep.d[None, None, ...], r)
        # complex amplitude
        amp = torch.einsum('ijk->ij', torch.exp(((op[None, None, :] + dr) * wave_num) * (0 + 1j)))
        psf = torch.real(amp * torch.conj(amp)).permute(1, 0)
        psf = psf/psf.sum()
        return psf
    
    def psf_kirchoff(self, pupil_sampling, image_sampling,
                    image_delta, wavelength=None, view=None, sample_distribution='hexapolar'):
        """
        add tilt factor to the coherent superposition on the exit pupil plane
        """
        if wavelength is None:
            wavelength = self.wavelengths[len(self.wavelengths)//2] 
        if view is None:
            view = self.views[-1]
        # rays sampling
        ray, ray_chief = self.image_cache(pupil_sampling, view, wavelength, sample_distribution)
        # ray, ray_chief = self.pupil_cache(pupil_sampling, view, wavelength, 'hexapolar')
        # optical path calculation with propagate
        valid, valid_chief = torch.ones_like(ray.t).bool(), torch.ones_like(ray_chief.t).bool()
        ## if input object plane is infinite and tilt       ##
        ## we assume the optical path of the chief ray is 0 ##
        op = torch.einsum('ij,ij->i', ray.o - ray_chief.o, ray_chief.d)
        op_chief = torch.einsum('ij,ij->i', ray_chief.o - ray_chief.o, ray_chief.d)

        # propagate to the last surface of lens (before image plane)
        for s in self.system[1:None]:
            valid, ray = s.propagate(ray, valid)
            op += ray.t * s.refractive_index_prev(wavelength)
            valid_chief, ray_chief = s.propagate(ray_chief, valid_chief)
            op_chief += ray_chief.t * s.refractive_index_prev(wavelength)

        # calculate the intersection of ray and optical axis to judge the exit pupil
        # NOTE: Only y direction cosine calculation is supported
        if view == 0:
            self.system._paraxial_info(self.views)
            ep_roc = - self.system.Exit_Pupil_Position
        else:
            ep_roc = ray_chief.o[..., 1] / ray_chief.d[..., 1]
            ep_roc = ep_roc.squeeze()

        # origin_exit_pupil = ray_chief.o - t_ep * ray_chief.d
        ep_distance = self.system[-1].d - ep_roc
        ep_distance_prev = -ep_roc
        ep_radius = 2. * ep_roc

        ep_surf = Spheric(roc=ep_roc,
                          conic=None,
                          radius=ep_radius,
                          distance=ep_distance,
                          distance_prev=ep_distance_prev,
                          distance_after=-ep_distance_prev,
                          material='vacuum',
                          material_prev='vacuum',
                          device=self.device
                          )  # exit pupil surface related to the center of sensor
        # shift rays on image plane to the center of sensor
        op = op[valid]
        o_final = torch.zeros_like(
            ray.o[valid],
            dtype=self.dtype,
            device=self.device)
        o_final[..., 0:2] = ray.o[valid][..., 0:2] - ray_chief.o[..., 0:2]
        o_final[..., 2] = ray.o[valid][..., 2]
        # form the ray backpropagate to the sensor plane
        ray_shift = Ray(
            o=o_final,
            d=ray.d[valid],
            wavelength=ray.wavelength,
            dtype=self.dtype,
            device=self.device)
        ray_ep = ep_surf.propagate(ray_shift, torch.ones_like(ray_shift.t).bool())[1]
        ray_ep.o[..., 0:2] += ray_chief.o[..., 0:2]
        op += ray_ep.t
        # the norm on the ep surface of each intersection
        n_p = ep_surf.surface_normal(x=ray_ep.o[..., 0], y=ray_ep.o[..., 1])
        cos_ref = torch.einsum('ij,ij->i', ray.d[valid], n_p) # [rays_num]
        # calculate the optical path difference of the sampled point on image
        # initialize the sample complex amplitude mat on image plane
        intensity_map = torch.zeros((image_sampling, image_sampling))
        wave_num = torch.tensor(
            2 * np.pi / wavelength.item(),
            dtype=self.dtype,
            device=self.device)  # 10000

        # calculate the complex amplitude of each sampled image position
        line_sample = torch.linspace(- int((intensity_map.shape[0] - 1) / 2), int(
            (intensity_map.shape[0] - 1) / 2), image_sampling, device=self.device) * image_delta
        x, y = torch.meshgrid(line_sample, line_sample, indexing='ij')
        rel_coor = ray_chief.o + torch.stack([x, y, torch.zeros_like(x, dtype=self.dtype, device=self.device)], dim=-1)
        r = rel_coor[:, :, None, :] - ray_ep.o[None, None, ...]
        u_pi = r / length(r)[..., None] # [h_image, w_image, rays_num on ep, xyz]
        cos_pi = torch.einsum('ijkl,ijkl->ijk', u_pi, n_p[None, None, ...]) # [h, w, rays_num]
        k = - 0.5 * (cos_pi + cos_ref[None, None, ...].repeat(image_sampling, image_sampling, 1)) # [h, w, rays_num] 
        # inner production for the final length
        dr = torch.einsum('ijkl,ijkl->ijk', ray_ep.d[None, None, ...], r) # [h, w, rays_num]
        # complex amplitude
        amp = torch.einsum('ijk->ij', torch.exp(((op[None, None, :] + dr) * wave_num) * (0 + 1j)) * k)
        psf = torch.real(amp * torch.conj(amp)).permute(1, 0)
        psf = psf/psf.sum()
        return psf

    def mtf(self, pupil_sampling, image_sampling, image_delta,
            method='coherent', wavelength=None, view=None, show=False):
        if wavelength is None:
            wavelength = self.wavelengths[len(self.wavelengths)//2] 
        if view is None:
            view = self.views[-1]

        if method == 'coherent':
            psf = self.psf_coherent(
                pupil_sampling,
                image_sampling,
                image_delta,
                wavelength,
                view)
        elif method == 'huygens':
            psf = self.psf_huygens(
                pupil_sampling,
                image_sampling,
                image_delta,
                wavelength,
                view)
        elif method == 'kirchoff':
            psf = self.psf_kirchoff(
                pupil_sampling,
                image_sampling,
                image_delta,
                wavelength,
                view)
        elif method == 'spots':
            psf = self.psf_spot(
                pupil_sampling,
                image_sampling,
                image_delta,
                wavelength,
                view)
        elif method == 'gausslets':
            raise NotImplementedError
        else:
            raise Exception('method={} is not available!'.format(method))

        # if (image_sampling % 2) == 0:
        num_points = np.int64(np.ceil(np.log2(image_sampling / 32) + 1) * 50)
        pad_points = np.int64(num_points - image_sampling / 2)
        # else:
        #     num_points = np.int64(np.ceil(np.log2((image_sampling) / 32) + 1) * 50)+1
        #     pad_points = np.int64(num_points - (image_sampling) / 2)
        
        psf = torch.nn.functional.pad(psf, [pad_points, pad_points, pad_points, pad_points])

        T = torch.abs(torch.fft.fftshift(torch.fft.fft2(psf)))[
            :, num_points - image_sampling % 2][num_points - image_sampling % 2:]
        S = torch.abs(torch.fft.fftshift(torch.fft.fft2(psf)))[
            num_points - image_sampling % 2, :][num_points - image_sampling % 2:]
        freq = torch.linspace(0, 1 / image_delta / 2, num_points)

        if show:
            plt.plot(freq, T, '-', label='T')
            plt.plot(freq, S, '-.', label='S')
            plt.xlim(0)
            plt.ylim(0)
            plt.legend()

        return freq, T, S

    def mtf_loss_from_psf(self, psf, max_freq=500):
        """calculate mtf loss from psf"""
        num_points = max_freq // 5
        pad_points = np.int64(num_points - psf.shape[0] + 1) // 2
        psf = torch.nn.functional.pad(psf, [pad_points, pad_points, pad_points, pad_points])

        T = torch.abs(torch.fft.fft2(psf))[0:num_points//2, 0]
        S = torch.abs(torch.fft.fft2(psf))[0, 0:num_points//2]

        # 四分之一奈奎斯特频率
        # loss_mtf = -T[T.shape[0]//4] - S[S.shape[0]//4]
        loss_mtf = -torch.abs(torch.fft.fft2(psf)).sum()
        return loss_mtf

    def mtf_huygens_through_focus(self,
                                  pupil_sampling,
                                  image_sampling,
                                  image_delta,
                                  frequency,
                                  delta_focus,
                                  steps,
                                  wavelength=None,
                                  view=None):
        """
        Computes the diffraction modulation transfer function (MTF) data
        using Huygens PSFs and displays the data as a function of delta focus.

        pupil_sampling: the size of the grid of rays to trace to perform the computation.
        image_sampling: The size of the grid of points on which to compute the diffraction image intensity.
        image_delta: The distance in micrometers between points in the image grid.
        frequency: The spatial frequency (cycle per micrometers).
        wavelength: The wavelength number to be used in the calculation.
        view: field of view.
        delta_focus: delta focus is the ± Z-axis range of the plot in micrometers.
        steps: The number of focal planes at which the data is computed.
        todo: through focus psf, such as edit ep_surf for different defocus; zrr: too tired to clean code, try later
        """
        if wavelength is None:
            wavelength = self.wavelengths[len(self.wavelengths)//2] 
        if view is None:
            view = self.views[-1]

        # rays sampling
        ray, ray_chief = self.image_cache(
            pupil_sampling, view, wavelength, 'hexapolar')

        # optical path calculation with propagate
        valid = torch.ones_like(ray.t).bool()
        valid_chief = torch.ones_like(ray_chief.t).bool()
        ## if input object plane is infinite and tilt       ##
        ## we assume the optical path of the chief ray is 0 ##
        op = torch.einsum('ij,ij->i', ray.o - ray_chief.o, ray_chief.d)
        op_chief = torch.einsum(
            'ij,ij->i', ray_chief.o - ray_chief.o, ray_chief.d)

        # propagate to the last surface of lens (before image plane)
        for s in self.system[1:None]:
            valid, ray = s.propagate(ray, valid)
            op += ray.t * s.refractive_index_prev(wavelength)
            valid_chief, ray_chief = s.propagate(ray_chief, valid_chief)
            op_chief += ray_chief.t * s.refractive_index_prev(wavelength)

        # calculate the intersection of ray and optical axis to judge the exit pupil
        # NOTE: Only y direction cosine calculation is supported

        if view == 0:
            self.system._paraxial_info(self.views)
            ep_roc = - self.system.Exit_Pupil_Position
        else:
            ep_roc = ray_chief.o[..., 1] / ray_chief.d[..., 1]
            ep_roc = ep_roc.squeeze()

        # origin_exit_pupil = ray_chief.o - t_ep * ray_chief.d
        ep_distance = self.system[-1].d - ep_roc
        ep_distance_prev = -ep_roc
        ep_radius = 2. * ep_roc

        ep_surf = Spheric(roc=ep_roc,
                          conic=None,
                          radius=ep_radius,
                          distance=ep_distance,
                          distance_prev=ep_distance_prev,
                          distance_after=-ep_distance_prev,
                          material='vacuum',
                          material_prev='vacuum'
                          )  # exit pupil surface related to the center of sensor
        # shift rays on image plane to the center of sensor
        op = op[valid]
        o_final = torch.zeros_like(
            ray.o[valid],
            dtype=self.dtype,
            device=self.device)
        o_final[..., 0:2] = ray.o[valid][..., 0:2] - ray_chief.o[..., 0:2]
        o_final[..., 2] = ray.o[valid][..., 2]
        # form the ray backpropagate to the sensor plane
        ray_shift = Ray(o=o_final, d=ray.d[valid], wavelength=ray.wavelength,
                        dtype=self.dtype, device=self.device)
        ray_ep = ep_surf.propagate(ray_shift, torch.ones_like(ray_shift.t).bool())[1]
        ray_ep.o[..., 0:2] += ray_chief.o[..., 0:2]
        op += ray_ep.t

        wave_num = torch.tensor(
            2 * np.pi / wavelength.item(),
            dtype=self.dtype,
            device=self.device)  # 10000
        MTF_T_through_focus = torch.zeros(
            steps, dtype=self.dtype, device=self.device)
        MTF_S_through_focus = torch.zeros(
            steps, dtype=self.dtype, device=self.device)
        # traverse the defocus position along the +-Z-axis
        z_idx = 0
        for z_delta in torch.linspace(-delta_focus, delta_focus, steps):
            # calculate the optical path difference of the sampled point on image
            # initialize the sample complex amplitude mat on image plane
            intensity_map = torch.zeros((image_sampling, image_sampling))

            # calculate the complex amplitude of each sampled image position
            line_sample = torch.linspace(- int((intensity_map.shape[0] - 1) / 2),
                                         int((intensity_map.shape[0] - 1) / 2),
                                         image_sampling) * image_delta
            x, y = torch.meshgrid(line_sample, line_sample, indexing='ij')
            # calculate the chief ray position after sensor plane has shifted
            t_delta = z_delta / ray_chief.d[..., 2]
            o_delta = ray_chief.o + t_delta * ray_chief.d
            rel_coor = o_delta + \
                torch.stack([x, y, torch.zeros_like(x, dtype=self.dtype, device=self.device)], dim=-1)
            r = rel_coor[:, :, None, :] - ray_ep.o[None, None, ...]
            # inner production for the final length
            dr = torch.einsum('ijkl,ijkl->ijk', ray_ep.d[None, None, ...], r)
            # complex amplitude
            amp = torch.einsum(
                'ijk->ij', torch.exp(((op[None, None, :] + dr) * wave_num) * (0 + 1j)))
            psf = torch.real(amp * torch.conj(amp)).permute(1, 0)
            psf = psf / psf.sum()

            num_points = np.int64(
                np.ceil(
                    np.log2(
                        image_sampling /
                        32) +
                    1) *
                50)
            pad_points = np.int64(num_points - image_sampling / 2)
            psf = torch.nn.functional.pad(
                psf, [pad_points, pad_points, pad_points, pad_points])
            # MTF in T and S directions
            T = torch.abs(torch.fft.fftshift(torch.fft.fft2(psf)))[
                :, num_points][num_points:]
            S = torch.abs(torch.fft.fftshift(torch.fft.fft2(psf)))[
                num_points, :][num_points:]
            freq = torch.linspace(0, 1 / image_delta / 2, num_points)
            # find the position for interpolation
            freq_idx = (freq < frequency).sum()  # the number of true
            T_itp = (T[freq_idx-1] * (freq[freq_idx] - frequency) + T[freq_idx] * (frequency - freq[freq_idx-1]))/\
                    (freq[freq_idx] - freq[freq_idx-1])
            S_itp = (S[freq_idx-1] * (freq[freq_idx] - frequency) + S[freq_idx] * (frequency - freq[freq_idx-1]))/\
                    (freq[freq_idx] - freq[freq_idx-1])

            MTF_T_through_focus[z_idx] = T_itp
            MTF_S_through_focus[z_idx] = S_itp
            z_idx += 1

        return MTF_T_through_focus, MTF_S_through_focus

    def relative_illumination(self, field_density, wavelength=None):
        """
        The Relative Illumination analysis computes the relative illumination
        as a function of radial field coordinate for a uniform Lambertian scene.

        Field_density: The number of points along the radial field coordinate
            to compute the relative illumination for. Larger field densities yield smoother curves.

        Wavelength: Selects the wavelength for computation.
            Relative illumination is a monochromatic entity.
        """
        if wavelength is None:
            wavelength = self.wavelengths[len(self.wavelengths)//2] 

        # update the field for calculation
        ri_list = []
        # then go through the sampled field of view
        field_sample = torch.linspace(0., self.views[-1], field_density)
        for fov_idx, fov in enumerate(field_sample):
            self.system[0].update(fov)
            R = np.tan(self.system[0].angle_fov) * self.system[1].surface(self.system[1].r, 0.) + self.system[1].r
            APERTURE_SAMPLING = 101  # enumerate sampling density
            x_e, y_e = torch.meshgrid(torch.linspace(-R, R, APERTURE_SAMPLING, dtype=self.dtype),
                                      torch.linspace(-R, R, APERTURE_SAMPLING, dtype=self.dtype),
                                      indexing='ij')
            x_e, y_e = x_e.reshape(APERTURE_SAMPLING * APERTURE_SAMPLING, -
                                   1), y_e.reshape(APERTURE_SAMPLING * APERTURE_SAMPLING, -1)
            o = torch.stack((x_e, y_e, torch.zeros_like(x_e)), dim=2)
            o = o.reshape(APERTURE_SAMPLING * APERTURE_SAMPLING, -1)
            d = self.system[0].map().unsqueeze(0).\
                repeat(APERTURE_SAMPLING, APERTURE_SAMPLING, 1).reshape(APERTURE_SAMPLING * APERTURE_SAMPLING, -1)
            ray_to_image = Ray(o, d, wavelength=wavelength)
            valid, _ = self.system.propagate(ray_to_image, start=1, stop=None)  # to the final plane of the optical system

            # find the valid position on image plane
            x_valid, y_valid = x_e[valid], y_e[valid]
            ctr_ray_fov_o = torch.tensor([x_valid.mean(), y_valid.mean(), 0.])[None, ...]  # chief ray
            top_ray_fov_o = torch.tensor([x_valid.mean(), y_valid.max(), 0.])[None, ...]  # top ray
            bot_ray_fov_o = torch.tensor([x_valid.mean(), y_valid.min(), 0.])[None, ...]  # bot ray
            rit_ray_fov_o = torch.tensor([x_valid.min(), y_valid.mean(), 0.])[None, ...]  # rit ray

            ray_fov_o = torch.concatenate([top_ray_fov_o, ctr_ray_fov_o, bot_ray_fov_o, rit_ray_fov_o], dim=0)
            ray_fov = Ray(ray_fov_o, self.system[0].map()[None, ...].repeat(4, 1), wavelength=wavelength)
            valid, ray_fov = self.system[1].propagate(ray_fov, torch.ones_like(ray_fov.t).bool())  # intersection on the first surface

            # calculate the intersection from the propagate
            xwa_t1 = ray_fov.o[3, 0]
            ywb_t1, zwb_t1, ywa_t1, zwa_t1 = ray_fov.o[0,1], ray_fov.o[0, 2], ray_fov.o[2, 1], ray_fov.o[2, 2]
            fov_cosine = ray_fov.d[1, 2]
            norm = self.system[1].surface_normal(ray_fov.o[1, 0], ray_fov.o[1, 1])
            theta_cosine = torch.sum(ray_fov.d[1, ...] * -norm, dim=-1) / (length(ray_fov.d[1, ...]) * length(-norm))

            if fov_idx == 0:
                y0a_t1 = ray_fov.o[0, 1]  # y of the top ray
            # relative illumination
            ri = (torch.abs(theta_cosine) * (fov_cosine ** 3) * torch.abs(xwa_t1) *
                  torch.sqrt((ywb_t1 - ywa_t1) ** 2 + (zwb_t1 - zwa_t1) ** 2)) / \
                (2 * y0a_t1 ** 2)
            ri_list.append(ri)

        return ri_list

    # ====================================================================================
    # Updates
    # ====================================================================================
    @torch.no_grad()
    def update_radius(self):
        """
        Fix the radius of pupil, update the radius of the all elements
        Ensuring the rays of all wavelengths in the largest fov could pass
        """
        # update the field-of-view of system
        self.system[0].update(torch.max(self.views))
        R_up = - torch.sign(self.system[0].angle_fov) * torch.tan(self.system[0].angle_fov) * \
            self.system[1].surface(self.system[1].r, 0.) + self.system[1].r
        R_dn = - torch.sign(self.system[0].angle_fov) * torch.tan(self.system[0].angle_fov) * \
            self.system[1].surface(self.system[1].r, 0.) - self.system[1].r
        # R_up =  torch.tan(self.system[0].angle_fov) * self.system[1].surface(self.system[1].r, 0.) + self.system[1].r
        APERTURE_SAMPLING = 1001

        # here we sample ray on the first plane for pupil calculation
        y_e = torch.linspace(R_dn, R_up, APERTURE_SAMPLING, dtype=self.dtype, device=self.device)
        o = torch.stack((torch.zeros_like(y_e), y_e, torch.zeros_like(y_e)), dim=1)
        d = self.system[0].map().unsqueeze(0).repeat(APERTURE_SAMPLING, 1).reshape(
                APERTURE_SAMPLING, -1)
        wavelength = self.wavelengths[len(self.wavelengths) // 2] # chief wavelength
        ray_to_pupil = Ray(o, d, wavelength=wavelength)
        valid, _ = self.system.propagate(
            ray_to_pupil, start=1, stop=self.system.aperture_ind)

        r_pupil_max = torch.max(ray_to_pupil.o[valid, 1])
        r_pupil_min = torch.min(ray_to_pupil.o[valid, 1])
        pupil_cache = [r_pupil_min, r_pupil_max]

        # r_max = [] # record the max radius of different surface
        y_e = torch.linspace(pupil_cache[0], pupil_cache[1],
                             APERTURE_SAMPLING, dtype=self.dtype, device=self.device)
        o = torch.stack((torch.zeros_like(y_e), y_e, torch.zeros_like(y_e)), dim=1)
        d = self.system[0].map().unsqueeze(0).repeat(APERTURE_SAMPLING, 1).reshape(APERTURE_SAMPLING, -1)
        ray = Ray(o, d, wavelength=wavelength)

        valid = None
        for idx, surf in enumerate(self.system[1:None]):
            valid, ray = surf.propagate(ray, valid)
            o_valid = ray.o[valid].clone()
            r_valid_max = torch.max(torch.sqrt(torch.sum(torch.square(o_valid[..., 0:2]), dim=1)))
            o_invalid = ray.o[~valid].clone()
            if o_invalid.shape[0] != 0:
                o_invalid = torch.where(torch.isnan(o_invalid),
                                        torch.tensor(0., dtype=self.dtype, device=self.device),
                                        o_invalid)
                r_invalid_max = torch.max(torch.sqrt(torch.sum(torch.square(o_invalid[..., 0:2]), dim=1)))
            else:
                r_invalid_max = torch.tensor(0., dtype=self.dtype, device=self.device)

            # r_max.append(torch.max(r_valid_max, r_invalid_max)) # max radius of this surface
            r_max = torch.max(r_valid_max, r_invalid_max)
            if surf.r < torch.max(r_valid_max, r_invalid_max):
                # set the max radius as the image plane
                setattr(self.system[idx+1], 'r', r_max)

    # ====================================================================================
    # Reports
    # ====================================================================================
