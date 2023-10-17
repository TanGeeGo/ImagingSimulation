import torch
import numpy as np
from .utils import PrettyPrinter, normalize, sagittal_meridional

# ==========================================================================================
# Basic
# ==========================================================================================

class NameMixin:
    _types = {}
    _default_type = None
    _nickname = None
    _type = None
    _typeletter = None

    @classmethod
    def register(cls, sub):
        if sub._type is None:
            sub._type = sub.__name__.lower()
        k = cls, sub._type
        assert k not in cls._types, (k, sub, cls._types)
        cls._types[k] = sub
        return sub

    def dict(self):
        dat = {}
        if self._type != self._default_type:
            dat["type"] = self._type
        if self._nickname:
            dat["nickname"] = self.nickname
        return dat

    @classmethod
    def make(cls, data):
        if isinstance(data, cls):
            return data
        typ = data.pop("type", cls._default_type)
        sub = cls._types[(cls, typ)]
        return sub(**data)

    @property
    def type(self):
        return self._type

    @property
    def typeletter(self):
        return self._typeletter or self._type[0].upper()

    @property
    def nickname(self):
        return self._nickname or hex(id(self))

    @nickname.setter
    def nickname(self, name):
        self._nickname = name

    def __str__(self):
        return f"<{self.typeletter}/{self.nickname}>"

# ==========================================================================================
# Pupil 
# ==========================================================================================
class Pupil(NameMixin):
    _default_type = 'radius'

    def __init__(self, distance=1., update_distance=True,
                 update_radius=True, aim=True, telecentric=False,
                 refractive_index=1., projection="rectilinear"):
        self.distance = distance
        self.update_distance = update_distance
        self.update_radius = update_radius
        self.refractive_index = refractive_index
        self.aim = aim
        self.telecentric = telecentric
        self.projection = projection

    def rescale(self, scale):
        self.distance *= scale

    def update(self, distance, radius):
        if self.update_distance:
            self.distance = distance
        if self.update_radius:
            self.radius = radius

    def dict(self):
        dat = super().dict()
        dat["distance"] = float(self.distance)
        if not self.update_distance:
            dat["update_distance"] = self.update_distance
        if self.update_radius:
            dat["update_radius"] = self.update_radius
        if self.aim:
            dat["aim"] = self.aim
        if self.projection != "rectilinear":
            dat["projection"] = self.projection
        if self.telecentric:
            dat["telecentric"] = self.telecentric
        if self.refractive_index != 1.:
            dat["refractive_index"] = float(self.refractive_index)
        return dat

    def text(self):
        yield "Pupil Distance: %g" % self.distance
        if self.telecentric:
            yield "Telecentric: %s" % self.telecentric
        if self.refractive_index != 1.:
            yield "Refractive Index: %g" % self.refractive_index
        if self.projection != "rectilinear":
            yield "Projection: %s" % self.projection
        if not self.update_distance:
            yield "Track Distance: %s" % self.update_distance
        if self.update_radius:
            yield "Update Radius: %s" % self.update_radius
        if self.aim:
            yield "Aim: %s" % self.aim

    @property
    def radius(self):
        return self.slope*self.distance

    @property
    def slope(self):
        return self.radius/self.distance

    @property
    def na(self):
        return self.sinarctan(self.slope)*self.refractive_index

    @property
    def fno(self):
        return 1/(2.*self.na)

    def sinarctan(self, u, v=None):
        u2 = np.square(u)
        if u2.ndim == 2:
            if u2.shape[1] >= 3:
                v = u[:, 3]
                u, u2 = u[:, :2], u2[:, :2]
            u2 = u2.sum(1)[:, None]
        u2 = 1/np.sqrt(1 + u2)
        u1 = u*u2
        if v is not None:
            u1 = np.concatenate((u1, np.sign(v)[:, None]*u2), axis=1)
        return u1

    def map(self, y, a, filter=True):
        # FIXME: projection
        # a = [[-sag, -mer], [+sag, +mer]]
        am = np.fabs(a).max()
        y = np.atleast_2d(y)*am
        if filter:
            c = np.sum(a, axis=0)/2
            d = np.diff(a, axis=0)/2
            r = ((y - c)**2/d**2).sum(1)
            y = y[r <= 1]
        return y

@Pupil.register
class RadiusPupil(Pupil):
    _type = "radius"
    radius = None

    def __init__(self, radius=1., **kwargs):
        super().__init__(**kwargs)
        self.radius = radius

    def dict(self):
        dat = super().dict()
        dat["radius"] = float(self.radius)
        return dat

    def text(self):
        yield from super().text()
        yield "Radius: %g" % self.radius

    def rescale(self, scale):
        super().rescale(scale)
        self.radius *= scale

# ==========================================================================================
# Conjugates 
# ==========================================================================================
class Conjugate(PrettyPrinter):
    _types = {}
    _default_type = 'infinite'
    _nickname = None
    _type = None
    _typeletter = None
    finite = None

    def __init__(self, 
                 pupil=None, 
                 projection='rectilinear',
                 update_radius=False,
                 dtype=torch.float64,
                 device=torch.device('cpu')):
        if pupil is None:
            self.pupil = RadiusPupil(radius=0.)
        self.projection = projection 
        self.update_radius = update_radius
        self.dtype = dtype
        self.device = device

    @property
    def wideangle(self):
        # FIXME: elaborate this
        return self.projection != "rectilinear"

    def text(self):
        if self.projection != "rectilinear":
            yield "Projection: %s" % self.projection
        if self.update_radius:
            yield "Update Radius: %s" % self.update_radius
        yield "Pupil:"
        for _ in self.pupil.text():
            yield "  %s" % _

    def dict(self):
        dat = super().dict()
        dat["pupil"] = self.pupil.dict()
        if self.projection != "rectilinear":
            dat["projection"] = self.projection
        return dat

    def rescale(self, scale):
        self.pupil.rescale(scale)

    def aim(self, xy, pq, z=None, a=None):
        """
        xy 2d fractional xy object coordinate (object knows meaning)
        pq 2d fractional sagittal/meridional pupil coordinate

        aiming should be aplanatic (the grid is by solid angle
        in object space) and not paraxaxial (equal area in entrance
        beam plane)

        z pupil distance from "surface 0 apex" (also for infinite object)
        a pupil aperture (also for infinite object or telecentric pupils,
        then from z=0)

        if z, a are not provided they are takes from the (paraxial data) stored
        in object/pupil
        """
        raise NotImplementedError
    
@Conjugate.register
class FiniteConjugate(Conjugate):
    _type = "finite"
    finite = True

    def __init__(self, radius=0., **kwargs):
        super().__init__(**kwargs)
        self.radius = radius

    @property
    def point(self):
        return not self.radius

    @property
    def slope(self):
        return self.radius/self.pupil.distance

    @slope.setter
    def slope(self, c):
        self.radius = self.pupil.distance*c

    def dict(self):
        dat = super().dict()
        if self.radius:
            dat["radius"] = float(self.radius)
        return dat

    def text(self):
        yield "Radius: %.3g" % self.radius
        yield from super().text()

    def update(self, radius, pupil_distance, pupil_radius):
        self.pupil.update(pupil_distance, pupil_radius)
        if self.update_radius:
            self.radius = radius

    def rescale(self, scale):
        super().rescale(scale)
        self.radius *= scale

    def aim(self, yo, yp=None, z=None, a=None, surface=None, filter=True):
        if z is None:
            z = self.pupil.distance
        yo = np.atleast_2d(yo)
        if yp is not None:
            if a is None:
                a = self.pupil.radius
                a = np.array(((-a, -a), (a, a)))
            a = np.arctan2(a, z)
            yp = np.atleast_2d(yp)
            yp = self.pupil.map(yp, a, filter)
            yp = z*np.tan(yp)
            yo, yp = np.broadcast_arrays(yo, yp)

        y = np.zeros((yo.shape[0], 3))
        y[..., :2] = -yo*self.radius
        if surface is not None:
            y[..., 2] = -surface.surface_sag(y)
        uz = (0, 0, z)
        if self.pupil.telecentric:
            u = uz
        else:
            u = uz - y
        if yp is not None:
            s, m = sagittal_meridional(u, uz)
            u += yp[..., 0, None]*s + yp[..., 1, None]*m
        normalize(u)
        if z < 0:
            u *= -1
        return y, u

@Conjugate.register
class InfiniteConjugate(Conjugate):
    _type = "infinite"
    finite = False

    def __init__(self, index=0, angle_fov=14., angle_azimuth=None, **kwargs):
        super().__init__(**kwargs)
        self.index = index
        self.angle_fov = angle_fov * np.pi / 180.
        if angle_azimuth is None:
            self.azimuth = 0. # direction from -y to +y direction
        else:
            # angle from the direction vector's projecton to -y axis
            self.azimuth = angle_azimuth * np.pi / 180.

    @property
    def slope(self):
        return np.tan(self.angle_fov)

    def dict(self):
        dat = super().dict()
        if self.angle_fov:
            dat["angle_fov"] = float(self.angle_fov)
        if self.angle_azimuth:
            dat["angle_azimuth"] = float(self.angle_azimuth)
        return dat

    def update(self, angle_fov, angle_azimuth=None):
        self.angle_fov = torch.tensor((angle_fov * np.pi / 180.).item(), dtype=self.dtype, device=self.device)
        if angle_azimuth is None:
            self.azimuth = torch.tensor(0., dtype=self.dtype, device=self.device) # direction from -y to +y direction
        else:
            # angle from the direction vector's projecton to -y axis
            self.azimuth = torch.tensor(angle_azimuth * np.pi / 180., dtype=self.dtype, device=self.device)

    def text(self):
        yield "Semi-Angle: %.3g deg" % self.angle_fov
        yield from super().text()

    def map(self):
        """
        Get the direction from the fov and the azimuth
        """
        # assuming the length of direction vector is 1
        z_proj = 1 * np.cos(self.angle_fov)
        x_proj = 1 * np.sin(self.angle_fov) * np.sin(self.azimuth)
        y_proj = 1 * np.sin(self.angle_fov) * np.cos(self.azimuth)
        # form tensor of direction (not need for normalization)
        return normalize(torch.tensor([x_proj, y_proj, z_proj], dtype=self.dtype, device=self.device))

    def aim(self, yo, yp=None, z=None, a=None, surface=None, filter=True, l=None, n0=1.000):
        if z is None:
            z = self.pupil.distance
        yo = np.atleast_2d(yo)
        u = self.map(yo, self.angle)
        if yp is not None:
            if a is None:
                a = self.pupil.radius + 0.6412080331894 * np.tan(np.arccos(u[0, -1]))
                a = np.array(((-a, -a), (a, a)))
            
            yp = np.atleast_2d(yp)
            yp = self.pupil.map(yp, a, filter)
            yo, yp = np.broadcast_arrays(yo, yp)
        # u = np.expand_dims(u, 0).repeat(num_rays, axis=0).squeeze(1)  # 不用这么麻烦，直接重新算！
        u = self.map(yo, self.angle)
        # print('u:', u)
        yz = (0, 0, z)
        y = yz - z * u
        y[:, 0:2] += yp
        # if yp is not None:
        #     s, m = sagittal_meridional(u, yz)
        #     y += yp[..., 0, None]*s + yp[..., 1, None]*m

        if surface is not None:
            y += surface.intercept(y, u)[..., None]*u
            i = u
            n, mu = surface.get_n_mu(n0, l)
            if mu:
                u = surface.refract(y, i, mu)

        return y, u
