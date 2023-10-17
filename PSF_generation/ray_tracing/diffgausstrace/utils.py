import numpy as np
import torch
import math

fraunhofer = dict(   # http://en.wikipedia.org/wiki/Abbe_number
    i=365.01e-6,  # Hg UV
    h=404.66e-6,  # Hg violet
    g=435.84e-6,  # Hg blue
    Fp=479.99e-6,  # Cd blue
    F=486.1327e-6,  # H  blue
    e=546.07e-6,  # Hg green
    Gy=555.00e-6,  # greenish-yellow
    d=587.5618e-6,  # He yellow
    D=589.30e-6,  # Na yellow
    Cp=643.85e-6,  # Cd red
    C=656.2725e-6,  # H  red
    r=706.52e-6,  # He red
    Ap=768.20e-6,  # K  IR
    s=852.11e-6,  # Cs IR
    t=1013.98e-6,  # Hg IR
)  # unit: [mm]

lambda_F = fraunhofer["F"]
lambda_d = fraunhofer["d"]
lambda_C = fraunhofer["C"]

class PrettyPrinter:
    """ 
    Basic class for printing information and to device operations
    """

    def __str__(self):
        lines = [self.__class__.__name__ + ':']
        for key, val in vars(self).items():
            if val.__class__.__name__ in ('list', 'tuple'):
                for i, v in enumerate(val):
                    lines += '{}[{}]: {}'.format(key, i, v).split('\n')

            elif val.__class__.__name__ in 'dict':
                pass
            elif key == key.upper() and len(key) > 5:
                pass
            else:
                lines += '{}: {}'.format(key, val).split('\n')
        return '\n    '.join(lines)

    def to(self, device=torch.device('cpu')):
        for key, val in vars(self).items():
            if torch.is_tensor(val):
                exec('self.{x} = self.{x}.to(device)'.format(x=key))
            elif issubclass(type(val), PrettyPrinter):
                exec(f'self.{key}.to(device)')
            elif val.__class__.__name__ in ('list', 'tuple'):
                for i, v in enumerate(val):
                    if torch.is_tensor(v):
                        exec(
                            'self.{x}[{i}] = self.{x}[{i}].to(device)'.format(
                                x=key, i=i))
                    elif issubclass(type(v), PrettyPrinter):
                        exec('self.{}[{}].to(device)'.format(key, i))

    @classmethod
    def register(cls, sub):
        if sub._type is None:
            sub._type = sub.__name__.lower()
        k = cls, sub._type
        assert k not in cls._types, (k, sub, cls._types)
        cls._types[k] = sub
        return sub


class Ray(PrettyPrinter):
    """
    Definition of a geometric ray.

    - o is the ray position
    - d is the ray direction (normalized)
    - t is the optical path (accumulated during propagate)
    """

    def __init__(self, o, d, wavelength,
                 dtype=torch.float64,
                 device=torch.device('cpu')):
        self.dtype = dtype
        self.device = device
        self.o = o.to(self.dtype) if torch.is_tensor(o) else \
            torch.tensor(np.asarray(o), dtype=self.dtype).to(self.device)
        self.d = d.to(self.dtype) if torch.is_tensor(d) else \
            torch.tensor(np.asarray(d), dtype=self.dtype).to(self.device)
        self.t = torch.zeros(
            (self.o.shape[0:-1]), dtype=self.dtype).to(self.device)

        # scalar-version
        self.wavelength = wavelength.to(self.dtype) if torch.is_tensor(wavelength) else \
            torch.tensor(wavelength, dtype=self.dtype).to(self.device)  # unit：mm
        self.mint = torch.tensor(1e-9, dtype=self.dtype).to(self.device)  # 0.001nm
        self.maxt = torch.tensor(1e3, dtype=self.dtype).to(self.device)  # 1m
        self.to(self.device)  # better to device individually

    def __call__(self, t):
        # invoke to return the point under this t
        return self.o + t[..., None] * self.d

    def clone(self):
        # clone a new ray for later propagate
        return Ray(o=self.o.clone().detach(), 
                   d=self.d.clone().detach(), 
                   wavelength=self.wavelength.clone().detach())

class Transformation(PrettyPrinter):
    """
    Rigid Transformation.

    - R is the rotation matrix.
    - t is the translational vector.
    """

    def __init__(self, R, t, dtype=torch.float64, device=torch.device('cpu')):
        self.dtype = dtype
        self.device = device
        self.R = R if torch.is_tensor(R) else \
            torch.tensor(np.asarray(R), dtype=self.dtype).to(self.device)
        self.t = t if torch.is_tensor(t) else \
            torch.tensor(np.asarray(t), dtype=self.dtype).to(self.device)

    def transform_point(self, o):
        # save the hw dimension even if only one ray
        return (self.R @ o[..., None]).squeeze(-1) + self.t

    def transform_vector(self, d):
        # save the hw dimension even if only one ray
        return (self.R @ d[..., None]).squeeze(-1)

    def transform_ray(self, ray):
        o = self.transform_point(ray.o)
        d = self.transform_vector(ray.d)
        if o.is_cuda:
            return Ray(o, d, ray.wavelength, device=torch.device('cuda'))
        else:
            return Ray(o, d, ray.wavelength)

    def inverse(self):
        RT = self.R.T
        t = self.t
        return Transformation(RT, -RT @ t)

    # def propa_tran(self):
    #     R = self.R
    #     t = torch.zeros_like(self.t)
    #     t[0] = self.t[0]
    #     t[1] = self.t[1]
    #     return Transformation(R, t)


# ----------------------------------------------------------------------

class Material(PrettyPrinter):
    """
    Optical materials for computing the refractive indices.

    support several categories of material

    1. first is the abbe material, where the formulation can easily expressed as:

    n(\lambda) = A + (\lambda - \lambda_ref) / (\lambda_long - \lambda_short) * (1 - A) / B

    where the two constants A and B can be computed from nD (index at 589.3 nm) and V (abbe number).

    2. second is the coeff material, where the formulation is defined by different equations:

    schott / sellmeier_1 / sellmeier_squared / sellmeier_squared_transposed / conrady / ...

    for more details of the coeff material, please refer to the following.

    """

    def __init__(self, name=None, dtype=torch.float64, device=torch.device('cpu')):
        self.name = 'vacuum' if name is None else name.lower()
        self.dtype = dtype
        self.device = device

        # format the material as follows:
        # "material_name": {
        #   "type": <type-of-material>, e.g., "abbe", "gas", "sellmeier_1", ...
        #   "coeff": <coefficient-of-material>,
        #   e.g., [nd, abbe number] for "abbe", [K1, L1, K2, L2, K3, L3] for "sellmeier_1", ...
        # }
        self.MATERIAL_TABLE = {
            "vacuum": {
                "type": "abbe",
                "coeff": [1., math.inf],  # [nd, abbe number]
            },
            "air": {
                "type": "gas",
                # [1.000293, math.inf]
                "coeff": [.05792105, .00167917, 238.0185, 57.362]
            },
            "occluder": [1.,  math.inf],
            "f2": [1.620, 36.37],
            "f15": [1.60570, 37.831],
            "uvfs": [1.458, 67.82],

            # https://shop.schott.com/advanced_optics/
            "bk10": [1.49780, 66.954],
            "n-baf10": [1.67003, 47.11],
            "n-bk7": {
                "type": "sellmeier_1",
                "coeff": [1.03961212E+000, 6.00069867E-003, 2.31792344E-001,
                          2.00179144E-002, 1.01046945E+000, 1.03560653E+002],  # [1.51680, 64.17],
            },
            "n-sf1": {
                "type": "sellmeier_1",
                "coeff": [1.608651580E+00, 1.196548790E-02, 2.377259160E-01,
                          5.905897220E-02, 1.515306530E+00, 1.355216760E+02],  # [1.71736, 29.62],
            },
            "n-sf2": [1.64769, 33.82],
            "n-sf4": [1.75513, 27.38],
            "n-sf5": [1.67271, 32.25],
            "n-sf6": [1.80518, 25.36],
            "n-sf6ht": [1.80518, 25.36],
            "n-sf8": [1.68894, 31.31],
            "n-sf10": [1.72828, 28.53],
            "n-sf11": [1.78472, 25.68],
            "sf1": [1.71736, 29.51],
            "sf2": [1.64769, 33.85],
            "sf4": [1.75520, 27.58],
            "sf5": [1.67270, 32.21],
            "sf6": [1.80518, 25.43],
            "sf18": [1.72150, 29.245],

            # HIKARI.AGF
            "baf10": [1.67, 47.05],

            # SUMITA.AGF / SCHOTT.AGF
            "sk1": [1.61030, 56.712],
            "sk2": {
                "type": "sellmeier_1",
                "coeff": [1.281890120E+00, 7.271916400E-03, 2.577382580E-01,
                          2.428235270E-02, 9.681860400E-01, 1.103777730E+02],  # [1.6074, 56.65],
            },
            "sk16": {
                "type": "sellmeier_1",
                "coeff": [1.343177740E+00, 7.046873390E-03, 2.411443990E-01,
                          2.290050000E-02, 9.943179690E-01, 9.275085260E+01],  # [1.62040, 60.306],
            },
            "ssk4": [1.61770, 55.116],
            "f5": {
                "type": "sellmeier_1",
                "coeff": [1.310446300E+00, 9.586330480E-03, 1.960342600E-01,
                          4.576276270E-02, 9.661297700E-01, 1.150118830E+02],  # [1.6034, 38.03],
            },

            # https://www.pgo-online.com/intl/B270.html
            "b270": [1.52290, 58.50],

            # https://refractiveindex.info, nD at 589.3 nm
            "s-nph1": [1.8078, 22.76],
            "d-k59": [1.5175, 63.50],

            "flint": [1.6200, 36.37],
            "pmma": [1.491756, 58.00],
            "polycarb": [1.585470, 30.00],

            # honor20
            "mc-pcd4-40": {
                "type": "schott",
                "coeff": [2.58314710E+000, -9.75023440E-003, 1.36153740E-002,
                          3.63461220E-004, -2.11870820E-005, 1.15361320E-006],  # [1.6192, 63.855],
            },
            "ep9000": {
                "type": "schott",
                "coeff": [2.67158942E+000, -9.88033522E-003, 2.31490098E-002,
                          9.46210022E-003, -1.30260155E-003, 1.19691096E-004],  # [1.6707, 19.238],
            },
            "apl5014cl": {
                "type": "schott",
                "coeff": [2.39240344E+000, -3.17013191E-002, -1.76719919E-002,
                          9.49949989E-003, -1.27481919E-003, 6.65182214E-005],  # [1.5439, 55.951],
            },
            "k26r": {
                "type": "schott",
                "coeff": [2.39341385E+000, -5.52710285E-002, -3.05566524E-002,
                          1.20870398E-002, -1.51685332E-003, 7.48343683E-005],  # [1.5348, 55.664],
            },
        }

        self.material = self.MATERIAL_TABLE.get(self.name)
        if not torch.is_tensor(self.material['coeff']):
            self.material['coeff'] = \
                torch.tensor(self.material['coeff'],
                             dtype=self.dtype).to(self.device)

        self.nd = self.refractive_index(lambda_d).to(self.device)

    # ==========================================================================
    # return the refractive index and abbe number
    # ==========================================================================

    def refractive_index(self, wavelength):
        n_fn = getattr(self, "n_%s" % self.material['type'])
        n = n_fn(wavelength*1e3, self.material['coeff'])
        if 'mirror' in self.material.keys():
            n = -n if self.material['mirror'] else n

        # return torch.tensor(np.asarray(n), dtype=self.dtype, device=self.device)
        return n

    def abbe_number(self):
        return torch.tensor(np.asarray((self.refractive_index(lambda_d)-1) /
                                       (self.refractive_index(lambda_F) - self.refractive_index(lambda_C))),
                            dtype=self.dtype, device=self.device)

    # ==========================================================================
    # calculating refractive index with different dispersion equation
    # ==========================================================================

    def n_abbe(self, w, c):
        return c[0] + (w - lambda_d) / (lambda_C - lambda_F) * (1 - c[0]) / c[1]

    def n_schott(self, w, c):
        n = c[0] + c[1] * w**2
        for i, ci in enumerate(c[2:]):
            n += ci * w**(-2*(i + 1))
        return torch.sqrt(n)

    def n_sellmeier(self, w, c):
        w2 = w**2
        c0, c1 = c.reshape(-1, 2).T
        return torch.sqrt(1. + (c0*w2 / (w2 - c1**2)).sum())

    def n_sellmeier_1(self, w, c):
        w2 = w**2
        c0, c1 = c.reshape(-1, 2).T
        return torch.sqrt((1. + (c0*w2 / (w2 - c1)).sum()))

    def n_sellmeier_squared_transposed(self, w, c):
        w2 = w**2
        c0, c1 = c.reshape(2, -1)
        return torch.sqrt(1. + (c0*w2/(w2 - c1)).sum())

    def n_conrady(self, w, c):
        return c[0] + c[1]/w + c[2]/w**3.5

    def n_herzberger(self, w, c):
        l = 1./(w**2 - .028)
        return c[0] + c[1]*l + c[2]*l**2 + c[3]*w**2 + c[4]*w**4 + c[5]*w**6

    def n_sellmeier_offset(self, w, c):
        w2 = w**2
        c0, c1 = c[1:1 + (c.shape[0] - 1)//2*2].reshape(-1, 2).T
        return torch.sqrt(1. + c[0] + (c0*w2/(w2 - c1**2)).sum())

    def n_sellmeier_squared_offset(self, w, c):
        w2 = w**2
        c0, c1 = c[1:1 + (c.shape[0] - 1)//2*2].reshape(-1, 2).T
        return torch.sqrt(1. + c[0] + (c0*w2/(w2 - c1)).sum())

    def n_handbook_of_optics1(self, w, c):
        return torch.sqrt(c[0] + (c[1]/(w**2 - c[2])) - (c[3]*w**2))

    def n_handbook_of_optics2(self, w, c):
        return torch.sqrt(c[0] + (c[1]*w**2/(w**2 - c[2])) - (c[3]*w**2))

    def n_extended2(self, w, c):
        n = c[0] + c[1]*w**2 + c[6]*w**4 + c[7]*w**6
        for i, ci in enumerate(c[2:6]):
            n += ci*w**(-2*(i + 1))
        return torch.sqrt(n)

    def n_hikari(self, w, c):
        n = c[0] + c[1]*w**2 + c[2]*w**4
        for i, ci in enumerate(c[3:]):
            n += ci*w**(-2*(i + 1))
        return torch.sqrt(n)

    def n_gas(self, w, c):
        c0, c1 = c.reshape(2, -1)
        return 1. + (c0 / (c1 - w**-2)).sum()

    def n_gas_offset(self, w, c):
        return c[0] + self.n_gas(w, c[1:])

    def n_refractiveindex_info(self, w, c):
        c0, c1 = c[9:].reshape(-1, 2).T
        return torch.sqrt(c[0] + c[1]*w**c[2]/(w**2 - c[3]**c[4]) +
                       c[5]*w**c[6]/(w**2 - c[7]**c[8]) + (c0*w**c1).sum())

    def n_retro(self, w, c):
        w2 = w**2
        a = c[0] + c[1]*w2/(w2 - c[2]) + c[3]*w2
        return torch.sqrt(2 + 1/(a - 1))

    def n_cauchy(self, w, c):
        c0, c1 = c[1:].reshape(-1, 2).T
        return c[0] + (c0*w**c1).sum()

    def n_polynomial(self, w, c):
        return torch.sqrt(self.n_cauchy(w, c))

    def n_exotic(self, w, c):
        return torch.sqrt(c[0] + c[1]/(w**2 - c[2]) +
                       c[3]*(w - c[4])/((w - c[4])**2 + c[5]))

    # ==========================================================================
    # output API for checking
    # ==========================================================================

    def dict(self):
        dat = {}
        if self.name:
            dat['name'] = self.name
        if self.material['type']:
            dat['type'] = self.material['type']
        if self.material['coeff'] is not None:
            dat['coeff'] = self.material['coeff']
        if 'mirror' in self.material.keys():
            dat['mirror'] = self.material['mirror']

        return dat

    def __str__(self):
        material_info = self.dict()
        output_base = material_info['name'].upper() + '\n' + \
            f'Material type: ' + material_info['type'] + '\n' + \
            f'Material coeff: ' + \
            np.array_str(np.asarray(material_info['coeff'])) + '\n'
        if 'mirror' in material_info.keys():
            if material_info['mirror']:
                output_mirror = f'Material MIRROR: True' + '\n'
        else:
            output_mirror = f'Material MIRROR: False' + '\n'
        return output_base + output_mirror


def rodrigues_rotation_matrix(k, theta, dtype=torch.float64):  # theta: [rad]
    """
    This function implements the Rodrigues rotation matrix.
    旋转轴(kx,ky,kz) 旋转角度theta
    """
    # cross-product matrix
    kx, ky, kz = k[0], k[1], k[2]
    K = torch.tensor([
        [0., -kz, ky],
        [kz, 0., -kx],
        [-ky, kx, 0.]
    ]).to(k.device)
    if not torch.is_tensor(theta):
        theta = torch.tensor(np.asarray(theta), dtype=dtype).to(k.device)
    return torch.eye(3, device=k.device) + torch.sin(theta) * K + (1 - torch.cos(theta)) * K @ K


def length2(d):
    return torch.sum(d ** 2, dim=-1)


def length(d):
    return torch.sqrt(length2(d))


def normalize(d):
    return d / length(d)[..., None]


def sagittal_meridional(u, z):
    # 根据输入孔径角u和光轴方向z，利用两次点积来确定子午和弧矢矢量
    # if pupil is after the first surface
    if z[-1] < 0:
        z = (0, 0, -z[-1])
    s = torch.cross(u, z)
    axial = np.all(s == 0, axis=-1)[..., None]  # 判断子午方向是否出现异常
    s = np.where(axial, (1., 0, 0), s)
    m = np.cross(u, s)
    normalize(s)
    normalize(m)
    return s, m


def init():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("DiffMetrology is using: {}".format(device))
    torch.set_default_tensor_type('torch.FloatTensor')
    return device


def calculate_u(ray1, ray2):
    """计算ray1在参考ray2上的u"""
    return ray1.d - torch.mul(torch.einsum('...k,...k', ray1.d, ray2.d)[..., None], ray2.d)


def calculate_h(ray1, ray2, device=torch.device('cpu')):
    """计算ray1落在参考ray2横截面上的t,然后得出相对ray2.o的位置关系"""
    # if torch.allclose(ray1.o, ray2.o):
    #     return torch.Tensor([0, 0, 0]).to(device)
    t = -torch.einsum('...k,...k', ray2.d, ray1.o - ray2.o) / \
        torch.einsum('...k,...k', ray2.d, ray1.d)
    return ray1.o + torch.mul(t[..., None], ray1.d) - ray2.o


def calculate_hp(ray1, ray2, point):
    """计算ray1落在参考ray2过point横截面上的t,然后返回位置"""
    t = -torch.einsum('...k,...k', ray2.d, ray1.o - point) / \
        torch.einsum('...k,...k', ray2.d, ray1.d)
    return ray1.o + torch.mul(t[..., None], ray1.d)


def calculate_t(ray, point):
    """
    计算point落在ray横截面上的t
    point.shape(m,3)
    (ray.d ray.o).shape(n,3)
    return (rays,points,value).shape(n,m,1)
    v2 return (points,rays,value).shape(n,m,1)
    """
    # return -torch.einsum('...k,...k', ray.d, ray.o - point)
    t = torch.einsum('...ik,...jk->...ji', ray.d, point) - \
        torch.einsum('...ik,...ik->...i', ray.d, ray.o)
    # return (torch.einsum('...jk->...kj', t)).unsqueeze(-1)
    return t.unsqueeze(-1)


def make_sensor(radius, pixel_size, z):
    resolution = int(2 * radius / pixel_size)
    y_ = torch.linspace(-radius, radius, resolution)
    x_ = torch.linspace(-radius, radius, resolution)
    x, y = torch.meshgrid(x_, y_, indexing='ij')
    z = torch.ones_like(x) * z
    return torch.stack((x, y, z), dim=-1)


def make_psf_sample(x_width, y_width, x_offset, y_offset, z, x_size=31, y_size=31, device=torch.device('cpu')):
    x_ = torch.linspace(x_offset - x_width / 2, x_offset + x_width / 2, x_size)
    y_ = torch.linspace(y_offset - y_width / 2, y_offset + y_width / 2, y_size)
    x, y = torch.meshgrid(x_, y_, indexing='ij')
    z = torch.ones_like(x) * z
    return torch.stack((x, y, z), dim=-1).to(device)


def make_patch(x, y, z, pixel_size=1e-3, x_points=101, y_points=101, device=torch.device('cpu')):
    # 奇数的points准确划分 否则右边是points//2-1
    x_ = torch.linspace(x - pixel_size * (x_points // 2),
                        x + pixel_size * (x_points // 2), x_points).to(device)
    y_ = torch.linspace(y - pixel_size * (y_points // 2),
                        y + pixel_size * (y_points // 2), y_points).to(device)
    x, y = torch.meshgrid(x_, y_, indexing='ij')
    z = torch.ones_like(x).to(device) * z
    return torch.stack((x, y, z), dim=-1)


def pupil_distribution(rays_h, rays_w, distribution, dtype=torch.float64, device=torch.device('cpu')):
    """
    Calculate the ray sample under the normalized pupil distribution, e.g., "suqare", "triangular" 
    Represent the ray_pupil with x and y coordinates. Normalized, so [-1, 1]
    Input Args:
        rays_h: (int) number of rays in h direction, e.g., 101, 1001, ...
        rays_w: (int) number of rays in w direction, e.g., 101, 1001, ...
        distribution: (str) name of distribution, e.g., 'square' 
    Returns Args:
        o_p: (2D Torch.tensor) sampled rays on pupil
        ref: (int) chief ray index
    """
    d = distribution
    h = rays_h
    w = rays_w
    if d == 'square':
        h_p = torch.linspace(-1, 1, h, dtype=dtype, device=device)
        w_p = torch.linspace(-1, 1, w, dtype=dtype, device=device)
        h_p, w_p = torch.meshgrid(h_p, w_p, indexing='ij')
        o_p = torch.stack([h_p, w_p], dim=2)
    elif d == 'triangular':
        h_p = torch.linspace(-1, 1, h, dtype=dtype, device=device)
        w_p = torch.linspace(-1, 1, w, dtype=dtype, device=device)
        h_p, w_p = torch.meshgrid(h_p, w_p, indexing='ij')
        w_p = w_p * (h_p + 1.) / 2.  # squeeze w direction
        o_p = torch.stack([h_p, w_p], dim=2)
    elif d == 'hexapolar':
        h_p = torch.linspace(-1, 1, h, dtype=dtype, device=device)
        w_p = torch.linspace(-1, 1, w, dtype=dtype, device=device)
        h_p, w_p = torch.meshgrid(h_p, w_p, indexing='ij')
        theta = - torch.arctan2(h_p, w_p)
        o_p = torch.stack((torch.sin(theta), torch.cos(theta)), dim=-1)
        o_p[..., 0] *= torch.max(torch.abs(h_p), torch.abs(w_p))
        o_p[..., 1] *= torch.max(torch.abs(h_p), torch.abs(w_p))
    elif d == 'fibonacci':
        rays_num = rays_h * rays_w
        R = torch.sqrt(torch.linspace(1 / 2, rays_num - 1 / 2, rays_num, dtype=dtype, device=device)) / \
            np.sqrt(rays_num - 1 / 2)
        T = 4 / (1 + np.sqrt(5)) * torch.pi * \
            torch.linspace(1, rays_num, rays_num, dtype=dtype, device=device)
        x = R * torch.cos(T)
        y = R * torch.sin(T)
        o_p = torch.stack((x, y), dim=-1)
    elif d == 'ring':
        rings_num = rays_h
        x = [0.]
        y = [0.]
        rings = torch.linspace(0., 1., rings_num+1, dtype=dtype, device=device)
        for i, r in enumerate(rings[1:]):
            if i == 0:
                angle = torch.linspace(0, 2 * torch.pi, 8 + 1, dtype=dtype, device=device)[1:]
            elif i == 1:
                angle = torch.linspace(0, 2 * torch.pi, 16 + 1, dtype=dtype, device=device)[1:]
            else:
                angle = torch.linspace(0, 2 * torch.pi, 6*(i + 1) + 1, dtype=dtype, device=device)[1:]
            x.extend(r * torch.cos(angle))
            y.extend(r * torch.sin(angle))
        o_p = torch.stack((torch.tensor(x), torch.tensor(y)), dim=-1)
    else:
        raise ValueError('Unknown ray distribution', d)

    return o_p


if __name__ == "__main__":
    # test material
    # coef = Material(name='sk2')
    # print(coef.refractive_index(wavelength=587.6))

    # test pupil
    out1, out2 = pupil_distribution(5, 5, 'hexapolar')
