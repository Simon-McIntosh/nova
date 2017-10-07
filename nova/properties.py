import numpy as np
from amigo.pyplot import plt
from amigo import geom


class second_moment(object):

    def __init__(self):
        self.patch = []
        self.hole = []
        self.C = {'y': 0, 'z': 0}
        self.I = {'yy': 0, 'zz': 0, 'xx': 0}
        self.A = 0

    def translate(self, C, I, A, poly, dy, dz, append=True):  # parrallel axis
        C['y'] += dy
        C['z'] += dz
        I['yy'] += A * C['z']**2
        I['zz'] += A * C['y']**2
        I['xx'] += A * (C['z']**2 + C['y']**2)
        if append:
            self.add_patch(poly, dy, dz)
        else:
            self.add_hole(poly, dy, dz)
        return C, I

    def circ(self, r=1, ro=0):
        r, ro = abs(r), abs(ro)
        if ro > r:
            raise ValueError('\'r\' must be greater than \'ro\'')
        C = {'y': 0, 'z': 0}  # centroid
        A = np.pi * (r**2 - ro**2)  # area
        Iyy = np.pi / 4 * (r**4 - ro**4)
        Izz = Iyy
        Ixx = Iyy + Izz
        t = np.linspace(0, 2 * np.pi, 20)  # draw
        y = np.append(r * np.cos(t), ro * np.cos(t[::-1]))
        z = np.append(r * np.sin(t), ro * np.sin(t[::-1]))
        patch = {'y': y, 'z': z}
        return C, {'yy': Iyy, 'zz': Izz, 'xx': Ixx}, A, patch

    def rect(self, b=1, h=1):  # width,height
        b, h = abs(b), abs(h)
        C = {'y': 0, 'z': 0}  # centroid
        A = b * h  # area
        Iyy = b * h**3 / 12
        Izz = b**3 * h / 12
        Ixx = Iyy + Izz
        y = np.array([-b / 2, b / 2, b / 2, -b / 2])
        z = np.array([-h / 2, -h / 2, h / 2, h / 2])
        patch = {'y': y, 'z': z}
        return C, {'yy': Iyy, 'zz': Izz, 'xx': Ixx}, A, patch

    def tri(self, b=1, h=1, double=False, flip_y=False, flip_z=False):
        b, h = abs(b), abs(h)
        C = {'y': b / 3, 'z': h / 3}  # centroid
        A = 0.5 * b * h  # area
        Iyy = b * h**3 / 6
        Izz = b**3 * h / 6
        y = np.array([b, 0, 0], dtype='float')
        z = np.array([0, h, 0], dtype='float')
        if double:
            A *= 2
            Izz *= 2
            C['y'] = 0
            y[-1] = -b
        if flip_y:
            C['y'] *= -1
            y *= -1
        if flip_z:
            C['z'] *= -1
            z *= -1
        Ixx = Iyy + Izz
        patch = {'y': y, 'z': z}
        return C, {'yy': Iyy, 'zz': Izz, 'xx': Ixx}, A, patch

    def add_patch(self, patch, dy, dz):
        patch['y'] += dy
        patch['z'] += dz
        self.patch.append(patch)

    def add_hole(self, hole, dy, dz):
        hole['y'] += dy
        hole['z'] += dz
        self.hole.append(hole)

    def update(self, C, I, A):
        for ax in self.C:  # adjust centroid
            self.C[ax] = (self.C[ax] * self.A + C[ax] * A) / (self.A + A)
        self.A += A  # increment area
        for ax in self.I:  # increment second moments
            self.I[ax] += I[ax]

    def downdate(self, C, I, A):
        for ax in self.C:  # adjust centroid
            self.C[ax] = (self.C[ax] * self.A - C[ax] * A) / (self.A + A)
        self.A -= A  # increment area
        for ax in self.I:  # increment second moments
            self.I[ax] -= I[ax]

    def get_shape(self, shape):
        try:
            gen = getattr(self, shape)
        except AttributeError:
            raise AttributeError('shape {} not found'.format(shape))
        return gen

    def remove_shape(self, shape, dy=0, dz=0, **kwargs):
        gen = self.get_shape(shape)
        C, I, A, hole = gen(**kwargs)
        C, I = self.translate(C, I, A, hole, dy, dz, append=False)
        self.downdate(C, I, A)  # update properties

    def add_shape(self, shape, dy=0, dz=0, **kwargs):
        gen = self.get_shape(shape)
        C, I, A, patch = gen(**kwargs)
        C, I = self.translate(C, I, A, patch, dy, dz)
        self.update(C, I, A)  # update properties

    def set_torsion(self, b, t, circular=False):
        if circular:  # b == r
            ro = b + t / 2
            ri = b - t / 2
            self.J = np.pi(ro**4 - ri**4) / 2
        else:
            if not isinstance(b, list) or not isinstance(t, list):
                errtxt = 'inputs b and t required as list '
                errtxt += 'for non-circular sections'
                raise ValueError(errtxt)
            if len(b) != 4 or len(t) != 4:
                raise ValueError('4 entries requred for b and t lists')
            b_mp = np.zeros(4)
            for i in range(3):
                b_mp[i] = b[i] - (t[i + 1] + t[i - 1]) / 2
            b_mp[-1] = b[-1] - (t[0] + t[2]) / 2

    def report(self):
        return self.C, self.I, self.A

    def plot(self, centroid=True):
        for p in self.patch:
            geom.polyfill(p['y'], p['z'], alpha=1)
        for h in self.hole:
            geom.polyfill(h['y'], h['z'], color=np.ones(3))
        plt.axis('equal')
        if centroid:
            plt.plot(0, 0, 's')
            plt.plot(self.C['y'], self.C['z'], 'o')

    def get_pnt(self):
        y, z = [], []
        for p in self.patch:
            y.append(p['y'])
            z.append(p['z'])
        return [y, z]


if __name__ == '__main__':

    w, d = 0.625, 1.243
    i, o, s = 0.04, 0.19, 0.1
    sm = second_moment()
    sm.add_shape('rect', b=d + 2 * s, h=w + i + o, dz=(i - o) / 2)
    sm.remove_shape('rect', b=d, h=w)
    sm.plot()

    sm = second_moment()
    sm.add_shape('circ', r=1, ro=0.5)
    sm.plot()
