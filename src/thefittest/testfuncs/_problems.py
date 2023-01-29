import numpy as np
import os


path = os.path.dirname(__file__) + '/shifts_data/'

fbias_data = np.loadtxt(path+'fbias_data.txt')
# 1
sphere_func_data = np.loadtxt(path+'sphere_func_data.txt')
# 2
schwefel_102_data = np.loadtxt(path+'schwefel_102_data.txt')
# 3
high_cond_elliptic_rot_data = np.loadtxt(path
                                         + 'high_cond_elliptic_rot_data.txt')[:50]
elliptic_M_D2 = np.loadtxt(path+'elliptic_M_D2.txt')
elliptic_M_D10 = np.loadtxt(path+'elliptic_M_D10.txt')
elliptic_M_D30 = np.loadtxt(path+'elliptic_M_D30.txt')
elliptic_M_D50 = np.loadtxt(path+'elliptic_M_D50.txt')
# 5
schwefel_206_data = np.loadtxt(path+'schwefel_206_data.txt')
o_206 = schwefel_206_data[0]
A_206 = schwefel_206_data[1:]
# 6
rosenbrock_func_data = np.loadtxt(path+'rosenbrock_func_data.txt')
# 7
griewank_func_data = np.loadtxt(path+'griewank_func_data.txt')[:50]
griewank_M_D2 = np.loadtxt(path+'griewank_M_D2.txt')
griewank_M_D10 = np.loadtxt(path+'griewank_M_D10.txt')
griewank_M_D30 = np.loadtxt(path+'griewank_M_D30.txt')
griewank_M_D50 = np.loadtxt(path+'griewank_M_D50.txt')
# 8
ackley_func_data = np.loadtxt(path+'ackley_func_data.txt')[:50]
ackley_M_D2 = np.loadtxt(path+'ackley_M_D2.txt')
ackley_M_D10 = np.loadtxt(path+'ackley_M_D10.txt')
ackley_M_D30 = np.loadtxt(path+'ackley_M_D30.txt')
ackley_M_D50 = np.loadtxt(path+'ackley_M_D50.txt')
# 9-10
rastrigin_func_data = np.loadtxt(path+'rastrigin_func_data.txt')[:50]
rastrigin_M_D2 = np.loadtxt(path+'rastrigin_M_D2.txt')
rastrigin_M_D10 = np.loadtxt(path+'rastrigin_M_D10.txt')
rastrigin_M_D30 = np.loadtxt(path+'rastrigin_M_D30.txt')
rastrigin_M_D50 = np.loadtxt(path+'rastrigin_M_D50.txt')
# 11
weierstrass_M_D2 = np.loadtxt(path+'weierstrass_M_D2.txt')
weierstrass_M_D10 = np.loadtxt(path+'weierstrass_M_D10.txt')
weierstrass_M_D30 = np.loadtxt(path+'weierstrass_M_D30.txt')
weierstrass_M_D50 = np.loadtxt(path+'weierstrass_M_D50.txt')
weierstrass_data = np.loadtxt(path+'weierstrass_data.txt')[:50]
# 12
schwefel_213_data = np.loadtxt(path+'schwefel_213_data.txt')
# 13
EF8F2_func_data = np.loadtxt(path+'EF8F2_func_data.txt')
a_213 = schwefel_213_data[:100]
b_213 = schwefel_213_data[100:200]
alpha_213 = schwefel_213_data[-1]
# 14
E_ScafferF6_func_data = np.loadtxt(path+'E_ScafferF6_func_data.txt')[:50]
E_ScafferF6_M_D2 = np.loadtxt(path+'E_ScafferF6_M_D2.txt')
E_ScafferF6_M_D10 = np.loadtxt(path+'E_ScafferF6_M_D10.txt')
E_ScafferF6_M_D30 = np.loadtxt(path+'E_ScafferF6_M_D30.txt')
E_ScafferF6_M_D50 = np.loadtxt(path+'E_ScafferF6_M_D50.txt')
# 15
hybrid_func1_data = np.loadtxt(path+'hybrid_func1_data.txt')
# 16
hybrid_func1_M_D2 = np.loadtxt(path+'hybrid_func1_M_D2.txt')
hybrid_func1_M_D10 = np.loadtxt(path+'hybrid_func1_M_D10.txt')
hybrid_func1_M_D30 = np.loadtxt(path+'hybrid_func1_M_D30.txt')
hybrid_func1_M_D50 = np.loadtxt(path+'hybrid_func1_M_D50.txt')
# 18
hybrid_func2_data = np.loadtxt(path+'hybrid_func2_data.txt')
hybrid_func2_M_D2 = np.loadtxt(path+'hybrid_func2_M_D2.txt')
hybrid_func2_M_D10 = np.loadtxt(path+'hybrid_func2_M_D10.txt')
hybrid_func2_M_D30 = np.loadtxt(path+'hybrid_func2_M_D30.txt')
hybrid_func2_M_D50 = np.loadtxt(path+'hybrid_func2_M_D50.txt')
# 21
hybrid_func3_data = np.loadtxt(path+'hybrid_func3_data.txt')
hybrid_func3_M_D2 = np.loadtxt(path+'hybrid_func3_M_D2.txt')
hybrid_func3_M_D10 = np.loadtxt(path+'hybrid_func3_M_D10.txt')
hybrid_func3_M_D30 = np.loadtxt(path+'hybrid_func3_M_D30.txt')
hybrid_func3_M_D50 = np.loadtxt(path+'hybrid_func3_M_D50.txt')
# 22
hybrid_func3_HM_D2 = np.loadtxt(path+'hybrid_func3_HM_D2.txt')
hybrid_func3_HM_D10 = np.loadtxt(path+'hybrid_func3_HM_D10.txt')
hybrid_func3_HM_D30 = np.loadtxt(path+'hybrid_func3_HM_D30.txt')
hybrid_func3_HM_D50 = np.loadtxt(path+'hybrid_func3_HM_D50.txt')
# 24
hybrid_func4_data = np.loadtxt(path+'hybrid_func4_data.txt')
hybrid_func4_M_D2 = np.loadtxt(path+'hybrid_func4_M_D2.txt')
hybrid_func4_M_D10 = np.loadtxt(path+'hybrid_func4_M_D10.txt')
hybrid_func4_M_D30 = np.loadtxt(path+'hybrid_func4_M_D30.txt')
hybrid_func4_M_D50 = np.loadtxt(path+'hybrid_func4_M_D50.txt')


class TestFunction:
    def __call__(self, x):
        return self.f(x)

    def build_grid(self, x, y):
        x1, y1 = np.meshgrid(x, y)
        xy = np.concatenate(
            [x1[:, :, np.newaxis], y1[:, :, np.newaxis]], axis=2)
        z = np.zeros(shape=xy.shape[:-1])
        for i, x_i in enumerate(xy):
            z[i] = self(x_i)
        return z


class TestShiftedFunction:
    def __init__(self, fbias, x_shift):
        self.fbias = fbias
        self.x_shift = x_shift

    def shift(self, x):
        shape = x.shape
        axis = [1]*(len(shape)-1) + [-1]
        return x - self.x_shift[:shape[-1]].reshape(axis)

    def __call__(self, x):
        z = self.shift(x)
        return self.f(z) + self.fbias

    def build_grid(self, x, y):
        x1, y1 = np.meshgrid(x, y)
        xy = np.concatenate(
            [x1[:, :, np.newaxis], y1[:, :, np.newaxis]], axis=2)
        z = np.zeros(shape=xy.shape[:-1])
        for i, x_i in enumerate(xy):
            z[i] = self(x_i)
        return z


class TestShiftedRotatedFunction:
    def __init__(self, fbias, x_shift, rotate_M_D2,
                 rotate_M_D10, rotate_M_D30, rotate_M_D50):
        self.fbias = fbias
        self.x_shift = x_shift
        self.rotate_M_D2 = rotate_M_D2
        self.rotate_M_D10 = rotate_M_D10
        self.rotate_M_D30 = rotate_M_D30
        self.rotate_M_D50 = rotate_M_D50

    def shift(self, x):
        shape = x.shape
        axis = [1]*(len(shape)-1) + [-1]
        return x - self.x_shift[:shape[-1]].reshape(axis)

    def rotate(self, x):
        if x.shape[1] == 2:
            z = x@self.rotate_M_D2
        elif x.shape[1] == 10:
            z = x@self.rotate_M_D10
        elif x.shape[1] == 30:
            z = x@self.rotate_M_D30
        elif x.shape[1] == 50:
            z = x@self.rotate_M_D50
        return z

    def __call__(self, x):
        z = self.shift(x)
        z_rotated = self.rotate(z)
        return self.f(z_rotated) + self.fbias

    def build_grid(self, x, y):
        x1, y1 = np.meshgrid(x, y)
        xy = np.concatenate(
            [x1[:, :, np.newaxis], y1[:, :, np.newaxis]], axis=2)
        z = np.zeros(shape=xy.shape[:-1])
        for i, x_i in enumerate(xy):
            z[i] = self(x_i)
        return z


class SampleHybridCompositionFunction():
    def __init__(self, basic_functions, sigmas, lambdas,
                 fbias,
                 x_shift,
                 M_D2, M_D10, M_D30, M_D50):
        self.basic_functions = basic_functions
        self.sigmas = sigmas
        self.lambdas = lambdas
        self.M_D2 = np.split(M_D2, range(
            M_D2.shape[1], M_D2.shape[0], M_D2.shape[1]))
        self.M_D10 = np.split(M_D10, range(
            M_D10.shape[1], M_D10.shape[0], M_D10.shape[1]))
        self.M_D30 = np.split(M_D30, range(
            M_D30.shape[1], M_D30.shape[0], M_D30.shape[1]))
        self.M_D50 = np.split(M_D50, range(
            M_D50.shape[1], M_D50.shape[0], M_D50.shape[1]))
        self.fbias = fbias
        self.x_shift = x_shift

        self.bias = np.array(
            [0, 100, 200, 300, 400, 500, 600, 700, 800, 900], dtype=np.float64)
        self.C = 2000

    def shift(self, x, i):
        shape = x.shape
        axis = [1]*(len(shape)-1) + [-1]
        return x - self.x_shift[i][:shape[-1]].reshape(axis)

    def rotate(self, x, M_D2, M_D10, M_D30, M_D50):
        if x.shape[1] == 2:
            z = x@M_D2
        elif x.shape[1] == 10:
            z = x@M_D10
        elif x.shape[1] == 30:
            z = x@M_D30
        elif x.shape[1] == 50:
            z = x@M_D50
        return z

    def calc_w(self, x, shift_i, sigma_i, i):
        D = x.shape[1]
        shift_x = self.shift(x, i)

        up = np.sum(shift_x**2, axis=-1)
        down = 2*D*(sigma_i*sigma_i)

        w_i = np.exp(-(up/down))

        return w_i, shift_x

    def procces_i_function(self, x, func_i, sigma_i, lambda_i, shift_i,
                           M_D2i, M_D10i, M_D30i, M_D50i, i):
        y = np.full(shape=x.shape, fill_value=5)
        w_i, shift_x = self.calc_w(x, shift_i, sigma_i, i)

        arg1 = self.rotate(shift_x/lambda_i,  M_D2i, M_D10i, M_D30i, M_D50i)
        fit_i = func_i()(arg1)

        arg2 = self.rotate(y/lambda_i,  M_D2i, M_D10i, M_D30i, M_D50i)
        f_max_i = func_i()(arg2)

        fit_i = self.C*fit_i/f_max_i

        return w_i, fit_i

    def __call__(self, x):
        result = tuple(self.procces_i_function(x, func_i, sigma_i, lambda_i, shift_i,
                                               M_D2i, M_D10i, M_D30i, M_D50i, i)
                       for func_i, sigma_i, lambda_i, shift_i,
                       M_D2i, M_D10i, M_D30i, M_D50i, i in zip(self.basic_functions, self.sigmas, self.lambdas, self.x_shift,
                                                               self.M_D2, self.M_D10, self.M_D30, self.M_D50, range(len(self.M_D50))))
        w_i, fit_i = list(zip(*result))
        w_i = np.array(w_i, np.float64).T
        fit_i = np.array(fit_i, np.float64).T
        max_w = np.max(w_i, axis=1)

        for i, w_i_j in enumerate(w_i.T):
            cond = w_i_j != max_w
            w_i[:, i][cond] = w_i[:, i][cond]*(1.0 - max_w[cond]**10)

        w_i = w_i/np.sum(w_i, axis=-1)[:, np.newaxis]

        return np.sum(w_i*(fit_i + self.bias), axis=-1) + self.fbias

    def test(self):
        y = self(self.x_shift.reshape(1, -1))
        return y - self.fbias < self.fixed_accuracy

    def build_grid(self, x, y):
        x1, y1 = np.meshgrid(x, y)
        xy = np.concatenate(
            [x1[:, :, np.newaxis], y1[:, :, np.newaxis]], axis=2)
        z = np.zeros(shape=xy.shape[:-1])
        for i, x_i in enumerate(xy):
            z[i] = self(x_i)
        return z


class OneMax(TestFunction):
    def f(self, x):
        return np.sum(x, axis=1)


class Sphere(TestFunction):
    def f(self, x):
        return np.sum(x**2, axis=-1)


class Schwefe1_2(TestFunction):
    def f(self, x):
        return np.sum(np.add.accumulate(x, axis=-1)**2, axis=-1)


class HighConditionedElliptic(TestFunction):
    def f(self, x):
        i = np.arange(1, x.shape[1]+1)
        demension = x.shape[1]
        return np.sum((1e6**((i - 1)/(demension - 1)))*x**2, axis=-1)


class Rosenbrock(TestFunction):
    def f(self, x):
        value = 100*((x.T[:-1]**2 - x.T[1:])**2) + (x.T[:-1] - 1)**2
        return np.sum(value.T, axis=-1)


class Rastrigin(TestFunction):
    def f(self, x):
        return np.sum(x**2 - 10*np.cos(2*np.pi*x) + 10, axis=-1)


class Griewank(TestFunction):
    def f(self, x):
        sqrt_i = np.sqrt(np.arange(1, x.shape[1]+1))
        sum_ = np.sum((x**2)/4000, axis=-1)
        prod_ = np.prod(np.cos(x/sqrt_i), axis=-1)
        return sum_ - prod_ + 1


class Ackley(TestFunction):
    def f(self, x):
        a = 20
        c = 2*np.pi
        b = 0.2
        D = x.shape[1]
        left = -a*np.exp(-b*np.sqrt(np.sum(x**2, axis=1)/D))
        right = -np.exp((1/D)*np.sum(np.cos(c*x), axis=1))
        return left + right + a + np.exp(1)


class Weierstrass(TestFunction):
    def f(self, x):
        a = 0.5
        b = 3
        k_max = np.arange(20, dtype=np.int64)
        D = x.shape[1]

        two_pi_power_b = 2*np.pi*(b**k_max)
        a_power_k = a**k_max
        arg1 = two_pi_power_b*(x[:, :, np.newaxis]+0.5)
        arg2 = two_pi_power_b*0.5
        left = a_power_k*np.cos(arg1)
        right = a_power_k*np.cos(arg2)

        left_sum = np.sum(np.sum(left, axis=-1), axis=-1)
        right_sum = np.sum(right, axis=-1)
        return left_sum - D*right_sum


class F8F2(TestFunction):
    def __init__(self):
        self.rosenbrock_f = Rosenbrock()
        self.griewank_f = Griewank()

    def f(self, x):
        indexes = np.kron(
            np.arange(1, x.shape[1]-1, dtype=np.int64), np.array([1, 1]))
        indexes = np.insert(indexes, [0], [0])
        indexes = np.append(indexes, np.array([x.shape[1]-1, x.shape[1]-1, 0]))

        x_indexes = x[:, indexes]

        vertical_X = x_indexes.reshape(-1, 2)
        F2 = self.rosenbrock_f(vertical_X)
        F8 = self.griewank_f(F2.reshape(-1, 1))
        horizontal_X = F8.reshape(x.shape[0], -1)
        return np.sum(horizontal_X, axis=-1)


class ExpandedScaffers_F6(TestFunction):
    def Scaffes_F6(self, x):
        sum_of_power = x[:, 0]**2 + x[:, 1]**2
        up = np.sin(np.sqrt(sum_of_power))**2 - 0.5
        down = (1 + 0.001*(sum_of_power))**2
        return 0.5 + up/down

    def f(self, x):
        indexes = np.kron(
            np.arange(1, x.shape[1]-1, dtype=np.int64), np.array([1, 1]))
        indexes = np.insert(indexes, [0], [0])
        indexes = np.append(indexes, np.array([x.shape[1]-1, x.shape[1]-1, 0]))
        x_indexes = x[:, indexes]

        vertical_X = x_indexes.reshape(-1, 2)
        F = self.Scaffes_F6(vertical_X)
        horizontal_X = F.reshape(x.shape[0], -1)

        return np.sum(horizontal_X, axis=-1)


class NonContinuosRastrigin(Rastrigin):
    def __call__(self, x):
        cond = np.abs(x) >= 0.5
        x[cond] = (np.round(2*x)/2)[cond]
        return super().__call__(x)


class NonContinuosExpandedScaffers_F6(ExpandedScaffers_F6):
    def __call__(self, x):
        cond = np.abs(x) >= 0.5
        x[cond] = (np.round(2*x)/2)[cond]
        return super().__call__(x)


class SphereWithNoise(Sphere):
    def __call__(self, x):
        y = super().__call__(x)
        noise = 1 + 0.1*np.abs(np.random.normal(size=y.shape))
        return y*noise


# CEC05 #1
class ShiftedSphere(TestShiftedFunction, Sphere):
    def __init__(self):
        TestShiftedFunction.__init__(self,
                                     fbias=fbias_data[0],
                                     x_shift=sphere_func_data)


# CEC05 #2
class ShiftedSchwefe1_2(TestShiftedFunction, Schwefe1_2):
    def __init__(self):
        TestShiftedFunction.__init__(self,
                                     fbias=fbias_data[1],
                                     x_shift=schwefel_102_data)


# CEC05 #3
class ShiftedRotatedHighConditionedElliptic(TestShiftedRotatedFunction,
                                            HighConditionedElliptic):
    def __init__(self):
        TestShiftedRotatedFunction.__init__(
            self,
            fbias=fbias_data[2],
            x_shift=high_cond_elliptic_rot_data,
            rotate_M_D2=elliptic_M_D2,
            rotate_M_D10=elliptic_M_D10,
            rotate_M_D30=elliptic_M_D30,
            rotate_M_D50=elliptic_M_D50)


# CEC05 #4
class ShiftedSchwefe1_2WithNoise(TestShiftedFunction, Schwefe1_2):
    def __init__(self):
        TestShiftedFunction.__init__(self,
                                     fbias=fbias_data[3],
                                     x_shift=schwefel_102_data)

    def __call__(self, x):
        y = super().__call__(x)
        noise = 1 + 0.4*np.abs(np.random.normal(size=y.shape))
        return y*noise


# CEC05 #5
class Schwefel2_6(TestShiftedFunction):
    def __init__(self):
        TestShiftedFunction.__init__(self,
                                     fbias=fbias_data[4],
                                     x_shift=o_206)

    def __call__(self, x):
        D = x.shape[1]
        self.x_shift[np.floor(3*D/4).astype(np.int64)-1:D] = 100
        self.x_shift[:np.ceil(D/4).astype(np.int64)] = -100
        return self.f(x) + self.fbias

    def f(self, x):
        o = self.x_shift[:x.shape[1]]
        A = A_206[:x.shape[1], :x.shape[1]]
        Ax = A@x.T
        B = A@o
        fx = np.abs(Ax - B[:, np.newaxis])
        return np.max(fx, axis=0)


# CEC05 #6
class ShiftedRosenbrock(TestShiftedFunction, Rosenbrock):
    def __init__(self):
        TestShiftedFunction.__init__(self,
                                     fbias=fbias_data[5],
                                     x_shift=rosenbrock_func_data)

    def shift(self, x):
        shape = x.shape
        axis = [1]*(len(shape)-1) + [-1]
        return x - self.x_shift[:shape[-1]].reshape(axis) + 1


# CEC05 #7
class ShiftedRotatedGriewank(TestShiftedRotatedFunction,
                             Griewank):
    def __init__(self):
        TestShiftedRotatedFunction.__init__(
            self,
            fbias=fbias_data[6],
            x_shift=griewank_func_data,
            rotate_M_D2=griewank_M_D2,
            rotate_M_D10=griewank_M_D10,
            rotate_M_D30=griewank_M_D30,
            rotate_M_D50=griewank_M_D50)


# CEC05 #8
class ShiftedRotatedAckley(TestShiftedRotatedFunction,
                           Ackley):
    def __init__(self):
        TestShiftedRotatedFunction.__init__(
            self,
            fbias=fbias_data[7],
            x_shift=ackley_func_data,
            rotate_M_D2=ackley_M_D2,
            rotate_M_D10=ackley_M_D10,
            rotate_M_D30=ackley_M_D30,
            rotate_M_D50=ackley_M_D50)

    def __call__(self, x):
        D = x.shape[1]
        self.x_shift[:D:2] = -32.0
        return super().__call__(x)


# CEC05 #9
class ShiftedRastrigin(TestShiftedFunction, Rastrigin):
    def __init__(self):
        TestShiftedFunction.__init__(self,
                                     fbias=fbias_data[8],
                                     x_shift=rastrigin_func_data)


# CEC05 #10
class ShiftedRotatedRastrigin(TestShiftedRotatedFunction,
                              Rastrigin):
    def __init__(self):
        TestShiftedRotatedFunction.__init__(
            self,
            fbias=fbias_data[9],
            x_shift=rastrigin_func_data,
            rotate_M_D2=rastrigin_M_D2,
            rotate_M_D10=rastrigin_M_D10,
            rotate_M_D30=rastrigin_M_D30,
            rotate_M_D50=rastrigin_M_D50)


# CEC05 #11
class ShiftedRotatedWeierstrass(TestShiftedRotatedFunction,
                                Weierstrass):
    def __init__(self):
        TestShiftedRotatedFunction.__init__(
            self,
            fbias=fbias_data[10],
            x_shift=weierstrass_data,
            rotate_M_D2=weierstrass_M_D2,
            rotate_M_D10=weierstrass_M_D10,
            rotate_M_D30=weierstrass_M_D30,
            rotate_M_D50=weierstrass_M_D50)


# CEC05 #12
class Schwefel2_13(TestShiftedFunction):
    def __init__(self):
        TestShiftedFunction.__init__(self,
                                     fbias=fbias_data[11],
                                     x_shift=alpha_213)

    def __call__(self, x):
        return self.f(x) + self.fbias

    def f(self, x):
        a = a_213[:x.shape[1], :x.shape[1]]
        b = b_213[:x.shape[1], :x.shape[1]]
        alpha = self.x_shift[:x.shape[1]]

        A = np.sum(a*np.sin(alpha)[np.newaxis:,]
                   + b*np.cos(alpha)[np.newaxis:,], axis=1)

        B1 = a[np.newaxis]*np.sin(x[:, np.newaxis])
        B2 = b[np.newaxis]*np.cos(x[:, np.newaxis])
        B = np.sum(B1 + B2, axis=-1)

        return np.sum((A[np.newaxis:,] - B)**2, axis=-1)


# CEC05 #13
class ShiftedExpandedGriewankRosenbrock(TestShiftedFunction):
    def __init__(self):
        TestShiftedFunction.__init__(
            self,
            fbias=fbias_data[12],
            x_shift=EF8F2_func_data)
        self.rosenbrock_f = Rosenbrock()
        self.griewank_f = Griewank()

    def shift(self, x):
        shape = x.shape
        axis = [1]*(len(shape)-1) + [-1]
        return x - self.x_shift[:shape[-1]].reshape(axis) + 1

    def f(self, x):
        indexes = np.kron(
            np.arange(1, x.shape[1]-1, dtype=np.int64), np.array([1, 1]))
        indexes = np.insert(indexes, [0], [0])
        indexes = np.append(indexes, np.array([x.shape[1]-1, x.shape[1]-1, 0]))

        x_indexes = x[:, indexes]

        vertical_X = x_indexes.reshape(-1, 2)
        F2 = self.rosenbrock_f(vertical_X)
        F8 = self.griewank_f(F2.reshape(-1, 1))
        horizontal_X = F8.reshape(x.shape[0], -1)
        return np.sum(horizontal_X, axis=-1)


# CEC05 #14
class ShiftedRotatedExpandedScaffes_F6(TestShiftedRotatedFunction):
    def __init__(self):
        TestShiftedRotatedFunction.__init__(
            self,
            fbias=fbias_data[13],
            x_shift=E_ScafferF6_func_data,
            rotate_M_D2=E_ScafferF6_M_D2,
            rotate_M_D10=E_ScafferF6_M_D10,
            rotate_M_D30=E_ScafferF6_M_D30,
            rotate_M_D50=E_ScafferF6_M_D50)

    def Scaffes_F6(self, x):
        sum_of_power = x[:, 0]**2 + x[:, 1]**2
        up = np.sin(np.sqrt(sum_of_power))**2 - 0.5
        down = (1 + 0.001*(sum_of_power))**2
        return 0.5 + up/down

    def f(self, x):
        indexes = np.kron(
            np.arange(1, x.shape[1]-1, dtype=np.int64), np.array([1, 1]))
        indexes = np.insert(indexes, [0], [0])
        indexes = np.append(indexes, np.array([x.shape[1]-1, x.shape[1]-1, 0]))
        x_indexes = x[:, indexes]

        vertical_X = x_indexes.reshape(-1, 2)
        F = self.Scaffes_F6(vertical_X)
        horizontal_X = F.reshape(x.shape[0], -1)

        return np.sum(horizontal_X, axis=-1)


# CEC05 #15
class HybridCompositionFunction1(SampleHybridCompositionFunction):
    def __init__(self):
        SampleHybridCompositionFunction.__init__(
            self,
            basic_functions=(Rastrigin,
                             Rastrigin,
                             Weierstrass,
                             Weierstrass,
                             Griewank,
                             Griewank,
                             Ackley,
                             Ackley,
                             Sphere,
                             Sphere),
            sigmas=np.ones(10),
            lambdas=np.array([1, 1, 10, 10, 5/60, 5/60, 5/32,
                             5/32, 5/100, 5/100], dtype=np.float64),
            fbias=fbias_data[14],
            x_shift=hybrid_func1_data,
            M_D2=np.vstack([np.eye(2) for _ in range(10)]),
            M_D10=np.vstack([np.eye(10) for _ in range(10)]),
            M_D30=np.vstack([np.eye(30) for _ in range(10)]),
            M_D50=np.vstack([np.eye(50) for _ in range(10)]))


# CEC05 #16
class RotatedVersionHybridCompositionFunction1(SampleHybridCompositionFunction):
    def __init__(self):
        SampleHybridCompositionFunction.__init__(
            self,
            basic_functions=(Rastrigin,
                             Rastrigin,
                             Weierstrass,
                             Weierstrass,
                             Griewank,
                             Griewank,
                             Ackley,
                             Ackley,
                             Sphere,
                             Sphere),
            sigmas=np.ones(10),
            lambdas=np.array([1, 1, 10, 10, 5/60, 5/60, 5/32,
                             5/32, 5/100, 5/100], dtype=np.float64),
            fbias=fbias_data[15],
            x_shift=hybrid_func1_data,
            M_D2=hybrid_func1_M_D2,
            M_D10=hybrid_func1_M_D10,
            M_D30=hybrid_func1_M_D30,
            M_D50=hybrid_func1_M_D50)


# CEC05 #17
class RotatedVersionHybridCompositionFunction1Noise(SampleHybridCompositionFunction):
    def __init__(self):
        SampleHybridCompositionFunction.__init__(
            self,
            basic_functions=(Rastrigin,
                             Rastrigin,
                             Weierstrass,
                             Weierstrass,
                             Griewank,
                             Griewank,
                             Ackley,
                             Ackley,
                             Sphere,
                             Sphere),
            sigmas=np.ones(10),
            lambdas=np.array([1, 1, 10, 10, 5/60, 5/60, 5/32,
                             5/32, 5/100, 5/100], dtype=np.float64),
            fbias=fbias_data[16],
            x_shift=hybrid_func1_data,
            M_D2=hybrid_func1_M_D2,
            M_D10=hybrid_func1_M_D10,
            M_D30=hybrid_func1_M_D30,
            M_D50=hybrid_func1_M_D50)

    def __call__(self, x):
        G_x = RotatedVersionHybridCompositionFunction1()(x) - fbias_data[15]
        noise = np.abs(np.random.normal(size=G_x.shape))
        return G_x*(1 + 0.2*noise) + fbias_data[16]


# CEC05 #18
class RotatedHybridCompositionFunction(SampleHybridCompositionFunction):
    def __init__(self):
        SampleHybridCompositionFunction.__init__(
            self,
            basic_functions=(Ackley,
                             Ackley,
                             Rastrigin,
                             Rastrigin,
                             Sphere,
                             Sphere,
                             Weierstrass,
                             Weierstrass,
                             Griewank,
                             Griewank),
            sigmas=np.array([1, 2, 1.5, 1.5, 1, 1, 1.5,
                            1.5, 2, 2], dtype=np.float64),
            lambdas=np.array([2*(5/32), 5/32, 2*1, 1, 2*(5/100),
                             5/100, 2*10, 10, 2*(5/60), 5/60], dtype=np.float64),
            fbias=fbias_data[17],
            x_shift=hybrid_func2_data,
            M_D2=hybrid_func2_M_D2,
            M_D10=hybrid_func2_M_D10,
            M_D30=hybrid_func2_M_D30,
            M_D50=hybrid_func2_M_D50)
        self.x_shift[9] = 0


# CEC05 #19
class RotatedHybridCompositionFunctionNarrowBasin(SampleHybridCompositionFunction):
    def __init__(self):
        SampleHybridCompositionFunction.__init__(self, basic_functions=(Ackley,
                                                                        Ackley,
                                                                        Rastrigin,
                                                                        Rastrigin,
                                                                        Sphere,
                                                                        Sphere,
                                                                        Weierstrass,
                                                                        Weierstrass,
                                                                        Griewank,
                                                                        Griewank),
                                                 sigmas=np.array(
                                                     [0.1, 2, 1.5, 1.5, 1, 1, 1.5, 1.5, 2, 2], dtype=np.float64),
                                                 lambdas=np.array(
                                                     [0.1*(5/32), 5/32, 2*1, 1, 2*(5/100), 5/100, 2*10, 10, 2*(5/60), 5/60], dtype=np.float64),
                                                 fbias=fbias_data[17],
                                                 x_shift=hybrid_func2_data,
                                                 M_D2=hybrid_func2_M_D2,
                                                 M_D10=hybrid_func2_M_D10,
                                                 M_D30=hybrid_func2_M_D30,
                                                 M_D50=hybrid_func2_M_D50)
        self.x_shift[9] = 0


# CEC05 #20
class RotatedHybridCompositionFunctionOptimalBounds(SampleHybridCompositionFunction):
    def __init__(self):
        SampleHybridCompositionFunction.__init__(
            self, basic_functions=(Ackley,
                                   Ackley,
                                   Rastrigin,
                                   Rastrigin,
                                   Sphere,
                                   Sphere,
                                   Weierstrass,
                                   Weierstrass,
                                   Griewank,
                                   Griewank),
            sigmas=np.array([1, 2, 1.5, 1.5, 1, 1, 1.5,
                            1.5, 2, 2], dtype=np.float64),
            lambdas=np.array([2*(5/32), 5/32, 2*1, 1, 2*(5/100),
                             5/100, 2*10, 10, 2*(5/60), 5/60], dtype=np.float64),
            fbias=fbias_data[19],
            x_shift=hybrid_func2_data,
            M_D2=hybrid_func2_M_D2,
            M_D10=hybrid_func2_M_D10,
            M_D30=hybrid_func2_M_D30,
            M_D50=hybrid_func2_M_D50)
        self.x_shift[9] = 0

    def __call__(self, x):
        D = x.shape[1]
        self.x_shift[0][1:int(D/2):2] = 5
        return super().__call__(x)


# CEC05 #21
class HybridCompositionFunction3(SampleHybridCompositionFunction):
    def __init__(self):
        SampleHybridCompositionFunction.__init__(
            self,
            basic_functions=(ExpandedScaffers_F6,
                             ExpandedScaffers_F6,
                             Rastrigin,
                             Rastrigin,
                             F8F2,
                             F8F2,
                             Weierstrass,
                             Weierstrass,
                             Griewank,
                             Griewank),
            sigmas=np.array([1., 1., 1., 1., 1., 2., 2.,
                            2., 2., 2.], dtype=np.float64),
            lambdas=np.array([5*5/100, 5/100, 5*1, 1, 5*1, 1,
                             5*10, 10, 5*5/200, 5/200], dtype=np.float64),
            fbias=fbias_data[20],
            x_shift=hybrid_func3_data,
            M_D2=hybrid_func3_M_D2,
            M_D10=hybrid_func3_M_D10,
            M_D30=hybrid_func3_M_D30,
            M_D50=hybrid_func3_M_D50)


# CEC05 #22
class HybridCompositionFunction3H(SampleHybridCompositionFunction):
    def __init__(self):
        SampleHybridCompositionFunction.__init__(
            self,
            basic_functions=(ExpandedScaffers_F6,
                             ExpandedScaffers_F6,
                             Rastrigin,
                             Rastrigin,
                             F8F2,
                             F8F2,
                             Weierstrass,
                             Weierstrass,
                             Griewank,
                             Griewank),
            sigmas=np.array([1., 1., 1., 1., 1., 2., 2.,
                            2., 2., 2.], dtype=np.float64),
            lambdas=np.array([5*5/100, 5/100, 5*1, 1, 5*1, 1,
                             5*10, 10, 5*5/200, 5/200], dtype=np.float64),
            fbias=fbias_data[21],
            x_shift=hybrid_func3_data,
            M_D2=hybrid_func3_HM_D2,
            M_D10=hybrid_func3_HM_D10,
            M_D30=hybrid_func3_HM_D30,
            M_D50=hybrid_func3_HM_D50)


# CEC05 #23
class NonContinuousHybridCompositionFunction3(SampleHybridCompositionFunction):
    def __init__(self):
        SampleHybridCompositionFunction.__init__(
            self,
            basic_functions=(ExpandedScaffers_F6,
                             ExpandedScaffers_F6,
                             Rastrigin,
                             Rastrigin,
                             F8F2,
                             F8F2,
                             Weierstrass,
                             Weierstrass,
                             Griewank,
                             Griewank),
            sigmas=np.array([1., 1., 1., 1., 1., 2., 2.,
                            2., 2., 2.], dtype=np.float64),
            lambdas=np.array([5*5/100, 5/100, 5*1, 1, 5*1, 1,
                             5*10, 10, 5*5/200, 5/200], dtype=np.float64),
            fbias=fbias_data[22],
            x_shift=hybrid_func3_data,
            M_D2=hybrid_func3_M_D2,
            M_D10=hybrid_func3_M_D10,
            M_D30=hybrid_func3_M_D30,
            M_D50=hybrid_func3_M_D50)

    def __call__(self, x):
        shape = x.shape
        axis = [1]*(len(shape)-1) + [-1]
        o = self.x_shift[0][:shape[-1]].reshape(axis)
        cond = np.abs(x - o) >= 0.5
        x[cond] = (np.round(2*x)/2)[cond]
        return super().__call__(x)


# CEC05 #24
class HybridCompositionFunction4(SampleHybridCompositionFunction):
    def __init__(self):
        SampleHybridCompositionFunction.__init__(
            self,
            basic_functions=(Weierstrass,
                             ExpandedScaffers_F6,
                             F8F2,
                             Ackley,
                             Rastrigin,
                             Griewank,
                             NonContinuosExpandedScaffers_F6,
                             NonContinuosRastrigin,
                             HighConditionedElliptic,
                             SphereWithNoise),
            sigmas=np.array([2., 2., 2., 2., 2., 2., 2.,
                            2., 2., 2.], dtype=np.float64),
            lambdas=np.array([10, 5/20, 1, 5/32, 1, 5/100,
                             5/50, 1, 5/100, 5/100], dtype=np.float64),
            fbias=fbias_data[23],
            x_shift=hybrid_func4_data,
            M_D2=hybrid_func4_M_D2,
            M_D10=hybrid_func4_M_D10,
            M_D30=hybrid_func4_M_D30,
            M_D50=hybrid_func4_M_D50)


# CEC05 #25
class HybridCompositionFunction4withoutbounds(SampleHybridCompositionFunction):
    def __init__(self):
        SampleHybridCompositionFunction.__init__(
            self,
            basic_functions=(Weierstrass,
                             ExpandedScaffers_F6,
                             F8F2,
                             Ackley,
                             Rastrigin,
                             Griewank,
                             NonContinuosExpandedScaffers_F6,
                             NonContinuosRastrigin,
                             HighConditionedElliptic,
                             SphereWithNoise),
            sigmas=np.array([2., 2., 2., 2., 2., 2., 2.,
                            2., 2., 2.], dtype=np.float64),
            lambdas=np.array([10, 5/20, 1, 5/32, 1, 5/100,
                             5/50, 1, 5/100, 5/100], dtype=np.float64),
            fbias=fbias_data[24],
            x_shift=hybrid_func4_data,
            M_D2=hybrid_func4_M_D2,
            M_D10=hybrid_func4_M_D10,
            M_D30=hybrid_func4_M_D30,
            M_D50=hybrid_func4_M_D50)