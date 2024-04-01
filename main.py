import numpy as np
import matplotlib.pyplot as plt
from numpy import sin, cos, pi, sqrt, linspace
from celluloid import Camera

# variables
AX, BX, AY, BY = -0.353, 0.353, 0.3, 0.3
X_LIM, Y_LIM = [-0.6, 0.6], [-0.3, 0.7]
C = 3*pi/8 
NUM_VARS = 5
N_MAX = 10**4
EPS = 10**(-6)
DT, DFT = 0.01, 0.005
P, M, G = 2000, 100, 9.80665

# initial approximation
vec = np.array([1] * NUM_VARS)


def stop_criterion(vec):
    return all(res <= EPS for res in f(vec))

# (x1 ; x2 ; y ; fi1 ; fi2)
def f(vec):
    f1 = vec[0] - vec[2]*sin(vec[3]) - AX
    f2 = vec[1] + vec[2]*sin(vec[4]) - BX
    f3 = vec[2] - vec[2]*cos(vec[3]) - AY
    f4 = vec[2]*(vec[3] + vec[4]) + (vec[1] - vec[0]) - C
    f5 = vec[2] - vec[2]*cos(vec[4]) - BY

    return np.array([f1, f2, f3, f4, f5])


def jacobian(vec):
    return np.array([[1, 0, -sin(vec[3]), -vec[2] * cos(vec[3]), 0],
                     [0, 1, sin(vec[4]), 0, vec[2] * cos(vec[4])],
                     [0, 0, 1 - cos(vec[3]), vec[2] * sin(vec[3]), 0],
                     [-1, 1, vec[3] + vec[4], vec[2], vec[2]],
                     [0, 0, 1 - cos(vec[4]), 0, vec[2] * sin(vec[4])]])

            
def solve_equations_2():
    global vec
    n = 0
    while not(stop_criterion(vec)) and n < N_MAX:
        fx = f(vec)
        J = jacobian(vec)
        dx = np.linalg.solve(J, -fx)
        vec = vec + dx
        n += 1


def solve_equations():
    global vec
    n = 0
    Fx = f(vec)
    x0 = vec - Fx*DFT

    while not(stop_criterion(x0)) and n < N_MAX:
        Fx = f(x0)
        x0 = x0 - Fx*DFT
        n += 1
    vec = x0


def plot_circle(x, y, r, ang_from, ang_to, col = 'black'):
    #plt.plot(x, y, color = 'r', marker = '.')
    angles = linspace(ang_from, ang_to, 100)
    xs = x + r*cos(angles)
    ys = y + r*sin(angles)
    plt.plot(xs, ys, color=col)
    plt.gca().set_aspect('equal')


def plot_line(x_1, y_1, x_2, y_2, col = 'black', st = '-'):
    plt.plot([x_1, x_2], [y_1, y_2], color=col, linestyle=st)


def plot_cushion_state():
    global vec
    alpha_1, alpha_2 = 3*pi/2 - vec[3], 3*pi/2
    r_1 = sqrt((AX - vec[0])**2 + (AY - vec[2])**2)
    r_2 = sqrt((BX - vec[1])**2 + (BY - vec[2])**2)

    plot_circle(vec[0], vec[2], r_1, alpha_1, alpha_2)
    plot_circle(vec[1], vec[2], r_2, -pi/2, vec[4] - pi/2)
    plot_line(AX, AY, BX, BY)
    plot_line(vec[0], vec[2] - r_1, vec[1], vec[2] - r_2)
    plot_line(X_LIM[0], 0, X_LIM[1], 0, 'blue', '--')

    plt.gca().annotate('<--- {:.3f} --->'.format(vec[1] - vec[0]), xy=(-0.13 + vec[0] + (vec[1] - vec[0])/2, 0 - 0.07), xycoords='data', fontsize=9)
    plt.gca().annotate('A ({:.2f}, {:.2f})'.format(AX, AY), xy=(AX - 0.1, AY + 0.07), xycoords='data', fontsize=9)
    plt.gca().annotate('B ({:.2f}, {:.2f})'.format(BX, BY), xy=(BX - 0.1, BY + 0.07), xycoords='data', fontsize=9)
    plt.plot(AX, AY, color='red', marker='.')
    plt.plot(BX, BY, color='red', marker='.')
    plt.plot(vec[0], 0, color='red', marker='.')
    plt.plot(vec[1], 0, color='red', marker='.')
    plt.xlim(X_LIM)
    plt.ylim(Y_LIM)


def animate_plots():
    global vec, AY, BY
    camera = Camera(plt.figure())
    tm, v = 0, 0
    while tm <= 2.5:
        #plt.legend(f, [f'f(x) = x^{i}'])
        AY, BY = AY + v*DT, BY + v*DT
        v = v + ((P*(vec[1] - vec[0])/M) - G)*DT
        tm = tm + DT
        solve_equations()
        plot_cushion_state()
        camera.snap()
        
    anim = camera.animate(interval=DT, repeat=True)
    anim.save('animation.gif')


animate_plots()
