
import math
import numpy as np
import scipy.optimize as optimize
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as axes3d
np.random.seed(2016)

mean = np.array([0.0,0.0,0.0])
cov = np.array([[1.0,-0.5,0.8], [-0.5,1.1,0.0], [0.8,0.0,1.0]])
xyz = np.random.multivariate_normal(mean, cov, 50)
x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]

#print variable type for x and print x
print(type(x))
print(x)

def minimize_z_error(x, y, z):
    # Best-fit linear plane, for the Eq: z = a*x + b*y + c.
    # See: https://gist.github.com/amroamroamro/1db8d69b4b65e8bc66a6
    A = np.c_[x, y, np.ones(x.shape)]
    C, resid, rank, singular_values = np.linalg.lstsq(A, z)

    # Coefficients in the form: a*x + b*y + c*z + d = 0.
    return C[0], C[1], -1., C[2]

def minimize_perp_distance(x, y, z):
    def model(params, xyz):
        a, b, c, d = params
        x, y, z = xyz
        length_squared = a**2 + b**2 + c**2
        return ((a * x + b * y + c * z + d) ** 2 / length_squared).sum()

    def unit_length(params):
        a, b, c, d = params
        return a**2 + b**2 + c**2 - 1

    # constrain the vector perpendicular to the plane be of unit length
    cons = ({'type':'eq', 'fun': unit_length})
    sol = optimize.minimize(model, initial_guess, args=[x, y, z], constraints=cons)
    return tuple(sol.x)

initial_guess = 0.28, -0.14, 0.95, 0.
vert_params = minimize_z_error(x, y, z)
perp_params = minimize_perp_distance(x, y, z)

def z_error(x, y, z, a, b, d):
    return math.sqrt((((a*x + b*y + d) - z)**2).sum())

def perp_error(x, y, z, a, b, c, d):
    length_squared = a**2 + b**2 + c**2
    return ((a * x + b * y + c * z + d) ** 2 / length_squared).sum()

def report(kind, params):
    a, b, c, d = params
    paramstr = ','.join(['{:.2f}'.format(p) for p in params])
    print('{:7}: params: ({}), z_error: {:>5.2f}, perp_error: {:>5.2f}'.format(
        kind, paramstr, z_error(x, y, z, a, b, d), perp_error(x, y, z, a, b, c, d)))

report('vert', vert_params)
report('perp', perp_params)
report('guess', initial_guess)

X, Y = np.meshgrid(np.arange(-3.0, 3.0, 0.5), np.arange(-3.0, 3.0, 0.5))
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

def Z(X, Y, params):
    a, b, c, d = params
    return -(a*X + b*Y + d)/c

ax.plot_surface(X, Y, Z(X, Y, initial_guess), rstride=1, cstride=1, alpha=0.3, color='magenta')
ax.plot_surface(X, Y, Z(X, Y, vert_params), rstride=1, cstride=1, alpha=0.3, color='yellow')
ax.plot_surface(X, Y, Z(X, Y, perp_params), rstride=1, cstride=1, alpha=0.3, color='green')
ax.scatter(x, y, z, c='r', s=50)
plt.xlabel('X')
plt.ylabel('Y')
ax.set_zlabel('Z')
ax.axis('equal')
ax.axis('tight')
plt.show()