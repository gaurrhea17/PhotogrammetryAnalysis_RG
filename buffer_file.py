import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy import linalg
import scipy.optimize

# import .dat type file into pandas dataframe, columns are separated by whitespace
df2 = pd.read_csv('parameters/dat/pmt_position_polygon.txt', header=None, delim_whitespace=True)

# df = pd.read_csv('parameters/SK_all_PMT_locations.txt', header=None, delim_whitespace=True)

pmts = ['00810', '00809', '00808', '00759', '00758', '00757', '00708', '00707', '00706', '00657', '00656', '00655',
        '00606', '00605', '00604', '00555', '00554', '00553', '00504', '00503', '00502', '00453', '00452', '00451']

# create dataframe with only the rows where the 0-th column's first 5 characters are in the list pmts
df2 = df2[df2[0].str[:5].isin(pmts)]

def fit_plane(dataframe):
    """
    p, n = planeFit(points)

    Given an array, points, of shape (d,...)
    representing points in d-dimensional space,
    fit an d-dimensional plane to the points.
    Return a point, p, on the plane (the point-cloud centroid),
    and the normal, n.
    """

    # Extract x, y, z coordinates from the DataFrame
    points = dataframe.iloc[:, 1:].values

    assert points.shape[0] <= points.shape[0], "There are only {} points in {} dimensions.".format(points.shape[1],
                                                                                                   points.shape[0])

    # make an array G with the x and y values of the points
    G = np.ones((points.shape[0], 3))
    G[:, 0] = points[:, 0]  # X
    G[:, 1] = points[:, 1]  # Y

    # make an array Z with the z values of the points
    Z = points[:, 2]
    (a, b, c), resid, rank, s = np.linalg.lstsq(G, Z)
    normal = (a, b, -1)

    # # Calculate the centroid
    # ctr = points.mean(axis=0)
    #
    # # Center the points
    # x = points - ctr
    #
    # # Calculate covariance matrix
    # M = np.dot(x.T, x)  # Could also use np.cov(x) here.
    #
    # normal = linalg.svd(M)[0][:, -1] # normal vector of the plane
    nn = np.linalg.norm(normal) # length of the vector

    normal = normal / nn # a**2 + b**2 + 1 = 1, so divide by sqrt(a**2+b**2)
    ctr = c/nn

    return ctr, normal

# separate the dataframe into 2 sub-dataframes, one for each supermodule
df1 = df2.iloc[:12]
df2 = df2.iloc[12:]

# get the max/min x and y values of the supermodules
def max_min(df):
    """
    Get the max and min x and y values of the supermodule
    :param df: dataframe
    :return: max and min x and y values
    """
    maxx = np.max(df[1])
    maxy = np.max(df[2])
    minx = np.min(df[1])
    miny = np.min(df[2])
    return maxx, maxy, minx, miny

maxx1, maxy1, minx1, miny1 = max_min(df1)
maxx2, maxy2, minx2, miny2 = max_min(df2)

# fit a plane to each supermodule
c1, n1 = fit_plane(df1) # only z coordinate of c1?
c2, n2 = fit_plane(df2)


# get the coordinates of the first point from the dataframe using iloc
point1 = np.array([df1.iloc[0, 1], df1.iloc[0, 2], c1]) # point on the plane
point2 = np.array([df2.iloc[0, 1], df2.iloc[0, 2], c2]) # point on the plane

d1 = -point1.dot(n1) # distance from the origin to the plane
d2 = -point2.dot(n2) # distance from the origin to the plane

# compute needed points for plane plotting
xx1, yy1 = np.meshgrid([minx1, maxx1], [miny1, maxy1])
z1 = (-n1[0]*xx1 - n1[1]*yy1 - d1)*1. / n1[2]

xx2, yy2 = np.meshgrid([minx2, maxx2], [miny2, maxy2])
z2 = (-n2[0]*xx2 - n2[1]*yy2 - d2)*1. / n2[2]


# plot the points from df1 and the corresponding plane
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(z1, yy1, xx1, color = 'r', alpha=0.2)
ax.scatter(df1[1], df1[2], df1[3], marker='o')
# add labels to the points
for i in range(len(df1)):
    ax.text(df1.iloc[i, 1], df1.iloc[i, 2], df1.iloc[i, 3], df1.iloc[i, 0])
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
# set x, y and z axis limits
# ax.set_xlim(minx1-10, maxx1+10)
# ax.set_ylim(miny1-10, maxy1+10)
# ax.set_zlim(np.min(df1[3]), np.max(df1[3]))
plt.show()


#%% Plot a plane that covers the points from the supermodule and find its equation

xx1, yy1 = np.meshgrid([minx1, maxx1], [miny1, maxy1])
# z should span the range of the z values of the supermodule
z1 = np.array([np.min(df1[3]), np.max(df1[3])])
z1, z1 = np.meshgrid(z1, z1)

# plot this plane
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(xx1, yy1, z1, color = 'r', alpha=0.2)
ax.scatter(df1[1], df1[2], df1[3], marker='o')




#%%
# plot the points from df2 and the corresponding plane
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df2[1], df2[2], df2[3], marker='o')
# plot the normal vector of the plane
# ax.quiver(c2[0], c2[1], c2[2], n2[0], n2[1], n2[2], length=10, normalize=True, color='r')
# add labels to the points
for i in range(len(df2)):
    ax.text(df2.iloc[i, 1], df2.iloc[i, 2], df2.iloc[i, 3], df2.iloc[i, 0])
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
# set x, y and z axis limits
ax.set_xlim(minx2-10, maxx2+10)
ax.set_ylim(miny2-10, maxy2+10)
ax.set_zlim(np.min(df2[3]), np.max(df2[3]))
plt.show()


#%% Computing angles between two supermodules to verify polygonal seed geometry

# get the points with labels 02015-00, 01964-00, 01913-00, 01862-00 from the df2_zero dataframe in a new sub dataframe

def get_sub_df(df, df2, labels):
    """
    Get a sub dataframe from the dataframe df with the labels in the list labels
    :param df: dataframe
    :param labels: list of labels
    :return: sub dataframe
    """
    df_sub = df[df[0].isin(labels)]
    df2_sub = df2[df2[0].isin(labels)]
    return df_sub, df2_sub

df_sub, df2_sub = get_sub_df(df_zero, df2_zero, ['02015-00', '01964-00', '01913-00', '01862-00']) #supermodule 1 row
df_sub2, df2_sub2 = get_sub_df(df_zero, df2_zero, ['01811-00', '01760-00', '01709-00', '01658-00']) #supermodule 2 row
df_sub3, df2_sub3 = get_sub_df(df_zero, df2_zero, ['02066-00', '02117-00', '02168-00', '02219-00']) #supermodule 3 row

def plot_sub_df(df, df2):
    """
    Plot the sub dataframe
    :param df: sub dataframe cylinder
    :param df2: sub dataframe polygon
    :return: plot
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(df[1], df[2], marker='o', label = "Cylinder")
    ax.scatter(df2[1], df2[2], marker='o', label = "Polygon")

    fit = np.polyfit(df[1], df[2], 1)
    fit_fn = np.poly1d(fit)
    ax.plot(df[1], fit_fn(df[1]), c='k')

    # fit a curve to the polygon
    fit2 = np.polyfit(df2[1], df2[2], 1)
    fit_fn2 = np.poly1d(fit2)
    ax.plot(df2[1], fit_fn2(df2[1]), c='k')

    print("Line of best fit equation cylinder: ", fit_fn)
    print("Line of best fit equation polygon: ", fit_fn2)
    plt.legend()
    plt.show()
    return fit_fn, fit_fn2

line_1, line2_1 = plot_sub_df(df_sub, df2_sub)
line_2, line2_2 = plot_sub_df(df_sub2, df2_sub2)
line_3, line2_3 = plot_sub_df(df_sub3, df2_sub3)

# compute angle between the lines
def compute_angle(line1, line2):
    """
    Compute angle between two lines in degrees
    :param line1: line of best fit of the first sub dataframe
    :param line2: line of best fit of the second sub dataframe
    :return: angle between the two lines
    """
    m1 = line1[1]
    m2 = line2[1]
    angle = np.arctan(np.abs((m1-m2)/(1+m1*m2)))
    angle = np.rad2deg(angle)
    return angle

angle1 = compute_angle(line1, line2)
angle2 = compute_angle(line1, line3)

print(angle1)
print(angle2)

#%%
import math
from scipy import optimize

# import .dat type file into pandas dataframe, columns are separated by whitespace
df2 = pd.read_csv('parameters/dat/pmt_position_polygon.txt', header=None, delim_whitespace=True)

# df = pd.read_csv('parameters/SK_all_PMT_locations.txt', header=None, delim_whitespace=True)

pmts = ['00810', '00809', '00808', '00759', '00758', '00757', '00708', '00707', '00706', '00657', '00656', '00655',
        '00606', '00605', '00604', '00555', '00554', '00553', '00504', '00503', '00502', '00453', '00452', '00451']

# create dataframe with only the rows where the 0-th column's first 5 characters are in the list pmts
df2 = df2[df2[0].str[:5].isin(pmts)]

# only trying with the first supermodule
df1 = df2.iloc[:12]
x, y, z = df1[1], df1[2], df1[3]
# convert x, y and z to numpy arrays
x = np.array(x)
y = np.array(y)
z = np.array(z)


def estimate_initial_guesses(points):
    # points is a list of lists [[x,y,z], [x,y,z], ...]
    p1 = points[0]
    p2 = points[1]
    p3 = points[2]
    a1 = p2[0] - p1[0]
    b1 = p2[1] - p1[1]
    c1 = p2[2] - p1[2]
    a2 = p3[0] - p1[0]
    b2 = p3[1] - p1[1]
    c2 = p3[2] - p1[2]
    a = b1 * c2 - b2 * c1
    b = a2 * c1 - a1 * c2
    c = a1 * b2 - b1 * a2
    d = (- a * p1[0] - b * p1[1] - c * p1[2])
    return [a, b, c, d]
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

# create list points where it is a list of lists [[x,y,z], [x,y,z], ...] out of the x, y and z values for indices 1, 5, 6
points = [[x[i], y[i], z[i]] for i in [1, 5, 6]]

initial_guess = estimate_initial_guesses(points)
initial_guess = initial_guess[0], initial_guess[1], initial_guess[2], initial_guess[3]

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
report('Initial guess ', initial_guess)


minx1, maxx1, miny1, maxy1 = np.min(x), np.max(x), np.min(y), np.max(y)

X, Y = np.meshgrid([minx1, maxx1], [miny1, maxy1])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

def Z(X, Y, params):
    a, b, c, d = params
    return -(a*X + b*Y + d)/c

ax.plot_surface(Z(X, Y, initial_guess), Y, X, rstride=1, cstride=1, alpha=0.3, color='magenta')
ax.plot_surface(Z(X, Y, vert_params), Y, X, rstride=1, cstride=1, alpha=0.3, color='yellow')
ax.plot_surface(Z(X, Y, perp_params), Y, X, rstride=1, cstride=1, alpha=0.3, color='green')
ax.scatter(x, y, z, c='r', s=50)
plt.xlabel('X')
plt.ylabel('Y')
ax.set_zlabel('Z')
ax.axis('equal')
ax.axis('tight')
# set z limit to 20 above largest z value and 20 below smallest z value
ax.set_zlim(np.min(z)-20, np.max(z)+20)
plt.show()
