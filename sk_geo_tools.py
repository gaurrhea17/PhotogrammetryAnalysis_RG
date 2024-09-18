
import numpy as np
from scipy import linalg
import re
import matplotlib.pyplot as plt
import pandas as pd

pmt_bolt_offset = 15
bolt_ring_radius = 29.8
bolt_distance = 2 * bolt_ring_radius * np.sin(np.pi / 24)
bolt_count = 24

def fit_plane(points):  # https://stackoverflow.com/questions/12299540/plane-fitting-to-4-or-more-xyz-points

    # convert points to numpy array of floats
    points = np.array(points, dtype=float)

    # assert that the number of points is greater than the number of dimensions
    assert points.shape[0] <= points.shape[0], "There are only {} points in {} dimensions.".format(points.shape[1],
                                                                                                   points.shape[0])
    p = points[0] # point on the plane

    ctr = points.mean(axis=0)  # center of the points
    x = points - ctr  # center the points
    M = np.dot(x.T, x)  # dot product of the centered points to get the covariance matrix. Could also use np.cov(x.T)
    _, _, n = linalg.svd(M)  # singular value decomposition of the covariance matrix

    n = n[-1]  # last column of unitary matrix n is the normal vector to the plane

    if n[0] > 0 and n[1] > 0:
        n = -n

    a, b, c = n
    d = np.dot(n,p) # Equation of the plane is ax + by + cz + d = 0

    return a, b, c, d # plane equation coefficients; the normal vector is simply (a, b, c)


def get_bolt_locations_barrel(pmt_locations):
    bolt_locations = {}
    step = 24 // bolt_count
    for f, pmt in pmt_locations.items():
        match = re.fullmatch(r"[0-1][0-9]{4}-00", f)
        if not match:
            continue
        phi = np.arctan2(pmt[1], pmt[0])
        bolt_locations.update({
            re.sub(r"-00$", "-" + str(i + 1).zfill(2), f): np.array([
                pmt[0] - pmt_bolt_offset * np.cos(phi) + bolt_ring_radius * np.sin(i * np.pi / 12.) * np.sin(phi),
                pmt[1] - pmt_bolt_offset * np.sin(phi) - bolt_ring_radius * np.sin(i * np.pi / 12.) * np.cos(phi),
                pmt[2] + bolt_ring_radius * np.cos(i * np.pi / 12.)])
            for i in range(0, 24, step)})
    return bolt_locations


def get_bolt_locations_poly(pmt_locations):

    bolt_locations = {}

    # create df with pmt_id, sm_id, x, y, z

    df_pmt_locations = pd.DataFrame.from_dict(pmt_locations, orient='index', columns=['SM_ID', 'x', 'y', 'z'])
    df_pmt_locations.index.name = 'PMT_ID'
    df_pmt_locations.reset_index(inplace=True)

    # remove first value from all keys in pmt_locations
    pmt_locations = {k: v[1:] for k, v in pmt_locations.items()}

    # get dataframe with the a, b, c, d coefficients of the plane for each SM using fit_sm_plane
    sm_planes = df_pmt_locations.groupby('SM_ID').apply(lambda group: fit_plane(group[['x', 'y', 'z']].values))
    sm_planes = pd.DataFrame(sm_planes.to_list(), columns=['a', 'b', 'c', 'd'], index=sm_planes.index)  # name columns in sm_planes dataframe: SM_ID, a, b, c, d

    # normalize the normal vectors in sm_planes
    sm_planes['a'], sm_planes['b'], sm_planes['c'] = sm_planes['a'] / np.linalg.norm(sm_planes[['a', 'b', 'c']], axis=1), sm_planes['b'] / np.linalg.norm(sm_planes[['a', 'b', 'c']], axis=1), sm_planes['c'] / np.linalg.norm(sm_planes[['a', 'b', 'c']], axis=1)

    print("The supermodule plane coefficients are: ", sm_planes)

    theta = np.linspace(2*np.pi, 0, 24, endpoint=False)  # 24 points around the bolt circle

    # for each supermodule, calulate two vectors perpendicular to the normal vector of the plane
    v1 = pd.DataFrame(columns=['a', 'b', 'c'], index=sm_planes.index)
    v1['a'], v1['b'], v1['c'] = 1, 0, sm_planes['a'] / sm_planes['c'] # calculate v1 by setting a = 1, b = 0, c = a/c
    v1 = v1.div(np.linalg.norm(v1, axis=1), axis=0)     # normalize v1

    v2 = pd.DataFrame(columns=['a', 'b', 'c'], index=sm_planes.index)
    v2['a'], v2['b'], v2['c'] = np.cross(sm_planes[['a', 'b', 'c']], v1).T     # get v2 by taking the cross product of the normal vector and v1
    v2 = v2.div(np.linalg.norm(v2, axis=1), axis=0)     # normalize v2

    # calculate 24 bolt points around each pmt using normal vector and v1 and v2 for respective supermodules
    bolt_locations = {} # dictionary to hold bolt locations

    for index, row in df_pmt_locations.iterrows():
        pmt_id = row['PMT_ID']
        sm_id = row['SM_ID']

        # pmt coordinates
        pmt_coords = np.array([row['x'], row['y'], row['z']])

        # Get corresponding v1 and v2 for the supermodule

        # sm_index = df_pmt_locations['SM_ID'].unique().tolist().index(sm_id)
        # v1_sm = np.array(v1.iloc[sm_index].values, dtype=float)
        # v2_sm = np.array(v2.iloc[sm_index].values, dtype=float)
        normal_sm = np.array(sm_planes.loc[sm_id, ['a', 'b', 'c']], dtype=float)

        v1_sm = np.array([0, 0, 1])
        # if np.dot(normal_sm, v1_sm) < 0:
        #     v1_sm = -v1_sm

        v2_sm = np.cross(normal_sm, v1_sm)
        v2_sm = v2_sm / np.linalg.norm(v2_sm)

        v1_sm = np.cross(v2_sm, normal_sm)
        v1_sm = v1_sm / np.linalg.norm(v1_sm)

        print("PMT ID: ", pmt_id)
        print("SM ID: ", sm_id)
        print("v1 and v2 are ", v1_sm, v2_sm)

        # shift pmt coordinates in front of sm plane by pmt_bolt_offset
        shifted_pmt_coords = pmt_coords + pmt_bolt_offset * normal_sm

        # Calculate bolt points clockwise around the pmt starting from the top
        bolt_points = np.array([shifted_pmt_coords + bolt_ring_radius * np.cos(t) * v1_sm + bolt_ring_radius * np.sin(t) * v2_sm for t in theta])

        # Update bolt_locations
        bolt_locations.update({f"{pmt_id[:6]}{j + 1:02d}": bolt_points[j] for j in range(24)})

    return pmt_locations, df_pmt_locations, sm_planes, bolt_locations


def get_bolt_distances(bolt_locations):
    bolt_distances = []
    for b, l in bolt_locations.items():
        next_bolt = b[:-2] + str(int(b[-2:]) % 24 + 1).zfill(2)
        if next_bolt in bolt_locations:
            bolt_distances.append(linalg.norm(l - bolt_locations[next_bolt]))
    return bolt_distances

def get_bolt_dists_new(feature_locations):
    # from dataframe feature_locations, remove PMTs
    bolt_df = feature_locations[~feature_locations['feature_id'].str.endswith('00')]
    bolt_df = bolt_df.sort_values(by='feature_id')
    bolt_df.reset_index(drop=True, inplace=True)

    # add a column for 'adjacent_bolt_dist'
    bolt_df['adjacent_bolt_dist'] = np.nan

    # calculate the distance between adjacent bolts for the same PMT. Bolts belong to same PMT have the same first 6 characters in feature_id
    for i in range(0, len(bolt_df)):
        next_bolt = bolt_df.loc[i, 'feature_id'][:-2] + str(int(bolt_df.loc[i, 'feature_id'][-2:]) % 24 + 1).zfill(2)

        if next_bolt in bolt_df['feature_id'].values:
            bolt_df.loc[i, 'adjacent_bolt_dist'] = np.linalg.norm(bolt_df.loc[i, ['x', 'y', 'z']].values - bolt_df.loc[bolt_df[bolt_df['feature_id'] == next_bolt].index[0], ['x', 'y', 'z']].values)
    return bolt_df


def get_unique_pmt_ids(feature_locations):
    return set(f[:5] for f in feature_locations.keys())


def get_bolt_ring_centres(bolt_locations):
    pmt_ids = get_unique_pmt_ids(bolt_locations)
    return {p: np.mean([bolt_locations[p + "-" + str(b).zfill(2)]
                        for b in range(1, 25) if p + "-" + str(b).zfill(2) in bolt_locations.keys()], axis=0)
            for p in pmt_ids}


def get_bolt_ring_radii(bolt_locations):
    bolt_ring_centres = get_bolt_ring_centres(bolt_locations)
    return [linalg.norm(l - bolt_ring_centres[b[:5]]) for b, l in bolt_locations.items()]

def get_bolt_ring_radii_new(feature_locations):

    # from dataframe feature_locations, remove PMTs
    bolt_df = feature_locations[~feature_locations['feature_id'].str.endswith('00')]
    bolt_df = bolt_df.sort_values(by='feature_id')
    bolt_df.reset_index(drop=True, inplace=True)

    # add a column for 'bolt_ring_radius'
    bolt_df['bolt_ring_radius'] = np.nan

    # calculate the distance between the bolt and the centre of the bolt ring
    for i in range(0, len(bolt_df)):
        pmt_id = bolt_df.loc[i, 'feature_id'][:5]

        # get mean of all bolt coordinates with the first 5 characters the same as pmt_id
        bolt_ring_centre = bolt_df[bolt_df['feature_id'].str.contains(pmt_id)]['x'].mean(), bolt_df[bolt_df['feature_id'].str.contains(pmt_id)]['y'].mean(), bolt_df[bolt_df['feature_id'].str.contains(pmt_id)]['z'].mean()
        bolt_df.loc[i, 'bolt_ring_radius'] = np.linalg.norm(bolt_df.loc[i, ['x', 'y', 'z']].values - bolt_ring_centre)

    return bolt_df


# def get_bolt_ring_planes(bolt_locations):
#     pmt_ids = get_unique_pmt_ids(bolt_locations)
#     planes = {}
#     for p in pmt_ids:
#         c, n = fit_plane(np.array([bolt_locations[p + "-" + str(b).zfill(2)]
#                                    for b in range(1, 25) if p + "-" + str(b).zfill(2) in bolt_locations.keys()]))
#         # flip normal if it is directed away from tank centre
#         if n[0] > 0 and n[1] > 0:
#             n = -n
#         planes[p] = c, n
#     return planes
#
#
# def get_supermodule_plane(bolt_locations, min_pmt, max_pmt):
#     c, n = fit_plane(np.array([l for b, l in bolt_locations.items() if min_pmt <= int(b[:5]) <= max_pmt]))
#     # flip normal if it is directed away from tank centre
#     if n[0] > 0 and n[1] > 0:
#         n = -n
#     return c, n

## Plotting functions


def plot_plane(a, b, c, d, points, ax):

    # grid of points
    # get coordinate with minimum x coordinate from points
    x_range = np.linspace(np.min(points[:,0]), np.max(points[:,0]))
    z_range = np.linspace(np.min(points[:,2]), np.max(points[:,2]))
    X, Z = np.meshgrid(x_range, z_range)

    # calculate y values
    Y = (-a*X - c*Z + d) / b

    ax.plot_surface(X,Y,Z, alpha=0.5)

    plt.show()

def plot_seed_poly(df_pmt_locations, sm_planes, bolt_locations, top=False, planes = True):
    # plot bolts
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    bolt_array = np.stack(list(bolt_locations.values()))
    ax.scatter(bolt_array[:, 0], bolt_array[:, 1], bolt_array[:, 2], marker='.', label="Bolt (seed position)")
    # add text labels to the bolts
    for i, f in bolt_locations.items():
        ax.text(f[0], f[1], f[2], i[6:], size=8, zorder=4, color='k')

    # plot pmts
    for sm, points in df_pmt_locations.groupby('SM_ID'):
        ax.scatter(points['x'], points['y'], points['z'], marker='.', label=f"SM{sm} PMT locations")
        # add text labels to the PMTs
        for i, f in points.iterrows():
            ax.text(f['x'], f['y'], f['z'], f['PMT_ID'][:5], size=8, zorder=4, color='k')

    ## plot the fitted planes
    if planes == True:
        for sm, plane in sm_planes.iterrows():
            plot_plane(*plane, df_pmt_locations[df_pmt_locations['SM_ID'] == sm][['x', 'y', 'z']].values, ax)
    elif planes == False:
        pass
    plt.legend(loc=0)
    plt.show()

    if top == True:
        fig, ax = plt.subplots(figsize=(9, 9))
        bolt_array = np.stack(list(bolt_locations.values()))
        ax.scatter(bolt_array[:, 0], bolt_array[:, 1], marker='.', label="Bolt (seed position)")
        for sm, points in df_pmt_locations.groupby('SM_ID'):
            ax.scatter(points['x'], points['y'], marker='.', label=f"SM{sm} PMT locations")
            for i, f in points.iterrows():
                ax.text(f['x'], f['y'], f['PMT_ID'][:5], size=6, zorder=4, color='k')
        plt.legend(loc=0)
        ax.set_xlabel("x [cm]", fontsize=12)
        ax.set_ylabel("y [cm]", fontsize=12)
        ax.set_title("Seed Positions (Top View)")
        plt.show()


def plot_cam_poses(pmt_locations, common_feature_locations, camera_positions):
    fig = plt.figure(figsize=(12, 9))
    pmt_array = np.stack(list(pmt_locations.values()))
    feat_array = np.stack(list(common_feature_locations.values()))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(feat_array[:, 0], feat_array[:, 1], feat_array[:, 2],
               marker='o', color='#1f77b4', label="Seed Positions", zorder=2, s=15)
    ax.scatter(camera_positions[:, 0], camera_positions[:, 1], camera_positions[:, 2],
               marker='*', color='#ff7f0e', label="Camera Position Estimate", zorder=1, s=50)

    for i, f in enumerate(pmt_locations.keys()):
        ax.text(pmt_array[i, 0], pmt_array[i, 1], pmt_array[i, 2], f[:5],
                size=8, zorder=4, color='black')

    legend = ax.legend(loc='upper right', fontsize=12)
    legend.set_bbox_to_anchor((1.15, 1))

    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    ax.set_facecolor('whitesmoke')

    ax.set_xlabel('X Axis', fontsize=12, labelpad=10)
    ax.set_ylabel('Y Axis', fontsize=12, labelpad=10)
    ax.set_zlabel('Z Axis', fontsize=12, labelpad=10)
    ax.set_title('Camera and Seed Positions', fontsize=16, pad=20)

    fig.tight_layout()
    ax.view_init(30, -100)
    plt.show()

def hist_res(df_res):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(df_res['RE'], bins='auto', color='green', edgecolor='black', linewidth=1, histtype='bar', alpha=0.7)
    ax.set_title("Histogram of reprojection errors")

    ax.set_xlabel("Reprojection error [px]")
    ax.set_ylabel("Number of features")
    # put the mean and max reprojection errors on the plot without overlapping the histogram
    textstr = '\n'.join((f'Mean: {df_res["RE"].mean():.2f} px', f'Max: {df_res["RE"].max():.2f} px'))
    props = dict(boxstyle='round', facecolor='lightyellow', alpha=0.5)
    plt.text(0.95, 0.95, textstr, transform=ax.transAxes, fontsize=12, verticalalignment='top', horizontalalignment='right', bbox=props)

    ## cosmetics
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.show()

def hist_residuals(df_residuals):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(df_residuals['residual'], bins='auto', color='green', edgecolor='black', linewidth=1, histtype='bar', alpha=0.7)
    ax.set_title("Histogram of residuals")

    ax.set_xlabel("Residual [cm]")
    ax.set_ylabel("Number of features")
    # put the mean and max residuals on the plot without overlapping the histogram
    textstr = '\n'.join((f'Mean: {df_residuals["residual"].mean():.2f} cm', f'Max: {df_residuals["residual"].max():.2f} cm'))
    props = dict(boxstyle='round', facecolor='lightyellow', alpha=0.5)
    plt.text(0.95, 0.95, textstr, transform=ax.transAxes, fontsize=12, verticalalignment='top', horizontalalignment='right', bbox=props)

    ## cosmetics
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.show()

def hist_bolt_dists(bolt_dists):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(bolt_dists['adjacent_bolt_dist'], bins='auto', color='lightcoral', edgecolor='black', linewidth=1, histtype='bar', alpha=0.7)

    # show mean and standard deviation on the plot
    textstr = '\n'.join((f'Mean: {bolt_dists["adjacent_bolt_dist"].mean():.2f} cm', f'Std. Dev.: {bolt_dists["adjacent_bolt_dist"].std():.2f} cm'))
    props = dict(boxstyle='round', facecolor='lightyellow', alpha=0.5)
    plt.text(0.95, 0.95, textstr, transform=ax.transAxes, fontsize=12, verticalalignment='top', horizontalalignment='right', bbox=props)

    ax.set_title("Reconstructed distance between adjacent bolts (cm)", fontsize=14)
    ax.set_xlabel("Distance between adjacent bolts (cm)", fontsize=13)
    ax.set_ylabel("Number of bolts", fontsize=13)

    ## cosmetics
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.show()

def hist_bolt_ring_radii(bolt_ring_radii):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(bolt_ring_radii['bolt_ring_radius'], bins='auto', color='lightcoral', edgecolor='black', linewidth=1, histtype='bar', alpha=0.7)

    # show mean and standard deviation on the plot
    textstr = '\n'.join((f'Mean: {bolt_ring_radii["bolt_ring_radius"].mean():.2f} cm', f'Std. Dev.: {bolt_ring_radii["bolt_ring_radius"].std():.2f} cm'))
    props = dict(boxstyle='round', facecolor='lightyellow', alpha=0.5)
    plt.text(0.95, 0.95, textstr, transform=ax.transAxes, fontsize=12, verticalalignment='top', horizontalalignment='right', bbox=props)

    ## titles and labels
    ax.set_title("Reconstructed distance between bolts and the centre of the bolt ring (cm)")
    ax.set_xlabel("Distance between bolt and the centre of the bolt ring (cm)")
    ax.set_ylabel("Number of bolts")

    ## cosmetics
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.show()