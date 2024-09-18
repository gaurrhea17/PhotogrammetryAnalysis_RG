from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))


%load_ext autoreload
%autoreload 2


import numpy as np
from scipy import linalg
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import cv2
import os
from collections import Counter
import pg_fitter_tools_new as fit
import sk_geo_tools as sk


# Model used by Nick in 24 PMT and ring reconstruction
fx, fy, cx, cy, k1, k2, k3, p1, p2, skew = 2.760529621789217e3, 2.767014510543478e3, 1.914303537872458e3, 1.596386868474348e3, -0.2398, 0.1145, 0, 0, 0, 0

focal_length = np.array([fx, fy])
principle_point = np.array([cx, cy])
radial_distortion = np.array([k1, k2, k3])
tangential_distortion = np.array([p1, p2])


all_pmt_locations = fit.read_3d_feature_locations("parameters/dat/pmt_position_polygon.txt") ## polygonal seed geometry
offset = np.array([0, 250])


path = 'source/LI_points/' # GUI feature detection and labeling
# path = 'BarrelSurveyFar_TopInjector_PD3/BarrelSurveyFar_TopInjector_median_texts/' # Manual feature detection and labeling

image_feature_locations = {}

# pass all of the text files in path to read_image_feature_locations
for file in os.listdir(path):
    if file.endswith(".txt"):
        image_feature_locations.update(fit.read_image_feature_locations(path+file, offset=offset))


# choose features that appear in 2+
feature_counts = Counter([f for i in image_feature_locations.values() for f in i.keys()])
common_features = [f for f in feature_counts if feature_counts[f] > 1]
pmt_locations = {k: p for k, p in all_pmt_locations.items() if k in common_features}

print("The number of features that appear in 2+ images is", len(common_features))


# generate bolt locations from PMT locations
bolt_locations = sk.get_bolt_locations_barrel(pmt_locations)
common_bolt_locations = {k: b for k, b in bolt_locations.items() if k in common_features}
common_feature_locations = {**pmt_locations, **bolt_locations}
common_image_pmt_locations = {
    k: {j: f for j, f in i.items() if j in common_features and j in pmt_locations}
    for k, i in image_feature_locations.items()}

common_image_feature_locations = {
    k: {j: f for j, f in i.items() if j in common_features and j in common_feature_locations}
    for k, i in image_feature_locations.items()}

common_image_bolt_locations = {
    k: {j: f for j, f in i.items() if j in common_features and j in bolt_locations}
    for k, i in image_feature_locations.items()}

nimages = len(common_image_feature_locations) # number of images with common features
nfeatures = len(common_feature_locations) # number of features that appear in at least 2 images
print("The number of images and common features is", nimages, nfeatures)


fitter_pmts = fit.PhotogrammetryFitter(common_image_pmt_locations, pmt_locations,
                                       focal_length, principle_point, skew, radial_distortion, tangential_distortion)
fitter_bolts = fit.PhotogrammetryFitter(common_image_bolt_locations, common_bolt_locations,
                                       focal_length, principle_point, skew, radial_distortion, tangential_distortion)
fitter_all = fit.PhotogrammetryFitter(common_image_feature_locations, common_feature_locations,
                                       focal_length, principle_point, skew, radial_distortion, tangential_distortion)


camera_rotations, camera_translations, reprojected_points = fitter_all.estimate_camera_poses(flags=cv2.SOLVEPNP_EPNP)


camera_orientations, camera_positions = fit.camera_world_poses(camera_rotations, camera_translations)


## Saving necessary results to text files

# change the shape of camera_orientations such that it is each 3x3 matrix in the array is a row
camera_orientations = camera_orientations.reshape((camera_orientations.shape[0], 9))

# change shape of camera_positions to be each row is a camera position
camera_positions = camera_positions.reshape((camera_positions.shape[0], 3))

# save camera positions and camera orientations to a text file called "reproduce_camera_poses.txt" and put it in the 'results/24_PMT_Study/' folder. Add a column in the beginning with the image name
np.savetxt('results/LI_camera_poses.txt', np.hstack((np.array([fitter_all.index_image[i] for i in range(len(fitter_all.index_image))]).reshape((len(fitter_all.index_image), 1)), camera_positions, camera_orientations)), fmt='%s', delimiter='\t')

# save the reprojected points to a text file called "reproduce_reprojected_points.txt" and put it in the 'results/24_PMT_Study/' folder. Add a column in the beginning with the image name
np.savetxt('results/LI_reprojected_points.txt', np.hstack((np.array([fitter_all.index_image[i] for i in range(len(fitter_all.index_image))]).reshape((len(fitter_all.index_image), 1)), np.array(list(reprojected_points.values())).reshape((len(fitter_all.index_image), -1)))), fmt='%s', delimiter='\t')



camera_rotations, camera_translations, reco_locations, opt = fitter_all.bundle_adjustment(camera_rotations, camera_translations, use_sparsity=True, fit_cam=False, return_opt=True)

## fitter error calculations
# J=opt.jac[:,fitter_all.nimages*6:].toarray() # Jacobian matrix for the feature locations
# U, s, Vh = linalg.svd(J, full_matrices=False) # singular value decomposition. U and Vh are unitary matrices, s contains J's singular values
# tol = np.finfo(float).eps*s[0]*max(J.shape) # tolerance for zero singular values
# w = s > tol # indices of nonzero singular values
# cov = (Vh[w].T/s[w]**2) @ Vh[w]  # robust covariance matrix
# perr = np.sqrt(np.diag(cov))     # 1sigma uncertainty on fitted parameters
# feature_location_errors = perr.reshape((-1, 3))
#
# opt.fun.size # number of residuals
# opt.x.size # number of parameters
# opt.nfev # number of function evaluations
# opt.njev # number of jacobian evaluations
# perr.size # number of fitted parameters
#
# chi2dof = np.sum(opt.fun**2)/(opt.fun.size - perr.size) # chi2 per degree of freedom
# cov_scaled = cov*chi2dof # scaled covariance matrix
# perr = np.sqrt(np.diag(cov_scaled))    # 1sigma uncertainty on fitted parameters
# feature_location_errors = perr.reshape((-1, 3)) # reshape to match feature locations
#
#
# # separating out the errors into x, y, z components
# x_variance = np.diag(cov_scaled)[0::3]
# y_variance = np.diag(cov_scaled)[1::3]
# z_variance = np.diag(cov_scaled)[2::3]
# variance_3d = x_variance + y_variance + z_variance
# xy_covariance = cov_scaled[range(0,cov_scaled.shape[0],3),range(1,cov_scaled.shape[1],3)]
# xz_covariance = cov_scaled[range(0,cov_scaled.shape[0],3),range(2,cov_scaled.shape[1],3)]
# yz_covariance = cov_scaled[range(1,cov_scaled.shape[0],3),range(2,cov_scaled.shape[1],3)]
# radial_positions = linalg.norm(fitter_all.reco_locations[:,:2], axis=1)
# x_positions = fitter_all.reco_locations[:,0]
# y_positions = fitter_all.reco_locations[:,1]
# z_positions = fitter_all.reco_locations[:,2]
# radial_variance = ((x_positions**2)*x_variance+(y_positions**2)*y_variance+2*x_positions*y_positions*xy_covariance)/radial_positions**2
# tangential_variance = variance_3d-radial_variance
#
#
# fig, ax = plt.subplots(figsize=(8,6))
# ax.hist(np.sqrt(x_variance), bins=20, histtype="step", color="r", label=f"x position error\nmean = {np.mean(np.sqrt(x_variance)):.3} cm")
# ax.hist(np.sqrt(y_variance), bins=20, histtype="step", color="b", label=f"y position error\nmean = {np.mean(np.sqrt(y_variance)):.3} cm")
# ax.hist(np.sqrt(z_variance), bins=20, histtype="step", color="g", label=f"z position error\nmean = {np.mean(np.sqrt(z_variance)):.3} cm")
# ax.set_xlabel("Fitter error on feature position [cm]", fontsize=15)
# ax.set_ylabel("Number of features", fontsize=15)
# plt.legend(loc="upper right", prop={'size': 15})
# plt.title("X, Y, Z Direction Fitter Errors")
#
# # save xyz plot to results folder
# plt.savefig("results/LI_xyz_fitter_errors.png", fig)
#
# fig, ax = plt.subplots(figsize=(8,6))
# ax.hist(np.sqrt(variance_3d), bins=20, histtype="step", label=f"3D position error\nmean = {np.mean(np.sqrt(variance_3d)):.3} cm")
# ax.set_xlabel("Fitter error on 3D feature position [cm]", fontsize=15)
# ax.set_ylabel("Number of features", fontsize=15)
# plt.legend(loc="upper right", prop={'size': 15})
# ax.set_title("Fitter Error on 3D Feature Positions", fontsize=15)
#
# plt.savefig("results/LI_3D_fitter_errors.png", fig)
#
# fig, ax = plt.subplots(figsize=(8,6))
# ax.hist(np.sqrt(radial_variance), bins=20, histtype="step", color="b", label=f"Radial position error\nmean = {np.mean(np.sqrt(radial_variance)):.3} cm")
# ax.hist(np.sqrt(tangential_variance), bins=20, histtype="step", color="r", label=f"Tangential position error\nmean = {np.mean(np.sqrt(tangential_variance)):.3} cm")
# ax.set_xlabel("Fitter error on radial feature position [cm]", fontsize=15)
# ax.set_ylabel("Number of features", fontsize=15)
# plt.legend(loc="upper right", prop={'size': 15})
# ax.set_title("Fitter Error on Radial and Tangential Feature Positions", fontsize=15)
#
# plt.savefig("results/LI_rad_tan_errors.png", fig)
#
#
#
# # save the fitter errors to a text file called "newFD_fitter_errors.txt" and put it in the 'results/24_PMT_Study/' folder
# np.savetxt('results/LI_3D_fitter_errors.txt', np.hstack((np.array([fitter_all.index_feature[i] for i in range(len(fitter_all.index_feature))]).reshape((len(fitter_all.index_feature), 1)), np.sqrt(variance_3d).reshape((len(fitter_all.index_feature), 1)))), fmt='%s', delimiter='\t')
#
# # save the x, y, z, fitter position errors to a text file called "newFD_xyz_errors.txt" and put it in the 'results/24_PMT_Study/' folder
# np.savetxt('results/LI_xyz_fitter_errors.txt', np.hstack((np.array([fitter_all.index_feature[i] for i in range(len(fitter_all.index_feature))]).reshape((len(fitter_all.index_feature), 1)), np.sqrt(x_variance).reshape((len(fitter_all.index_feature), 1)), np.sqrt(y_variance).reshape((len(fitter_all.index_feature), 1)), np.sqrt(z_variance).reshape((len(fitter_all.index_feature), 1)))), fmt='%s', delimiter='\t')
#
# # save the radial and tangential fitter position errors to a text file called "newFD_radial_tangential_errors.txt" and put it in the 'results/24_PMT_Study/' folder
# np.savetxt('results/LI_rad_tan_fitter_errors.txt', np.hstack((np.array([fitter_all.index_feature[i] for i in range(len(fitter_all.index_feature))]).reshape((len(fitter_all.index_feature), 1)), np.sqrt(radial_variance).reshape((len(fitter_all.index_feature), 1)), np.sqrt(tangential_variance).reshape((len(fitter_all.index_feature), 1)))), fmt='%s', delimiter='\t')


errors, reco_transformed, scale, R, translation, location_mean = fit.kabsch_errors(
    common_feature_locations, reco_locations)
print("mean reconstruction error:", linalg.norm(errors, axis=1).mean())
print("max reconstruction error:", linalg.norm(errors, axis=1).max())


camera_orientations, camera_positions = fit.camera_world_poses(camera_rotations, camera_translations)
camera_orientations = np.matmul(R, camera_orientations)
camera_positions = camera_positions - translation
camera_positions = scale*R.dot(camera_positions.transpose()).transpose() + location_mean

# save the reconstructed positions to a text file called "reproduce_reconstructed_positions.txt" and put it in the 'results/24_PMT_Study/' folder. Add a column in the beginning with the feature name
np.savetxt('results/LI_reconstructed_positions.txt', np.hstack((np.array([fitter_all.index_feature[i] for i in range(len(fitter_all.index_feature))]).reshape((len(fitter_all.index_feature), 1)), reco_transformed)), fmt='%s', delimiter='\t')

# save the reprojection errors to a text file called "reproduce_reprojection_errors.txt" and put it in the 'results/24_PMT_Study/' folder. Add a column in the beginning with the feature name
np.savetxt('results/LI_reprojection_errors.txt', np.hstack((np.array([fitter_all.index_feature[i] for i in range(len(fitter_all.index_feature))]).reshape((len(fitter_all.index_feature), 1)), reprojection_errors)), fmt='%s', delimiter='\t')

# save the residuals to a text file called "reproduce_residuals.txt" and put it in the 'results/24_PMT_Study/' folder. Add a column in the beginning with the feature name
np.savetxt('results/LI_residuals', np.hstack((np.array([fitter_all.index_feature[i] for i in range(len(fitter_all.index_feature))]).reshape((len(fitter_all.index_feature), 1)), errors)), fmt='%s', delimiter='\t')

bolt_dict = {b: reco_transformed[fitter_all.feature_index[b]] for b in common_bolt_locations.keys()}
bolt_dists = sk.get_bolt_distances(bolt_dict)
fig, ax = plt.subplots(figsize=(8,6))
ax.hist(bolt_dists, bins='auto')
ax.set_title("Reconstructed distance between adjacent bolts (cm)")
ax.axvline(linewidth=2, color='r', x=sk.bolt_distance)
fig.tight_layout()

# include the mean and std of bolt distances on the plot
ax.text(0.05, 0.95, "mean = {:.2f} cm\nstd = {:.2f} cm".format(np.mean(bolt_dists), np.std(bolt_dists)), transform=ax.transAxes, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
ax.set_xlabel("Distance between adjacent bolts (cm)")
ax.set_ylabel("Number of bolts")

# print mean and std of bolt distances
print ('mean and std of bolt distances:', np.mean(bolt_dists), np.std(bolt_dists))

plt.savefig("results/LI_bolt_dists.png", fig)

bolt_radii = sk.get_bolt_ring_radii(bolt_dict)
fig, ax = plt.subplots(figsize=(8,6))
ax.hist(bolt_radii, bins='auto')
ax.set_title("Reconstructed distance between bolts and centre of bolt ring (cm)")
ax.axvline(linewidth=2, color='r', x=sk.bolt_ring_radius)
fig.tight_layout()

# include the mean and std of bolt radii on the plot
ax.text(0.05, 0.95, "mean = {:.2f} cm\nstd = {:.2f} cm".format(np.mean(bolt_radii), np.std(bolt_radii)), transform=ax.transAxes, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

ax.set_xlabel("Distance between bolts and centre of bolt ring (cm)")
ax.set_ylabel("Number of bolts")

# print mean and std of bolt distances
print ('mean and std of bolt radii:', np.mean(bolt_radii), np.std(bolt_radii))

plt.savefig("results/LI_bolt_radii.png", fig)

## Save the bolt distances and bolt radii to text files

# save the bolt distances to a text file called "reproduce_bolt_distances.txt" and put it in the 'results/24_PMT_Study/' folder
np.savetxt('results/LI_bolt_dists.txt', bolt_dists, fmt='%s', delimiter='\t')

# save the bolt radii to a text file called "reproduce_bolt_radii.txt" and put it in the 'results/24_PMT_Study/' folder
np.savetxt('results/LI_bolt_radii.txt', bolt_radii, fmt='%s', delimiter='\t')