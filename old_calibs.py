# import necessary packages
import numpy as np
import h5py

def read_h5py(filename):
    f = h5py.File(filename, 'r')
    images = np.array(f.get('images'))
    imgpoints = np.array(f.get('imgpoints'))
    objpoints = np.array(f.get('objpoints'))
    f.close()
    return images, imgpoints, objpoints


# define path to h5py file
filename = r'C:\Users\gaurr\OneDrive - TRIUMF\Hyper-K\Calibration\old_drone_covs.h5'

# read in the h5py file
images, imgpoints, objpoints = read_h5py(filename)

# print the shape of each array
print(images.shape)
print(imgpoints.shape)
print(objpoints.shape)

#%%

import matplotlib.pyplot as plt
import numpy as np

# load in file
file = r'C:\Users\gaurr\OneDrive - TRIUMF\UBC\RG_Masters_Thesis\MATLAB_camcal_xy_RE.txt'

# load in the data in variables x, y and RE
x, y, RE = np.loadtxt(file, unpack=True)

# plot RE vs distance from the centre of the image (2000, 1500)
plt.plot(np.sqrt((x - 2000)**2 + (y - 1500)**2), RE, 'o', markersize=0.7)
plt.xlabel('Distance from centre of image [pixels]')
plt.ylabel('Reprojection error [pixels]')
plt.title('Reprojection error vs distance from centre of image')

# fit a function to the data
# define the function
def func(x, a, b, c):
    return a * np.exp(-b * x) + c

# fit the function to the data
from scipy.optimize import curve_fit
popt, pcov = curve_fit(func, np.sqrt((x - 2000)**2 + (y - 1500)**2), RE)

# plot the fit
plt.plot(np.sqrt((x - 2000)**2 + (y - 1500)**2), func(np.sqrt((x - 2000)**2 + (y - 1500)**2), *popt), 'r-', label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))

# print the fit parameters
print(popt)

plt.show()

#%%
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')
import numpy as np
from scipy import linalg

# load in the points from results/24_PMT_study/Reproduce/reproduce_reprojection_errors.txt
# define path to text file
# filename = r'C:\Users\gaurr\OneDrive - TRIUMF\Super-K\Reconstruction\PhotogrammetryAnalysis_RG\results\24_PMT_study\Reproduce\reproduce_bolt_distances.txt'
filename = r'C:\Users\gaurr\OneDrive - TRIUMF\Super-K\Reconstruction\PhotogrammetryAnalysis_RG\results\LI'
filename2 = r'C:\Users\gaurr\OneDrive - TRIUMF\Super-K\Reconstruction\PhotogrammetryAnalysis_RG\results\24_PMT_study\OpenCV Bootstrap\OpenCV - Polygonal Seed\newFD_bolt_distances_opencvmodel_polygon.txt'

# load in data
data1 = np.loadtxt(filename)
# data2 = np.loadtxt(filename2)


# histogram of residuals divided by the 3D fitter errors from 'data2'. Plot also the mean and standard deviation and write them on the plot
# plt.figure()
# plt.hist(data1/data2, bins='auto')
# plt.axvline(np.mean(data1/data2), color='r', linestyle='dashed', linewidth=1.5)
# plt.axvline(np.mean(data1/data2) + np.std(data1/data2), color='k', linestyle='dashed', linewidth=1)
# plt.axvline(np.mean(data1/data2) - np.std(data1/data2), color='k', linestyle='dashed', linewidth=1)
# plt.text(2.5, 80, 'Mean = {:.2f}\nStandard Deviation = {:.2f}'.format(np.mean(data1/data2), np.std(data1/data2)), verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
# plt.xlabel("Residuals/3D Fitter Errors")
# plt.ylabel("Frequency")
# plt.title("Residuals in units of 1-sigma 3D Fitter Errors for MATLAB Calibrated Model (Polyhedron Seed)")
# plt.show()

# plot 3D fitter errors
# plt.figure()
# plt.hist(data2, bins=20)
# plt.xlabel("3D Fitter Errors (cm)")
# plt.ylabel("Frequency")
# plt.title("3D Fitter Errors for Manually Detected 24 PMTs")
# plt.show()

# load in the second column of the text files
# data_rad = np.loadtxt(filename, usecols=1)
# data2_rad = np.loadtxt(filename2, usecols=1)
# data_tan = np.loadtxt(filename, usecols=2)
# data2_tan = np.loadtxt(filename2, usecols=2)


# plot both histograms on the same plot. Label the first one, "Manually detected" and the second one, "Semi-automatically detected"
# plt.hist(data_rad, bins=20, alpha=0.5, label='Radial Error: MATLAB')
# plt.hist(data2_rad, bins=20, alpha=0.5, label='Radial Error: OpenCV')
# plt.hist(data_tan, bins=20, alpha=0.5, label='Tangential Error: MATLAB')
# plt.hist(data2_tan, bins=20, alpha=0.5, label='Tangential Error: OpenCV')
# plt.xlabel("Fitter Error on Position [cm]")
# plt.ylabel("Frequency")
# plt.legend()
# plt.title("Comparing the Radial and Tangential Position Errors for the MATLAB and\nOpenCV Calibrated Reconstructions")

# load in the data
# data = np.loadtxt(filename)
# data2 = np.loadtxt(filename2)
#
# # plot both histograms on the same plot. Label the first one, "Manually detected" and the second one, "Semi-automatically detected"
# plt.hist(data1, bins=20, histtype = 'step', label='MATLAB')
# plt.hist(data2, bins=20, histtype = 'step', label='OpenCV')
# plt.legend()
# plt.xlabel("Distance between adjacent bolts [cm]")
# plt.ylabel("Number of features")
# plt.title("Comparing the Distance Adjacent Bolts for the\nMATLAB and OpenCV Calibrated Models (Polyhedron Seed)")
# #
#%%
import os
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
from scipy import linalg
import matplotlib.pyplot as plt
import pg_fitter_tools_new as fit
import pandas as pd

# load in original detected points
path1 = 'BarrelSurveyFar_TopInjector_PD3/BarrelSurveyFar_TopInjector_median_texts/' # Manual feature detection and labeling
path2 = 'source/LI_Points/'

offset = np.array([0,250])

manual_positions = {}
newFD_positions = {}

# pass all text files from path1 and path2 into manual_positions and newFD_positions respectively

for file in os.listdir(path1):
    if file.endswith(".txt"):
        manual_positions.update(fit.read_image_feature_locations(path1+file, offset=offset))

for file in os.listdir(path2):
    if file.endswith(".txt"):
        newFD_positions.update(fit.read_image_feature_locations(path2+file, offset=offset))

# swap "-25" and "-00" pairs
for im in manual_positions.values():
    for feat, loc in im.items():
        if feat[-2:] == "00" and feat[:-2]+"25" in im:
            tmp = loc
            im[feat] = im[feat[:-2]+"25"]
            im[feat[:-2]+"25"] = tmp

# delete all entries in manual_positions which end in "-25"
for im in manual_positions.values():
    for feat in list(im.keys()):
        if feat[-2:] == "25":
            del im[feat]

# look through image_feature_locations and keep only the keys with the following values '1045', '1046', '1047', '1048', '1086', '1087', '1124', '1125', '1126', '1127', '1236', '1237', '1238', '1239', '1240'
newFD_positions = {k: v for k, v in newFD_positions.items() if k in ['1045', '1046', '1047', '1048', '1086', '1087', '1236', '1237', '1238', '1239', '1240']}

# go through each image in image_feature_locations. If the feature does not begin with '00810', '00809', '00808', '00807', '00759', '00758', '00757', '00708', '00707', '00706', '00657', '00656', '00655', '00606', '00605', '00604', '00555', '00554', '00553', '00504', '00503', '00502', '00453', '00452', or '00451', remove it from the image
for im in newFD_positions.values():
    for feat in list(im):
        if feat[:5] not in ['00810', '00809', '00808', '00759', '00758', '00757', '00708', '00707', '00706', '00657', '00656', '00655', '00606', '00605', '00604', '00555', '00554', '00553', '00504', '00503', '00502', '00453', '00452', '00451']:
            del im[feat]

for im in manual_positions.values():
    for feat in list(im):
        if feat[:5] not in ['00810', '00809', '00808', '00759', '00758', '00757', '00708', '00707', '00706', '00657', '00656', '00655', '00606', '00605', '00604', '00555', '00554', '00553', '00504', '00503', '00502', '00453', '00452', '00451']:
            del im[feat]


# convert manual_positions into a pandas dataframe, keeping that each key is an image with its own dataframe of labels and positions
manual_positions = pd.DataFrame.from_dict({(i,j): manual_positions[i][j]
                            for i in manual_positions.keys()
                            for j in manual_positions[i].keys()},
                          orient='index')

# convert newFD_positions into a pandas dataframe, keeping that each key is an image with its own dataframe of labels and positions
newFD_positions = pd.DataFrame.from_dict({(i,j): newFD_positions[i][j]
                            for i in newFD_positions.keys()
                            for j in newFD_positions[i].keys()},
                            orient='index')

# separate the image number and label into two separate columns
manual_positions.reset_index(inplace=True)
manual_positions[['image', 'label']] = pd.DataFrame(manual_positions['index'].tolist(), index=manual_positions.index)
manual_positions.drop(columns=['index'], inplace=True)


# now for newFD_positions
newFD_positions.reset_index(inplace=True)
newFD_positions[['image', 'label']] = pd.DataFrame(newFD_positions['index'].tolist(), index=newFD_positions.index)
newFD_positions.drop(columns=['index'], inplace=True)

# make the image number and label the first two columns
cols = manual_positions.columns.tolist()
cols = cols[-2:] + cols[:-2]
manual_positions = manual_positions[cols]

# name the last two columns 'x' and 'y'
manual_positions.columns = ['image', 'label', 'x', 'y']

cols = newFD_positions.columns.tolist()
cols = cols[-2:] + cols[:-2]
newFD_positions = newFD_positions[cols]

newFD_positions.columns = ['image', 'label', 'x', 'y']

# change the image numbers in newFD_positions to retain only the last three characters
newFD_positions['image'] = newFD_positions['image'].str[-3:]

print(manual_positions)
print(newFD_positions)

# compute the differences in positions between the two dataframes for features which have the same label and image number

# check each image number and label in manual_positions. If it exists in newFD_positions, compute the difference in position using the x and y columns
# in manual_positions and newFD_positions. Store this difference in a dataframe called 'differences'
differences = pd.DataFrame(pd.np.empty((0, 5)))
# rename the columns of differences
differences.columns = ['Image', "label", "dx", 'dy', 'distance']

for i in range(len(manual_positions)):
    for j in range(len(newFD_positions)):
        if manual_positions['image'][i] == newFD_positions['image'][j] and manual_positions['label'][i] == newFD_positions['label'][j]:
            differences = differences.append({'Image': manual_positions['image'][i], 'label': manual_positions['label'][i], 'dx': manual_positions['x'][i] - newFD_positions['x'][j], 'dy': manual_positions['y'][i] - newFD_positions['y'][j], 'distance': np.sqrt((manual_positions['x'][i] - newFD_positions['x'][j])**2 + (manual_positions['y'][i] - newFD_positions['y'][j])**2)}, ignore_index=True)

#%%

# print the mean and standard deviation of the differences
print(np.mean(differences['distance']))
print(np.std(differences['distance']))

# print the fraction of differences which are greater than 1 standard deviation
print(len(differences[differences['distance'] > np.std(differences['distance']) + np.mean(differences['distance'])])/len(differences))


#%%

# histogram the distances from the differences dataframe
plt.figure()
plt.hist(differences['distance'], bins=20, histtype='step')
plt.xlabel("Distance Between the Detected Positions [pixels]")
plt.ylabel("Frequency")
plt.title("Distance Between the Detected Positions for the Manually and\nSemi-Automatically Detected Points")

# save differences dataframe to a pickle file
differences.to_pickle('differences.pkl')

#%% print the average distance for each image

# create a list of the unique image numbers
images = differences['Image'].unique()

# create a list of the average distances for each image
average_distances_images = []

# for each image, compute the average distance and append it to the list
for image in images:
    average_distances_images.append(np.mean(differences[differences['Image'] == image]['distance']))

# print the average distances
print(average_distances_images)

#%% print the average distance for each label

# create a list of the unique labels
labels = differences['label'].unique()

# create a list of the average distances for each label
average_distances = []

# for each label, compute the average distance and append it to the list
for label in labels:
    average_distances.append(np.mean(differences[differences['label'] == label]['distance']))

# find which labels have the largest average distance
print(average_distances)

# find the index of the largest average distance
print(np.argmax(average_distances))

# find the label with the largest average distance
print(labels[np.argmax(average_distances)])

#%% plot the average distance for each image

# plot the average distance for each image
plt.figure()
plt.plot(images, average_distances_images, 'o')
plt.xlabel("Image Number")
plt.ylabel("Average Distance Between the Detected Positions [pixels]")
plt.title("Average Distance Between the Detected Positions for the Manually and\nSemi-Automatically Detected Points")

#%%

path = "results/LI_bolt_radii.txt"

radii = np.loadtxt(path)

fig, ax = plt.subplots(figsize=(7,6))
ax.hist(radii, bins=60, histtype='step')
fig.tight_layout()
ax.set_xlabel("Distance between bolts and centre of bolt ring [cm]")
ax.set_ylabel("Number of features")
ax.set_yscale('log')
ax.set_title("Reconstructed Distance Between Bolts and Centre of Bolt Ring\nfor the LI Column (Log-Y Scale)")

#%%
count = 0
for i in range(len(radii)):
    if radii[i] < 25:
        count+=1
print(count)

#%% fit line to column of PMTs

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import matplotlib
matplotlib.use('Qt5Agg')


path = "results/LI_reconstructed_positions.txt"
data = pd.read_csv(path, sep="\t", header=None)
data.columns = ["Labels", "X", "Y", "Z"]

# find max z value and label from "Labels" column for it
max_z = np.max(data["Z"])
max_z_label = data["Labels"][np.argmax(data["Z"])]

# in data, find all the entries with labels that end in "00"
new_data = data[data["Labels"].str[-2:] == "00"]

# amend new_data to only keep entries which have labels where the first 5 characters are between 00663 and 00613 inclusive
new_data = new_data[(new_data["Labels"].str[:5] >= "00613") & (new_data["Labels"].str[:5] <= "00663")]


# make 3D plot with new_data
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(new_data["X"], new_data["Y"], new_data["Z"])
ax.set_xlabel("X [cm]")
ax.set_ylabel("Y [cm]")
ax.set_zlabel("Z [cm]")
ax.set_title("Reconstructed Positions of the PMTs in the LI Column")


# add labels to the points
for i in range(len(new_data)):
    ax.text(new_data.iloc[i]["X"], new_data.iloc[i]["Y"], new_data.iloc[i]["Z"], new_data.iloc[i]["Labels"])

# calculate the polar coordinates for new_data and add them to the dataframe in columns "radius" and "theta"
new_data["radius"] = np.sqrt(new_data["X"]**2 + new_data["Y"]**2)
new_data["theta"] = np.arctan2(new_data["Y"], new_data["X"])

# find the central values of the polar coordinates
central_radius = np.mean(new_data["radius"])
central_theta = np.mean(new_data["theta"])
# convert central_theta to degrees
central_theta = central_theta * 180 / np.pi

print("Central Radius: ", central_radius)
print("Central Theta: ", central_theta)

# include a vertical line going from the minimum z value to the maximum z value at the central radius and theta
ax.plot([central_radius * np.cos(central_theta * np.pi / 180), central_radius * np.cos(central_theta * np.pi / 180)], [central_radius * np.sin(central_theta * np.pi / 180), central_radius * np.sin(central_theta * np.pi / 180)], [np.min(new_data["Z"]), np.max(new_data["Z"])],
        color='r', linewidth=3)


# find the standard deviations of the polar coordinates
radius_std = np.std(new_data["radius"])
theta_std = np.std(new_data["theta"])
# convert theta_std to degrees
theta_std = theta_std * 180 / np.pi

print("Radius Standard Deviation: ", radius_std)
print("Theta Standard Deviation: ", theta_std)

#%% arclength for standard deviation

x, y = 1690, 2
# convert x and y to polar coordinates
r = np.sqrt(x**2 + y**2)

# find arclength using r and theta_std
arclength = r * theta_std * np.pi / 180

print(arclength)

#%%
# plot the histogram of the radii
plt.figure()
plt.hist(new_data["radius"], bins=20)
plt.xlabel("Radius [cm]")
plt.ylabel("Frequency")
plt.title("Histogram of the Radii of the PMTs in the LI Column")

# plot the histogram of the theta values in degrees
plt.figure()
plt.hist(new_data["theta"] * 180 / np.pi, bins=20)
plt.xlabel("Theta [degrees]")
plt.ylabel("Frequency")
plt.title("Histogram of the Theta Values of the PMTs in the LI Column")

# plot a histogram of the residuals of the radii
plt.figure()
plt.hist(new_data["radius"] - central_radius, bins=20)
plt.xlabel("Radius Residuals [cm]")
plt.ylabel("Frequency")
plt.title("Histogram of the Residuals of the Radii of the PMTs in the LI Column")

# plot a histogram of the residuals of the theta values in degrees
plt.figure()
plt.hist((new_data["theta"] * 180 / np.pi) - central_theta, bins=20)
plt.xlabel("Theta Residuals [degrees]")
plt.ylabel("Frequency")
plt.title("Histogram of the Residuals of the Theta Values of the PMTs in the LI Column")




#%%
# fit a line to the z values vs the labels
from scipy.optimize import curve_fit
def func(x, a, b):
    return a*x + b
popt, pcov = curve_fit(func, new_data["Labels"], new_data["Z"])

# plot the fit
plt.plot(new_data["Labels"], func(new_data["Labels"], *popt), 'r-', label='fit: a=%5.3f, b=%5.3f' % tuple(popt))
plt.legend()

# print the fit parameters
print(popt)

# plot the residuals
plt.figure()
plt.plot(new_data["Labels"], new_data["Z"] - func(new_data["Labels"], *popt), 'o')
plt.xlabel("Label")
plt.ylabel("Residuals [cm]")
plt.title("Residuals for the Z Position vs Label Fit for the LI Column")

# print the mean and standard deviation of the residuals
print(np.mean(new_data["Z"] - func(new_data["Labels"], *popt)))
print(np.std(new_data["Z"] - func(new_data["Labels"], *popt)))

#%%

