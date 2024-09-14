import pandas as pd
import cv2
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from sklearn.metrics import roc_auc_score
from numpy import mean
from numpy import std
from pandas import read_csv
from matplotlib import pyplot
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import make_scorer
from imblearn.metrics import geometric_mean_score
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.over_sampling import ADASYN
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from imblearn.ensemble import BalancedBaggingClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.pipeline import Pipeline
from xgboost import XGBClassifier
#%% Feature extraction
import os
import glob
import cv2
from natsort import natsorted
from scipy.ndimage import find_objects
import mat73
import skimage.measure as measure
def cir(area, perimeter):
    return (perimeter ** 2) / area

def dm(area, average_opd):
    alpha = 0.2
    return (area *0.275*10**-12 * abs(average_opd*0.275*10**-9))/(alpha*10**-18)


def pv(area, average_opd):
    alpha = 0.2
    dm2 = dm(area,average_opd)
    return dm2 * alpha

def psa(img,area):
    gray = img.astype('float32')
    gX_Scharr = cv2.Scharr(gray, ddepth=cv2.CV_32F, dx=1, dy=0)
    gY_Scharr = cv2.Scharr(gray, ddepth=cv2.CV_32F, dx=0, dy=1)
    gX_Scharr = cv2.convertScaleAbs(gX_Scharr)
    gY_Scharr = cv2.convertScaleAbs(gY_Scharr)
    inner = np.sqrt(1 + (gX_Scharr[:]*10**-9)**2 + (gY_Scharr[:]*10**-9)**2)
    return np.sum(inner[:]) + (area*0.275**2 )

    
def psa2dm(img,area,average_opd):
    psa1 = psa(img,area)
    dm1 = dm(area, average_opd)
    return psa1/dm1

def a2v(area, average_opd):
    pv1 = pv(area, average_opd)
    return area*0.275**2 / pv1

def speri(img, area, average_opd):
    pv1 = pv(area, average_opd)
    psa1 = psa(img, area)
    return (np.pi)**0.33 * ((6*pv1)**0.667 / psa1)

def energy(img):
    img2 = img**2
    return 10**-6 * np.sum(img2[:])

def ellipticity(ma,mi):
    return mi/(ma+0.000001)
mask_path = "D:/Skin Cancer/dataset/masks" 
output_folder = "D:/Skin Cancer/dataset/Properties"
opd_path = "D:/Skin Cancer/dataset/opd_values"
mask_names = natsorted(os.listdir(mask_path))
opd_files = natsorted(os.listdir(opd_path))
pixels_to_um = 5.5/20
# Iterate over the sorted file names
for mask_name, opd_file in zip(mask_names, opd_files):
        mask_path_1 = os.path.join(mask_path, mask_name)
        opd_path_1 = os.path.join(opd_path, opd_file)
        # Load the .npy file using numpy's load function
        dat = np.load(mask_path_1, allow_pickle=True).item()
        f11 = mat73.loadmat(opd_path_1)
        # image = f11['opd_a375more']
        image = f11['opd_same']        
        mask = dat['masks']
        mask, num_objects = measure.label(mask, return_num=True)
        from skimage import measure, color, io
        regions = measure.regionprops(mask, intensity_image=image)
        images_mask = []
        for region in regions:
            images_mask.append(region.image)
        plt.imshow(images_mask[15], cmap='gray')
        plt.show()
        df = pd.DataFrame()
        df2 = pd.DataFrame()
        circularity = []
        dry_mass = []
        Phase_surface_area = []
        psa2drymass = []
        spericity = []
        variances = []
        skewness = []
        kurtosis = []
        energy_terture = []
        ellipse = []
        for region in regions:
            # Calculate circularity using perimeter and area
            perimeter = region.perimeter
            area = region.area
            mean = region.intensity_mean
            img_intensity = region.image_intensity
            major = region.axis_major_length
            minor = region.axis_minor_length
            circularity1 = cir(area,perimeter)
            circularity.append(circularity1)
            # Print circularity for this region
            # print(f"Region {region.label} has circularity {circularity:.2f}.")
            dm_1 = dm(area,mean)
            dry_mass.append(dm_1)
            dm_ad1 = dm_ad(area,mean)           
            pv_1 = pv(area,mean)           
            psa_1 = psa(img_intensity, area)
            Phase_surface_area.append(psa_1)           
            psa2v_1 = psa2vr(img_intensity, area, mean)
            psa2dm_1 = psa2dm(img_intensity, area, mean)
            psa2drymass.append(psa2dm_1)
            
            sa2v_1 = a2v(area,mean)
            spericity_1 = speri(img_intensity,area, mean)
            spericity.append(spericity_1)            
            variances1 = region.moments_central[2, 0]*0.275 / (region.moments_central[0, 0])
            variances.append(variances1)            
            skewness1 = region.moments_normalized[3, 0]
            skewness.append(skewness1)            
            kurtosis1 = region.moments_hu[3]
            kurtosis.append(kurtosis1)            
            energy_terture1 = energy(img_intensity)
            energy_terture.append(energy_terture1)            
            ellipse1 = ellipticity(major, minor)
            ellipse.append(ellipse1)
            
        my_dict2 = {'Circularity': circularity, 'Ellipticity': ellipse ,'dry_mass': dry_mass ,  'PSA': Phase_surface_area, 'PSA2DM': psa2drymass,  'Sphericity':spericity, 'Energy': energy_terture, 'Variance':variances, 'Kurtosis': kurtosis, 'Skewness': skewness }
        
    
        # Iterate over the regions and extract properties
        for region in regions:
            properties = {
                'area': region.area * pixels_to_um**2,
                'equivalent_diameter': region.equivalent_diameter_area* pixels_to_um,
                'Major axis': region.axis_major_length* pixels_to_um,
                'Minor axis': region.axis_minor_length* pixels_to_um,
                'perimeter': region.perimeter* pixels_to_um,
                'Min Intensity':region.intensity_min,
                'Mean Intensity': region.intensity_mean,
                'Max Intensity': region.intensity_max,
                'Eccentricity': region.eccentricity
                
                
                # Add more properties as needed
            }
            df = df.append(properties, ignore_index=True)
        for col_name, col_data in my_dict2.items():
            # df2 = df2.append(pd.DataFrame(col_data, columns=[col_name]), ignore_index=True,axis=1)
            df = pd.concat([df, pd.DataFrame(col_data, columns=[col_name])], axis=1)
        # Create the output CSV filename
        # df_final = pd.concat([df, df2],axis= 1, ignore_index=False)

        output_filename = os.path.splitext(mask_name)[0] + '.csv'
        output_path = os.path.join(output_folder, output_filename)
        
        # Save the DataFrame to a CSV file
        df.to_csv(output_path, index=False)
#%%
data = pd.read_csv("D:/Skin Cancer/dataset/Properties/combined_mid.csv")
target = data['label']
target.value_counts()
original_features = data.drop('label', axis = 1)
original_features.head()
target = target.map({'HDF': 1, 'A375': 0})

oversample = ADASYN()
original_features, target = oversample.fit_resample(original_features, target)
target.value_counts()

#%% Feature selection
standard_features = (original_features - original_features.mean()) / original_features.std()
X_train, X_test, Y_train, Y_test = train_test_split(standard_features, target, test_size = 0.1, random_state = 42)
# Fit model to data
rf = BalancedRandomForestClassifier(n_estimators=1000)
steps = [('p', PowerTransformer()), ('m',rf)]
    # define pipeline
pipeline = Pipeline(steps=steps)
pipeline.fit(X_train, Y_train)
importances = rf.feature_importances_
std = np.std([tree.feature_importances_ for tree in rf.estimators_], axis = 0)
indices = np.argsort(importances)[::-1]
# Evaluate feature importances
mean_std = std.mean()
selected_indices = [i for i in range(len(importances)) if importances[i] > mean_std]
new_features = original_features.iloc[:, selected_indices]
plt.figure(figsize=(15, 8))
plt.bar(range(X_train.shape[1]), importances[indices], color="r", align="center")
plt.xticks(range(X_train.shape[1]), X_train.columns[indices], rotation=90, fontsize=15)
plt.yticks(fontsize=15)
# plt.xlabel("Features", fontsize=15)
plt.ylabel("Importance", fontsize=15)
plt.xlim([-1, X_train.shape[1]])
plt.show()
#%%
def evaluate_model(X, y, model):
    # define evaluation procedure
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    # evaluate model
    scores = cross_val_score(model, X, y, scoring='roc_auc', cv=cv, n_jobs=-1)
    return scores
# define models to test
def get_models():
    models, names = list(), list()
    # LR
    models.append(LogisticRegression(solver='lbfgs', class_weight='balanced'))
    names.append('LR')
    # SVM
    models.append(SVC(gamma='scale', class_weight='balanced', probability=True))
    names.append('SVM')
    # Bagging
    models.append(BalancedBaggingClassifier(n_estimators=1000))
    names.append('DT')
    #BRF
    models.append(BalancedRandomForestClassifier(n_estimators=1000))
    names.append('RF')
    # XGBoost
    models.append(XGBClassifier(scale_pos_weight=1.01))
    names.append('XGBoost')
    return models, names
models, names = get_models()
results = list()
# evaluate each model
for i in range(len(models)):
# defines pipeline steps
    steps = [('p', PowerTransformer()), ('m',models[i])]
    # define pipeline
    pipeline = Pipeline(steps=steps)
    # evaluate the pipeline and store results
    scores = evaluate_model(new_features, target, pipeline)
    results.append(scores)
    # summarize and store
    print('>%s %.3f (%.3f)' % (names[i], mean(scores), std(scores)))
# plot the results
pyplot.boxplot(results, labels=names, showmeans=True)
pyplot.xlabel('ML Algorithms',fontsize = 15)
pyplot.ylabel('AUC Score',fontsize = 15)
plt.ylim([0.83, 1])
pyplot.xticks(fontsize = 15)
pyplot.yticks(fontsize = 15)
pyplot. grid(False)
pyplot.show()

#%% Testing
