import mir3.modules.features as feat
import mir3.modules.tool.wav2spectrogram as spec
import mir3.modules.features.centroid as cent
import mir3.modules.features.rolloff as roll
import mir3.modules.features.flatness as flat
import mir3.modules.features.flux as flux
import mir3.modules.features.mfcc as mfcc
import mir3.modules.features.diff as diff
import mir3.modules.features.stats as stats
reload(stats)
import mir3.modules.features.join as join
import mir3.modules.tool.to_texture_window as tex
from sklearn.preprocessing import normalize
from sklearn import preprocessing
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import ShuffleSplit
from sklearn.grid_search import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
import copy
from scipy.stats import friedmanchisquare as friedman
from scipy.stats import wilcoxon as wilcoxon
from scipy.stats import ttest_ind as ttest
from scipy.stats import f_oneway as f_oneway
from matplotlib import pyplot as plt
import numpy as np
import csv


def dataset_characterization(dataset_file):
    dataset = {} # Dictionary index = filename, content = class
    with open('base.csv', 'rb') as f:
        reader = csv.reader(f)
        for row in reader:
            dataset[row[0]] = row[2]
    return dataset

def dataset_class_histogram(dataset):
    histogram = {}
    for data in dataset:
        if dataset[data] not in histogram:
            histogram[dataset[data]] = 1
        else:
            histogram[dataset[data]] += 1
    return histogram



def features_gtzan(filename, directory=""):
    # Calculate spectrogram (normalizes wavfile)
    converter = spec.Wav2Spectrogram()
    print (directory+filename)    
    s = converter.convert(open(directory + filename), window_length=2048, dft_length=2048,
                window_step=1024, spectrum_type='magnitude', save_metadata=True)
    
    # Extract low-level features, derivatives, and run texture windows    
    
    d = diff.Diff()
    features = (cent.Centroid(), roll.Rolloff(), flat.Flatness(), flux.Flux(), mfcc.Mfcc())
  
    all_feats = None
    for f in features:
        track = f.calc_track(s) # Feature track
        all_feats = join.Join().join([all_feats, track])
        dtrack = d.calc_track(track) # Differentiate
        all_feats = join.Join().join([all_feats, dtrack])
        ddtrack = d.calc_track(dtrack) # Differentiate again
        all_feats = join.Join().join([all_feats, ddtrack]) 
        
        # Texture window
        t = tex.ToTextureWindow().to_texture(all_feats, 40)
        
    #Statistics
    s = stats.Stats()
    d = s.stats([t], mean=True, variance=True)    
    return d



def low_level_features(dataset_file, dataset_dir=""): # Estimate low-level features. 
                                                      # Returns sklearn-compatible structures.
    d = dataset_characterization(dataset_file)
    labels = []
    features = []
    for f in d:
        feat = features_gtzan(filename=f, directory=dataset_dir)
        features.append(feat.data)
        labels.append(d[f])
        # print(features, labels)
    
    return features, labels
    
dataset_dir = "./datasets/gtzan/"
dataset_file = 'base.csv'

dataset = dataset_characterization(dataset_file)
print dataset_class_histogram(dataset)

features, labels = low_level_features(dataset_file, dataset_dir)

def f1_from_confusion(c):
    ret = []
    for i in xrange(c.shape[0]):
        n_i = np.sum(c[i,:])
        est_i = np.sum(c[:,i])
        if n_i > 0:
            R = c[i,i] / float(n_i)
        else:
            R = 0.0
        if est_i > 0:
            P = c[i,i] / float(est_i)
        else:
            P = 0.0
            
        if (R+P) > 0:
            F = 2*R*P/(R+P)
        else:
            F = 0.
        ret.append([R, P, F])
    return ret



def model_comparison(features, labels, models, parameters_to_optimize, gender_parameter):
    #norm_features = normalize(features)
    # print (gender_parameter)
    features = np.array(features)
    # print (features)
    #use gender as a parameter
    if gender_parameter == True:
        gender_parameter = True
    skf = StratifiedKFold(labels, n_folds=10)

    f1 = np.zeros((10,len(models)))
    r = np.zeros((10,len(models)))
    p = np.zeros((10,len(models)))
   
    for m in xrange(len(models)):
        n = 0
        for train_index, test_index in skf:
            X_train, X_test = features[train_index,:], features[test_index,:]
            Y_train, Y_test = [labels[i] for i in train_index], [labels[i] for i in test_index]
            scaler = preprocessing.StandardScaler().fit(X_train)
            X_train = scaler.transform(X_train) 
            X_test = scaler.transform(X_test)
            
            cv = ShuffleSplit(X_train.shape[0], n_iter=10, test_size=0.2, random_state=0) # 80% train / 20% validation
            classifier = GridSearchCV(estimator=copy.deepcopy(models[m]), cv=cv, param_grid=parameters_to_optimize[m], scoring='f1_weighted')
            
            classifier.fit(X_train, Y_train)
            Y_pred = classifier.predict(X_test)
            confusion = confusion_matrix(Y_test, Y_pred)
            conf = f1_from_confusion(confusion)
            conf_all = np.average(conf, axis=0)

            r[n,m] = conf_all[0]
            p[n,m] = conf_all[1]
            f1[n,m] = conf_all[2]
            n += 1
            
    return r, p, f1

def label_to_numbers(labels, d):
    return [d.keys().index(i) for i in labels]

def numbers_to_labels(numbers, d):
    return [d.keys[i] for i in numbers]


gammas = np.logspace(-6, -2, 2)
Cs = np.array([1, 1000, 10000])
n_neighbors_s = np.array([1, 5, 10, 15, 20])

params_knn = dict(n_neighbors=n_neighbors_s)
params_svm = dict(gamma=gammas, C=Cs)
parameters_to_optimize = [params_knn, params_svm]

models = [KNeighborsClassifier(), SVC(class_weight='balanced')]

#using model_comparison without gender as a feature
recall, precision, f1_score = model_comparison(features, label_to_numbers(labels, dataset_class_histogram(dataset)), models, parameters_to_optimize, False)

#printing results without gender as a parameter
print ("Gender is not a feature \n")
print np.average(f1_score, axis=0)

if len(models) > 2:
#    print [f1_score[:,i].T for i in range(len(models))]
    print "Anova: ", f_oneway( *f1_score.T  )[1]
    
    
else:
    
    print "T-test:", ttest( f1_score[:,0].T,  f1_score[:,1].T)[1]


# #using model_comparison with gender as a feature
# recall, precision, f1_score = model_comparison(features, label_to_numbers(labels, dataset_class_histogram(dataset)), models, parameters_to_optimize, True)

# #printing results with gender as a parameter
# print ("Gender is a feature \n")
# print np.average(f1_score, axis=0)

# if len(models) > 2:
# #    print [f1_score[:,i].T for i in range(len(models))]
#     print "Anova: ", f_oneway( *f1_score.T  )[1]
    
    
# else:
    
#     print "T-test:", ttest( f1_score[:,0].T,  f1_score[:,1].T)[1]



# #COLOCAR APRENDIZADO SO COM GENERO
# #COLOCAR GENERO COMO FEATURE
