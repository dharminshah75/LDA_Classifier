
# coding: utf-8

# In[28]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
from sklearn import metrics
import scikitplot as skplt
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from matplotlib import style
import warnings
from sklearn.model_selection import train_test_split
style.use('fivethirtyeight')
from sklearn.neighbors import KNeighborsClassifier
warnings.filterwarnings("ignore")


# In[29]:


# 0. Load in the data and split the descriptive and the target feature
df = pd.read_csv('data/sat.trn',sep=' ' , header = None)
X_train = df.iloc[:,:-1].copy()
y_train = df.iloc[:,-1].copy()
classes = np.unique(df.iloc[:,-1])


# In[30]:


print('X_train.shape : ', X_train.shape, '\ny_train.shape : ' , y_train.shape)


# In[31]:


num_features = X_train.shape[1]
print('Number of features : ' , num_features)


# In[32]:


# 1. Standardize the data
for col in X_train.columns:
    X_train[col] = StandardScaler().fit_transform(X_train[col].values.reshape(-1,1))


# In[33]:


# 2. Compute the mean vector mu and the mean vector per class mu_k
mu = np.mean(X_train,axis=0).values.reshape(num_features,1) # Mean vector mu --> Since the data has been standardized, the data means are zero 


mu_k = []

for i,orchid in enumerate(np.unique(df.iloc[:,-1])):
    mu_k.append(np.mean(X_train.where(df.iloc[:,-1]==orchid),axis=0))
mu_k = np.array(mu_k).T


# In[34]:


#print(mu.shape,mu_k.shape)


# In[35]:


list_mu_k = []
for i in range(classes.__len__()):
    list_mu_k.append(mu_k[:,i].reshape(num_features,1))


# In[36]:


#3 Compute Covariance matrix 'E' for entire dataset => Shape (p,p)
X_t = X_train.T
E = np.cov(X_t)
#print(E.shape)


# In[37]:


#4 Compute prior probabilites
num_total = df.shape[0]
num = []
for i in classes:
    num.append(y_train.where(df.iloc[:,-1]==i).dropna().shape[0])
for i in range(num.__len__()):
    num[i] /= num_total
#print(num)


# In[38]:


def classification(x):
    '''
    Inputs:
    x - new instance to be classified
        Shape = (p,1)     #p = number of features
    Returns:
    class - integer ranging between 1 and 7 depicting the class to which input x belong
    '''
    
    ###############################################
    #Check for dimensions of x
    if x.shape[0]!=num_features and x.shape[1]!=1:
        print('Error : Incorrect shape of input x')
        return -1
    ###############################################
    
    current_max = -99999999999
    current_class = -1
    for i in range(classes.__len__()):
        #Calculate delta_kx
        term_1 = np.dot(np.dot(x.T,np.linalg.inv(E)),list_mu_k[i])
        term_2 = 0.5 * np.dot(np.dot(list_mu_k[i].T,np.linalg.inv(E)),list_mu_k[i])
        term_3 = np.log10(num[i])
        delta_kx = term_1 - term_2 +term_3
        if delta_kx > current_max:
            current_max = delta_kx
            current_class = classes[i]

    return current_class


# In[39]:


#Testing on sat.tst dataset
df = pd.read_csv('data/sat.tst',sep=' ' , header = None)
X_test = df.iloc[:,:-1].copy()
y_test = df.iloc[:,-1].copy()


# In[40]:


#Standardize test dataset
for col in X_test.columns:
    X_test[col] = StandardScaler().fit_transform(X_test[col].values.reshape(-1,1))


# In[41]:


print('X_test.shape : ' , X_test.shape , '\ny_test.shape : ' , y_test.shape)


# In[42]:


y_true = list(y_test)


# In[21]:

print('Running predictions on test set')
y_pred = []
for i in range(y_true.__len__()):
    print('Running test on index ', i , '/2000' , end='\r')
    x = np.array(X_test.iloc[i,:]).reshape(num_features,1)
    if(classification(x) == -1):
        print('-1 at index ',i)
    y_pred.append(classification(x))
print('\n')

# In[22]:


#Create confusion matrix
confusion_matrix = metrics.confusion_matrix(y_true,y_pred)
print('Confusion Matrix : ')
print(confusion_matrix)


# In[23]:


print(metrics.classification_report(y_true,y_pred,digits=3))


# In[24]:


print('Kappa Score :' , metrics.cohen_kappa_score(y_true,y_pred))


# In[28]:


def plot_roc_curve():
    #plot ROC curve for all the classes
    for j in range(classes.__len__()):
        #calculations for class i
        y = []
        scores = []
        for i in range(y_true.__len__()):
            if y_true[i] == j:
                y.append(1)
            else:
                y.append(0)
            if y_pred[i] == j:
                scores.append(1)
            else:
                scores.append(0)
        fpr, tpr, thresholds = metrics.roc_curve(y, scores)
        plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color='darkorange',
                 lw=lw, label='ROC curve for class '+str(classes[j]))
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        plt.show()


# In[30]:


plot_roc_curve()


# In[25]:


gnb = GaussianNB()
y_pred = gnb.fit(X_train, y_train).predict(X_test)


# In[26]:


#Confusion matrix for Naive Bayes
confusion_matrix = metrics.confusion_matrix(y_true,y_pred)
print('Confusion Matrix for Naive Bayes: ')
print(confusion_matrix)


# In[27]:


print(metrics.classification_report(y_true,y_pred,digits=3))

