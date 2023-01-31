#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import librosa
import numpy as np
from sklearn.preprocessing import StandardScaler
from processing import *


# In[ ]:


def prep_one_sample(sample_path):
    
    X, sample_rate = librosa.load(sample_path,
                                  duration=2.5,
                                  offset=0.6)
    
    X = extract_features(X, sample_rate)
    
    scaler = StandardScaler()
    
    X = scaler.fit_transform(X)
    
    X = np.expand_dims(X, axis=2)
    
    return X

    
    

