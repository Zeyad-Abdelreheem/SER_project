#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from tensorflow import keras


# In[2]:


def load_model(saved_model_path):
    model = keras.models.load_model(saved_model_path)
    
    return model
    
    


# In[ ]:




