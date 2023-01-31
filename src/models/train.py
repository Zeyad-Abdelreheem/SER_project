#!/usr/bin/env python
# coding: utf-8

# In[3]:


from create_model import create_model


# In[2]:


def train_model(data_path, EPOCHS = 25, batch_size = 32, path_to_save_model):
    df = pd.read_csv(data_path)
    
    X = df.drop(labels="labels", axis=1)
    
    Y = df["labels"]
    
    lb = LabelEncoder()
    
    Y = np_utils.to_categorical(lb.fit_transform(Y))
    
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        Y,
                                                        random_state=42,
                                                        test_size=0.2,
                                                        shuffle=True)
    
    X_train, X_val, y_train, y_val = train_test_split(X_train,
                                                      y_train,
                                                      random_state=42,
                                                      test_size=0.1,
                                                      shuffle=True)
    
   
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    X_val = scaler.transform(X_val)
    
    
    
    X_train = np.expand_dims(X_train, axis=2)
    X_val = np.expand_dims(X_val, axis=2)
    X_test = np.expand_dims(X_test, axis=2)
    X_train.shape
    
    
    earlystopping = EarlyStopping(monitor ="val_acc",
                              mode = 'auto', patience = 5,
                              restore_best_weights = True)

    
    learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc',
                                            patience=3,
                                            verbose=1,
                                            factor=0.5,
                                            min_lr=0.00001)
    
    model = create_model(X_train.shape[1])
    
    
    history = model.fit(X_train, y_train,
                        validation_data=(X_val, y_val),
                        epochs = EPOCHS, batch_size=batch_size,
                        callbacks=[earlystopping, learning_rate_reduction])
    
    
    model.save(path_to_save_model)
    
    

    
    

    
    
    


# In[ ]:




