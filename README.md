# Speech Emotion Recognition 

 
## **Description**   
The project is about creating a model that can recognize emotions based on an input audio signal. <br />
The model can recognize the following emotions:-
- Anger
- Disgust
- Fear
- Happiness
- Neutral
- Sadness
- Surprise

the model was trained on 4 popular datasets found on 
[Kaggle](https://www.kaggle.com/datasets/dmitrybabko/speech-emotion-recognition-en)

## **How to run**   
First, install dependencies found in `requirements.txt`


```bash   
pip install -r requirements.txt
 ```
Then
from the `main_script` call the predict function with needed paths


**Example**

```
trained_model_path = './models/res_model.h5'

weights_path = './models/model_weights.h5'

samples_path = './test_samples'

scaler_model_path = './scaler.pkl'

predict(samples_path, trained_model_path, weights_path, scaler_model_path)

```

**sample output** <br />
1/1 [==============================] - 0s 353ms/step  <br />
the model predicted  **sad**  and the true label was  **sad**  <br />

1/1 [==============================] - 0s 120ms/step <br />
the model predicted  **angry**  and the true label was  **angry**  <br />

1/1 [==============================] - 0s 123ms/step <br />
the model predicted  **neutral**  and the true label was  **disgust** <br />


## Model Summary

## Results

## References

