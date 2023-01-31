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
First, install dependencies found in requirements.txt


```bash   
pip install -r requirements.txt
 ```
Then
run the main script with the needed parameters

## Imports
This project is setup as a package which means you can now easily import any file into any other file like so:
```python
from project.datasets.mnist import mnist
from project.lit_classifier_main import LitClassifier
from pytorch_lightning import Trainer

# model
model = LitClassifier()

# data
train, val, test = mnist()

# train
trainer = Trainer()
trainer.fit(model, train, val)

# test using the best model!
trainer.test(test_dataloaders=test)
```

### Citation   
```
@article{YourName,
  title={Your Title},
  author={Your team},
  journal={Location},
  year={Year}
}
```   
