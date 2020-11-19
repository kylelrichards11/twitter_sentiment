# CIL Project: Text Sentiment Classification
## Authors:

* Aurélia Autem
* Adrien Chabert
* Guillaume Comte
* Kyle Richards

# Setup
Run the following commands in the terminal (inside project folder). Note that the `git remote add` may take 5 to 10 minutes as there are several large files.
```bash
pip install --user -r requirements.txt
python setup_lib.py
```

# Data Preparations
There are many data preparations we use in this project. We have created classes for loading, cleaning, and tokenizing to remove any need for repetitively recoding the same trivial tasks.

## Loading
`data_preparations/loader.py` contains code to simply load the tweets as pandas DataFrames. Different arguments allow different datasets to be loaded. The arguments are documented for each function in the file.

Example:
```python
from data_preparations.loader import load_data, load_indiv_data
full_train = load_data(True)
small_train = load_data(False)
test = load_indiv_data("test")
```

## Cleaning
`data_preparations/cleaner.py` contains a class to clean the data. Because of the size of the datasets, it was important to optimize the code as much as possible. Taking advantage of the parallelization offered by the Leonhard system, we created parallelizable versions of nearly every method. To organize this, we created a base class `_BaseCleaner` with the unparallelized version of each cleaning method. The class `Cleaner` is then essentially an interface to the base cleaner that parallelizes the desired cleaning method with the given data. This way, the user never has to worry about parallelizing anything, it is all automatic. The cleaning methods are documented in the class description.

Example:
```python
from data_preparations.cleaner import Cleaner
cleaner = Cleaner()
data = cleaner.remove_digits(data)
data = cleaner.remove_punctuation(data)
data = cleaner.stem(data)
```

## Text Features
`data_preparations/text_features.py` contains a class to create manual features and tokenizations of the tweets. Similarly to the `Cleaner` class it uses two classes to implement parallelization. The features are documented in the class description.

Example:
```python
from data_preparations.text_features import text_feature_maker
tfm = text_feature_maker()
tokenized = tfm.tokenize(data)
emoticon_feature = tfm.extract_emoji(data)
```


# Methods
## CNN
The CNN method (and extensions) implementation can be found in `classifiers/vector_classifier.py` and `make_vector_model.py`. There is actually very little specific to CNNs in this code and is able to use any model that uses a vectorized version of the tweets. The vector_classifier.py file contains classes that are used to define a model that uses word vectors. It abstracts all of the vector conversions, training, and predicting into an easy interface. The following code creates and trains a CNN + LSTM classification model given only the training data.

```python
from classifiers.vector_classifier import VectorModel
custom_model = VectorModel(train_data, embedding_dim=100, lstm=True)
train_args = {'epochs':3, 'shuffle':True, 'validation_split':0.1, 'batch_size':32, 'verbose':False}
history = custom_model.fit(train_args)
```

`VectorModel` also contains functionality to save the entire model to a file and then reload the file. To load, simply call the load class method with the folder of the saved model. An example of this can be seen in the file `vector_model_load.py`

```python
from classifiers.vector_classifier import VectorModel
my_model = VectorModel.load("path_to_model")
test_probabilities = my_model.get_probabilities(test_data)
```

`make_vector_model.py` is the script used to train different variations of the VectorModels. There are many input options which can be seen by running

```bash
python make_vector_model.py --help
```
