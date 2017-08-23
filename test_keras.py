import pandas as pd
from keras.models import model_from_json
from keras.optimizers import RMSprop

test = pd.read_csv('test.csv')
X = test.values.astype('float32') # pixels
assert test.isnull().sum().sum() == 0, 'Null values found in test.'

del test
X /= 255
X = X.reshape(-1, 28, 28, 1)

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

loaded_model.load_weights('weights.hdf5')

optimizer = RMSprop()
loaded_model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

predictions = loaded_model.predict_classes(X, verbose=0)

preds = pd.DataFrame({'ImageId': list(range(1, len(predictions) + 1)), 'Label': predictions})
preds.to_csv('predictions.csv', index=False, header=True)
