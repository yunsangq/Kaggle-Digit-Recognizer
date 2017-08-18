import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image


X_test = (pd.read_csv('test.csv').values).astype('float32')

X_test = X_test.reshape(28000, 28, 28)

image = Image.fromarray(X_test[1331])
plt.imshow(image)
plt.show()


