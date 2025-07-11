import numpy as np
from keras.models import load_model
from matplotlib import pyplot as plt


model = load_model("ganarator.keras")



sample_noise=np.random.normal(0, 1, (100, 100))


result=model.predict(sample_noise)
print(result)

plt.imshow(result[11],cmap='gray')
plt.show()

