import cv2
from keras.models import load_model
from PIL import Image
import numpy as np

model = load_model('models/BrainTumor-10-0.98.model')
image = cv2.imread('tests/0.jpg')
img = Image.fromarray(image)
img = img.resize((64, 64))
img = np.array(img)
# print(img)

input_img = np.expand_dims(img, axis=0)

result = model.predict(input_img)
print(result)