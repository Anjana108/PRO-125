import numpy as np
import seaborn as sns
from PIL import Image
import PIL.ImageOps
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pandas as pd

x = np.load("image.npz")["arr_0"]
y = pd.read_csv("labels.csv")["labels"]

classes = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z" ]
nclasses = len(classes)

x_train, x_test, y_train, y_test = train_test_split(x, y , random_state = 9, train_size = 3500, test_size = 500)

clf = LogisticRegression(solver = "saga", multi_class = "multinomial").fit(x_train, y_train)

def prediction(image):
    pil_image = Image.open(image)
    bwImage = pil_image.convert("L")
    bwImageResized = bwImage.resize((22, 30), Image.ANTIALIAS)
    pixel = 20
    min = np.percentile(bwImageResized, pixel)
    img_inverted = np.clip(bwImageResized-min, 0, 255)
    max = np.max(bwImageResized)
    img_inverted = np.asarray(img_inverted/max)
    test_sample = np.array(img_inverted).reshape(1, 660)
    test_pred = clf.predict(test_sample)

    return test_pred[0]