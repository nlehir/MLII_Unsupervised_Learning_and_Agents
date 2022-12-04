import os
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt


sample_index = 12
digits = load_digits()
plt.imshow(digits.data[sample_index].reshape(8, 8))
figpath = os.path.join("images", f"sample_{sample_index}.pdf")
plt.savefig(figpath,)
