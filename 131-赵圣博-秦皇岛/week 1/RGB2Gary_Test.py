from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2


# 灰度化 手动实现
img = cv2.imread("lenna.png")            # 载入图片
img_ = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)      # 将BGR通道转化为RGB
h, w = img.shape[:2]
img_gray = np.zeros([h, w], img.dtype)
for i in range(h):
    for j in range(w):
        m = img_[i, j]
        img_gray[i, j] = int(m[0]*0.3 + m[1]*0.59 + m[2]*0.11)
print(img_gray)
print("image show gray: %s" % img_gray)
cv2.imshow("image show gary", img_gray)

plt.subplot(221)
img = plt.imread("lenna.png")
plt.imshow(img)
print("----image lenna----")
print(img)

# 灰度化 直接调用函数
img_gary = rgb2gray(img)
plt.subplot(222)
plt.imshow(img_gary, cmap="gray")
print("----image gray----")
print(img_gary)

# 二值化 手动实现
# rows, cols = img_gray.shape
# for i in range(rows):
#     for j in range(cols):
#         if (img_gray[i, j] <= 0.5):
#             img_gray[i, j] = 0
#         else:
#             img_gray[i, j] = 1

# 二值化 调用函数
img_binary = np.where(img_gary >= 0.5, 1, 0)
print(img_binary)
print(img_binary.shape)

plt.subplot(223)
plt.imshow(img_binary, cmap="gray")
plt.show()
