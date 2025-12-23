import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
# 데이터 생성
x = np.arange(start=-2 * np.pi, stop=2 * np.pi, step=0.1)
y1, y2 = np.sin(x), np.cos(x)

plt.grid()
plt.plot(x, y1, color='r')
plt.plot(x, y2, color='b')
plt.xlabel('x')
plt.ylabel('y')
plt.title('cos & sin in [-2*pi, 2*pi]')
plt.legend(['sin', 'cos'])
plt.show()
plt.close()

# 사진 출력
file_path = os.path.join(os.getcwd(), 'chapter_1/data/money.jpg')
img = plt.imread(file_path)
plt.axis('off')
plt.imshow(img)
plt.show()
plt.close()
