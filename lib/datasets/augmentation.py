import albumentations as A
from albumentations.core.transforms_interface import DualTransform
import cv2
import numpy as np
from typing import Any

# TODO: MixUP3D https://github.com/Traffic-X/MonoLSS/blob/main/lib/datasets/kitti.py

class AugTest(DualTransform):
    def apply(self, img: np.ndarray, *args: Any, **params: Any) -> np.ndarray:
        return np.zeros_like(img)

    def apply_to_bboxes(
        self, bboxes: np.ndarray, *args: Any, **params: Any
    ) -> np.ndarray:
        return np.zeros_like(bboxes)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    transform = A.Compose(
        [
            A.RandomCrop(width=256, height=256),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            # A.TemplateTransform(),
            AugTest(p=0.2),
        ]
    )

    images = []
    images_transformed = []

    # 读取原始图像
    for i in range(10):
        image = cv2.imread("temp.jpg")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        images.append(image)

    # 将图像应用转换
    for image in images:
        transformed = transform(image=image)
        transformed_image = transformed["image"]
        images_transformed.append(transformed_image)

    # 将图像显示至plt
    for i in range(len(images)):
        # subplot(行, 列, 图像编号) # 编号从1开始,优先遍历行
        plt.subplot(2, len(images), i + 1)
        plt.imshow(images[i])
        plt.subplot(2, len(images), i + len(images) + 1)
        plt.imshow(images_transformed[i])

    plt.show()
