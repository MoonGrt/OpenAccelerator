
## 📌 卷积输出

对于输入大小 $W \times H \times C$ （宽、高、通道），卷积输出大小为：

$$
\text{Output Size} = \left\lfloor \frac{(W - F + 2P)}{S} \right\rfloor + 1
$$

* $F$：卷积核大小（filter size）
* $P$：padding（在输入边缘补零的宽度）
* $S$：stride（卷积窗口移动的步长）
* $W, H$：输入宽、高

通道数由 **卷积核个数** 决定。

## 🎯 例子：输入 28x28x3, filter=3x3

| Padding | Stride | 输出大小    |
| ------- | ------ | ------- |
| 0       | 1      | 26×26×8 |
| 1       | 1      | 28×28×8 |
| 1       | 2      | 14×14×8 |
| 0       | 2      | 13×13×8 |
