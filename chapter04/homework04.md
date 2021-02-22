### 4 求神经网络的VC Dimension

#### 4.1 线性激活函数

由于对于线性分类器而言，其VC Dimension是其输入向量的维度数+1。

对第一层而言，其输入特征的维度数是d，所以VC Dimension 是d +1。

对第二层而言，其输入特征的维度是n, 所以其VC Dimension 是n + 1。

综合以上，VC Dimension为d +1和n + 1中的较小者。

#### 4.2 ReLU激活函数

如果输入特征维度数是1，通过提取bias值可得，VC Dimension >= n + 1。当输入特征维度数大于1时，VC Dimension 更加大。

又由于线性分类器VC Dimension <= n + 1。

综合以上，得到VC Dimension为n+1。