对于一个直角三角形，我们选定一个非直角的角，这个角的度数为$\theta$，那么：
该角正对的直角边，我们称之为$a$,
该角相邻的的直角边，我们称之为$b$
直角三角形的斜边，我们称之为$c$

- 正弦
$$
sin\theta = \frac{a}{c}
$$

- 余弦
$$
cos\theta = \frac{b}{c}
$$

直角三角形中有几个特殊的三角形：
- 等腰直角三角形的三角度数是：$45^\circ\;45^\circ\;\;90^\circ\;$
- 角度是$30^\circ\;60^\circ\;\;90^\circ\;$的三角形，这个三角形的特点是两条直角边，短直角边是斜边的二倍

有如下关系：
- $sin⁡30^\circ = \frac{1}{2}\qquad \cos30^\circ = \frac{\sqrt3}{2}\;$
- $sin⁡45^\circ = \frac{\sqrt{2}}{2}\qquad cos⁡45^\circ = \frac{\sqrt{2}}{2}$
- $sin⁡60^\circ = \frac{\sqrt3}{2}\qquad cos⁡60^\circ = \frac{1}{2}$


再以单位圆的方式理解，那么角度 $\theta$ 从x轴出发，逆时针旋转 ，那么其对应的的正余弦就是单位圆上的对应点坐标 $(\sin\theta, \cos\theta)$
因此：$sin⁡90^\circ = 1\qquad sin⁡0^\circ = 0\qquad  cos⁡90^\circ = 0\qquad  cos⁡0^\circ = 1$  


差角公式：
正弦差角公式：
$sin(\alpha - \beta) = sin(\alpha)cos(\beta) - cos(\alpha)sin(\beta)$

余弦差角公式：
$cos(\alpha - \beta) = cos(\alpha)cos(\beta) + sin(\alpha)sin(\beta)$



那么我们针对某一个token，假定该token的位置为m，则其对应的Q向量为$Q_m=(x_m, y_m)$。  
现在我们计算该token与另外一个token的相关性，假定另外一个token的位置为n，则其对应的K向量为$K_n=(x_n, y_n)$。

在计算两者的相关性时，仅仅通过计算两者的点积是无法区分两者之间的相对位置关系的。因此我们给Q和K都套上一个旋转矩阵。


其中：
$q_m = R_{m\theta}Q_m = \begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix} Q_m^T$

为什么是这个公式，可以通过数学公式进行推断。
假定现在有一个长度为$r$的向量，


那么我们针对一个Q向量(我们假定他是二维向量)，假设





$\begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix}$