import random, math
import numpy as np


# ========== 基础变量 ==========


class MyTensor:
    """张量"""
    def __init__(self, data, op=None):
        """初始化"""
        self.data = np.array(data, dtype=float)
        self.grad = np.zeros(len(data), dtype=float)
        self.op = op

    def backward(self, clean=True):
        """反向传播"""
        self.one_grad()
        op_list = [self.op]
        for op in op_list:
            if op is None:
                continue
            if op.inputs is None:
                continue
            for t in op.inputs:
                if t is None:
                    continue
                if t.op in op_list:
                    continue
                if t.op is None:
                    continue
                op_list.append(t.op)
        op_list.sort(key=lambda x: MyOperator.global_operator_list.index(x), reverse=True)
        for op in op_list:
            op.deriv()
        if clean:
            MyOperator.clean()

    def grad_descent(self, k):
        """梯度下降"""
        self.data -= k * self.grad

    def __repr__(self):
        """字符串"""
        return f"<data:{self.data},grad:{self.grad}>"

    def __len__(self):
        """长度"""
        return len(self.data)

    def __add__(self, other):
        """加法"""
        op = MyAdd()
        return op.compute(self, other)

    def __sub__(self, other):
        """减法"""
        op = MySub()
        return op.compute(self, other)

    def __mul__(self, other):
        """乘法"""
        op = MyMul()
        return op.compute(self, other)

    def __truediv__(self, other):
        """整除"""
        op = MyDiv()
        return op.compute(self, other)

    def __pow__(self, power, modulo=None):
        """求幂"""
        op = MyPow()
        return op.compute(self, power)

    def sum(self):
        """求和"""
        op = MySum()
        return op.compute(self)

    def exp(self):
        """指数"""
        op = MyExp()
        return op.compute(self)

    def log(self):
        """对数"""
        op = MyLog()
        return op.compute(self)

    def softmax(self):
        """归一化"""
        a = self.exp()
        s = a.sum()
        s = s ** -1 if s.data[0] != 0 else MyTensor([0])
        c = MyTensor.connect([s for _ in range(len(a))])
        return a * c

    def cut(self, start, end):
        """切片"""
        op = MyCut()
        return op.compute(self, start, end)

    def expand(self, times):
        """广播"""
        out = MyTensor([self for _ in range(times)])
        out.zero_grad()
        return out

    def relu(self):
        """Relu激活函数"""
        op = MyRelu()
        return op.compute(self)

    def sigmoid(self):
        """Sigmoid激活函数"""
        op = MySigmoid()
        return op.compute(self)

    def tanh(self):
        """Tanh激活函数"""
        op = MyTanh()
        return op.compute(self)

    def zero_grad(self):
        """梯度置0"""
        self.grad = np.zeros(len(self.data), dtype=float)

    def one_grad(self):
        """梯度置1"""
        self.grad = np.ones(len(self.data), dtype=float)

    @classmethod
    def connect(cls, x):
        """连接"""
        op = MyConnect()
        return op.compute(x)

    @classmethod
    def zero(cls, size):
        """创建一个长度为size的填充0"""
        return MyTensor([0 for _ in range(size)])

    @classmethod
    def one(cls, size):
        """创建一个长度为size的填充1"""
        return MyTensor([1 for _ in range(size)])

    @classmethod
    def mse(cls, a, b):
        """均方误差"""
        return ((a - b) ** 2).sum() / MyTensor([len(a.data)])

    @classmethod
    def sse(cls, a, b):
        """平方误差"""
        return ((a - b) ** 2).sum()

    @classmethod
    def nll(cls, out, target):
        """交叉熵误差, 负对数似然"""
        return MyTensor([0]) - (target * out.log()).sum()


# ========== 运算符 ==========


class MyOperator:
    """运算符, 重写 compute 和 deriv """

    global_operator_list = []  # 全局运算符列表

    def __init__(self):
        """初始化"""
        MyOperator.global_operator_list.append(self)
        self.inputs = None
        self.output = None

    def compute(self, *args):
        """计算, 重写"""
        ...

    def deriv(self):
        """求导, 重写"""
        ...

    @classmethod
    def backward(cls, last_1_grad=True):
        """反向传播, """
        MyOperator.global_operator_list.reverse()
        if last_1_grad:
            MyOperator.global_operator_list[0].output.one_grad()
        for op in MyOperator.global_operator_list:
            op.deriv()
            op.inputs = None
            op.output = None
        MyOperator.global_operator_list.reverse()

    @classmethod
    def clean(cls):
        """清空运算符列表"""
        MyOperator.global_operator_list.clear()


class MyAdd(MyOperator):
    """加法"""
    def __init__(self):
        super().__init__()

    def compute(self, a: MyTensor, b: MyTensor):
        self.inputs = [a, b]
        out = MyTensor(a.data + b.data, self)
        self.output = out
        return out

    def deriv(self):
        self.inputs[0].grad += self.output.grad
        self.inputs[1].grad += self.output.grad


class MySub(MyOperator):
    """减法"""
    def __init__(self):
        super().__init__()

    def compute(self, a: MyTensor, b: MyTensor):
        self.inputs = [a, b]
        out = MyTensor(a.data - b.data, self)
        self.output = out
        return out

    def deriv(self):
        self.inputs[0].grad += self.output.grad
        self.inputs[1].grad += -self.output.grad


class MyMul(MyOperator):
    """乘法"""
    def __init__(self):
        super().__init__()

    def compute(self, a: MyTensor, b: MyTensor):
        self.inputs = [a, b]
        out = MyTensor(a.data * b.data, self)
        self.output = out
        return out

    def deriv(self):
        self.inputs[0].grad += self.inputs[1].data * self.output.grad
        self.inputs[1].grad += self.inputs[0].data * self.output.grad


class MyDiv(MyOperator):
    """除法"""
    def __init__(self):
        super().__init__()

    def compute(self, a: MyTensor, b: MyTensor):
        self.inputs = [a, b]
        out = MyTensor(a.data / b.data, self)
        self.output = out
        return out

    def deriv(self):
        self.inputs[0].grad += self.output.grad / self.inputs[1].data
        self.inputs[1].grad += -self.output.grad * (self.inputs[0].data / self.inputs[1].data ** 2)


class MySum(MyOperator):
    """求和"""
    def __init__(self):
        super().__init__()

    def compute(self, a: MyTensor):
        self.inputs = [a]
        out = MyTensor([sum(a.data)], self)
        self.output = out
        return out

    def deriv(self):
        for i in range(len(self.inputs[0].grad)):
            self.inputs[0].grad[i] += self.output.grad[0]


class MyPow(MyOperator):
    """幂"""
    def __init__(self):
        super().__init__()
        self.mi = None

    def compute(self, a: MyTensor, b: float):
        self.inputs = [a]
        self.mi = b
        out = MyTensor(a.data ** b, self)
        self.output = out
        return out

    def deriv(self):
        self.inputs[0].grad += np.array([self.mi * self.inputs[0].data[i] ** (self.mi - 1)
            * self.output.grad[i] for i in range(len(self.inputs[0].grad))])


class MyExp(MyOperator):
    """指数"""
    def __init__(self):
        super().__init__()

    def compute(self, a: MyTensor):
        self.inputs = [a]
        out = MyTensor([math.exp(i) for i in a.data], self)
        self.output = out
        return out

    def deriv(self):
        self.inputs[0].grad += self.output.grad * self.output.data


class MyLog(MyOperator):
    def __init__(self):
        super().__init__()

    def compute(self, a: MyTensor):
        self.inputs = [a]
        out = MyTensor([math.log(i) if i != 0 else float("inf") for i in a.data], self)
        self.output = out
        return out

    def deriv(self):
        self.inputs[0].grad += self.output.grad / self.inputs[0].data


class MyCut(MyOperator):
    def __init__(self):
        super().__init__()
        self.start = None
        self.end = None

    def compute(self, a: MyTensor, start: int, end: int):
        self.inputs = [a]
        self.start = start
        self.end = end
        out = MyTensor(a.data[start:end], self)
        self.output = out
        return out

    def deriv(self):
        for i in range(len(self.output.data)):
            self.inputs[0].grad[i+self.start] += self.output.grad[i]


class MyConnect(MyOperator):
    def __init__(self):
        super().__init__()

    def compute(self, a: list[MyTensor]):
        self.inputs = a
        out = MyTensor([], self)
        out.data = np.concatenate([t.data for t in a])
        out.grad = np.concatenate([t.grad for t in a])
        self.output = out
        return out

    def deriv(self):
        seg = self.output.grad
        for i in range(len(self.inputs)):
            self.inputs[0].grad += self.output.grad[i]
            seg = seg[len(self.inputs[i].grad)]


class MyRelu(MyOperator):
    def __init__(self):
        super().__init__()

    def compute(self, a: MyTensor):
        self.inputs = [a]
        out = MyTensor([max(0, i) for i in a.data], self)
        self.output = out
        return out

    def deriv(self):
        for i in range(len(self.inputs[0].data)):
            if self.inputs[0].data[i] > 0:
                self.inputs[0].grad[i] += self.output.grad[i]


class MySigmoid(MyOperator):
    def __init__(self):
        super().__init__()

    def compute(self, a: MyTensor):
        self.inputs = [a]
        out = MyTensor([1 / (1 + math.exp(-i)) for i in a.data], self)
        self.output = out
        return out

    def deriv(self):
        self.inputs[0].grad += (self.output.grad * self.output.data *
            (np.array([1 for _ in range(len(self.output))]) - self.output.data))


class MyTanh(MyOperator):
    def __init__(self):
        super().__init__()

    def compute(self, a: MyTensor):
        self.inputs = [a]
        out = MyTensor([math.tanh(i) for i in a.data], self)
        self.output = out
        return out

    def deriv(self):
        self.inputs[0].grad += (self.output.grad *
            np.array([1 - i ** 2 for i in self.output.data]))


# ========== 神经网络层 ==========


class MyLayer:
    """数据层。所有要保存数据的类都需要继承此类"""

    global_layer_list = []  # 保存所有层
    is_load = False  # 是否处于读取状态
    pointer = 0  # 读取数据用的指针, 记录当前正在读取的索引

    def __init__(self, *args):
        """当处于读取状态时，继承实例会按顺序读取global_layer_list中的内容
        一般情况下，继承它的类的init中需最后调用super().init()，防止数据被覆盖"""
        if MyLayer.is_load and MyLayer.pointer != len(MyLayer.global_layer_list):
            self.load(MyLayer.global_layer_list[MyLayer.pointer])
            MyLayer.global_layer_list[MyLayer.pointer] = self
            MyLayer.pointer += 1
        else:
            MyLayer.global_layer_list.append(self)

    def grad_descent_and_zero(self, k):
        """梯度下降和梯度清零"""
        ...

    def save(self):
        """保存数据"""
        ...

    def load(self, *args):
        """读取数据"""
        ...

    @classmethod
    def save_all(cls, path: str):
        """存储所有实例"""
        with open(path, "w") as f:
            for i in MyLayer.global_layer_list:
                f.write(i.save() + "\n")

    @classmethod
    def load_all(cls, path: str):
        """从文件中读取保存的内容"""
        MyLayer.is_load = True
        with open(path, "r") as f:
            MyLayer.global_layer_list = f.readlines()


class MyLinear(MyLayer):
    """全连接层"""
    def __init__(self, in_size: int, out_size: int, bias=True):
        """
        初始化全连接层
        :param in_size: int 作为输入的Ten的维度
        :param out_size: int 作为输出的Ten的维度
        :param bias: bool 是否加上偏置
        """
        super().__init__()
        if not MyLayer.is_load:
            self.w = [MyTool.randinit(in_size) for _ in range(out_size)]
            self.b = [0 for _ in range(out_size)]
            self.bias = bias
            if self.bias:
                self.b = [MyTool.randinit(1) for _ in range(out_size)]

    def __call__(self, x: MyTensor):
        out = [(self.w[i] * x).sum() + self.b[i] for i in range(len(self.w))]
        out = MyTensor.connect(out)
        return out

    def grad_descent_and_zero(self, k: float):
        for i in range(len(self.w)):
            self.w[i].grad_descent(k)
            self.b[i].grad_descent(k)
            self.w[i].zero_grad()
            self.b[i].zero_grad()

    def save(self):
        txt = str(list(i.data) for i in self.w)
        txt += "/" + str(list(i.data) for i in self.b)
        return txt

    def load(self, txt: str):
        txt = txt.split("/")
        self.w = [MyTensor(i) for i in eval(txt[0])]
        self.b = [MyTensor(i) for i in eval(txt[1])]
        self.bias = sum(self.b) != 0


class MyTensorLayer(MyTensor, MyLayer):
    """可保存的张量"""
    def __init__(self, data: list):
        MyTensor.__init__(self, data)
        MyLayer.__init__(self)

    def __call__(self, *args, **kwargs):
        return self

    def grad_descent_and_zero(self, k: float):
        self.grad_descent(k)
        self.zero_grad()

    def save(self):
        txt = str(list(self.data))
        return txt

    def load(self, txt: str):
        self.data = np.array(eval(txt))


class MyConv(MyLayer):
    """卷积层"""
    def __init__(self, width: int, height: int, stride_w=1, stride_h=1, pad=True, bias=True):
        """
        初始化2d卷积层
        :param width: int 卷积核的宽度（如填充，请设为奇数）
        :param height: int 卷积核的高度（如填充，请设为奇数）
        :param stride_w: int 横向的步长
        :param stride_h: int 纵向的步长
        :param pad: bool 是否进行填充（使运算的输入和输出的大小一样）
        :param bias: bool 是否加上偏置
        """
        super().__init__()
        if not MyLayer.is_load:
            self.width = width
            self.height = height
            self.stride_w = stride_w
            self.stride_h = stride_h
            self.pad = pad
            self.bias = bias
            self.kernel = MyTool.randinit(width * height)
            self.b = [0 for _ in range(width * height)]
            if self.bias:
                self.b = MyTool.randinit(1)

    def __call__(self, x: list[MyTensor]):
        if self.pad:
            x = self.padding(x)
        new_x = []
        for _y in range(0, len(x)-self.height+1, self.stride_h):
            line = []
            for _x in range(0, len(x[0])-self.width+1, self.stride_w):
                window = MyTensor.connect(
                    [x[_y + i].cut(_x, _x + self.width) for i in range(self.height)]
                )
                s = (window * self.kernel).sum() + self.b
                line.append(s)
            new_x.append(MyTensor.connect(line))
        return new_x

    def grad_descent_and_zero(self, k: float):
        self.kernel.grad_descent(k)
        self.kernel.zero_grad()

    def save(self):
        t = f"{self.width}/{self.height}/{list(self.kernel.data)}/{self.stride_w}/{self.stride_h}/{self.pad}"
        t += f"/{self.b.data}"
        return t

    def load(self, txt: str):
        txts = txt.split("/")
        self.width = int(txts[0])
        self.height = int(txts[1])
        self.kernel = MyTensor(eval(txts[2]))
        self.stride_w = int(txts[3])
        self.stride_h = int(txts[4])
        self.pad = eval(txts[5])
        self.b = MyTensor(eval(txts[6]))
        self.bias = self.b.sum() != 0

    def padding(self, x: list[MyTensor]):
        pad_x = (self.stride_w * (len(x[0]) - 1) - len(x[0]) + self.width) // 2
        pad_y = (self.stride_h * (len(x) - 1) - len(x) + self.height) // 2
        new_x = []
        for i in range(pad_y):
            new_x.append(MyTensor.zero(len(x[0]) + pad_x * 2))
        for i in range(len(x)):
            new_x.append(MyTensor.connect([MyTensor.zero(pad_x), x[i], MyTensor.zero(pad_x)]))
        for i in range(pad_y):
            new_x.append(MyTensor.zero(len(x[0]) + pad_x * 2))
        return new_x


class MyMultiConv(MyLayer):
    """多通道卷积层"""
    def __init__(self, in_channels: int, out_channels: int, width: int, height: int, stride_w=1, stride_h=1, pad=True, bias=True):
        """
        初始化多通道卷积层
        :param in_channels: int 输入通道数
        :param out_channels: int 输出通道数
        :param width: int 卷积核的宽度（如填充，请设为奇数）
        :param height: int 卷积核的高度（如填充，请设为奇数）
        :param stride_w: int 横向的步长
        :param stride_h: int 纵向的步长
        :param pad: bool 是否进行填充（使运算的输入和输出的大小一样）
        :param bias: bool 是否加上偏置
        """
        super().__init__()
        self.cores = [
            [MyConv(width, height, stride_w, stride_h, pad=pad, bias=bias)
             for _ in range(in_channels)]
            for _ in range(out_channels)
        ]
        self.pad = pad

    def __call__(self, x: list[list[MyTensor]]):
        if self.pad:
            x = [self.cores[0][0].padding(i) for i in x]
        new_x = []
        for channel in self.cores:
            x_channel = MyTool.sum_channels(
                [channel[i](x[i]) for i in range(len(channel))]
            )
            new_x.append(x_channel)
        return new_x

    def grad_descent_and_zero(self, k: float):
        for i in self.cores:
            for j in i:
                j.grad_descent_and_zero(k)

    def save(self):
        ...

    def load(self, txt: str):
        ...


class MyAttention(MyLayer):
    """单头自注意力模块"""
    def __init__(self, embed_size: int, qk_size:int=None, v_size:int=None):
        """
        初始化单头自注意力模块
        :param embed_size: int 输入词向量维度
        :param qk_size: int q、k维度
        :param v_size: int 输出词向量维度，默认与输入相同
        """
        super().__init__()
        if qk_size is None:
            qk_size=embed_size//2
        if v_size is None:
            v_size=embed_size
        self.q = MyLinear(embed_size,qk_size)
        self.k = MyLinear(embed_size,qk_size)
        self.v = MyLinear(embed_size,v_size)
        self.embed_size = embed_size
        self.qk_size = qk_size
        self.out_size = v_size

    def __call__(self, x: list[MyTensor], mask_list=None, tri_mask=False):
        q_list = []
        k_list = []
        v_list = []
        for w in x:
            q_list.append(self.q(w))
            k_list.append(self.k(w))
            v_list.append(self.v(w))
        at_list = []
        for i in range(len(q_list)):
            line = []
            for j in range(len(q_list)):
                if (mask_list is not None and (mask_list[i] == 1 or mask_list[j] == 1)) \
                        or (tri_mask and j > i):
                    line.append(MyTensor([float("-inf")]))
                else:
                    line.append((q_list[i] * k_list[j]).sum() / MyTensor([self.qk_size ** 0.5]))
            at_list.append(MyTensor.connect(line).softmax())
        new_v_list = []
        for i in range(len(q_list)):
            line = MyTensor.zero(self.out_size)
            for j in range(len(q_list)):
                line += v_list[j] * (at_list[i].cut(j, j + 1).expand(self.out_size))
            new_v_list.append(line)
        return new_v_list

    def grad_descent_and_zero(self, k: float):
        self.q.grad_descent_and_zero(k)
        self.k.grad_descent_and_zero(k)
        self.v.grad_descent_and_zero(k)

    def save(self):
        ...

    def load(self, txt: str):
        ...


class MyMultiAtt(MyLayer):
    """多头注意力模块"""
    def __init__(self, embed_size: int, head_num: int, qk_size:int=None, v_size:int=None):
        """
        初始化多头注意力模块
        :param embed_size: int 输入词向量维度
        :param head_num: int 多头注意力的个数
        :param qk_size: int q、k维度
        :param v_size: int 输出向量维度
        """
        super().__init__()
        self.heads = [MyAttention(embed_size, qk_size, v_size) for _ in range(head_num)]
        self.embed_size = embed_size

    def __call__(self, x: list[MyTensor], mask_list=None,tri_mask=False):
        out = [h(x, mask_list, tri_mask) for h in self.heads]
        out = MyTool.sum_channels(out)
        return out

    def grad_descent_and_zero(self, k: float):
        for i in self.heads:
            i.grad_descent_and_zero(k)

    def save(self):
        ...

    def load(self, txt: str):
        ...


class MyLSTM(MyLayer):
    """LSTM层"""
    def __init__(self, embed_size: int, output_size: int):
        """
        初始化LSTM层
        :param embed_size:
        :param output_size:
        """
        super().__init__()
        self.forget_gate = MyLinear(embed_size + output_size, output_size)
        self.input_gate = MyLinear(embed_size + output_size, output_size)
        self.input_gate2 = MyLinear(embed_size + output_size, output_size)
        self.output_gate = MyLinear(embed_size + output_size, output_size)
        self.h = MyTensor.zero(output_size)
        self.s = MyTensor.zero(output_size)

    def __call__(self, x: list[MyTensor]):
        out = []
        for i in x:
            i = MyTensor.connect([i, self.h])
            self.s *= self.forget_gate(i).sigmoid()
            self.s += self.input_gate(i).sigmoid() * self.input_gate2(i).tanh()
            self.h = self.output_gate(i).sigmoid() * self.s.tanh()
            out.append(self.h)
        return out

    def grad_descent_and_zero(self, k: float):
        self.forget_gate.grad_descent_and_zero(k)
        self.input_gate.grad_descent_and_zero(k)
        self.input_gate2.grad_descent_and_zero(k)
        self.output_gate.grad_descent_and_zero(k)

    def save(self):
        ...

    def load(self, txt: str):
        ...


class MyNorm(MyLayer):
    """标准化层"""
    def __init__(self):
        """初始化标准化层"""
        super().__init__()
        self.w = MyTensorLayer([random.gauss(0, 0.04)])
        self.b = MyTensorLayer([random.gauss(0, 0.04)])

    def __call__(self, x: MyTensor, eps=0.0001):
        n = MyTensor([len(x)])
        mean = (x.sum() / n).expand(len(x))
        sigma = (((x - mean) ** 2).sum() / n) ** 0.5
        std = (x - mean) / (MyTensor([eps]) + sigma).expand(len(x))
        out = self.w.expand(len(x)) * std + self.b.expand(len(x))
        return out

    def grad_descent_and_zero(self, k: float):
        self.w.grad_descent(k)
        self.b.grad_descent(k)

    def save(self):
        ...

    def load(self, txt: str):
        ...


# ========== 模型 ==========


class MyMiniTransformer:
    def __init__(self, head_num, emb_size, window_size, lowrank=False):
        self.a = MyMultiAtt(head_num, emb_size)
        if lowrank:
            self.f1 = MyMiniLinear(emb_size * window_size, emb_size * window_size)
            self.f2 = MyMiniLinear(emb_size * window_size, emb_size * window_size)
        else:
            self.f1 = MyLinear(emb_size * window_size, emb_size * window_size)
            self.f2 = MyLinear(emb_size * window_size, emb_size * window_size)
        self.n1 = MyNorm()
        self.n2 = MyNorm()
        self.emb_size = emb_size
        self.window_size = window_size

    def __call__(self, x: list[MyTensor], mask_list=None, tri_mask=False):
        x2 = x  # 2dTen
        x = MyTensor.connect(self.a(x, mask_list, tri_mask))
        x = self.n1(x)
        x += MyTensor.connect(x2)
        x2 = x  # 1dTen
        x = self.f2(self.f1(x).relu())  # 应为gelu
        x = self.n2(x)
        x += x2
        x = MyTool.resize2d(x, self.emb_size, self.window_size)
        return x

    def grad_descent_and_zero(self, k: float):
        self.a.grad_descent_and_zero(k)
        self.f1.grad_descent_and_zero(k)
        self.f2.grad_descent_and_zero(k)
        self.n1.grad_descent_and_zero(k)
        self.n2.grad_descent_and_zero(k)

    def save(self):
        ...

    def load(self, txt: str):
        ...


class MyMiniLinear:
    def __init__(self, input_size, out_size, midsize=None, bias=True):
        if midsize is None:
            midsize = round(((input_size + out_size) / 2) ** 0.5)
        self.f1 = MyLinear(input_size, midsize, bias)
        self.f2 = MyLinear(midsize, out_size, bias)

    def __call__(self, x: MyTensor):
        x = self.f1(x)
        x = self.f2(x)
        return x

    def grad_descent_and_zero(self, k: float):
        self.f1.grad_descent_and_zero(k)
        self.f2.grad_descent_and_zero(k)

    def save(self):
        ...

    def load(self, txt: str):
        ...


# ========== 工具函数 ==========


class MyTool:
    @staticmethod
    def randinit(size):
        """
        初始化权重
        :param size: int 权重的维度大小
        :return: Ten
        """
        sigma = (2 / size) ** 0.5 / 5  # 0.04 只有定值才能使GAN正常训练？
        return MyTensor([random.gauss(0, sigma) for _ in range(size)])

    @staticmethod
    def sum_channels(x):
        """
        对多通道的2dTen求和，变成单通道2dTen
        :param x: list[list[Ten,Ten...],list[Ten,Ten...]...]
        :return: list[Ten,Ten...]
        """
        out = x[0][:]
        for pic in range(1, len(x)):
            for line in range(len(x[0])):
                out[line] += x[pic][line]
        return out

    @staticmethod
    def resize2d(x, emb_size, window_size):
        """
        把一维Ten转为2dTen
        :param x: Ten
        :param emb_size: int 每个Ten的大小
        :param window_size: int 多少个Ten
        :return: list[Ten,Ten...]
        """
        x2 = []
        for i in range(window_size):
            x2.append(x.cut(i * emb_size, (i + 1) * emb_size))
        return x2


class Test:
    @staticmethod
    def run():
        x = MyTensor([1])
        y = MyTensor([1])
        z = MyTensor([1])

        for i in range(1000):
            print(f"{i + 1}=x{x},y{y},z{z}")
            s1 = ((x * y + z) - MyTensor([40])) ** 2
            s2 = ((x * z + y) - MyTensor([51])) ** 2
            s3 = ((x + y + z) - MyTensor([19])) ** 2
            if s1.data[0] < 0.01 and s2.data[0] < 0.01 and s3.data[0] < 0.01:
                break
            s1.backward(clean=False)
            s2.backward(clean=False)
            s3.backward()
            # s1.grad = np.array([1])
            # s2.grad = np.array([1])
            # s3.grad = np.array([1])
            # MyOperator.backward()
            x.data -= x.grad * np.array([0.002])
            y.data -= y.grad * np.array([0.002])
            z.data -= z.grad * np.array([0.002])
            # print(f"{i+1}=x{x},y{y},z{z}")
            x.zero_grad()
            y.zero_grad()
            z.zero_grad()

    @staticmethod
    def start():
        x = MyTensor([1])
        y = MyTensor([1])

        for i in range(100):
            print(f"{i + 1}=x{x},y{y}")

            s = x ** 2

            s.backward()

            x.data -= x.grad * np.array([0.01])

            x.zero_grad()

if __name__ == '__main__':
    Test.run()







