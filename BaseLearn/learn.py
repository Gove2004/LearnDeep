import torch
import numpy as np
def test_1():  # torch和numpy
    np_data = np.arange(6).reshape((2, 3))
    torch_data = torch.from_numpy(np_data)
    np_data = torch_data.numpy()
    print("完成！")

from torch.autograd import Variable
def test_2():  # tensor和Variable
    tensor = torch.FloatTensor([[1, 2], [3, 4]])
    Variabe = Variable(tensor, requires_grad=True)
    t_out = torch.mean(tensor * tensor)
    v_out = torch.mean(Variabe * Variabe)
    v_out.backward()
    print(Variabe.grad)
    print(Variabe.data)
    print(Variabe.data.numpy())
    print("完成！")

import torch.nn.functional as F  # 激励函数
import matplotlib.pyplot as plt  # 绘图模块
def test_3():  # 激励函数图像绘制,relu(),sigmoid(),tanh(),softplus()
    # 创建一个张量包含从 -5 到 5 的 200 个均匀间隔的数值。
    x = torch.linspace(-5, 5, 200)
    x = Variable(x, requires_grad=True)
    x_np = x.data.numpy()
    # 定义四种激励函数
    y_relu = F.relu(x).data.numpy()  # 负数置为0
    y_sigmoid = F.sigmoid(x).data.numpy()  # 0到1
    y_tanh = F.tanh(x).data.numpy()  # -1到1
    y_softplus = F.softplus(x).data.numpy()  #log(1 + exp(x))
    # 绘制四个激励函数的图像
    plt.figure(1, figsize=(8, 6))
    plt.subplot(221)
    plt.plot(x_np, y_relu, c='red',label='relu')  # 画出relu函数
    plt.ylim((-1, 5))
    plt.legend(loc='best')
    plt.subplot(222)
    plt.plot(x_np, y_sigmoid, c='red', label='sigmoid')  # 画出sigmoid函数
    plt.ylim((-0.2, 1.2))
    plt.legend(loc='best')
    plt.subplot(223)
    plt.plot(x_np, y_tanh, c='red', label='tanh')  # 画出tanh函数
    plt.ylim((-1.2, 1.2))
    plt.legend(loc='best')
    plt.subplot(224)
    plt.plot(x_np, y_softplus, c='red', label='softplus')  # 画出softplus函数
    plt.ylim((-0.2, 6))
    plt.legend(loc='best')
    # 显示图像
    plt.show()
    print("完成！")

# 激活函数改成tanh,训练次数改成500效果更加！亲测有效！！！
def test_4(gf=False):  # 训练一个简单的神经网络（拟合）
    i = input("输入1启用gf模式\n")
    if i == '1':
        gf = True
        print("启用gf模式！")
    
    #  创建数据：自变量x，元数据y（随机噪声）
    x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)
    y = x.pow(2) + 0.2 * torch.rand(x.size())
    x, y = Variable(x), Variable(y)

    class Net(torch.nn.Module):  # 定义网络结构
        def __init__(self, n_feature, n_hidden, n_output):  # 初始化网络结构(输入层维数，隐藏层维数，输出层维数)
            super(Net, self).__init__()  # 继承父类的初始化方法
            self.hidden = torch.nn.Linear(n_feature, n_hidden)  # 隐藏层
            self.predict = torch.nn.Linear(n_hidden, n_output)  # 输出层
        def forward(self, x):
            if gf:
                x = F.tanh(self.hidden(x))  # 激活函数tanh
            else:
                x = F.relu(self.hidden(x))  # 激活函数relu
            x = self.predict(x)
            return x
    net = Net(1, 10, 1)  # 实例化网络

    plt.ion()  # 打开交互模式，允许动态更新图
    plt.show()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.2)  # 定义随机梯度下降优化器SGD，net.parameters()为网络参数, lr为学习率
    loss_func = torch.nn.MSELoss()  # 定义损失函数:均方误差损失函数MSELoss()
    if gf:
        step_number = 500  # 训练500次
    else:
        step_number = 100  # 训练100次
    for t in range(step_number):
        # 标准五个步骤：向前传播，计算损失，梯度清零，反向传播，更新权重
        prediction = net(x)  # 向前传播（预测）
        loss = loss_func(prediction, y)  # 计算损失
        optimizer.zero_grad()  # 因为梯度在多次调用backward() 时累加，不清零的话旧的梯度会干扰新梯度
        loss.backward()  # 反向传播
        optimizer.step()  # 更新，权重的优化
        
        plt.cla()
        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=3)  # 画出预测值（折线），r-表示红色，lw=3表示线宽
        plt.text(0, 1, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size': 20, 'color': 'red'})
        # 显示损失，0,1为text坐标，'Loss=%.4f'%loss.data.numpy()为内容（四位小数），fontdict为字体设置(大小，颜色)
        plt.pause(0.01)
    print("完成！")
    plt.ioff()  # 关闭交互模式
    plt.show()

def test_5(quickly_create=False):  # [快速]训练一个简单的神经网络（分类）
    i = input("输入1启用快速创建模式\n")
    if i == '1':
        quickly_create = True
        print("启用快速创建模式！")
    
    # 生成100个包含数据，x是2维（样本/数据），y是1维（标签/类别）
    n_data = torch.ones(100, 2)  # 值为1的数据
    x0 = torch.normal(2 * n_data, 1)  # 均值为2*n_data，方差为1
    x1 = torch.normal(-2 * n_data, 1)  # 均值为-2*n_data，方差为1
    x = torch.cat((x0, x1), 0).type(torch.FloatTensor)  # 将x0和x1拼接并转换为浮点型作为样本/数据
    y0 = torch.zeros(100)  # 一类标签
    y1 = torch.ones(100)  # 二类标签
    y = torch.cat((y0, y1), 0).type(torch.LongTensor)  # 将y0和y1拼接并转换为长整型作为标签/类别
    x, y = Variable(x), Variable(y)
    
    if quickly_create:  # 快速创建网络
        net = torch.nn.Sequential(  
            torch.nn.Linear(2, 10), # 第一层：输入2维，输出10维
            torch.nn.ReLU(),  # 第二层：激活函数ReLU
            torch.nn.Linear(10, 2))  # 第一三层：输入10维，输出2维
    else:  # 手动创建网络
        class Net(torch.nn.Module):
            def __init__(self, n_feature, n_hidden, n_output):
                super(Net, self).__init__()
                self.hidden = torch.nn.Linear(n_feature, n_hidden)
                self.predict = torch.nn.Linear(n_hidden, n_output)
            def forward(self, x):
                x = F.relu(self.hidden(x))
                x = self.predict(x)
                return x
        net = Net(2, 10, 2)  # 两个输入特征(x0, x1)和两类输出标签(0，1)

    plt.ion()
    plt.show()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01)  # 适当的lr方便可视化演示
    loss_func = torch.nn.CrossEntropyLoss()  # 交叉熵损失函数，
    #交叉熵（Cross Entropy）用于衡量一个概率分布（例如[0.3, 0.7]）与另一个概率分布（真实分布[0, 1)）之间的距离
    for t in range(100):
        outcome = net(x)
        loss = loss_func(outcome, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        prediction = torch.max(F.softmax(outcome, dim=1), 1)[1]  # softmax函数将模型的原始输出转换为和为1的概率分布,概率最大的max()作为预测值，类别/标签为[1]
        pred_y = prediction.data.numpy().squeeze()  # squeeze()用于移除张量维度中所有的大小为1的维度
        target_y = y.data.numpy()  # 获取标签/类别
        accuracy = sum(pred_y == target_y) / 200.  # 预测正确的样本数/总样本数
        plt.cla()
        plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=pred_y, s=100, lw=0, cmap='RdYlGn')  # s为点的大小，cmap为颜色映射（根据c表示分类结果）
        plt.text(1.5, -4, 'Accuracy=%.2f' % accuracy, fontdict={'size': 20, 'color': 'red'})  # 显示准确率
        plt.pause(0.1)
    plt.ioff()
    plt.show()

    print("完成！")
# 参数gf作用见test_4()
def test_6(gf=False, _path=r'C:\Users\net.pkl'):  # 保存和加载整个网络或网络参数
    i = input("输入1启用gf模式\n")
    if i == '1':
        gf = True
        print("启用gf模式")
    _path = input(r"输入文件保存/读取路径(默认'C:\Users\net.pkl')")
    if _path == '':
        _path = r'C:\Users\net.pkl'
        print("启用默认路径")

    x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)
    y = x.pow(2) + 0.2 * torch.rand(x.size())
    x, y = Variable(x), Variable(y)
    # 采用快速创建网络的方式
    net1 = torch.nn.Sequential(  
        torch.nn.Linear(1, 10),
        torch.nn.Tanh() if gf else torch.nn.ReLU(),  # 适当的激活函数
        torch.nn.Linear(10, 1))

    optimizer = torch.optim.SGD(net1.parameters(), lr=0.2)  # 不适合的学习率结果不理想
    loss_func = torch.nn.MSELoss()  # 错误的损失函数无法得到好的结果
    for t in range(500 if gf else 100):
        prediction1 = net1(x)
        loss = loss_func(prediction1, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # 保存函数(整个网络或网络参数)
    def save_net(net, path=_path, params=False):  # 保存网络
        if params:  # 保存网络参数
            torch.save(net.state_dict(), path)
        else:  # 保存整个网络
            torch.save(net, path)
    # 加载函数(整个网络或网络参数) 
    def load_net(path=_path, params=False):  # 加载网络
        if params:  # 加载网络参数
            net = torch.nn.Sequential(    # 创建一个空且相同的网络
                torch.nn.Linear(1, 10),
                torch.nn.Tanh() if gf else torch.nn.ReLU(),
                torch.nn.Linear(10, 1))
            net.load_state_dict(torch.load(path))  # 向空网中写入参数
        else:  # 加载整个网络
            net = torch.load(path)
        return net
    # 保存和加载网络
    prediction1 = net1(x)
    save_net(net1)
    net2 = load_net()
    prediction2 = net2(x)
    save_net(net1, params=True)
    net3 = load_net(params=True)
    prediction3 = net3(x)
    # 画出结果
    plt.figure(1, figsize=(10, 3))
    plt.subplot(131)
    plt.title('net1')
    plt.scatter(x.data.numpy(), y.data.numpy())
    plt.plot(x.data.numpy(), prediction1.data.numpy(), 'r-', lw=3)
    plt.subplot(132)
    plt.title('net2')
    plt.scatter(x.data.numpy(), y.data.numpy())
    plt.plot(x.data.numpy(), prediction2.data.numpy(), 'r-', lw=3)
    plt.subplot(133)
    plt.title('net3')
    plt.scatter(x.data.numpy(), y.data.numpy())
    plt.plot(x.data.numpy(), prediction3.data.numpy(), 'r-', lw=3)
    plt.show()

    print("完成！")

import torch.utils.data as Data
def test_7():  # 数据批处理(批训练)
    BATCH_SIZE = 5  # 批大小

    x = torch.linspace(1, 10, 10)
    y = torch.linspace(10, 1, 10)
    
    torch_dataset = Data.TensorDataset(x, y)  # 创建数据集(x为训练y为目标)
    loader = Data.DataLoader(  # 迭代器
        dataset=torch_dataset,  # 数据集
        batch_size=BATCH_SIZE,  # 批大小
        shuffle=True,  # 是否打乱数据
        num_workers=2,  # 多线程
        drop_last=False)  # 丢弃最后一个不完整的批
      # epoch为训练/迭代次数，step为每次训练/迭代时的批次
    for epoch in range(3):
        # enumerate()接收一个可迭代对象作为输入，loader()并返回一个元组
        for step, (batch_x, batch_y) in enumerate(loader):  # 遍历元组，step接收元组的第一个元素（迭代的当前批次索引（从0开始））
            #(batch_x, batch_y)接收元组的第二个元素（一个包含两个元素的元组）batch_x代表当前批次的输入数据，batch_y代表对应的标签或目标数据
            print('epoch: ', epoch, 
                  '| step: ', step, 
                  '| batch x: ', batch_x.numpy(), 
                  '| batch y: ', batch_y.numpy())
    print("完成！")

def test_8():  # 更多的优化器(以及更好的误差/损失比较)
    # 超参数
    LR = 0.01  # 学习率太大不太好
    BATCH_SIZE = 32
    EPOCH = 12
    # 生成数据(拟合)
    x = torch.unsqueeze(torch.linspace(-1, 1, 1000), dim=1)
    y = x.pow(2) + 0.1 * torch.rand(x.size())
    # 批处理迭代器
    torch_dataset = Data.TensorDataset(x, y)
    loader = Data.DataLoader(
        dataset=torch_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        drop_last=False)
    # 定义网络
    class Net(torch.nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.hidden = torch.nn.Linear(1, 20)
            self.predict = torch.nn.Linear(20, 1)
        
        def forward(self, x):
            x = F.relu(self.hidden(x))
            x = self.predict(x)
            return x
    # 创建四个网络
    net_SGD = Net()  # 随机梯度下降
    net_Momentum = Net()  # 引入动量的随机梯度下降
    net_RMSprop = Net()  # 动态调整学习率的随机梯度下降
    net_Adam = Net()  # 结合引入动量和动态调整学习率的随机梯度下降
    nets = [net_SGD, net_Momentum, net_RMSprop, net_Adam]
    # 创建四个不同的优化器
    opt_SGD = torch.optim.SGD(net_SGD.parameters(), lr=LR)
    opt_Momentum = torch.optim.SGD(net_Momentum.parameters(), lr=LR, momentum=0.8)
    opt_RMSprop = torch.optim.RMSprop(net_RMSprop.parameters(), lr=LR, alpha=0.9)  # 较高的alpha/beta意味着更多的历史信息被保留
    opt_Adam = torch.optim.Adam(net_Adam.parameters(), lr=LR, betas=(0.9, 0.99))  # 较低的alpha/beta意味着更多关注近期信息
    optimizers = [opt_SGD, opt_Momentum, opt_RMSprop, opt_Adam]
    # 误差/损失函数并记录
    loss_func = torch.nn.MSELoss()
    losses_list = [[], [], [], []]
    # 开始训练
    for epoch in range(EPOCH):
        print('迭代次数：', epoch)  # 输出当前迭代次数（提示是否迭代结束）
        for step, (batch_x, batch_y) in enumerate(loader):
             b_x = Variable(batch_x)  # 将数据转换成需要的类型Variable
             b_y = Variable(batch_y)
             for net, opt, loss_list in zip(nets, optimizers, losses_list):  # 遍历四个网络,zip()
                 outcome = net(b_x)
                 loss = loss_func(outcome, b_y)
                 opt.zero_grad()
                 loss.backward()
                 opt.step()
                 loss_list.append(loss.data.numpy())  # 记录误差/损失
    
    Label = ['SGD', 'Momentum', 'RMSprop', 'Adam']  # 创建可视化标签
    for i, loss_list in enumerate(losses_list):
        plt.plot(loss_list, label=Label[i])
    plt.legend(loc='best')
    plt.xlabel('step')
    plt.ylabel('loss')
    plt.show()

    print("完成！")

import torch.nn as nn
import torchvision  # 提供许多预定义的图像数据集、预训练模型和图像转换工具,仅test_9()可下载训练数据
def test_9(download=False, _path='C:\\Users\\'):  # 卷积神经网络CNN
    i = input("输入1启用下载数据\n")
    if i == '1':
        download = True
    _path = input("输入数据路径（默认'C:\\Users\\'）\n")
    if _path == '':
        _path = 'C:\\Users\\'
        print("启用默认路径")

    EPOCH = 1
    BATCH_SIZE = 50
    LR = 0.001
    DOWNLOAD_MNIST = download  # 是否下载数据

    train_data = torchvision.datasets.MNIST(  # 加载训练数据
        root=_path,  # 数据存放位置
        train=True,  # 训练数据
        transform=torchvision.transforms.ToTensor(),  # 转换PIL.Image或numpy.ndarray为torch.FloatTensor -> (0, 1)
        download=DOWNLOAD_MNIST,)  # 下载数据
    train_loader = Data.DataLoader(
        dataset=train_data, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=2,
        drop_last=False)
    if DOWNLOAD_MNIST:
        plt.imshow(train_data.data[0].numpy(), cmap='gray')  # 显示训练数据中的第一张图片
        plt.title('%i' % train_data.targets[0])  # 显示训练数据中的第一张图片的标签
        plt.show()

    test_data = torchvision.datasets.MNIST(root=_path, train=False)  # 加载测试数据
    # 选取前[:2000]个数据作为测试数据，转化为合适的数据类型和数据范围
    test_x = Variable(torch.unsqueeze(test_data.data, dim=1)).type(torch.FloatTensor)[:2000]/255.  
    test_y = test_data.targets[:2000]  # 这是一道分类题

    class CNN(nn.Module):
        def __init__(self):  # 卷积层通常用于处理图像数据，通过学习局部特征来进行特征提取
            super(CNN, self).__init__()
            self.conv1 = nn.Sequential(  # 卷积层1
                nn.Conv2d(  # (1, 28, 28) -> (16, 28, 28)
                    in_channels=1,  # 输入图像的通道数为1，适用于灰度图像。
                    out_channels=16,  # 输出特征图的数量为16，即将学习16个不同的特征。
                    kernel_size=5,  # 卷积核的大小为5x5
                    stride=1,  # 卷积核在输入图像上移动的步长为1px
                    padding=2,), #在图像[边界]外添加2圈零填充，保持输出尺寸不变,与kernel_size和stride有关
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2),)  # 最大池化层，1x1取代2x2中最大值，长宽减半, (16, 28, 28) -> (16, 14, 14)
            self.conv2 = nn.Sequential(  # 卷积层2
                nn.Conv2d(16, 32, 5, 1, 2),  # (16, 14, 14) -> (32, 14, 14)
                nn.ReLU(),
                nn.MaxPool2d(2),)  #(32, 14, 14) -> (32, 7, 7)
            self.out = nn.Linear(32 * 7 * 7, 10)  # 全连接层， 输入32 * 7 * 7个维度数据，输出10个类别之一
        
        def forward(self, x):
            x = self.conv1(x)
            x = self.conv2(x)
            x = x.view(x.size(0), -1)  # 展平，(batch, 32, 7, 7) -> (batch, 32 * 7 * 7)
            x = self.out(x)
            return x
    cnn = CNN()

    optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)  # Adam优化器
    loss_func = nn.CrossEntropyLoss()
    for epoch in range(EPOCH):
        for step, (b_x, b_y) in enumerate(train_loader):
            outcome = cnn(b_x)
            print(outcome)
            loss = loss_func(outcome, b_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            test_output = cnn(test_x)
            pred_y = torch.max(test_output, 1)[1].data.squeeze()
            accuracy = sum(pred_y == test_y) / float(test_y.size(0))
            print('迭代次数：', epoch, '|批次：', step, '|误差损失：%.4f' % loss.data.numpy(), '|正确率：%.2f' % accuracy)
    print("完成！")

# 普通循环神经网络RNN无法回忆长期记忆，容易梯度消失或梯度爆炸，因此改进产生LSTMRNN，见test_10()和test_11()
def test_10():  # 长短期记忆循环神经网络LSTMRNN（分类）图像识别，将图片按行识别，记忆是之前行的延续）
    EPOCH = 1
    BATCH_SIZE = 64
    TIME_STEP = 28  # 延续的时间记忆数，列宽
    INPUT_SIZE = 28  # 每一行的输入数，行宽
    LR = 0.001
    # 假设你已经下载了数据，并保存在当前目录下的mnist.pkl.gz文件中
    train_data = torchvision.datasets.MNIST(
        root='C:\\Users\\Gove2\\Desktop\\', 
        train=True, 
        transform=torchvision.transforms.ToTensor(), 
        download=False)
    train_loader = Data.DataLoader(
        dataset=train_data, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=2,
        drop_last=False)
    
    test_data = torchvision.datasets.MNIST(root='C:\\Users\\Gove2\\Desktop\\', train=False)
    test_x = Variable(test_data.data).type(torch.FloatTensor)[:2000]/255.  
    test_y = test_data.targets.squeeze()[:2000]
    # 
    class RNN(nn.Module):
        def __init__(self):
            super(RNN, self).__init__()
            self.rnn = nn.LSTM(  # 如果是pack_padded_sequence，inputs应该为PackedSequence类型
                input_size=INPUT_SIZE,  # 输入的特征数
                hidden_size=64,  # 隐藏层的特征数
                num_layers=1,  # 层数
                batch_first=True,)  # 输入数据的格式，
            self.out = nn.Linear(64, 10)

        def forward(self, x):
            r_out, (h_state_n, h_state_c) = self.rnn(x, None)
            out = self.out(r_out[:, -1, :])
            return out
    rnn = RNN()

    optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)
    loss_func = nn.CrossEntropyLoss()
    for epoch in range(EPOCH):
        for step, (x, y) in enumerate(train_loader):
            b_x = Variable(x.view(-1, 28, 28))
            b_y = Variable(y)
            outcome = rnn(b_x)
            loss = loss_func(outcome, b_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            test_output = rnn(test_x)
            pred_y = torch.max(test_output, 1)[1].data.squeeze()
            accuracy = sum(pred_y == test_y) / float(test_y.size(0))
            print('迭代次数：', epoch, '|批次：', step, '|误差损失：%.4f' % loss.data.numpy(), '|正确率：%.2f' % accuracy)
    print("完成！")

def test_11(Dynamic = False):  # 长短期记忆循环神经网络LSTMRNN（拟合） # 动态dynamic
    TIME_STEP = 10  # 时间步的大小
    INPUT_SIZE = 1  # 输入特征数
    LR = 0.02

    class RNN(nn.Module):
        def __init__(self):
            super(RNN, self).__init__()
            self.rnn = nn.RNN(  # 在这里rnn足矣，不需要lstm
                input_size=INPUT_SIZE,
                hidden_size=32,
                num_layers=1,
                batch_first=True,)
            self.out = nn.Linear(32, 1)
        
        def forward(self, x, h_state):  # x=(batch_size, time_step, input_size)，h_state: RNN的隐藏状态
            r_out, h_state = self.rnn(x, h_state)  #r_out: RNN包含每个时间步的输出，h_state: 更新后的隐藏状态
            outs = []  # 创建一个空列表，用于存储每个时间步的输出
            for time_step in range(r_out.size(1)):
                outs.append(self.out(r_out[:, time_step, :]))  # 循环遍历每个时间步，将每个时间步的输出添加到outs列表中
            return torch.stack(outs, dim=1), h_state  # 将outs列表转换为张量，并返回
    rnn = RNN()

    optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)
    loss_func = nn.MSELoss()
    h_state = None  # 初始化隐藏状态
    for step in range(60):
        # 生成数据
        start, end = step * np.pi, (step+1)*np.pi
        if Dynamic:  # 动态步数
            dynamic_steps = np.random.randint(1, 4)  # 动态步数
            steps = np.linspace(start, end, 10 * dynamic_steps, dtype=np.float32)
        else:  # 固定步数
            steps = np.linspace(start, end, TIME_STEP, dtype=np.float32)
        print(len(steps))  # 输出步数：动态or固定
        x_np = np.sin(steps)
        y_np = np.cos(steps)
        x = torch.from_numpy(x_np[np.newaxis, :, np.newaxis])  # np.newaxis会插入一个新的轴（即增加一个维度）
        y = torch.from_numpy(y_np[np.newaxis, :, np.newaxis])
        # 标准步骤
        prediction, h_state = rnn(x, h_state)
        h_state = h_state.data
        loss = loss_func(prediction, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # 绘图
        plt.plot(steps, prediction.data.numpy().flatten(), 'b-')  # flatten()将多维数组拉直成一维数组以供绘图使用
        plt.plot(steps, y_np.flatten(), 'r-')
        plt.draw()  # 在之前的图像基础上绘制新的图像
        plt.pause(0.05)
    plt.show()
    print("完成！")

#（源数据->编码器编码->解码器解码->目标数据）
def test_12():  # 自编码AutoEncoder(无监督:数据压缩成特征，特征向量再压缩成数据，并与最初的数据比较以衡量效果)
    EPOCH = 10
    BATCH_SIZE = 64
    LR = 0.005
    N_TESt_IMG = 10  # 绘图演示的图片数量
    # 假设你已经下载了数据，并保存在'C:\\Users\\Gove2\\Desktop\\'中
    train_data = torchvision.datasets.MNIST(
        root='C:\\Users\\Gove2\\Desktop\\', 
        train=True, 
        transform=torchvision.transforms.ToTensor(), 
        download=False)
    train_loader = Data.DataLoader(
        dataset=train_data, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=2,)
    
    class AutoEncoder(nn.Module):
        def __init__(self):
            super(AutoEncoder, self).__init__()
            self.encoder = nn.Sequential(  # 编码
                nn.Linear(28*28, 128),  # 784->128
                nn.Tanh(),
                nn.Tanh(),
                nn.Linear(128, 64),   # 128->64
                nn.Tanh(),
                nn.Linear(64, 12),  # 64->12
                nn.Tanh(),)
            self.decoder = nn.Sequential(  # 解码
                nn.Linear(12, 64),  # 12->64
                nn.Tanh(),
                nn.Linear(64, 128),  # 64->128
                nn.Tanh(),
                nn.Linear(128, 28*28),  # 128->784
                nn.Sigmoid(),)

        def forward(self, x):
            encoded = self.encoder(x)
            decoded = self.decoder(encoded)
            return encoded, decoded
    autoencoder = AutoEncoder()
    # 创建一个2行N_TESt_IMG列的子图，每个子图(10, 2)英寸
    f, a = plt.subplots(2, N_TESt_IMG, figsize=(10, 2))  # f为整个画布，a为子图列表
    plt.subplots_adjust(wspace=0.1, hspace=0.1)  # 调整子图之间的间距
    plt.ion()
    plt.show()
    view_data = train_data.train_data[:N_TESt_IMG].view(-1, 28*28).type(torch.FloatTensor)/255.  # 绘图演示所需图片数据
    for i in range(N_TESt_IMG):
        a[0][i].imshow(np.reshape(view_data.data.numpy()[i], (28, 28)), cmap='gray')  # 将view_data转换为28*28的灰度图
        a[0][i].set_xticks(())  # 隐藏x轴的刻度
        a[0][i].set_yticks(())  # 隐藏y轴的刻度

    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=LR)
    loss_func = nn.MSELoss()
    for epoch in range(EPOCH):
        for step, (x, y) in enumerate(train_loader):
            b_x = Variable(x.view(-1, 28*28))
            b_y = Variable(x.view(-1, 28*28))

            encoded, decoded = autoencoder(b_x)
            loss = loss_func(decoded, b_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            print('Epoch: ', epoch, '| step: ', step, '| train loss: %.4f' % loss.data.numpy())
            _, decoded_data = autoencoder(view_data)  # 获取演示数据解码后的数据,用于绘制以作对比，下划线_返回将被忽略
            for i in range(N_TESt_IMG):
                a[1][i].clear()
                a[1][i].imshow(np.reshape(decoded_data.data.numpy()[i], (28, 28)), cmap='gray')
                a[1][i].set_xticks(())
                a[1][i].set_yticks(())
            plt.draw()
            plt.pause(0.01)
            plt.ioff()
            plt.show()
    print("完成！")

import gymnasium as gym    # gym库用于开发和比较强化学习算法。它提供了各种环境:经典的控制问题、模拟机器人任务以及 Atari 游戏
# 传统神经网络难以承担大量状态和动作的问题
def test_13():  # 强化学习DQN(深度Q网络)获取环境状态，基于奖励训练智能体（以训练小车平衡杆为例）
    BATCH_SIZE = 32  # 每次训练的样本数量  
    LR = 0.01
    EPSILON = 0.9  # 贪婪策略的概率,越高越倾向于探索，越低越倾向于利用已知的最佳策略
    GAMMA = 0.9  # 折扣因子，越高越重视长期收益（训练也越慢），越低越重视短期收益（效果也越差）
    TARGET_REPLACE_ITER = 100  # 目标网络的更新频率
    MEMORY_CAPACITY = 2000  # 经验池容量（循环队列）存储代理与环境交互的历史数据（状态、动作、奖励、下一个状态）
    N_ACTIONS = env.action_space.n  # 动作数量
    N_STATES = env.observation_space.shape[0]  # 状态数量
    ENV_A_SHAPE = 0 if isinstance(env.action_space.sample(), int) else env.action_space.sample().shape  # 判断动作空间是离散还是连续

    class Net(nn.Module):  # 普通神经网络结构
        def __init__(self, ):
            super(Net, self).__init__()
            self.fc1 = nn.Linear(N_STATES, 50)
            self.fc1.weight.data.normal_(0, 0.1)  # 权重初始化随机生成
            self.out = nn.Linear(50, N_ACTIONS)
            self.out.weight.data.normal_(0, 0.1)  # 权重初始化随机生成

        def forward(self, x):
            x = self.fc1(x)
            x = F.relu(x)
            actions_value = self.out(x)
            return actions_value
        
    class DQN(object):
        def __init__(self):
            self.eval_net, self.target_net = Net(), Net()  # 创建评估最优动作的网络（评估网络，目标网络）
            self.learn_step_counter = 0  # 记录学习次数，定期（TARGET_REPLACE_ITER）更新目标网络参数
            self.memory_counter = 0  # 记录记忆池容量
            self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))
            self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
            self.loss_func = nn.MSELoss()

        def choose_action(self, x):
            x = torch.unsqueeze(torch.FloatTensor(torch.from_numpy(x)), 0)  # 然后转换并增加维度
            if np.random.uniform() < EPSILON:  # 选取最优动作（不探索）
                actions_value = self.eval_net.forward(x)
                action = torch.max(actions_value.squeeze(), 0)[1].data.numpy()
                action = action if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)
            else:  # 随机选取动作（探索）
                action = np.random.randint(0, N_ACTIONS)
                action = action if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)
            return action
        
        def store_transition(self, s, a, r, s_):
            transition = np.hstack((s, [a, r], s_))  # 状态、动作、奖励、下一个状态
            index = self.memory_counter % MEMORY_CAPACITY  # 获取索引，当记忆超过容量则覆盖旧的记忆
            self.memory[index, :] = transition  # 存入记忆池
            self.memory_counter += 1  # 更新记忆池容量
        
        def learn(self):
            if self.learn_step_counter % TARGET_REPLACE_ITER == 0:  # 定期更新目标网络参数
                self.target_net.load_state_dict(self.eval_net.state_dict())
            self.learn_step_counter += 1
            # 经验回放，从MEMORY_CAPACITY个经验中随机选择BATCH_SIZE个
            sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)  # 随机抽样
            batch_memory = self.memory[sample_index, :]  # 批量记忆提取
            # 提取记忆中的信息
            b_s = torch.FloatTensor(batch_memory[:, :N_STATES])  # 状态
            b_a = torch.LongTensor(batch_memory[:, N_STATES:N_STATES+1].astype(int))  # 动作
            b_r = torch.FloatTensor(batch_memory[:, N_STATES+1:N_STATES+2])  # 奖励
            b_s_ = torch.FloatTensor(batch_memory[:, -N_STATES:])  # 下一个状态

            q_eval = self.eval_net(b_s).gather(1, b_a)  # 记忆中的动作b_a在当前评估网络中的Q值
            q_next = self.target_net(b_s_).detach()  # 下一状态在目标网络中的Q值，datach()不进行梯度传递更新target_net()
            q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)  # 计算目标Q值，基于记忆中的奖励和下一步最优动作
            loss = self.loss_func(q_eval, q_target)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
    dqn = DQN()
    
    env = gym.make('CartPole-v1', render_mode='human')  # 创建环境，CartPole-v1为小车平衡杆环境，render_mode='human'为可视化
    env = env.unwrapped  # 取消环境限制
    for i_episode in range(250):
        print(i_episode/250)  # 显示进度
        s, info = env.reset()  # 重置环境，s为环境状态，此处包括小车位置、小车速度、木棒角度、木棒角速度
        while True:
            env.render()   # 渲染
            a = dqn.choose_action(s)  # 获取动作
            s_, r, done, _, info = env.step(a)  # 执行动作，获取反馈（特别是新的环境状态）
            # 自定义奖励信号（原本的奖励算法太慢）
            x, x_v, theta, theta_v = s_  # 小车位置、小车速度、木棒角度、木棒角速度
            r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8  # 小车偏离中心情况奖励
            r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5  # 木棒偏离竖直情况奖励
            r = r1 + r2  # 奖励信号叠加

            dqn.store_transition(s, a, r, s_)  # 保存记忆

            if dqn.memory_counter > MEMORY_CAPACITY:  # 当经验池满了，就进行学习
                dqn.learn()
            if done:  # 判断是否完成（失败，这里指偏离过大）
                break
            s = s_  # 更新状态

# 生成器的目标是生成能够欺骗判别器的样本，即让判别器认为生成的样本是真实的
# 判别器的目标是区分真实数据和生成器生成的假数据
def test_14():  # GAN(生成对抗网络)（以新手画家和新手鉴赏家为例）
    BATCH_SIZE = 64  # 批处理大小
    LR_G = 0.0001  # 生成器学习率
    LR_D = 0.0001  # 判别器学习率
    N_IDEAS = 5  # 噪声/创意维度（每个创意将由多个点组成）
    ART_COMPLEXITY = 10  # 艺术作品维度（每个艺术作品将由多少个点构成）
    PAINT_POINTS = np.vstack([np.linspace(-1, 1, ART_COMPLEXITY) for _ in range(BATCH_SIZE)])  # 艺术作品维度

    def artist_works():  # 大画师作品(测试用例)
        a = np.random.uniform(1, 2, size=BATCH_SIZE)[:, np.newaxis]
        paintings = a * np.power(PAINT_POINTS, 2) + (a-1)  # 大画师作品在一个范围内
        paintings = torch.from_numpy(paintings).float()
        return paintings
    # 生成器，N_IDEAS->ART_COMPLEXITY
    G = nn.Sequential(
        nn.Linear(N_IDEAS, 128),
        nn.ReLU(),
        nn.Linear(128, ART_COMPLEXITY))
    # 判别器,ART_COMPLEXITY->1
    D = nn.Sequential(
        nn.Linear(ART_COMPLEXITY, 128),
        nn.ReLU(),
        nn.Linear(128, 1),
        nn.Sigmoid())  # 输出概率
    
    plt.ion()
    opt_G = torch.optim.Adam(G.parameters(), lr=LR_G)
    opt_D = torch.optim.Adam(D.parameters(), lr=LR_D)
    for step in range(10000):
        print(step/10000)  # 进度
        artist_paintings = artist_works()  # 大画师作品
        G_ideas = torch.rand(BATCH_SIZE, N_IDEAS, requires_grad=True)  # 噪声/创意
        G_paintings = G(G_ideas)  # 根据噪声/创意生成艺术作品
        # 生成器G的目标是最大化判别器认为生成作品是真实作品的概率，最小化-log(D(G(z)))或者最大化log(1-D(G(z)))
        prob_artist1 = D(G_paintings)
        G_loss = torch.mean(torch.log(1.0-prob_artist1))  # 取对数是为了有效区分（0，1）之间的量，求平均是为了反应整个批次
        opt_G.zero_grad()
        G_loss.backward()
        opt_G.step()
        # 判别器D的目的是尽可能准确地区分真实样本和生成样本
        prob_artist0 = D(artist_paintings)  # 大画师作品的鉴别情况/概率
        prob_artist1 = D(G_paintings.detach())  # 生成器作品的鉴别情况/概率
        D_loss = - torch.mean(torch.log(prob_artist0) + torch.log(1. - prob_artist1))  # 适当设计的损失函数
        opt_D.zero_grad()
        D_loss.backward(retain_graph=True)  # retain_graph：保留计算图
        opt_D.step()

        plt.cla()
        plt.plot(PAINT_POINTS[0], G_paintings.data.numpy()[0], c='#4AD631', lw=3, label='Generated painting',)
        plt.plot(PAINT_POINTS[0], 2 * np.power(PAINT_POINTS[0], 2) + 1, c='#74BCFF', lw=3, label='upper bound')  # 大画师作品上界
        plt.plot(PAINT_POINTS[0], 1 * np.power(PAINT_POINTS[0], 2) + 0, c='#FF9359', lw=3, label='lower bound')  # 大画师作品下界
        plt.text(-0.5, 2.3, 'D accuracy=%.2f (converge=0.5)' % prob_artist0.data.numpy().mean(), fontdict={'size': 13})
        plt.text(-0.5, 2, 'D score= %.2f (converge=-1.38)' % -D_loss.data.numpy(), fontdict={'size': 13})
        plt.ylim((0, 3))
        plt.legend(loc='upper right', fontsize=10)
        plt.draw()
        plt.pause(0.01)
    plt.ioff()
    plt.show()

    print("完成!")

def test_15():  # dropout方法缓解过拟合
    N_SAMPLES = 20
    N_HIDDEN = 300

    x = torch.unsqueeze(torch.linspace(-1, 1, N_SAMPLES), 1)
    y = x + 0.3 * torch.normal(torch.zeros(N_SAMPLES, 1), torch.ones(N_SAMPLES, 1))
    x, y = Variable(x), Variable(y)
    test_x = torch.unsqueeze(torch.linspace(-1, 1, N_SAMPLES), 1)
    test_y = test_x + 0.3 * 0.3 * torch.normal(torch.zeros(N_SAMPLES, 1), torch.ones(N_SAMPLES, 1))
    test_x, test_y = Variable(test_x), Variable(test_y)

    net_overfitting = nn.Sequential(
        nn.Linear(1, N_HIDDEN),
        nn.ReLU(),
        nn.Linear(N_HIDDEN, N_HIDDEN),
        nn.ReLU(),
        nn.Linear(N_HIDDEN, N_HIDDEN),
        nn.ReLU(),
        nn.Linear(N_HIDDEN, 1))
    
    net_dropped = nn.Sequential(
        nn.Linear(1, N_HIDDEN),
        nn.Dropout(0.5),
        nn.ReLU(),
        # nn.Dropout(0.5),
        nn.Linear(N_HIDDEN, N_HIDDEN),
        nn.Dropout(0.5),
        nn.ReLU(),
        nn.Linear(N_HIDDEN, 1))
    
    optimizer_ofit = torch.optim.Adam(net_overfitting.parameters(), lr=0.01)
    optimizer_drop = torch.optim.Adam(net_dropped.parameters(), lr=0.01)
    loss_func = torch.nn.MSELoss()

    plt.ion()
    for t in range(500):
        pred_overfitting = net_overfitting(x)
        pred_drop = net_dropped(x)
        loss_overfitting = loss_func(pred_overfitting, y)
        loss_drop = loss_func(pred_drop, y)
        optimizer_ofit.zero_grad()
        optimizer_drop.zero_grad()
        loss_overfitting.backward()
        loss_drop.backward()
        optimizer_ofit.step()
        optimizer_drop.step()

        net_overfitting.eval()
        net_dropped.eval()
        test_pred_ofit = net_overfitting(test_x)
        test_pred_drop = net_dropped(test_x)
        plt.cla()
        plt.scatter(x.data.numpy(), y.data.numpy(), c='magenta', s=50, alpha=0.4, label='real')
        plt.scatter(test_x.data.numpy(), test_y.data.numpy(), c='cyan', s=50, alpha=0.4, label='test')
        plt.plot(test_x.data.numpy(), test_pred_ofit.data.numpy(), 'r-', lw=3, label='overfitting')
        plt.plot(test_x.data.numpy(), test_pred_drop.data.numpy(), 'b--', lw=3, label='dropout(50%)')
        plt.text(0, -1.2, 'overfitting loss={:.6f}'.format(loss_overfitting.data.numpy()), fontdict={'size': 15})
        plt.text(0, -1.5, 'dropout(0.5) loss={:.6f}'.format(loss_drop.data.numpy()), fontdict={'size': 15})
        plt.legend(loc='upper left')
        plt.ylim((-2.5, 2.5))
        plt.pause(0.1)
        net_overfitting.train()
        net_dropped.train()
    plt.ioff()
    plt.show()

    print("完成！")
    
# 当神经网络太深时有（100，999，...）->tanh->（~1，~1，...）
# 训练深层神经网络十分困难，特别是在较短时间内收敛，批量归一化可以加速深层网络收敛
from torch.nn import init
def test_16():  # BN批标准化（Batch Normalization）
    N_SAMPLES = 2000  # 样本数量
    BATCH_SIZE = 64  # 批大小
    EPOCH = 24  # 训练轮数
    N_HIDDEN = 8  # 隐藏层数量
    LR = 0.03  # 学习率
    ACTIATION = F.tanh  # 激活函数
    B_INIT = -0.2  # 偏置初始值
    
    x = np.linspace(-7, 10, N_SAMPLES)[:, np.newaxis]  # 创建样本集数据
    noise = np.random.normal(0, 2, x.shape)
    y = np.square(x) -5 + noise
    train_x = torch.from_numpy(x).float()
    train_y = torch.from_numpy(y).float()
    train_dataset = Data.TensorDataset(train_x, train_y)
    train_loader = Data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_x = np.linspace(-7, 10, 200)[:, np.newaxis]  # 创建测试集数据
    noise = np.random.normal(0, 2, test_x.shape)
    test_y = np.square(test_x) -5 + noise
    test_x = torch.from_numpy(test_x).float()
    test_y = torch.from_numpy(test_y).float()

    class Net(nn.Module):
        def __init__(self, batch_normalization=False):
            super(Net, self).__init__()
            self.do_bn = batch_normalization  # 是否使用批标准化
            self.fcs = []  # 创建隐藏层列表
            self.bns = []  # 创建批标准化层列表
            self.bn_input = nn.BatchNorm1d(1, momentum=0.5)  # 创建批标准化层的输入层
            for i in range(N_HIDDEN):  # 添加隐藏层（和批标准化层）
                input_size = 1 if i == 0 else 10  # 如果是第一层则输入数量为1，否则为隐藏层神经元数量
                fc = nn.Linear(input_size, 10)  # 创建隐藏层
                setattr(self, 'fc%i' % i, fc)  # 动态地为实例添加属性fc_i
                self._set_init(fc)  # 调用下面的_set_init()设置隐藏层初始化权重
                self.fcs.append(fc)  # 将隐藏层添加到隐藏层列表中
                # 如果使用批标准化将添加批标准化层
                if self.do_bn:
                    bn = nn.BatchNorm1d(10, momentum=0.5)  # 创建批标准化层
                    setattr(self, 'bn%i' % i, bn)  # 动态地为实例添加属性bn_i
                    self.bns.append(bn)  # 将批标准化层添加到批标准化层列表中
            self.predict = nn.Linear(10, 1)  # 创建输出层
            self._set_init(self.predict)  # 调用下面的_set_init()设置输出层初始化权重
        
        def _set_init(self, layer):  # 设置layer层的初始化权重
            init.normal_(layer.weight, mean=0.0, std=0.1)  # 正态分布初始化（均值为0，标准差B_INIT为0.1）
            init.constant_(layer.bias, B_INIT)  # 常量初始化（所有偏置项都将被初始化为B_INIT）
            return layer

        def forward(self, x):  # 激活函数前是否进行批标准化
            pre_activation = [x]  # 批标准化【前】输出列表
            if self.do_bn:  # 如果使用批标准化则对输入进行批标准化
                x=self.bn_input(x)
            layer_input = [x]  # 批标准化【后】输出列表
            for i in range(N_HIDDEN):  # 遍历隐藏层
                x = self.fcs[i](x)  # x通过隐藏层
                pre_activation.append(x)  # 将输出添加到批标准化【前】输出列表
                if self.do_bn:  # 如果使用批标准化则对当前隐藏层输出进行批标准化
                    x = self.bns[i](x)  # x通过当前批标准化层
                x = ACTIATION(x)  # x通过激活函数
                layer_input.append(x)  # 将输出添加到批标准化【后】输出列表
            out = self.predict(x)  # 输出层
            return out, pre_activation, layer_input  # 返回输出层，激活函数前输出列表，激活函数前输出列表

    f, axs = plt.subplots(4, N_HIDDEN + 1, figsize=(10, 5))  # 创建子图,f为画布，axs为子图集
    plt.ion()
    plt.show()

    def plot_histogram(l_in, l_in_bn, pre_ac, pre_ac_bn):  # 绘制直方图
        '''
        l_in: 通常代表输入到某一层之前的原始数据。
        l_in_bn: 代表经过批标准化处理后的输入数据。
        pre_ac: 表示通过了某一层但还未经过激活函数的数据。
        pre_ac_bn: 指经过批标准化后，但在激活函数处理前的数据
        '''
        # ax_pa:批标准化前激活函数前输出,ax_pa_bn:批标准化后激活函数前输出,ax:批标准化前激活函数后输出,ax_bn:批标准化后激活函数后输出
        for i, (ax_pa, ax_pa_bn, ax, ax_bn) in enumerate(zip(axs[0, :], axs[1, :], axs[2, :], axs[3, :])):  # 遍历子图集
            [a.clear() for a in [ax_pa, ax_pa_bn, ax, ax_bn]]  # 清空子图
            if i == 0:  # 如果是第一层,有更大的范围显示
                p_range = (-7, 10)
                the_range = (-7, 10)
            else:
                p_range = (-4, 4)
                the_range = (-1, 1)  
            ax_pa.set_title('L' + str(i))  # 添加子图标题
            # hist(x一维数据数组，bins直方图柱状区间数量，range直方图显示范围，color颜色，alpha透明度)
            ax_pa.hist(pre_ac[i].data.numpy().ravel(), bins=10, range=p_range, color='#FF9359', alpha=0.5)
            ax_pa_bn.hist(pre_ac_bn[i].data.numpy().ravel(), bins=10, range=p_range, color='#74BCFF', alpha=0.5)
            ax.hist(l_in[i].data.numpy().ravel(), bins=10, range=the_range, color='#FF9359')
            ax_bn.hist(l_in_bn[i].data.numpy().ravel(), bins=10, range=the_range, color='#74BCFF')
            for a in [ax_pa, ax, ax_pa_bn, ax_bn]:  # 删除子图x轴和y轴刻度
                a.set_xticks(())
                a.set_yticks(())
            ax_pa_bn.set_xticks(p_range)  # 添加子图x轴刻度
            ax_bn.set_xticks(the_range)  # 添加子图x轴刻度
            axs[0, 0].set_ylabel('PreAct')  # 添加子图y轴标签
            axs[1, 0].set_ylabel('BN PreAct')
            axs[2, 0].set_ylabel('Act')
            axs[3, 0].set_ylabel('BN Act')
            plt.pause(0.01)
    # 创建网络实例（是/否批标准化）、优化器、损失函数、损失列表（用于绘制）
    nets = [Net(batch_normalization=False), Net(batch_normalization=True)]
    opts = [torch.optim.Adam(net.parameters(), lr=LR) for net in nets]
    loss_func = torch.nn.MSELoss()
    losses = [[], []]
    for epoch in range(EPOCH):
        print('Epoch:', epoch)
        layer_inputs, pre_acts = [], []  # 创建隐藏层输入列表，激活函数前输出列表（均用于绘制）
        for net, l in zip(nets, losses):
            net.eval()  # 设置网络为评估模式，Dropout 和 Batch Normalization情况下
            pred, pre_act, layer_input = net(test_x)
            l.append(loss_func(pred, test_y).data.item())
            layer_inputs.append(layer_input)
            pre_acts.append(pre_act)
            net.train()  # 设置网络为训练模式
        plot_histogram(*layer_inputs, *pre_acts)  # *将列表解包后提取参数
        # 训练
        for step, (b_x, b_y) in enumerate(train_loader):
            for net, opt in zip(nets, opts):
                pred, _, _ = net(b_x)
                loss = loss_func(pred, b_y)
                opt.zero_grad()
                loss.backward()
                opt.step()
    # 绘制Original和Batch Normalization的loss
    plt.ioff()
    plt.figure(2)
    plt.plot(losses[0], c='#FF9359', lw=3, label='Original')
    plt.plot(losses[1], c='#74BCFF', lw=3, label='Batch Normalization')
    plt.xlabel('step')
    plt.ylabel('test loss')
    plt.ylim((0, 2000))
    plt.legend(loc='best')
    # 绘制Original和Batch Normalization和test的数据
    [net.eval() for net in nets]
    preds = [net(test_x)[0] for net in nets]
    plt.figure(3)
    plt.plot(test_x.data.numpy(), preds[0].data.numpy(), c='#FF9359', lw=4, label='Original')
    plt.plot(test_x.data.numpy(), preds[1].data.numpy(), c='#74BCFF', lw=4, label='Batch Normalization')
    plt.scatter(test_x.data.numpy(), test_y.data.numpy(), c='r', s=50, alpha=0.2, label='test')
    plt.legend(loc='best')
    plt.show()

    print("完成！")

if __name__ == '__main__':  # 防止多线程错误
    while True:
        i = input("请输入操作：\n"
                  "1~16:执行对应测试\n"
                  "h:查看帮助\n"
                  "0:退出\n")
        if i == 'h':
            print("1: torch和numpy\n"
                  "2: tensor和Variable\n"
                  "3: 激励函数\n"
                  "4: 简单的神经网络（拟合）\n"
                  "5: 简单的神经网络（分类）\n"
                  "6: 保存和加载\n"
                  "7: 数据批处理\n"
                  "8: 更多的优化器\n"
                  "9: 卷积神经网络CNN\n"
                  "10: 长短期记忆循环神经网络LSTMRNN（分类）\n"
                  "11: 长短期记忆循环神经网络LSTMRNN（拟合）\n"
                  "12: 自编码AutoEncoder\n"
                  "13: 强化学习DQN\n"
                  "14: 生成对抗网络GAN\n"
                  "15: dropout缓解过拟合\n"
                  "16: BN批标准化")
        if i == '0':
            break
        if i in [str(j) for j in range(1, 17)]:
            globals()['test_' + i]()

# GPU加速
'''
GPU加速：没有NVIDIA显卡，无法使用GPU加速
使用方法:XX.cuda() 将XX放到GPU上运行
使用方法:XX.cpu() 将XX放到CPU上运行
XX可以是数据（原始，测试...）、模型、函数
模型和数据要放在同一个设备上，否则会报错
'''
