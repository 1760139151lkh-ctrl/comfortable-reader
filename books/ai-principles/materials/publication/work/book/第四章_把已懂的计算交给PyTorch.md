<a id="fourth-question"></a>
# 第四章　把已经懂的计算交给 PyTorch

第三章那张四点表，只用了九个参数，我们还能逐项写出导数。即使如此，代码也不得不自己记住每个中间活动、每条影响路径和每次平均了几张卡片。模型一大，靠人反复誊写这些关系既费力又容易把某个轴、符号或更新时刻弄错。我们想把重复劳动交给数值库，但有个先决条件：**交出去的仍须是第三章同一个计算。**若标签、初值、损失的平均方式或更新幅度变了，即使新程序也给出四个正确类别，两边也没有完成这次交接。

本章沿第三章种子 1 的**九个原始初值**重新出发，四张异或卡片、两处 \(\tanh\) 中间活动、末端半平方平均损失和第一步步长 0.2 都不换。先让数组接走逐张循环的算数，再让 PyTorch 的 Tensor 接走数组算数，再让自动求导接走手写链式导数；模型对象、优化器、数据批次和存盘恢复各接走另一件具体工作。每走一步，都能指出输入、目标、参数、活动、导数和真正改动参数的那一行。

<a id="fourth-history"></a>
## 从能手算链式法则，到能反复使用的工具

<a id="history-1989"></a>
1986 年的多层学习研究已经给出了正向与反向关系，实际任务随后又把问题推大了。1989 年 LeCun 等人把反向传播网络用于美国邮政编码的手写数字，强调把图像任务里的先验限制放进网络结构；1998 年 LeCun、Bottou、Bengio 与 Haffner 讨论的不只是一张数字图，还包括字符切分、识别与语言约束怎样组成文档系统，并报告了投入使用的支票读取系统。更多模块共同训练时，每一处局部计算都要把对最终目标的影响接下去。这里的历史作用是告诉我们为什么**可重复的导数与数组计算**成为实际需要，而不是用1998年的真实识别成绩替本章四张卡片背书。([LeCun 等，1989，原刊摘要](https://direct.mit.edu/neco/article/1/4/541/5515/Backpropagation-Applied-to-Handwritten-Zip-Code)；[LeCun 等，1998，摘要与引言](https://gwern.net/doc/ai/nn/cnn/1998-lecun.pdf))

<a id="history-2005"></a>
在软件这边，Python 的数值数组有自己的来路。NumPy 在 2005 年由 Numeric 与 Numarray 的前期工作发展而来：一整块同类型数值可以作为数组接受矩阵运算，不必为每个乘法都让 Python 逐个处理对象。它让本章第一个交接有了现成载体，却不负责替人求导。2010 年 Theano 的论文把另一层工作接上：研究者先在 Python 里组成符号表达式，Theano 沿表达式关系求导，再把计算编译给 CPU 或 GPU。它的示例还分得清声明变量、构造目标、编译函数和执行训练；当时的办法并非一句“库自动会了”。([NumPy 官方项目介绍](https://numpy.org/about/)；[Bergstra 等，2010，摘要及示例](https://papers.baulab.info/papers/also/Bergstra-2010.pdf))

<a id="history-2011"></a>
工具并没有沿唯一一条直线换代。2011 年的 Torch7 把张量、神经网络部件与 CPU/GPU 数值运算组织在 Lua 环境中；2015 年 Chainer 明确采用“边运行边定义”的方式，在一次实际前向执行中记录计算关系，使普通 Python 条件和循环也能参与建图。2019 年 PyTorch 论文说明它要同时保留 Python 的即时执行、自动求导和加速器运算。这里既有沿用，也有不同取舍：我们不必先宣告整张静态计算表，第三章的前向程序实际执行了哪些 Tensor 操作，本轮自动求导就记录哪些关系。PyTorch 官方文档也明确说每轮会重新建立这张图。不能从软件发表的先后次序推断任何两个项目之间的单向影响；眼前要检验的是本机这套接口究竟做了什么。([Torch7，2011，摘要与§§1—3](https://ronan.collobert.com/pub/matos/matos/2011_torch7_nipsw.pdf)；[Chainer Define-by-Run 官方说明](https://docs.chainer.org/en/stable/guides/define_by_run.html)及[项目所列 2015 年论文](https://github.com/chainer/chainer)；[PyTorch 2019 论文，摘要及§§1—2](https://papers.nips.cc/paper/2019/file/bdbca288fee7f92f2bfa9f7012727740-Paper.pdf)；[PyTorch 2.11 自动求导机制](https://docs.pytorch.org/docs/2.11/notes/autograd.html))

<a id="fourth-arrays"></a>
## 四张卡片一起算，轴各指什么

第三章一张卡片时，两个中间单元分别算 \(z_j=a_{j1}x_1+a_{j2}x_2+c_j\)，活动 \(h_j=\tanh z_j\)，末端给 \(s=b+v_1h_1+v_2h_2\)。把四张输入按行排成 \(X\)，四个目标排成 \(Y\)；把两处输入权重排成 \(A\) 的两行，两个隐藏偏置排成 \(c\)，末端权重排成 \(v\)。同一个关系可写为
\[
Z=XA^\top+c,\qquad H=\tanh Z,\qquad s=Hv+b,\qquad
L=\frac14\sum_{i=1}^{4}\frac12(s_i-Y_i)^2.
\]
转置 \(A^\top\) 让 \(X\) 的每一行同时与两个隐藏单元的权重相遇。\(c\) 只有两个数，却要在**每张卡片**的两个隐藏位置各加一次；\(b\) 只有一个数，在四个末端分数上各加一次。下表是这些数的形状，也是它们的任务：

| 数 | 形状 | 每个轴在数什么 |
|---|:---:|---|
| \(X\)、\(Y\) | \(4\times2\)、\(4\) | 四张卡片；每张有两项输入、一个目标 |
| \(A\)、\(c\) | \(2\times2\)、\(2\) | 两个隐藏单元；每个接两项输入、一个偏置 |
| \(Z\)、\(H\) | \(4\times2\) | 每张卡片在每个隐藏单元里的分数与活动 |
| \(v\)、\(b\) | \(2\)、标量 | 两项隐藏活动怎样合成一个末端分数 |
| \(s\)、\(L\) | \(4\)、标量 | 每张的末端分数；四张半平方误差的平均 |

程序开头用 import numpy as np 取得 NumPy；np.tanh 就是请它把同一个函数逐项用于数组。矩阵乘写作 @，.T 取转置；代码的小写 x、y、a 分别承接上面的 X、Y、A，只是书写习惯不同。于是正向计算几乎就是上面四个等号：

~~~python
z = x @ a.T + c
h = np.tanh(z)
scores = h @ v + b
residual = scores - y
loss = 0.5 * np.mean(residual ** 2)
~~~

NumPy 替我们按形状把 \(c\) 与 \(b\) 用在多张卡片上；这个按缺少的轴或长度为 1 的轴扩展的规则叫**广播**。广播只决定算术怎样排开，目标该与哪个分数配对仍由我们负责。导数也是第三章四张逐项导数的并排版本：若 \(E=s-Y\)，则 \(D_{ij}=E_i v_j(1-H_{ij}^2)\)，其中 \(i\) 数卡片、\(j\) 数隐藏单元。于是
\[
\frac{\partial L}{\partial A}=\frac{D^\top X}{4},\qquad
\frac{\partial L}{\partial c}=\operatorname{mean}_{i}D,\qquad
\frac{\partial L}{\partial v}=\frac{H^\top E}{4},\qquad
\frac{\partial L}{\partial b}=\operatorname{mean}_{i}E.
\]
每个“平均”都沿**四张卡片**，不把两个隐藏单元平均掉。代码中的 E[:, None] 把四项残差排成 \(4\times1\)，v[None, :] 把两项末端权重排成 \(1\times2\)，它们相乘才逐张、逐单元得到 \(4\times2\) 的影响。NumPy没有决定用半平方目标，也没有替我们写出这些导数；它接过的是数值数组和按轴执行的重复乘加。

<a id="fourth-tensor"></a>
## Tensor 先只接算数，导数仍由我们负责

现在把同一批数放进 PyTorch 的 Tensor。它是能保存多维数的对象，除了形状，还记录数值格式和所在设备；此处统一用 CPU 和 float64。我们暂时不设置 requires_grad——这个标志表示是否要求库追踪该数对最终结果的导数。前向式和 \(D\) 的手写关系逐项照搬，查看得到的 Tensor 形状与数值。[一步交接程序](../code/torch_handoff_one_step.py)实际做了这段中间实验，因此“换成 Tensor”不会被误说成“库已经反传”。此时损失 Tensor 的 requires_grad 为 false，导数仍是我们写出的 \(D^\top X/4\) 等式。

<a id="fourth-handoff-result"></a>
这份程序从[第三章真实运行记录](../results/two_hidden_units_run.json)读出种子 1 的四张输入、九项初值、初始损失和第一步参数，而不是重新抽一组碰巧容易比较的数。旧 Python 的起点损失是约 0.587636；NumPy、Tensor 手写、Tensor 自动求导以及稍后的模型对象加 SGD，九项梯度相对旧 Python 的最大绝对差不超过约 \(1.8\times10^{-17}\)，一步后的九项参数在这台电脑上逐项相同。这检验的是**同一对象的一步移交**；训练很多步或换设备仍需另说。

<a id="fourth-autograd"></a>
## 前向留下关系，反向才把导数算回参数

第三章手写反向时，必须记住 \(h=\tanh(XA^\top+c)\) 以及它沿哪些操作抵达损失。PyTorch 的自动求导做的是这件重复劳动：把四个待训练的 Tensor 设为 requires_grad=True，再执行矩阵乘、加偏置、tanh、末端合成和平均损失。实际执行的运算形成一张计算关系图；损失的 grad_fn 是回到这张图的入口。调用 loss.backward()，它沿图使用各基本运算的导数，把九项结果**累积到叶参数的 .grad**。输入卡片和目标没有请求训练导数，库也不替我们选择标签与平方目标。([PyTorch 2.11 自动求导机制](https://docs.pytorch.org/docs/2.11/notes/autograd.html))

这段程序把“算导数”和“改参数”放在两处：

~~~python
loss = loss_now()
loss.backward()                 # 把本次导数累积进 a.grad、c.grad、v.grad、b.grad
with torch.no_grad():
    a -= 0.2 * a.grad            # 此处才真正改变参数
    c -= 0.2 * c.grad
    v -= 0.2 * v.grad
    b -= 0.2 * b.grad
~~~

更新包在 no_grad 中，是因为这四次原地改数属于训练程序的操作，我们并不要求把“本轮改权”再接到下一轮的反向图里。新一轮前向会按那时的参数重新建图。如果做两次新的前向并各调用一次 backward()，中间又没有清 .grad，第二次读到的是两次导数之和；本机对同一状态的核对恰为第一次的两倍。因此每轮先清掉上轮的导数贡献，才符合“一批卡片的一次平均梯度”的含义。调用 backward() 本身不会改参数，调用 no_grad() 也不等于已经求导。([PyTorch 2.11 自动求导和 no_grad 说明](https://docs.pytorch.org/docs/2.11/notes/autograd.html))

<a id="fourth-module"></a>
## 把会改的数交给模型对象，再交给优化器

四个孤立 Tensor 能做一步对照；反复训练还要让程序知道哪些数构成模型。我们把同一前向式放进继承 nn.Module 的 TinyMLP，把 \(A,c,v,b\) 各自包装为 nn.Parameter 后赋给模型属性。Parameter 是特殊的 Tensor：赋作 Module 属性就会进入 model.parameters()、named_parameters() 和 model.state_dict()。普通 Tensor 即便 requires_grad=True，若只作对象属性，也不会被自动登记为待优化参数。本机用一个普通缓存和一个 Parameter 作了对照，登记表里只出现后者。([PyTorch 2.11 Parameter 说明](https://docs.pytorch.org/docs/2.11/generated/torch.nn.parameter.Parameter.html))

实际的模型类保留的是刚才同一公式：

~~~python
class TinyMLP(nn.Module):
    def __init__(self, initial, dtype=torch.float64, device="cpu"):
        super().__init__()
        self.a = nn.Parameter(torch.tensor(initial["a"], dtype=dtype, device=device))
        self.c = nn.Parameter(torch.tensor(initial["c"], dtype=dtype, device=device))
        self.v = nn.Parameter(torch.tensor(initial["v"], dtype=dtype, device=device))
        self.b = nn.Parameter(torch.tensor(initial["b"], dtype=dtype, device=device))

    def forward(self, x):
        hidden = torch.tanh(x @ self.a.T + self.c)
        return hidden @ self.v + self.b
~~~

class 给这类对象命名；self 指向眼前这个模型实例。构造模型时，__init__ 先执行，super().__init__() 建立 Module 管理参数所需的部分，随后四个 Parameter 才作为属性登记。调用 model(x) 时会运行 forward，把这一批输入送过同一隐藏层和末端。这里没有另加 PyTorch 默认网络层，也没有换掉第三章的 \(\tanh\) 与平方目标。

这时把注册的四组参数交给无动量、无权重衰减的 SGD，步长仍为 0.2：

~~~python
optimizer = torch.optim.SGD(model.parameters(), lr=0.2, momentum=0.0)
optimizer.zero_grad(set_to_none=True)
scores = model(x)
loss = 0.5 * (scores - y).square().mean()
loss.backward()
optimizer.step()
~~~

zero_grad(set_to_none=True) 使优化器管理的上轮 .grad 重新为空；backward() 按已执行的前向关系累计本轮导数；step() 才按这些导数和步长修改注册参数。这个 SGD 设置与第三章的一步 \(v\leftarrow v-0.2\nabla L\) 相同，所以一步结果可以逐项比较。今后若加动量或权重衰减，优化器会有额外状态或目标变化，不能继续用这条一步相等声称算法未变。([PyTorch 2.11 SGD](https://docs.pytorch.org/docs/2.11/generated/torch.optim.SGD.html) 与 [zero_grad](https://docs.pytorch.org/docs/2.11/generated/torch.optim.Optimizer.zero_grad.html))

<a id="fourth-shape"></a>
## 一个很像真结果的错误形状

模型给一批 \(B\) 张卡片的分数形状应是 \((B,)\)，目标也应是 \((B,)\)。若两者有一个误留为 \((B,1)\)，另一个仍是一维时，相减在 PyTorch 的广播规则下可能成为 \((B,B)\)：每张预测会错配上这一批里的**每一个**目标。此时再求 mean，照样得到一个能下降的标量，却已经换了训练题目。程序在每次计算损失前核对分数和目标形状完全相同。([PyTorch 2.11 广播规则](https://docs.pytorch.org/docs/2.11/notes/broadcasting.html))

这不是纯粹假想的难以察觉错误。本机起点四项正确配对的损失约 0.587636；故意让分数成为 \((4,1)\)、目标保留 \((4,)\)，广播后变成 \(4\times4\)，错误损失却约 0.588123。两个标量很接近，单看“损失没有离谱”可能漏掉问题；检查每个轴在数什么，才能发现它把四个标签互相交叉了。完整的形状与数值记录在[一步运行结果](../results/torch_handoff_one_step.json)。

<a id="fourth-batches"></a>
## 多个小批次，已经是另一个更新过程

一次用四张资料，梯度在同一旧参数处把四项相加除以 4。真实训练常把更多资料按批送入。TensorDataset 让第 \(i\) 张输入与第 \(i\) 个目标一起取出；我们还把卡片编号放进去，方便核对前后是不是读了同一批：

~~~python
ids = torch.arange(len(y))
dataset = TensorDataset(x, y, ids)
loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)
for xb, yb, row_ids in loader:
    assert xb.shape == (2, 2) and yb.shape == (2,)
    optimizer.zero_grad(set_to_none=True)
    scores = model(xb)
    loss = 0.5 * (scores - yb).square().mean()
    loss.backward()
    optimizer.step()
~~~

每轮给我们两批，每批 \(X\) 是 \(2\times2\)、\(Y\) 是长度 2。xb、yb、row_ids 分别接住当前批次的输入、目标和原卡片编号；for 循环取完两批便结束一轮。每批先清上一次的导数，算当前批损失，再求导和改参数。shuffle=False 让顺序始终是卡片 0、1，然后 2、3；num_workers=0 使这个小实验没有额外的取数进程。([PyTorch 2.11 数据载入说明](https://docs.pytorch.org/docs/2.11/data.html))

第一批求完平均梯度就更新，第二批遇到的已经是新参数。因此，哪怕仍叫“同一模型、同一四张卡片”，一轮两个小批次也不会与一次四张完整批次逐项相等。本章后半段把这个变化明说出来，再用步长 0.05、动量 0.9 训练 200 轮。这两个设置现在决定了下一步还依赖什么：数据顺序决定下一批何时到来，动量让优化器记住此前的更新方向。

“记住方向”可以写成这次代码实际使用的数。这里的 \(t\) 数的是已经完成的**批次更新**，不是轮数；把九项参数和下一批的平均梯度各按同一顺序排成向量 \(\theta_t\) 与 \(g_{t+1}\)，优化器还保存一组同样长的数 \(m_t\)，称为**动量缓冲**。这里没有权重衰减，也没有 Nesterov 变体，PyTorch 的 SGD 在已有缓冲时按

\[
m_{t+1}=0.9m_t+g_{t+1},\qquad
\theta_{t+1}=\theta_t-0.05m_{t+1}
\]

更新；第一次根本还没有缓冲时，它先用当次梯度作 \(m_1\)。因此同样的参数和同一批卡片，即使算出同样的 \(g\)，只要先前的 \(m\) 不同，下一步参数也会不同。前面的无动量一步核对相当于只用本批的 \(g\)；它与这里的多批过程不是同一个更新规则。([PyTorch 2.11 SGD 官方更新式及首步缓冲说明](https://docs.pytorch.org/docs/2.11/generated/torch.optim.SGD.html))

<a id="fourth-resume"></a>
## 关掉训练，再从确切位置接上

[保存恢复程序](../code/torch_training_resume.py)一条命令会依次启动新的 Python 进程：一条从头连续跑 200 轮；另一条先跑 50 轮，在整轮结束处保存模型 state_dict、优化器 state_dict 和轮次，退出，再由新进程读回继续 150 轮；最后再做一条只读模型权重、不读动量的对照。读取使用 weights_only=True、map_location=cpu，随后分别调用模型和优化器的 load_state_dict。保留轮次与固定资料顺序，是为了知道下一批到底应当是哪两张。([PyTorch 2.11 torch.load](https://docs.pytorch.org/docs/2.11/generated/torch.load.html) 与 [官方保存恢复指南](https://docs.pytorch.org/tutorials/beginner/saving_loading_models))

关键文件操作只有几行：

~~~python
torch.save({
    "epoch": 50,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
}, CHECKPOINT)

saved = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
model.load_state_dict(saved["model_state_dict"])
optimizer.load_state_dict(saved["optimizer_state_dict"])
~~~

state_dict 是一份有名字的当前状态；torch.save 把它写进文件，torch.load 从文件读回。CHECKPOINT 在完整程序开头指向工作区里的检查点文件。读回数据不会自行造出模型，程序先构造同一 TinyMLP 和 SGD，再把两份状态分别装进去。优化器的状态里此时有动量缓冲；只读模型那条对照恰好删掉了这部分历史。实际检查点还保存轮次和资料身份，[文件](../results/tiny_mlp_epoch50.pt)可由本机重新运行生成。

在项目根目录的 PowerShell 终端，先运行前面的一步交接，再运行这条完整恢复实验：

~~~powershell
$env:PYTHONUTF8='1'
python work/code/torch_handoff_one_step.py
python work/code/torch_training_resume.py --mode all
~~~

<a id="fourth-resume-result"></a>
本机这次连续训练共有 400 批；前 50 轮的 100 批与恢复后的 300 批拼起来，批次编号、卡片顺序、每批更新前损失和最终参数与不停机路线**逐项相同**。第 50 轮结束，也就是第 100 批更新后，存下的 \(m_{100}\) 长度约 0.248，完整读回后仍是这个数。只读权重时，恢复的第一批在更新前损失仍相同，因为当前权重和输入都一样。此时两条路线的梯度 \(g_{101}\) 也相同；按上式，完整恢复会在它之外加上 \(0.9m_{100}\)，只读权重则把本批梯度当作新缓冲。第一步之后两组参数的距离应为 \(0.05\times0.9\|m_{100}\|\)，本机约为 0.01116；程序分别实际走完这一个批次后测得的距离与公式值相差不到 \(10^{-12}\)。于是第二批更新前损失已相差约 0.00244，最终参数最大差约 0.0225。两条路线训练到 200 轮都能把这四点分对；这个对照证明的是**过程状态不同**，不是“动量保证泛化更好”。[完整恢复记录](../results/torch_training_resume.json)保留了数值与运行范围。

这份严格接续只在 CPU float64、固定四张资料、固定两张一批、无随机层、无多工人、**第 50 轮末**保存的条件下成立。若在半批中断，或打乱顺序，还得知道采样器走到哪里以及随机状态；跨版本或跨设备也不能根据这里的完全相同推断逐位复现。程序的入口同时能从头在这台电脑重跑这些分开的进程，不靠旧书稿里的报告数字。

<a id="fourth-device"></a>
## 数字存在哪儿，和用什么格式存

本机是 Python 3.13.14、NumPy 2.4.6、PyTorch 2.11.0+cu128；CUDA 可用，设备为 RTX 5090 Laptop GPU。前面严格比较故意用 CPU float64，让算式差异先于设备与舍入差异被查清。float32 用 32 位保存一个浮点数，float64 用 64 位，后者通常保留更多有效数字，也占更多空间。Tensor 的 dtype 指这种数值格式，device 指数存在哪个设备并由谁执行运算；模型参数、输入、目标要放在相容的设备和格式上，运算才是我们宣称的那个运算。把模型搬到 GPU，而输入仍留在 CPU，通常根本不能完成这次矩阵乘。

<a id="fourth-device-result"></a>
一步程序随后在 CPU float32、GPU float64、GPU float32 各跑一次相同前向和一步更新。它们的形状和目标都没有变，数值却不要求逐位相等；本机 GPU float32 相对 CPU float64 的九项梯度最大差约 \(2.0\times10^{-8}\)，CPU/GPU float64 的差小得多。这些数只核这九参数网络的一步，不是性能基准，也不能用于估算以后真实语言模型的显存或耗时。[本次设备与格式结果](../results/torch_handoff_one_step.json)留在后台，读者此刻应记住的是：同一数学式仍要指定数的表示和执行位置。

模型进入评价模式与停止记录导数也是两件事。调用 model.eval() 会通知有些模块改用评价行为；我们这张网只有加权和与 tanh，所以它的分数不因此改变，仍可在需要时求导。在 torch.no_grad() 内前向则不留下这次反向所需的关系，本机两种前向的分数相同，requires_grad 却不同。以后遇到 dropout 或批规范化，eval() 可能连数值行为也改变，不能把这里的数值相同推广过去。([PyTorch 2.11 对 eval 与 no_grad 的区分](https://docs.pytorch.org/docs/2.11/notes/autograd.html))

<a id="fourth-close"></a>
## 这一章真正交出去的劳动

同一四张卡片让交接可以逐处指认：NumPy 与 Tensor 接过按轴的数值计算；autograd 根据实际前向运算接过链式求导；Module 与 Parameter 让参数被发现、保存；优化器按已算出的梯度执行更新并可能保留动量；DataLoader 决定每一批是哪几条；存盘和恢复要把模型、优化器及训练位置重新接在一起。它们仍没有替我们选标签、平方目标或未见过的检验数据。我们也没有用四张全参加训练的异或卡片证明一般图像能力。

四点例子到此退出主文。接下来若要说一个模型对未见输入知道多少、犯错意味着什么，先得有一种比“对或错”更完整的表达不确定性的方法。下一章会从有限结果表建立概率、条件与期望；等这些量有了实际意思，再讨论训练成绩如何与新资料上的表现联系起来。

<a id="fourth-exercises"></a>
## 自己动手核对

1. 两张一批时，模型分数形状为 \((2,)\)，目标若误存成 \((2,1)\)，相减会得到什么形状？说出其中一个本不该出现的“预测对错标签”配对，并指出为何只看最后一个平均损失容易漏掉它。
2. 假设某叶参数本轮应收导数 \(g\)。第一次 backward() 后未清 .grad，又对相同数值状态做一次新的前向和 backward()，.grad 会是什么？这期间参数在什么操作之前保持不变？
3. 从第 50 轮末存盘后，为什么只恢复模型权重的路线在首个新批次**更新前**还能与完整恢复同损失？用保存的动量长度约 0.248，先预测两路线走完这个批次后的参数距离，再与运行结果比较。若把数据改为打乱顺序，除权重和动量外还要考虑什么？
4. 在本章模型里，model.eval() 后的分数为什么仍可求导？torch.no_grad() 又改变了哪一件事？

**核对。**第一题会得到 \(2\times2\)：第一张的分数也会与第二张的目标相减，反之亦然；平均后形状错误被一个标量遮住。第二题新前向给相同的 \(g\)，两次累计为 \(2g\)；backward() 不改参数，要到手写的改权行或 optimizer.step() 才改。第三题第一批尚未改权，权重和输入一致，故损失相同；参数距离按本章设置为 \(0.05\times0.9\times0.248\)，约 0.0112，本机精确值约 0.01116。打乱顺序时还需保存或重建采样器位置、随机生成状态等，不能拿本章的固定顺序结论照搬。第四题 eval() 控制模块的训练/评价行为，当前模型没有会改变输出的这类模块；no_grad() 控制是否记录反向图，两者作用不同。
