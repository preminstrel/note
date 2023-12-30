---
counter: True
comment: True
---

!!! abstract 
    CMU 11-785 IDL Homework, Project 和评价。

    Course Website：http://deeplearning.cs.cmu.edu

    11-785 的是 Bhiksha Raj 讲的，很多我之前以为自己掌握了，但是实际上一知半解的东西，在这节课上都得到了答案。

## 课程评价

11-785 的 workload 算比较大的，除了每周的 Quiz 、course project，每次的 Homework 都分为两个部分，第一个部分是从零开始实现自己的 MyTorch，Bonus 部分还包括了 Auto Gradient 的实现，有点像 Tianqi Chen 讲的 DLSys 的 needle，和 needle 的区别是不用自己写 numpy CPU backend 和 CUDA backend；第二个部分是打 Kaggle，和全班同学一起刷点数打榜。虽然这门课的 workload 很大，但是不得不说，这门课的 infra 相当完善，可以和著名的 CSAPP 媲美，有 25 个 TA、写得很好的 write-up、每周的 recitation、Hackthon等，Piazza 上的响应速度也是超级快。

从我自身的感受出发，我觉得这门课最大的价值是把很多东西讲清楚的同时，用比较好的 Homework 来巩固 coding 能力，每周的 Quiz 来巩固理论知识。说实话，在上这门课之前，我做 DL 的一些 Project 很多时候都是直接依葫芦画瓢，并不清楚这一行代码背后真正在运行什么（只知道这行代码是用来干什么的），也对深度学习框架只有一个比较模糊的认知（本课的 HW Part1 完美解决了这个问题），这门课绝对配得上是 CMU 的神课之一。正如 Bhiksha 在第一节课说的，如果这门课拿了 A，说明你已经初步具备了在 DL 工业界干活的能力了。

!!! comment
    Workload: ★★★★☆

    Reward: ★★★★☆

    Recommendation: ★★★★★


## Homework
这门课想要拿高分确实不难，因为没有考试，作业给的时间也比较充足，所以大家分数基本都很高。由于 Bhiksha 是在 LTI 学院做 Speech 的，所以这门课好像除了第二个 Homework 是 Face Classification & Verification，其他 Homework 都是关于 Speech Recognition 的（相比之下，ECE 学院的 Intro to DL 大部分的 Homework 和 Project 都是关于 Computer Vision 的）。

关于workload，这门课我感觉主要是前面花的时间稍微多一些，最后一个月我几乎没有花什么时间在上面。看起来 write-up 很吓人，但是其实做起来真的还好。 每个 Homework 的 Part1 都可以一天内完成（不过我记得 Bonus 中第一个 Autograd 我 Debug 花的时间有点长）。虽然 Part2 每个都给了快一个月的训练时间，但是我感觉只要模型结构设计得还行，适当加点 regularization 和其他的 trick 基本都可以达到 high cutoff 拿满分。反而我是觉得每周的 Quiz 还挺折磨的，每周都得花点时间去做一下。不过需要注意一下计算资源的使用情况，如果天天跑的话，400 刀的 GCP 和 150 刀的 AWS credits 远远不够。

### HW1P1

HW1P1 (Introduction to MyTorch, Autograd) 包括了 MyTorch 和 Autograd 两个部分，其中 Autograd 是 Bonus 部分。MyTorch 的实现基本上就是把 PyTorch 的 API 用 Numpy 实现一遍，包括 MLP、 一些常见的激活函数、常见的 loss、BatchNorm、SGD、AdamW、Dropout 等等。这部分的代码量不算很大，但是需要注意的是，这部分的代码需要写得很规范，因为后面的 HW 都是基于这部分的代码来实现的，所以需要保证这部分的代码没有 bug。Autograd 算是一个小难点，主要的工作量是去理解他的 framework，了解他是怎么写 automatic differentitation 和 autograd engine 的，下面算是一个最简单的 skeleton：

```python
class Operation:
    def __init__(self, inputs, output, gradients_to_update, backward_operation):
        self.inputs = inputs
        self.output = output
        self.gradients_to_update = gradients_to_update
        self.backward_operation = backward_operation


class Autograd:
    def __init__(self):
        if getattr(self.__class__, "_has_instance", False):
            raise RuntimeError("Cannot create more than 1 Autograd instance")
        self.__class__._has_instance = True

        self.gradient_buffer = GradientBuffer()
        self.operation_list = []

    def __del__(self):
        del self.gradient_buffer
        del self.operation_list
        self.__class__._has_instance = False

    def add_operation(self, inputs, output, gradients_to_update, backward_operation):
        if len(inputs) != len(gradients_to_update):
            raise Exception(
                "Number of inputs must match the number of gradients to update!"
            )
        for input in inputs:
            self.gradient_buffer.add_spot(input)
        self.operation_list.append(
            Operation(inputs, output, gradients_to_update, backward_operation)
        )

    def backward(self, divergence):
        for operation in reversed(self.operation_list):
            if operation == self.operation_list[-1]:
                grad_front = divergence
            else:
                grad_front = self.gradient_buffer.get_param(operation.output)
            grad_ret = operation.backward_operation(grad_front, *operation.inputs)

            for i in range(len(operation.inputs)):
                if operation.gradients_to_update[i] is None:
                    self.gradient_buffer.update_param(operation.inputs[i], grad_ret[i])
                else:
                    operation.gradients_to_update[i] += grad_ret[i]

    def zero_grad(self):
        self.gradient_buffer.clear()
        self.operation_list = []

```

### HW1P2
HW1P2 (Frame Level Classification of Speech) 就是简单用 MLP 搭积木，尝试各种网络的 structure，activations 和一些 tricks。因为当时 AWS 的 credit 还没到账，我就直接用了 GCP server 去做。我发现 slack 的 Client API 还是挺好用的，可以直接用来 log training 的结果实时发到 slack 上，这样就不用每次都去看 log 了。

```python
Asuka = WebClient(token="...")
try:
    response = Asuka.chat_postMessage(channel='#training', 
        text=f"Training Finished! best_precision: {best_precision} at epoch {best_epoch}. 
        The best result is submitted to Kaggle."
    )
    response = Asuka.files_upload(
        channels='#training',
        file=img_filename,
        title="Training Plot",
    )
    assert response["ok"]
except SlackApiError as e:
    print(f"Error: {e.response['error']}")
```

最后 submit 到 Autolab 的时候还需要提交 ablation study 的一些 wandb log 等结果，再写一个简单的 README 就可以了。Autolab 会用 MOSS 检测代码的相似度，所以往年的代码最好不要参考。最后 Part2 的结果是分两个部分的，一个是截止前可以看见的 public leaderboard，另一个是截止后的 private leaderboard；相当于可以去拟合的第二个 valid set 和 test 吧，需要两个都达到 high cutoff 才能拿满分。

### HW2P1
HW2P1 (MyTorch: Convolutional Neural Networks) 是手撸 CNNs 的，包括 Conv、Pooling、和 resampling layers 等的 forward/backward，最后还有个小 toy training task 可以稍微玩一下，这里面很重要的一个思想是 “Scanning MLPs == CNNs”。相比 MLP，CNNs 的 backpropagation 写法是可以让我更加感受到 gradient 是怎么传播的，写完后对反向传播的理解会更加深刻。

在 Bonus 里面，会让我们完成 Dropout2D 和 BatchNorm2D 的实现，这里面需要注意的是，Dropout2D 的实现和 Dropout1D 的实现是不一样的，Dropout2D 的实现需要保证每个 channel 的 dropout mask 是一样的，而 Dropout1D 的实现则不需要。最后还需要自己 implement 一下 ResNet；不过这次的 Autograd 确实比上次简单多了，可能是因为框架已经熟了，而且这次的 Autograd 的内容也和本身的 HW2P1 部分有 overlap。

### HW2P2

HW2P2 (Face Classification & Verification) 是我觉得花时间最长的一个 HWxP2了，因为他会有两个 Kaggle 比赛要去打，包括了 classification 和 verification。由于不能用 pretrained 权重，得 training from scratch，所以刚开始学习率一不小心就会让梯度爆炸，导致 loss 变成 nan 了，所以一开始只能用比较小的 learning rate去做。刚开始训练也会遇到准确率一直很低，loss 下降也比较慢的情况，但是训练了十几个 epoch 之后会有一个类似于 turning point 一样的位置，在这个位置之后模型似乎找到正确的优化方向，收敛快了很多。还有一个比较重要的点是，由于计算资源的限制，所以用 ResNet 去做这个任务是不太现实的，因为 FLOPs 太大了，训练相对慢很多。最后我选了 ConvNeXt，利用了他的 group convolution 的特性，可以用比较少的 FLOPs 去实现比较好的效果。

model 在分类的任务上训练得到了一个比较好的 backbone 之后就可以开始进行 verification 的 task，实际上就是加了一些额外的 loss，让他在 latent space 上 distance 更加优化，比如 ArcFace 之类的。

### HW3P1
HW3P1 的名字特别有意思，叫 “RNNs, GRUs and Search, Oh My!”。相比前两次的 HWxP1，这次要难一些，尤其是 search 部分。这次的 RNNs 和 GRUs 的部分主要是实现了 RNNs、GRUs 和 CTC loss 的 forward 和 backward。我记得多层 RNNs 的 backward 会比较复杂，还是需要停下来想一下的。而 GRUs 的 backward 则需要大量动手计算偏导数，也算是另外一种折磨。

search 部分中，greedy search 比较简单，beam search 算法算是比较难 debug 的了，关键是 write-up 刚开始还写错了，导致我花了很长时间，不过最后写出来还是挺有成就感的；最后在 Piazza 上问了 TA 才知道是题目有问题。

### HW3P2
HW3P2 (Utterance to Phoneme Mapping) 也是一个 speech 的 Kaggle 比赛，主要是用来让我们熟悉如何 set up GRU/LSTM based models 的，包括了 RNNs、GRUs、LSTMs、CTC loss等等。这次的 Kaggle 比赛的数据集比较大，所以训练时间也比较长，同时也用了一些外部的 package 来计算 CTC loss。这是我第一次处理不同长度的 mfcc 数据，在处理数据的时候经常需要 pack/unpack 来输入到 RNNs 里面，这个过程也是比较容易出错的。同时，这次的 model 还需要用 CNN 作为一个 feature extractor 去抓一个窗口的 mfcc，然后再输入到 RNNs。

### HW4P1
HW4P1 (Language Modeling using RNNs) 应该算是最杂的 HWxP1 了，从 attention 的 implementation 到整个的 training 都要求不低。最后还需要用 OpenAI 的 API 去测试一下 ppl，当做分数的一部分，另外一部分要在 Autolab 里面刷点数。不过我感觉这门课很多基本只需要结构设计的还可以，可以 overfit training set 之后加点 regularization 就可以达到要求了。我做的时候都没注意到 Autolab 的部分居然还限制了 submission 的次数，还是之后在 Piazza 上看到有人complain 才知道的。

### HW4P2

最后一个 HW4P2 (Attention-based Speech Recognition) 我觉得其实和 Part 1 很像，做的事情都几乎差不多。不过这次的 write-up 也有坑的地方：在 write-up 里面的结果是先建议我们用 LSTM/Transformer encoder 再在中间插入一个 CNNs 来抽取特征，但是实际上这样做的话，训练的时候收敛性并不是很好，所以我最后还是用了 CNN encoder + LSTM + Transformer 作为 encoder 的结构。这次的 training 也花了很长时间，因为这次的数据集巨大，train 一次得要几天，感觉对没有 GPU 的同学来说有点难度。

Write-up 的最后一个 section 也比较 touching:

!!! comment
    This homework is a true litmus test for your understanding of concepts in Deep Learning. It is not easy to implement, and the competition is going to be tough. But we guarantee you, that if you manage to complete this homework by yourself, you can confidently claim that you are now a Deep Learning Specialist!

    Good luck and enjoy the challenge!

## Project

我不是很喜欢这门课的 Project，给的时间不够充足，也不提供额外的计算资源，而且还是小组作业（这可能是罪魁祸首）。让我感觉有点在浪费时间生产所谓的科研垃圾，还不如 enroll 11-685 版本去做 HW5——implement a LLM from scratch。也听说之前有人靠着 Project 发了顶会，不过实际体验下来我感觉这部分的 support 做得还不够好。

虽然最后分数也还可以，不过体验不是很好，所以我也不想多说了，也许对完全没有接触过 Deep Learning Research 的同学来说还算有意思吧。

## Final Words

最后，我想引用一下 Bhiksha 在 Piazza 上发的一封邮件，感谢他和他的 TAs 。我们做的一切；在这门课上，我学到了很多，也很幸运这是我在 CMU 的第一门课。

!!! comment
    Dear All,

    Our semester has finally ended, the holidays are here, and a new year is upon us.

    I would like to thank all of you for all of the effort you have put into the course.  I know that the course was enormously tiring for many of you.  Yet, you kept with it, some of you through illness and loss,  while also managing the rest of your equally demanding academic schedules.  Thank you!

    It is my hope that you will eventually find that it was worth it.  That a few years from now, with full hindsight,  you will look back and think it was time and effort that was well spent.

    I would also like to thank my amazing TAs, without whom the semester would not be possible. The sheer number of hours they put in each day could not ever be compensated in any manner.  Their work began in May, before the previous semester ended, and continued till yesterday, and they have had little rest in the interim.

    Over the term, they have designed and tested a dozen homeworks, held or recorded dozens of recitations, several bootcamps, a dozen hackathons, conducted over 600 office hours, fielded over 25000 posts on Piazza with an average response time of 5 minutes, designed, verified and setup weekly quizzes, tracked study groups, mentored project teams, shadowed lectures, and ensured everything went off on time, all the while also managing their own equally demanding coursework.

    I am not entirely sure *why* they do it, given that they get paid little to nothing for what they do. Their dedication is inspiring even to me. Thank you, guys!

    And finally, a thank you to all the unseen staff who kept the clock ticking -- the mediatech folks who broadcast the lectures, my admin Jennifer who ensured that all the organization and scheduling was in order, and the folks at Eberly and the helpdesk who help us setup various things.

    I wish you all a very happy holiday season and a wonderful new year.  May it be all good.

    -Bhiksha