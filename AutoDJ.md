自动DJ混音系统
一、项目概述
项目名称： AutoDJ — 基于音乐信息检索的自动混音工具
一句话描述： 输入两首歌，程序自动分析BPM和调性，找到最佳切入点，输出一段平滑过渡的混音。

二、核心技术原理
项目涉及三个关键技术模块：
1. BPM检测（节拍匹配）

用 librosa 库分析音频的节拍帧，提取每首歌的每分钟节拍数（BPM）
用 phase vocoder（相位声码器） 对第二首歌做时间拉伸，使其BPM与第一首对齐
这是最核心的论文技术点，有大量学术文献支撑

2. 调性检测 + Camelot Wheel（和声混音）

用 librosa 检测每首歌的音乐调性（如 C大调、A小调），这是和声混音的关键——DJs通过在调性兼容的歌曲之间切换来创造更流畅的混合 StemSplit
Camelot Wheel 是一个帮助DJs进行和声混音的图表工具，灵感来源于音乐理论中的"五度圈"。DJs使用它来识别哪些曲目可以基于调性兼容性混合在一起 DJ.Studio
Camelot系统用字母数字代码标记每个调性：如8A = C小调，8B = 降E大调；相邻编号（7A/8A/9A）的歌曲是和谐兼容的 Mixed In Key
程序自动判断两首歌的Camelot编号，决定是否可以直接混、或需要变调

3. 交叉淡化（Crossfade）过渡

检测第一首歌的段落边界（通常在第8或16小节的末尾）作为切出点
在切出点叠加第二首歌，做5-10秒的音量交叉淡化
可选：用 pydub 加EQ处理，模拟DJ低频切换技巧


三、技术栈
工具用途librosaBPM检测、调性分析、音频特征提取pydub音频剪辑、音量淡化、格式转换numpy / scipy信号处理、时间拉伸matplotlib可视化波形、tempogramPython 3.x主语言

四、项目功能流程
输入: 歌曲A.mp3 + 歌曲B.mp3
        ↓
[分析模块]
  ├── 检测 BPM_A, BPM_B
  ├── 检测 Key_A, Key_B → 转换为Camelot编号
  └── 判断和声兼容性
        ↓
[处理模块]
  ├── 时间拉伸 B 使 BPM_B → BPM_A
  ├── 找最佳切出点（A的段落尾部）
  └── 交叉淡化混合
        ↓
输出: mixed_output.mp3 + 分析报告

五、论文结构建议（对应作业要求）

Introduction — DJ混音的背景，自动化的意义
Technology — MIR技术原理（BPM检测、调性分析、Camelot系统）
Implementation — 你怎么用Python实现的，每个模块的选择
Results & Reflection — 效果怎么样，哪些情况好/不好，局限性
References — 见下方


六、推荐学术参考文献（满足4篇要求）

Müller, M. (2015). Fundamentals of Music Processing. Springer. — 音乐信息检索教科书，BPM/调性检测理论基础
Böck, S., Krebs, F., & Widmer, G. (2014). "A Multi-Model Approach to Beat Tracking." ISMIR 2014. — 节拍追踪的学术论文
Librosa: McFee, B. et al. (2015). "librosa: Audio and Music Signal Analysis in Python." Proceedings of SciPy. — librosa官方学术论文，权威可引
Ishizaki, H. et al. (2009). "Full-Automatic DJ Mixing System with Optimal Tempo Adjustment." ISMIR 2009. — 这篇论文专门研究全自动DJ混音系统和最优节奏调整，发表于第10届国际音乐信息检索会议，和你的项目最直接相关 GitHub