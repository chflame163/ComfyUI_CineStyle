# ComfyUI CineStyle

ComfyUI_CineStyle 是一组为ComfyUI 视频工作流开发的，更易于使用的自定义节点。
本项目不能替代专业视频编辑软件。


### 安装插件
* 在CompyUI插件目录(例如“CompyUI\custom_nodes\”)中打开cmd窗口，键入    
```
git clone https://github.com/chflame163/ComfyUI_CineStyle.git
```


### 如何找到本节点组
* 在ComfyUI画布点击右键 - Add Node, 找到 "😺dzNodes/CineStyle"。    
<img src="images/node-menu.jpg" alt="CineStyle 节点菜单" width="480">

* 或者在ComfyUI画布双击, 在搜索框输入"cinestyle"。
<img src="images/node-search.jpg" alt="CineStyle 节点搜索" width="480">


## 更新说明

* 添加 [CS Video Segment (SAM3.1)](#cs-video-segment-sam31) 节点，在锚点帧用 Semantic、粗略 Mask、Point 或 BBox 定义对象，并自动传播 mask。
* 添加 [CS Video Segment (SeC-4B)](#cs-video-segment-sec-4b) 节点，使用 SeC-4B 的概念理解和 LongSAM2.1 记忆传播 mask。
* 添加 [CS SeC-4B Model Loader](#cs-sec-4b-model-loader) 节点，用于加载和复用 SeC-4B 推理模型。
* 添加 [CS Load Video](#cs-load-video) 节点，用于加载视频，支持出入点设置、更改尺寸、帧率等。
* 添加 [CS Save Video](#cs-save-video) 节点，支持可选 metadata 写入和 H.264 目标码率控制。

示例工作流及素材位于插件的 `workflows` 子目录。

## 节点说明

### CS Load Video
把视频文件加载到 ComfyUI，并提供一个可交互的时间线编辑窗口。节点执行时会读取视频帧、音频和帧率，根据工作流中保存的设置截取和调整内容，然后输出给下游节点。

- 从节点直接选择和上传视频。
- 点击节点上的`Edit Timeline`按钮进入时间线界面。
- 通过时间线拖动入点和出点，支持逐帧定位。
- 通过 `Set In` 和 `Set Out` 按钮，把视频预览窗口的当前帧快速设为入点或出点。
- 使用蓝色当前帧指针，在时间线上拖动即可同步预览对应视频帧。
- 出入点设置按钮组的 `Play` 只播放已设置的入点到出点范围。
- 视频预览窗口的白色播放键仍然播放完整视频，不受入出点限制。
- 开启保持宽高比后，修改宽度或高度会自动联动另一项。
- 输出尺寸可以自动四舍五入到指定倍数，适合视频模型常见的尺寸要求。
- 入点、出点、宽度、高度和 FPS 会保存到 ComfyUI 工作流中，重新打开工作流后仍可复现相同设置。
- 输出同时包含图像帧批次、音频和结构化视频信息。

#### 节点选项说明：
![CS Load Video 节点](images/CS_Load_Video_node.jpg)
- video： 选择 ComfyUI 输入目录中的视频，或使用上传控件上传视频文件。
- keep_aspect_ratio：布尔值，默认`true`。开启后，宽度和高度按照源视频的原始宽高比联动计算。 
- multiple：整数，默认 `32`。 输出尺寸的取整倍数。宽度和高度会按最近的倍数四舍五入。 
- start_frame： 整数，默认 `0`。 起始帧，使用从 `0` 开始的帧编号。
- end_frame：整数，默认`-1` |。 结束帧，`-1` 表示使用视频最后一帧。 
- width： 整数，默认`0`。 输出宽度。`0` 表示根据源视频尺寸和 `multiple` 自动计算。 
- height： 整数，默认`0`。 输出高度。`0` 表示根据源视频尺寸和 `multiple` 自动计算。 
- fps： 浮点数，默认 `0`。 输出帧率。`0` 表示保留源视频帧率；输入其他数值时会按目标帧率重新采样帧。 
- choose file to upload：点击按钮从本地加载视频。
- Edit Timeline：进入时间线界面。
其中 `start_frame`、`end_frame`、`width`、`height` 和 `fps` 可通过 `Edit Timeline` 窗口设置或在节点控件中直接编辑。

#### Edit Timeline 时间线界面

![Edit Timeline 时间线界面](images/CS_Load_Video_UI.jpg)

时间线界面从上到下依次包含视频预览、时间读数、当前帧指针、入出点标记栏、时间线操作按钮和输出参数。


##### 入点和出点标记

标记栏中的两个白色手柄分别表示入点和出点：

- 左侧手柄是入点。
- 右侧手柄是出点。
- 可以直接拖动手柄调整范围。
- 点击标记栏空白区域时，程序会根据点击位置距离哪个手柄更近来调整对应边界。

##### 时间线按钮
- `Set In`：将视频预览当前帧设为入点。如果当前帧晚于出点，会自动修正出点。
- `|<`： 跳转到上一帧。只受视频首帧限制，不受入点限制。 
- `Play`: 只播放从入点到出点的内容，播放到出点后自动暂停。再次播放时，如果当前帧不在范围内，会从入点重新开始。 
- `>|`: 跳转到下一帧。只受视频尾帧限制，不受出点限制。 
- `Set Out`： 将视频预览当前帧设为出点。如果当前帧早于入点，会自动修正入点。 

#### 输出说明
- video：标准 ComfyUI `VIDEO` 类型，包含时间线选段、尺寸、帧率和音频，可直接连接官方视频节点。
- IMAGE: 输出的video图像帧批次。
- frame_count: 实际输出帧数。修改 FPS 后，该数值可能与源视频选段帧数不同。 
- audio: 选定时间范围内的音频。没有音频轨道时输出为空。 
- video_info: 包含源视频和输出视频的 FPS、帧数、时长、宽高、入点和出点等信息。


### CS Save Video
基于 ComfyUI 官方 `Save Video` 节点，增加save metadata和H264 bitrate选项。

- `video`：标准 ComfyUI `VIDEO` 输入。
- `filename_prefix`：输出文件名前缀，支持官方的日期和节点控件格式化语法。
- `format`：输出容器格式，默认 `auto`。
- `codec`：视频编码方式，默认 `h264`。选择 H.264 时显示码率控件。
- `H.264 bitrate (Mbps)`：H.264 目标码率，浮点数保留 1 位小数，范围 `1.0–160.0 Mbps`，默认 `8.0 Mbps`。范围覆盖官方建议的低分辨率到 8K 高帧率视频；常见 1080p 视频可从 `8.0 Mbps` 开始，高帧率 1080p 可提高到约 `12.0 Mbps`。
- `save_metadata`：默认关闭。开启后保存的文件将写入工作流和源视频 metadata。


### CS SeC-4B Model Loader

加载SeC-4B 模型，为 `CS Video Segment (SeC-4B)` 节点和 Selector Preview 提供可复用的模型实例。节点扫描 `ComfyUI/models/sams/SeC-4B` 目录下的单文件权重。

![CS SeC-4B Model Loader 节点](images/CS_SeC-4B_Model_Loader_node.jpg)

#### 节点选项说明

- `model_file`：支持 `SeC-4B-bf16.safetensors` 和 `SeC-4B-fp16.safetensors`，节点按文件原生精度加载。
- `device`：运行设备，`auto` 自动选择 CUDA，`cpu` 使用 CPU，也可选择具体的 `gpu0`、`gpu1` 等设备。
- 选择 `cpu` 时，BF16/FP16 权重会自动转换为 `float32`。
- `use_flash_attn`：高级选项，默认开启。环境没有 FlashAttention 时会回退到标准注意力。
- `allow_mask_overlap`：高级选项，默认开启，允许多个对象的 Mask 重叠。
- `SEC_MODEL`：输出的 SeC-4B 模型连接到 Video Segment 节点的 `model`。

首次执行 Loader 或首次进行 SeC Preview 时，如果选择的权重文件不存在，节点会自动从 Hugging Face 下载对应的单文件权重到 `ComfyUI/models/sams/SeC-4B`。默认 BF16 文件约 7.35 GiB。自动下载需要当前环境可以访问 Hugging Face；失败时可手动下载：

[BF16 权重下载地址](https://huggingface.co/VeryAladeen/Sec-4B/resolve/main/SeC-4B-bf16.safetensors)

[FP16 权重下载地址](https://huggingface.co/VeryAladeen/Sec-4B/resolve/main/SeC-4B-fp16.safetensors)

将所选文件放置到：

`ComfyUI/models/sams/SeC-4B`

配置和 tokenizer 已随插件放在 `py/sec_model_config`，依赖声明位于项目根目录 `requirements.txt`，推理代码和 SAM2 配置位于 `py/sec_inference` 与 `py/sec_configs`。



### CS Video Segment (SeC-4B)
使用 OpenIXCLab SeC-4B 模型和内置 LongSAM2.1 视频记忆编码器，在锚点帧用多个 BBox、多个正负 Point 和粗略 Mask 定义对象，并传播到整段视频。

![CS Video Segment (SeC-4B) 节点](images/CS_Video_Segment(Sec-4B)_node.jpg)

#### 使用流程

1. 先执行 [CS SeC-4B Model Loader](#cs-sec-4b-model-loader)，再把输出的 `SEC_MODEL` 连接到节点的 `model`。
2. 将同一视频的 `IMAGE` 或 `VIDEO` 输出连接到节点。`images` 与 `video_input` 同时连接时，节点优先使用 `images`。
3. 点击 `Open Selector`，在锚点帧中定义一个或多个对象的提示。
4. 点击 `Preview Current Frame` 检查 SeC-4B 的当前帧分割结果，确认后点击 `Apply to Node`。
5. 执行节点，得到整段视频的 mask。默认情况下，节点执行结束会卸载 SeC-4B 的推理子模型以释放显存。

Selector 会递归查找上游的官方或第三方视频加载节点。对于运行后才生成的输入，先运行一次工作流建立缓存，再打开 Selector。

#### 节点选项说明

- `model`：`CS SeC-4B Model Loader` 输出的 `SEC_MODEL`，必需输入。
- `images`：可选 `IMAGE` 帧批次，连接后作为执行和 Selector 的首选视频来源。
- `video_input`：可选 `VIDEO` 输入，仅在 `images` 未连接时使用。
- `anchor_frame`：锚点帧在当前输入帧批次中的本地编号，从 `0` 开始，通常由 Selector 自动写入。
- `prompt_data`：Selector 序列化的多对象 Mask、BBox 和 Point 提示数据，不建议手动编辑。
- `tracking_direction`：传播方向，`bidirectional` 双向传播，`forward` 向后传播，`backward` 向前传播。
- `max_frames_to_track`：每个方向最多传播的帧数，`-1` 表示直到视频边界。
- `mllm_memory_size`：场景变化时用于提取对象概念的历史关键帧数量，默认 `12`。
- `offload_video_to_cpu`：将视频推理状态尽可能放到 CPU，以降低显存占用，但会降低速度。
- `auto_unload_model`：默认开启。节点执行结束后卸载 SeC-4B 子模型；下次执行或 Preview 时会自动重载。
- `Open Selector`：打开交互式提示编辑器。

#### 输出说明

- `mask`：形状为 `[帧数, 高, 宽]` 的整段视频 mask。
- `anchor_mask`：锚点帧的合并分割 mask。
- `video_info`：包含帧数、尺寸、锚点帧、传播方向和对象数量。

### CS Video Segment (SAM3.1)
使用 ComfyUI 官方 SAM3/SAM3.1 模型和推理内核，把 Selector 中定义在锚点帧的多个对象提示传播到整段视频。

SAM3.1 官方权重下载地址：[Comfy-Org/sam3.1](https://huggingface.co/Comfy-Org/sam3.1)。下载后放入 ComfyUI 的 `models/checkpoints`

![CS Video Segment (SAM3.1) 节点](images/CS_Video_Segment(SAM3.1)_node.jpg)

#### 使用流程

1. 使用官方 `CheckpointLoaderSimple`加载 SAM3/SAM3.1 模型，并连接节点的 `model`。
2. 将视频的 `IMAGE` 或 `VIDEO` 输出连接到节点。`images` 与 `video_input` 同时连接时，节点优先使用 `images`。
3. 点击节点上的 `Open Selector`，在实际输入视频的帧上定义提示。
4. 点击 Selector 的 `Preview Current Frame` 检查当前帧分割结果，确认后点击 `Apply to Node`。
5. 执行节点，得到整段视频的 mask。

Selector 会递归查找上游的官方或第三方视频加载节点。如果输入来自中间生成节点、无法在打开 Selector 时直接追溯到文件，先运行一次工作流；节点会缓存最近一次实际收到的帧，Selector 随后使用这份缓存。

#### 节点选项说明

- `model`：官方 SAM3/SAM3.1 模型，必需输入。
- `images`：可选 `IMAGE` 帧批次。连接后作为执行和 Selector 的首选视频来源。
- `video_input`：可选 `VIDEO` 输入。仅在 `images` 未连接时使用。
- `anchor_frame`：锚点帧在当前输入帧批次中的本地编号，从 `0` 开始。通常由 Selector 自动写入。
- `prompt_data`：Selector 序列化的 Mask、BBox、Point 和对象列表，不建议手动编辑。
- `propagation_direction`：传播方向，`both` 双向传播，`forward` 向后传播，`backward` 向前传播。
- `max_objects`：SAM3.1 的最大对象槽数量，默认 `16`。
- `Open Selector`：打开交互式提示编辑器。

#### 输出说明

- `mask`：形状为 `[帧数, 高, 宽]` 的整段视频 mask。
- `anchor_mask`：锚点帧的分割 mask。
- `video_info`：包含帧数、尺寸、锚点帧、传播方向和对象数量。



## Selector 使用说明

SAM3.1 和 SeC-4B 共用同一个 Selector 框架。两者都支持多对象；SAM3.1 由官方模型批量处理对象，SeC-4B 会逐对象建立跟踪条件后合并结果。Semantic 选项卡仅对 SAM3.1 显示，SeC-4B 使用 Draw Mask、Edit BBox 和 Edit Point。

![Selector 界面](images/CS_Video_Segment_seletor.jpg)

#### 时间控制和锚点帧

- 时间线上方的 `|<` 和 `>|` 用于逐帧移动，滑杆和帧号输入框用于快速定位。
- 只允许一个锚点帧存在。第一次在某帧产生有效的 Mask、BBox 或 Point 后，该帧成为锚点。
- 已存在编辑数据时切换帧，会提示“当前已存在编辑数据，切换锚点帧将自动清除，是否继续？”。确认后会清除全部对象提示和 Undo/Redo 历史。
- 切换到新帧后，只有重新添加提示，该帧才会成为新的锚点。
- 已经点击 `Apply to Node` 的工作流再次打开 Selector 时，会自动跳转到保存的锚点帧并恢复提示。
- `anchor_frame` 是当前输入帧批次的本地帧号。

#### Draw Mask

- 进入 `Draw Mask` 后，鼠标光标显示当前画笔大小的半透明空心圆。
- `Brush/Eraser` 切换键在画笔和橡皮擦之间切换；画笔状态为绿色，橡皮擦状态为红色。
- `Brush Size` 范围为 `2–100`。拖动滑杆或在视频画布上滚动鼠标滚轮快速调整；按住 `Shift` 滚轮可大步调整。
- `Clear Mask` 清除当前对象的粗略 Mask。
- 粗略 Mask 只作为模型提示，点击 Preview 后画布会隐藏粗略笔迹，仅显示模型返回的分割结果。

#### Edit BBox

- 点击 `Add BBox` 后，在视频画布上拖出一个新框；松开鼠标后自动退出添加状态。
- 鼠标移到四角时可拖拽调整宽高，移到水平或垂直边时可单独调整对应边的位置。
- 鼠标移到框内部时显示手形光标，拖拽可移动整个框。
- 在框内部点击右键，选择“删除BBox”删除当前对象的框。
- 使用公共的 `Add Object` 创建更多对象，再为每个对象添加自己的 BBox。
- `Clear All BBox` 清除所有对象的 BBox，但不会清除 Point 或粗略 Mask。

#### Edit Point

- 空白处左键添加 positive point。
- 空白处右键添加 negative point。
- 已有 Point 左键按住并拖动可移动；如果没有实际移动，释放鼠标不会新增或修改 Point。
- 已有 Point 右键打开菜单，可删除 Point，或在 positive/negative 之间切换。
- `Clear All Point` 清除所有对象的 Point，但不会清除 BBox 或粗略 Mask。

#### 对象和公共按钮

- `Object` 下拉框选择当前编辑对象。
- `Add Object` 增加对象；`Delete Object` 删除当前对象。
- `Undo` / `Redo` 撤销或恢复最近一次提示编辑。
- `Clear All Prompt` 清除所有对象的 Mask、BBox 和 Point。
- `Preview Current Frame` 只运行当前锚点帧的模型分割，结果用于检查提示是否合理，不会替代最终整段视频执行。
- `Cancel` 关闭窗口并放弃本次未应用的修改。
- `Apply to Node` 将提示数据和锚点帧写入节点。

#### 示例工作流

workflow JSON 和示例素材位于插件的 `workflows` 子目录。以下图片仅为示意。

![SAM3.1 示例工作流](images/CS_Video_Segment(SAM3.1)_workflow.jpg)

![SeC-4B 示例工作流](images/CS_Video_Segment(Sec-4B)_workflow.jpg)


##  声明
ComfyUI_CineStyle节点遵照MIT开源协议，有部分功能代码和模型来自其他开源项目。如果作为商业用途，请查阅原项目授权协议使用。
