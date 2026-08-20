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
![image](images/node-menu.jpg)    

* 或者在ComfyUI画布双击, 在搜索框输入"cinestyle"。
![image](images/node-search.jpg)    


## 更新说明

* 添加 [CS Load Video](#cs-load-video) 节点，用于加载视频，支持出入点设置、更改尺寸、帧率等。
* 添加 [CS Save Video](#cs-save-video) 节点，支持可选 metadata 写入和 H.264 目标码率控制。

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

##  声明
ComfyUI_CineStyle节点遵照MIT开源协议，有部分功能代码和模型来自其他开源项目。如果作为商业用途，请查阅原项目授权协议使用。
