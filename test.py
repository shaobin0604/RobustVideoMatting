import torch
from model import MattingNetwork
from inference import convert_video

model = MattingNetwork('mobilenetv3').eval()  # 或 "resnet50"
model.load_state_dict(torch.load('test/rvm_mobilenetv3.pth'))

dilate_iterations = 0

convert_video(
    model,                           # 模型，可以加载到任何设备（cpu 或 cuda）
    input_source='test/Archive/input.mp4',        # 视频文件，或图片序列文件夹
    output_type='video',             # 可选 "video"（视频）或 "png_sequence"（PNG 序列）
    output_composition='test/Archive/com.mp4',    # 若导出视频，提供文件路径。若导出 PNG 序列，提供文件夹路径
    output_alpha="test/Archive/pha.mp4",          # [可选项] 输出透明度预测
    output_foreground="test/Archive/fgr.mp4",     # [可选项] 输出前景预测
    output_video_mbps=5,             # 若导出视频，提供视频码率
    dilate_iterations=dilate_iterations,             # 使用 OpenCV 对 output_alpha 进行 dilate
    downsample_ratio=None,           # 下采样比，可根据具体视频调节，或 None 选择自动
    seq_chunk=12,                    # 设置多帧并行计算
)
