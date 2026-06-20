import os

def crop_radar_bin_final(input_file, output_file, crop_frames=100):
    num_rx = 4
    num_adc_samples = 256
    total_chirps_per_frame = 60
    bytes_per_sample = 4  # 复数 IQ 16bit

    # 计算一帧的精确大小
    bytes_per_frame = num_adc_samples * num_rx * total_chirps_per_frame * bytes_per_sample

    # 获取文件信息
    file_size = os.path.getsize(input_file)
    actual_total_frames = file_size // bytes_per_frame

    if actual_total_frames < crop_frames:
        print(f"错误：文件实际只有 {actual_total_frames} 帧，无法提取 {crop_frames} 帧。")
        return

    # 计算中间 100 帧的起始点
    start_frame = (actual_total_frames - crop_frames) // 2
    start_byte = start_frame * bytes_per_frame
    read_size = crop_frames * bytes_per_frame

    print(f"检测到文件包含 {actual_total_frames} 帧。")
    print(f"正在从第 {start_frame} 帧开始截取 {crop_frames} 帧...")

    try:
        with open(input_file, 'rb') as f:
            f.seek(start_byte)
            data = f.read(read_size)

        with open(output_file, 'wb') as f_out:
            f_out.write(data)

        final_size = os.path.getsize(output_file) / (1024 * 1024)
        print(f"裁剪成功！新文件大小: {final_size:.2f} MB")
        print(f"压缩比：约 {(final_size / (file_size / (1024 * 1024))) * 100:.1f}%")

    except Exception as e:
        print(f"处理过程中出错: {e}")


# 运行裁剪
# 确保你的文件路径正确
crop_radar_bin_final('adc_data.bin', 'cropped.bin', crop_frames=100)