import cv2
import numpy as np
from PIL import Image, ImageEnhance
import argparse
import os
from typing import Optional, Tuple, Dict
import sys
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
from datetime import datetime

# 方法1.1: 使用支持中文的字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像时负号'-'显示为方块的问题

def process_image(input_path, output_path, threshold=None, show_steps=False, target_uniformity=0.5, tolerance=0.01):
    """
    处理PNG图像：彩色转黑白、自动调整对比度/亮度/阈值，使比特均匀性接近50%
    
    Args:
        input_path: 输入图片路径
        output_path: 输出图片路径
        threshold: 二值化阈值，如果为None则自动调整
        show_steps: 是否显示处理步骤
        target_uniformity: 目标比特均匀性（默认0.5，即50%）
        tolerance: 允许的均匀性误差范围
    """
    try:
        # 1. 使用PIL读取PNG图像
        original_image = Image.open(input_path)
        
        # 2. 转换为灰度图像
        gray_image = original_image.convert('L')
        
        # 3. 初始图像处理参数
        best_image = None
        best_uniformity = 0
        best_params = {}
        
        # 4. 定义参数搜索空间
        contrast_factors = [1.5, 2.0, 2.5, 3.0]  # 对比度增强因子
        brightness_factors = [0.9, 1.0, 1.1, 1.2]  # 亮度调整因子
        
        # 如果指定了阈值，使用固定阈值
        if threshold is not None:
            # 5. 增强对比度
            enhancer_contrast = ImageEnhance.Contrast(gray_image)
            enhanced_image = enhancer_contrast.enhance(2.0)  # 使用默认对比度
            
            # 6. 调整分辨率为32×32像素
            resized_image = enhanced_image.resize((96, 96), Image.LANCZOS)
            
            # 7. 应用阈值进行二值化
            cv_image = np.array(resized_image)
            _, binary_image = cv2.threshold(cv_image, threshold, 255, cv2.THRESH_BINARY)
            
            # 8. 转换回PIL图像并保存
            final_image = Image.fromarray(binary_image)
            final_image.save(output_path)
            
            # 计算比特均匀性
            uniformity_info = calculate_bit_uniformity(output_path, threshold)
            if uniformity_info:
                print(f"比特均匀性: {uniformity_info['bit_uniformity']:.4f} (0: {uniformity_info['ratio_0']:.2%}, 1: {uniformity_info['ratio_1']:.2%})")
            
            if show_steps:
                display_processing_steps(original_image, gray_image, enhanced_image, 
                                  resized_image, binary_image)
            
            print(f"图像处理完成！输出文件：{output_path}")
            print(f"使用的阈值：{threshold}")
            return final_image
        
        # 9. 自动调整参数以获得接近50%的比特均匀性
        print("自动调整参数以获得接近50%的比特均匀性...")
        
        for contrast in contrast_factors:
            for brightness in brightness_factors:
                # 增强对比度
                enhancer_contrast = ImageEnhance.Contrast(gray_image)
                contrast_image = enhancer_contrast.enhance(contrast)
                
                # 调整亮度
                enhancer_brightness = ImageEnhance.Brightness(contrast_image)
                brightness_image = enhancer_brightness.enhance(brightness)
                
                # 调整分辨率
                resized_image = brightness_image.resize((96, 96), Image.LANCZOS)
                
                # 转换为OpenCV格式
                cv_image = np.array(resized_image)
                
                # 尝试不同的阈值
                threshold_candidates = list(range(50, 255, 1))  # 0-255，步长1
                
                for thresh in threshold_candidates:
                    # 应用阈值
                    _, binary_image = cv2.threshold(cv_image, thresh, 255, cv2.THRESH_BINARY)
                    
                    # 计算比特均匀性
                    binary_array = (binary_image > thresh).astype(np.uint8)
                    count_0 = np.sum(binary_array == 0)
                    count_1 = np.sum(binary_array == 1)
                    total_pixels = count_0 + count_1
                    ratio_0 = count_0 / total_pixels
                    #uniformity = 1.0 - abs(ratio_0 - 0.5) * 2
                    uniformity = ratio_0
                    
                    # 检查是否接近目标均匀性
                    if abs(uniformity - target_uniformity) < abs(best_uniformity - target_uniformity):
                        best_uniformity = uniformity
                        best_image = binary_image.copy()
                        best_params = {
                            'contrast': contrast,
                            'brightness': brightness,
                            'threshold': thresh,
                            'ratio_0': ratio_0,
                            'ratio_1': 1 - ratio_0,
                            'uniformity': uniformity
                        }
                    
                    # 如果已经非常接近目标，提前结束搜索
                    if abs(uniformity - target_uniformity) <= tolerance:
                        print(f"找到理想参数: 对比度={contrast:.1f}, 亮度={brightness:.1f}, 阈值={thresh}")
                        print(f"比特均匀性: {uniformity:.4f} (0: {ratio_0:.2%}, 1: {1-ratio_0:.2%})")
                        
                        # 保存图片
                        final_image = Image.fromarray(binary_image)
                        final_image.save(output_path)
                        
                        if show_steps:
                            display_processing_steps(original_image, gray_image, contrast_image, 
                                              resized_image, binary_image)
                        
                        print(f"图像处理完成！输出文件：{output_path}")
                        print(f"使用参数: 对比度={contrast}, 亮度={brightness}, 阈值={thresh}")
                        return final_image
        
        # 10. 使用找到的最佳参数
        if best_image is not None:
            print(f"使用最佳参数: 对比度={best_params['contrast']:.1f}, 亮度={best_params['brightness']:.1f}, 阈值={best_params['threshold']}")
            print(f"比特均匀性: {best_params['uniformity']:.4f} (0: {best_params['ratio_0']:.2%}, 1: {best_params['ratio_1']:.2%})")
            
            # 重新处理以获得最佳结果
            # 应用对比度和亮度调整
            enhancer_contrast = ImageEnhance.Contrast(gray_image)
            contrast_image = enhancer_contrast.enhance(best_params['contrast'])
            
            enhancer_brightness = ImageEnhance.Brightness(contrast_image)
            brightness_image = enhancer_brightness.enhance(best_params['brightness'])
            
            # 调整分辨率
            resized_image = brightness_image.resize((96, 96), Image.LANCZOS)
            
            # 应用阈值
            cv_image = np.array(resized_image)
            _, best_image = cv2.threshold(cv_image, best_params['threshold'], 255, cv2.THRESH_BINARY)
            
            # 保存图片
            final_image = Image.fromarray(best_image)
            final_image.save(output_path)
            
            if show_steps:
                display_processing_steps(original_image, gray_image, contrast_image, 
                                  resized_image, best_image)
            
            print(f"图像处理完成！输出文件：{output_path}")
            return final_image
        
        # 11. 如果没有找到合适参数，使用默认参数
        print("警告：未能找到理想参数，使用默认设置...")
        enhancer = ImageEnhance.Contrast(gray_image)
        enhanced_image = enhancer.enhance(2.0)
        resized_image = enhanced_image.resize((96, 96), Image.LANCZOS)
        cv_image = np.array(resized_image)
        
        # 使用Otsu自动阈值
        thresh, binary_image = cv2.threshold(cv_image, 50, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 计算比特均匀性
        binary_array = (binary_image > thresh).astype(np.uint8)
        count_0 = np.sum(binary_array == 0)
        count_1 = np.sum(binary_array == 1)
        total_pixels = count_0 + count_1
        ratio_0 = count_0 / total_pixels
        uniformity = 1.0 - abs(ratio_0 - 0.5) * 2
        
        print(f"Otsu自动阈值: {thresh}")
        print(f"比特均匀性: {uniformity:.4f} (0: {ratio_0:.2%}, 1: {1-ratio_0:.2%})")
        
        final_image = Image.fromarray(binary_image)
        final_image.save(output_path)
        
        if show_steps:
            display_processing_steps(original_image, gray_image, enhanced_image, 
                              resized_image, binary_image)
        
        print(f"图像处理完成！输出文件：{output_path}")
        return final_image
        
    except Exception as e:
        print(f"处理图像时出错：{e}")
        return None

def display_processing_steps(original, gray, enhanced, resized, binary):
    """显示图像处理的各个步骤"""
    import matplotlib.pyplot as plt
    
    images = [original, gray, enhanced, resized, binary]
    titles = ['原图', '灰度图', '对比度增强', '32×32分辨率', '二值化结果']
    
    plt.figure(figsize=(15, 5))
    for i, (img, title) in enumerate(zip(images, titles)):
        plt.subplot(1, 5, i+1)
        if i == 0:  # 原图是彩色的
            plt.imshow(img)
        else:  # 其他步骤是灰度的
            plt.imshow(img, cmap='gray')
        plt.title(title)
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def load_image_safely(image_path):
    """
    安全地读取图片，处理中文路径问题
    """
    try:
        from PIL import Image
        img_pil = Image.open(image_path)
        if img_pil.mode != 'L':
            img_pil = img_pil.convert('L')
        return np.array(img_pil)
        
    except Exception as e:
        print(f"无法读取图片 {image_path}: {e}")
        return None

def calculate_hamming_distance(img1_path: str, img2_path: str, 
                              normalize: bool = True, threshold=128) -> Tuple[float, int]:
    """
    计算两张二值化图片之间的汉明距离
    
    Args:
        img1_path: 第一张图片路径
        img2_path: 第二张图片路径
        normalize: 是否将距离归一化到[0,1]范围
    
    Returns:
        Tuple[距离值, 总位数]
    """
    try:
        # 读取两张图片
        #img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
        #img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
        # 使用安全的图片读取函数
        img1 = load_image_safely(img1_path)
        img2 = load_image_safely(img2_path)
        
        if img1 is None or img2 is None:
            raise ValueError("无法读取图片，请检查文件路径")
        
        # 确保两张图片大小相同
        if img1.shape != img2.shape:
            print("警告：图片尺寸不同，尝试调整到相同大小...")
            # 调整到相同大小（以第一张图片为准）
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
        
        # 确保图片是二值化的（0和255）
        if img1.dtype != np.uint8 or img2.dtype != np.uint8:
            print("警告：图片格式非uint8，正在进行转换...")
            img1 = img1.astype(np.uint8)
            img2 = img2.astype(np.uint8)
        
        # 将图片展平为一维数组
        img1_flat = img1.flatten()
        img2_flat = img2.flatten()
        
        # 将像素值二值化（大于threshold为1，否则为0）
        img1_binary = (img1_flat > threshold).astype(np.uint8)
        img2_binary = (img2_flat > threshold).astype(np.uint8)
        
        # 计算汉明距离（不同位的数量）
        hamming_dist = np.sum(img1_binary != img2_binary)
        total_bits = len(img1_binary)
        
        if normalize:
            # 归一化到[0,1]范围
            normalized_dist = hamming_dist / total_bits
            return normalized_dist, total_bits
        else:
            return hamming_dist, total_bits
            
    except Exception as e:
        print(f"计算汉明距离时出错：{e}")
        return -1, 0

def visualize_hamming_difference(img1_path: str, img2_path: str, 
                                output_path: Optional[str] = None, threshold=128):
    """
    可视化显示两张图片的差异位
    """
    try:
        # 读取图片
        img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
        
        if img1 is None or img2 is None:
            print("无法读取图片")
            return
        
        # 调整到相同大小
        if img1.shape != img2.shape:
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
        
        # 创建差异图
        img1_binary = (img1 > threshold).astype(np.uint8) * 255
        img2_binary = (img2 > threshold).astype(np.uint8) * 255
        
        # 差异图：红色表示img1为1，绿色表示img2为1，蓝色表示相同
        diff_image = np.zeros((img1.shape[0], img1.shape[1], 3), dtype=np.uint8)
        
        # 找出差异
        mask1 = (img1_binary == 255) & (img2_binary == 0)  # img1有但img2没有
        mask2 = (img1_binary == 0) & (img2_binary == 255)  # img2有但img1没有
        mask_same = (img1_binary == img2_binary)  # 相同
        
        # 设置颜色
        diff_image[mask1] = [255, 0, 0]  # 红色
        diff_image[mask2] = [0, 255, 0]  # 绿色
        diff_image[mask_same] = [0, 0, 255]  # 蓝色
        
        # 保存或显示结果
        if output_path:
            cv2.imwrite(output_path, diff_image)
            print(f"差异图已保存到：{output_path}")
        
        # 显示
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        
        axes[0, 0].imshow(img1, cmap='gray')
        axes[0, 0].set_title('图片1')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(img2, cmap='gray')
        axes[0, 1].set_title('图片2')
        axes[0, 1].axis('off')
        
        axes[1, 0].imshow(diff_image)
        axes[1, 0].set_title('差异图（红:img1独有, 绿:img2独有, 蓝:相同）')
        axes[1, 0].axis('off')
        
        # 计算并显示汉明距离
        hamming_dist, total_bits = calculate_hamming_distance(img1_path, img2_path, normalize=False)
        norm_dist = hamming_dist / total_bits
        
        axes[1, 1].text(0.1, 0.5, 
                       f'汉明距离: {hamming_dist}/{total_bits}\n'
                       f'归一化距离: {norm_dist:.4f}\n'
                       f'相似度: {(1-norm_dist):.2%}',
                       fontsize=12, 
                       verticalalignment='center')
        axes[1, 1].axis('off')
        axes[1, 1].set_title('距离统计')
        
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"可视化差异时出错：{e}")

def batch_process_images(input_dir, output_dir, threshold=128):
    """批量处理目录中的所有PNG图像"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    processed_count = 0
    for filename in os.listdir(input_dir):
        if filename.lower().endswith('.png'):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, f"processed_{filename}")
            
            result = process_image(input_path, output_path)
            if result:
                processed_count += 1
    
    print(f"批量处理完成！共处理 {processed_count} 张图像")

def calculate_bit_uniformity(image_path: str, threshold: int = 128) -> Dict:
    """
    计算单张图片的比特均匀性
    
    参数:
        image_path: 图片路径
        threshold: 二值化阈值
    
    返回:
        包含比特均匀性指标的字典
    """
    try:
        # 读取图片
        img = load_image_safely(image_path)
        if img is None:
            return None
        
        # 转换为二值化
        binary_array = (img > threshold).astype(np.uint8)
        
        # 获取图片尺寸
        height, width = binary_array.shape
        total_pixels = height * width
        
        # 计算0和1的数量
        count_0 = np.sum(binary_array == 0)
        count_1 = np.sum(binary_array == 1)
        
        # 计算比例
        ratio_0 = count_0 / total_pixels
        ratio_1 = count_1 / total_pixels
        
        # 计算比特均匀性（均匀性 = 1 - |ratio_0 - 0.5| * 2）
        # 值在0-1之间，越接近1表示比特分布越均匀
        uniformity = 1.0 - abs(ratio_0 - 0.5) * 2
        
        return {
            'filename': os.path.basename(image_path),
            'total_pixels': int(total_pixels),
            'image_width': int(width),
            'image_height': int(height),
            'count_0': int(count_0),
            'count_1': int(count_1),
            'ratio_0': float(ratio_0),
            'ratio_1': float(ratio_1),
            'bit_uniformity': float(uniformity),
            'threshold': int(threshold)
        }
        
    except Exception as e:
        print(f"计算图片 {image_path} 比特均匀性时出错: {e}")
        return None

def batch_bit_uniformity(input_dir: str, threshold: int = 128, output_excel: str = None) -> None:
    """
    批量计算目录下所有图片的比特均匀性，并保存到Excel
    
    参数:
        input_dir: 输入目录路径
        threshold: 二值化阈值
        output_excel: Excel输出文件路径，如果为None则自动生成
    """
    if not output_excel:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_excel = f"bit_uniformity_{timestamp}.xlsx"
    
    # 获取目录下所有PNG文件
    png_files = []
    for filename in os.listdir(input_dir):
        if filename.lower().endswith('.png'):
            png_files.append(os.path.join(input_dir, filename))
    
    if not png_files:
        print(f"目录 {input_dir} 中没有PNG文件")
        return
    
    print(f"找到 {len(png_files)} 个PNG文件")
    print(f"使用阈值: {threshold}")
    
    # 计算每张图片的比特均匀性
    uniformity_data = []
    
    for i, img_path in enumerate(png_files):
        filename = os.path.basename(img_path)
        print(f"正在处理 ({i+1}/{len(png_files)}): {filename}")
        
        result = calculate_bit_uniformity(img_path, threshold)
        if result:
            uniformity_data.append(result)
            print(f"  比特均匀性: {result['bit_uniformity']:.6f}")
            print(f"  0的比例: {result['ratio_0']:.6f}, 1的比例: {result['ratio_1']:.6f}")
    
    if uniformity_data:
        # 创建DataFrame
        df = pd.DataFrame(uniformity_data)
        
        # 保存到Excel
        try:
            df.to_excel(output_excel, index=False)
            print(f"\n比特均匀性分析结果已保存到: {output_excel}")
            
            # 打印汇总统计
            print("\n=== 汇总统计 ===")
            print(f"总图片数: {len(df)}")
            print(f"平均比特均匀性: {df['bit_uniformity'].mean():.6f}")
            print(f"最大比特均匀性: {df['bit_uniformity'].max():.6f}")
            print(f"最小比特均匀性: {df['bit_uniformity'].min():.6f}")
            print(f"平均0的比例: {df['ratio_0'].mean():.6f}")
            print(f"平均1的比例: {df['ratio_1'].mean():.6f}")
            
            # 打印比特均匀性最好的3张图片
            print("\n=== 比特均匀性最好的3张图片 ===")
            best_images = df.nlargest(3, 'bit_uniformity')
            for _, row in best_images.iterrows():
                print(f"  {row['filename']}: 均匀性={row['bit_uniformity']:.6f}")
            
        except Exception as e:
            print(f"保存Excel文件时出错: {e}")
    else:
        print("没有计算到任何比特均匀性数据")

def batch_hamming(input_dir, threshold=128, output_excel=None):
    """批量计算目录中图片的汉明距离"""
    if not output_excel:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_excel = f"hamming_distances_{timestamp}.xlsx"
    
    # 获取所有PNG文件
    png_files = []
    for filename in os.listdir(input_dir):
        if filename.lower().endswith('.png'):
            png_files.append(os.path.join(input_dir, filename))
    
    if not png_files:
        print(f"目录 {input_dir} 中没有PNG文件")
        return
    
    print(f"找到 {len(png_files)} 个PNG文件")
    
    # 准备数据存储
    data = []
    processed_pairs = 0
    total_pairs = len(png_files) * len(png_files)
    
    for i, filename1 in enumerate(png_files):
        file1_name = os.path.basename(filename1)
        
        for j, filename2 in enumerate(png_files):
            file2_name = os.path.basename(filename2)
            
            # 计算汉明距离
            hamming_dist, total_bits = calculate_hamming_distance(filename1, filename2, normalize=False, threshold=threshold)
            
            if hamming_dist >= 0:  # 计算成功
                norm_dist = hamming_dist / total_bits
                similarity = (1 - norm_dist) * 100
                
                # 添加到数据列表
                data.append({
                    '文件1': file1_name,
                    '文件2': file2_name,
                    '汉明距离': hamming_dist,
                    '总位数': total_bits,
                    '归一化距离': round(norm_dist, 6),
                    '相似度_百分比': round(similarity, 4),
                    '相似度_比值': f"{total_bits-hamming_dist}/{total_bits}",
                    '阈值': threshold
                })
                
                # 输出到控制台
                print(f"\n对比: {file1_name} 与 {file2_name}")
                print(f"汉明距离: {hamming_dist}/{total_bits}")
                print(f"归一化距离: {norm_dist:.6f}")
                print(f"相似度: {similarity:.4f}%")
            else:
                print(f"\n错误: 无法计算 {file1_name} 与 {file2_name} 的汉明距离")
            
            processed_pairs += 1
            progress = (processed_pairs / total_pairs) * 100
            print(f"进度: {progress:.1f}% ({processed_pairs}/{total_pairs})", end='\r')
    
    print()  # 换行
    
    if data:
        # 创建DataFrame
        df = pd.DataFrame(data)
        
        # 保存到Excel
        try:
            df.to_excel(output_excel, index=False)
            print(f"结果已保存到 Excel 文件: {output_excel}")
            print(f"共计算了 {len(data)} 对图片的汉明距离")
            
            # 打印汇总统计
            print("\n汇总统计:")
            print(f"平均汉明距离: {df['汉明距离'].mean():.2f}")
            print(f"平均归一化距离: {df['归一化距离'].mean():.6f}")
            print(f"平均相似度: {df['相似度_百分比'].mean():.4f}%")
            print(f"最小相似度: {df['相似度_百分比'].min():.4f}%")
            print(f"最大相似度: {df['相似度_百分比'].max():.4f}%")
            
        except Exception as e:
            print(f"保存Excel文件时出错: {e}")
    else:
        print("没有计算到任何有效的汉明距离数据")

def main():
    parser = argparse.ArgumentParser(description='PNG图像处理工具', 
                                    formatter_class=argparse.RawDescriptionHelpFormatter,
                                    epilog='''
示例:
  # 1. 处理单张图片
  python script.py process input.png output.png
  
  # 2. 批量处理图片
  python script.py batch input_folder output_folder
  
  # 3. 计算汉明距离
  python script.py hamming image1.png image2.png
  
  # 4. 可视化汉明距离差异
  python script.py visualize image1.png image2.png
  
  # 5. 批量计算汉明距离
  python script.py batchhamming input_folder --output results.xlsx
  
  # 6. 批量计算比特均匀性
  python script.py bituniformity input_folder --output uniformity.xlsx
''')
    
    subparsers = parser.add_subparsers(dest='command', help='命令')
    
    # 子命令1: 处理单张图片
    # 在 parser_process 部分添加新参数
    parser_process = subparsers.add_parser('process', help='处理单张图片')
    parser_process.add_argument('input', help='输入图像路径')
    parser_process.add_argument('output', help='输出图像路径')
    parser_process.add_argument('--threshold', type=int, default=None, 
                            help='二值化阈值 (0-255)，如果为None则自动调整以获得50%比特均匀性')
    parser_process.add_argument('--target-uniformity', type=float, default=0.5,
                            help='目标比特均匀性 (0-1)，默认0.5')
    parser_process.add_argument('--tolerance', type=float, default=0.05,
                            help='均匀性误差容忍度，默认0.05')
    parser_process.add_argument('--show-steps', action='store_true',
                            help='显示处理过程中的每个步骤')
    
    # 子命令2: 批量处理
    parser_batch = subparsers.add_parser('batch', help='批量处理目录中的所有PNG图像')
    parser_batch.add_argument('input', help='输入目录路径')
    parser_batch.add_argument('output', help='输出目录路径')
    parser_batch.add_argument('--threshold', type=int, default=128,
                            help='二值化阈值 (0-255)，默认128')
    
    # 子命令3: 计算汉明距离
    parser_hamming = subparsers.add_parser('hamming', help='计算两张图片的汉明距离')
    parser_hamming.add_argument('img1', help='第一张图片路径')
    parser_hamming.add_argument('img2', help='第二张图片路径')
    parser_hamming.add_argument('--normalize', action='store_true', default=True,
                              help='计算归一化汉明距离（0-1范围）')
    parser_hamming.add_argument('--no-normalize', action='store_false', dest='normalize',
                              help='不计算归一化汉明距离')
    parser_hamming.add_argument('--threshold', type=int, default=128,
                            help='二值化阈值 (0-255)，默认128')
    
    # 子命令4: 可视化差异
    parser_viz = subparsers.add_parser('visualize', help='可视化显示两张图片的差异')
    parser_viz.add_argument('img1', help='第一张图片路径')
    parser_viz.add_argument('img2', help='第二张图片路径')
    parser_viz.add_argument('--output', '-o', help='差异图输出路径')
    parser_viz.add_argument('--threshold', type=int, default=128,
                            help='二值化阈值 (0-255)，默认128')

    # 子命令5: 批量计算汉明距离
    parser_bhamming = subparsers.add_parser('batchhamming', help='批量计算目录中图片的汉明距离')
    parser_bhamming.add_argument('input', help='输入目录路径')
    parser_bhamming.add_argument('--threshold', type=int, default=128,
                            help='二值化阈值 (0-255)，默认128')
    parser_bhamming.add_argument('--output', '-o', help='Excel输出文件路径')
    
    # 子命令6: 批量计算比特均匀性
    parser_bituniformity = subparsers.add_parser('bituniformity', help='批量计算目录中图片的比特均匀性')
    parser_bituniformity.add_argument('input', help='输入目录路径')
    parser_bituniformity.add_argument('--threshold', type=int, default=128,
                                    help='二值化阈值 (0-255)，默认128')
    parser_bituniformity.add_argument('--output', '-o', help='Excel输出文件路径')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    if args.command == 'process':
        process_image(args.input, args.output, args.threshold, args.show_steps)
    
    elif args.command == 'batch':
        batch_process_images(args.input, args.output, args.threshold)
    
    elif args.command == 'hamming':
        if not os.path.exists(args.img1) or not os.path.exists(args.img2):
            print("错误：指定的图片文件不存在")
            return
        
        distance, total_bits = calculate_hamming_distance(args.img1, args.img2, args.normalize)
        
        if distance >= 0:
            if args.normalize:
                print(f"\n汉明距离统计：")
                print(f"归一化距离: {distance:.6f}")
                print(f"相似度: {(1-distance):.2%}")
                print(f"总位数: {total_bits}")
            else:
                print(f"\n汉明距离: {distance}/{total_bits} 位不同")
                print(f"归一化距离: {distance/total_bits:.6f}")
                print(f"相似度: {(1-distance/total_bits):.2%}")
    
    elif args.command == 'visualize':
        if not os.path.exists(args.img1) or not os.path.exists(args.img2):
            print("错误：指定的图片文件不存在")
            return
        
        visualize_hamming_difference(args.img1, args.img2, args.output)

    elif args.command == 'batchhamming':
        batch_hamming(args.input, args.threshold, args.output)
    
    elif args.command == 'bituniformity':
        batch_bit_uniformity(args.input, args.threshold, args.output)

if __name__ == "__main__":
    main()