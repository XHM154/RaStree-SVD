from PIL import Image
import sys
import os
# 打开图片

# 获取图片的尺寸
width = 178
height = 218

def main():
    if len(sys.argv) < 3:
        print("Usage: python script.py <arg1> <arg2>")
        return
    
    try:
        num = int(sys.argv[1])
        nParts = int(sys.argv[2])
    except ValueError:
        print("arguments errors.")
        return
    print(f"Total images number: {num}")
    print(f"nParts: {nParts}")
    
    common_group_size = num // nParts
    final_group_size  = num - (nParts - 1) * common_group_size
    prefix_path = f"partition_{nParts}/"
    
    try:
        os.mkdir(prefix_path)
    except FileExistsError:
        print("File Folder already created.")
    except PermissionError:
        print("No Permission")
        return
    
    sum = 0
    
    with open(f'{prefix_path}image_align_celeba_nnz.txt', 'w') as file_nnz:
        for part in range(nParts):
            print(f"{part} partition starts!")
            
            group_size = common_group_size
            start_col = part * common_group_size + 1
            if(part == nParts - 1):
                start_col = num - final_group_size + 1
                group_size = final_group_size
                
            nnz_part = group_size * width * height * 3
            
            with open(f'{prefix_path}image_align_celeba_{part}.txt', 'w') as file:
                for number in range(group_size):
                    index = number + start_col
                    formatted_number_filename = f"image_align_celeba/{index:06}.jpg"  # 格式化为6位长度，前面补零
                    
                    img = Image.open(formatted_number_filename)
                    # 获取像素数据
                    pixels = img.load()
                    
                    # file.write(f"{formatted_number_filename}:\n")
                    # 遍历每一个像素并打印 RGB 值
                    cnt = 1
                    for y in range(height):
                        for x in range(width):
                            r, g, b = pixels[x, y]
                            file.write(f"{index} {cnt} {r} ")
                            file.write(f"{index} {cnt + 1} {g} ")
                            file.write(f"{index} {cnt + 2} {b} ")
                            
                            cnt = cnt + 3
                            
            print(f"{part} partition done!")
            print(sum)
            file_nnz.write(f"{nnz_part}\n")

if __name__ == "__main__":
    main()