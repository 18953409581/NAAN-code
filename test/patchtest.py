import os.path
from PIL import Image
import sys
import torchvision.transforms as transforms

# 图片的切块
def cut_image(image, patch_num):
    width, height = image.size
    item_width = int(width / patch_num)
    box_list = []
    # (left, upper, right, lower)
    for i in range(0 ,patch_num)  :  # 两重循环，生成n张图片基于原图的位置
        for j in range(0 ,patch_num):
            # print((i*item_width,j*item_width,(i+1)*item_width,(j+1)*item_width))
            box = ( j *item_width , i *item_width ,( j +1 ) *item_width ,( i +1 ) *item_width)
            box_list.append(box)
    print(box_list)
    image_list = [image.crop(box) for box in box_list]  #Image.crop(left, up, right, below)
    return image_list


# 保存
def save_images(image_list, save_path):
    index = 1
    for image in image_list:
        image.save(os.path.join(save_path, str(index) + '.png'))
        index += 1

if __name__ == '__main__':
    file_path = r'./data/images/bus.jpg'
    save_path = r'data/folder/'
    image = Image.open(file_path)
    # image.show()
    transform = transforms.Resize(224)
    image = transform(image)
    image_list = cut_image(image, patch_num=7)
    save_images(image_list,save_path)
