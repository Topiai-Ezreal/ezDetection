import cv2
import os
from mmdet.apis import inference_detector, init_detector
import numpy as np
from PIL import Image, ImageDraw, ImageFont

font = cv2.FONT_HERSHEY_SIMPLEX
padding_size = 40
color_maps = {1:(104,221,230),2:(153,255,153),3:(254,0,0),4:(193,18,28),5:(209,91,143),6:(172,50,59),
              7:(213,109,86),8:(250,166,155),9:(255,220,164),10:(255,192,0),11:(146,209,79),12:(255,255,0),
              13:(166,94,47),14:(250,132,43),15:(198,54,120),16:(251,138,226),17:(114,159,178),18:(226,240,217),
              19:(219,226,254),20:(1,32,96),21:(0,247,0),22:(33,136,143),23:(135,115,161),24:(217,192,34),
              25:(204,197,143),26:(175,138,84),27:(190,78,32),28:(183,217,177),29:(158,160,161),30:(52,129,184),
              31:(152,64,225),32:(56,154,90),33:(254,94,47),34:(47,166,94),35:(94,47,166),36:(166,94,166),
              37:(220,11,200),38:(22,11,150), 100:(255,0,0),200:(255,0,0)}
ill_map = ['1', '2', '3', '4', '5', '7', '8', '9', '10', '11', '12', '13',
           '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '25',
           '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36','37', '38']
ill_name_map = {1:' 1：视网膜前膜',2:' 2：视网膜裂孔',3:' 3：玻璃体黄斑牵拉',4:' 4：视网膜囊性变样',5:' 5：视网膜劈裂',7:' 7：黄斑水肿（非囊性）',\
           8:' 8：视网膜内液',9:' 9：视网膜下积液',10:'10：弥漫性高反射病灶',11:'11：局部网膜内高反射点',12:'12：视网膜色素上皮（RPE）结构紊乱含玻璃疣',\
           13:'13：瘢痕与机化',14:'14：视网膜神经纤维层（RNFL）萎缩',15:'15：脉络膜增厚',16:'16：圆顶状色素上皮层脱落（PED）',\
           17:'17：视网膜萎缩',18:'18：纤维血管性色素上皮层脱离（含2型MNV）',19:'19：脉络膜曲度异常（例如肿瘤）',20:'20：双层征（扁平不规则PED，含1型MNV）',\
           21:'21：后巩膜葡萄肿',22:'22：椭圆体带（IS/OS）缺失',23:'23：脉络膜变薄',25:'25：视盘水肿',26:'26：视神经萎缩（视盘处）',\
           27:'27：视盘小凹',28:'28：神经节细胞层（GCL）萎缩',29:'29：有髓视神经纤维',30:'30：视网膜色素上皮（RPE）萎缩',\
           31:'31：视网膜大血管瘤',32:'32：视网膜脱离',33:'33：视网膜前出血',34:'34：视网膜内出血',35:'35：视网膜下出血',36:'36：视盘凹陷扩大',\
           37:'37：玻璃体后脱离',38:'38：玻璃体混浊（含玻璃体出血）', 100:'100：其它'}



def AddChineseText(img, text, left, top, textColor, textSize):
    if isinstance(img, np.ndarray):  # 判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # 创建一个可以在给定图像上绘图的对象
    draw = ImageDraw.Draw(img)
    # 字体的格式
    fontStyle = ImageFont.truetype("simsun.ttc", textSize, encoding="utf-8")
    # 绘制文本
    draw.text((left, top), text, textColor, font=fontStyle)
    # 转换回OpenCV格式
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


def draw_solid_box(img,box_pos,color,text=None):
    for k in range(box_pos[0], box_pos[2] + 1):
        img[box_pos[1], k] = color
        try:
            img[box_pos[3], k] = color
        except:
            print(1)
        img[box_pos[1]+1, k] = color
        img[box_pos[3]-1, k] = color
    for k in range(box_pos[1], box_pos[3] + 1):
        img[k, box_pos[0]] = color
        img[k, box_pos[2]] = color
        img[k, box_pos[0]+1] = color
        img[k, box_pos[2]-1] = color
    if text:
        img = cv2.putText(img, text, (box_pos[0] - 12, box_pos[1] - 10), font, 0.8, (255, 0, 0), 2)

    return img


def draw_dets(img, illness, pos_list):
    color = color_maps[int(illness)]
    pil_color = (color[2], color[1], color[0])  # RGB转为BGR
    text_shift = [10, 8]
    for box in pos_list:
        prob = box[-1]
        pos = [int(box[i]) + padding_size for i in range(4)]

        img = draw_solid_box(img, pos, pil_color)
        text_pos = (pos[0] - text_shift[0], pos[1] - text_shift[1])
        text = str(illness) + '|' + str(round(prob, 2))
        img = cv2.putText(img, text, text_pos, font, 0.8, pil_color, 2)

    # img = AddChineseText(img, illness, 0, y_ill, color, 18)
    # y_ill += 18
    return img


def visualize_on_cpu(config, checkpoint, img_path, save_path, gts_path):
    device = 'cpu'
    model = init_detector(config, checkpoint, device=device)

    for name in os.listdir(img_path):
        x_ill, y_ill = 0, 0
        temp = []
        raw_img = os.path.join(img_path, name)
        img = cv2.imread(raw_img)
        n_img = np.zeros((img.shape[0] + padding_size * 2, img.shape[1] + padding_size * 2, 3), dtype='uint8')
        n_img[padding_size:-padding_size, padding_size:-padding_size] = img
        img = n_img
        bbox_result = inference_detector(model, raw_img)
        for i in range(len(bbox_result)):
            if len(bbox_result[i]) != 0:
                illness = ill_map[i]
                pos_list = bbox_result[i]
                img = draw_dets(img, illness, pos_list)
                pil_color = color_maps[int(illness)]
                if illness not in temp:
                    temp.append(illness)
                    ill_name = ill_name_map[int(illness)]
                    img = AddChineseText(img, ill_name, x_ill, y_ill, pil_color, 18)
                    y_ill += 18

        if name in os.listdir(gts_path):
            gt_img = cv2.imread(os.path.join(gts_path, name))
            img = np.concatenate((gt_img, img), axis=1)
        cv2.imwrite(os.path.join(save_path, name), img)


if __name__ == '__main__':
    config_file = 'E:\\0726_ai_annotation\\all_data\\datasets\\config.py'
    checkpoint_file = 'E:\\0726_ai_annotation\\all_data\\datasets\\epoch_28.pth'
    img_path = 'E:\\0726_ai_annotation\\all_data\\datasets\\test_img'
    gts_path = 'E:\\0726_ai_annotation\\all_data\\gts_visualize'
    save_path = 'E:\\0726_ai_annotation\\all_data\\datasets\\visualize'
    visualize_on_cpu(config_file, checkpoint_file, img_path, save_path, gts_path)
    # conc(gts_path, img_path, save_path)
