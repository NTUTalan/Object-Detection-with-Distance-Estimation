import pickle
import random
import os
import cv2
object_real_height = {}
car_infos_file = "carInfos.pkl"

# 給CustomUtility用的
def CustomPlotBox(x: list, img, box_color: list=None, label: str=None, line_thickness: int=3) -> list[tuple]:
    '''
    從image的shape來判斷在左還是在右
    center_x = (shape[0] - 1 ) / 2
    '''
    img_width = img.shape[0]
    img_height = img.shape[1]
    tl = line_thickness or round(0.002 * (img_width + img_height) / 2) + 1  # line/font thickness
    box_color = box_color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3])) #左上、右下
    w = int(x[2]) - int(x[0])
    height = int(x[3]) - int(x[1])
    # ?? cv2.rectangle(img, c1, c2, box_color, thickness=tl, lineType=cv2.LINE_AA)
    ### Caculate Distance and plot
    if label:
        tf = max(tl - 1, 1)  # font thickness
        distance = 512 * object_real_height[label] / height ### Distance measuring in Inch 
        try:
            str_distance = str("{:.2f} Meters".format(distance)) ### 目前使用手機計算
        except:
            distance =  ''
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        d_size = cv2.getTextSize(distance, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0] + d_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, box_color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
        cv2.putText(img, distance, (c1[0] + t_size[0] + 1, c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
        return (GetPosition(x, img_width), distance)

def GetPosition(x: list, img_width: int):
    box_center_x = ( x[2] - x[0] ) / 2
    left_bound = (img_width - 1) / 3
    right_bound = (img_width - 1) * 2 / 3
    if(box_center_x < left_bound):
        return "left"
    elif(box_center_x > right_bound):
        return "right"
    else:
        return "center"
    

if(__name__ == "__main__"):
    if(os.path.isfile(car_infos_file)):
        with open("carInfos.pkl", "rb") as car_infos:
            object_real_height = pickle.load(car_infos)
    else:
        object_real_height = {'Car': 1.676, 'Van': 1.676, 'Truck': 4, 'Pedestrian': 1.671}
        with open("carInfos.pkl", "wb") as car_infos:
            pickle.dump(object_real_height, car_infos)
    print(CustomPlotBox.__annotations__)