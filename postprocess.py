from utils.general import scale_coords
from utils.custom_utils import CustomPlotBox

class PostProcessor():
    def __init__(self):
        self.nightMode = False
        # For 車道線
        # self.xxx = False
    
    def YoloPostProcess(self, predict, img, ori_img):
        position_arr = [1, 1, 1]
        for i, det in enumerate(predict):  # detections per image
            s, im0 = '', ori_img
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                # im0 = self.AdjustGamma(im0, 0.2)
                # Print results
                # for c in det[:, -1].unique():
                #     n = (det[:, -1] == c).sum()  # detections per class
                #     s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    # label = f'{self.names[int(cls)]} {conf:.2f}'
                    # label = f'{self.names[int(cls)]}'
                    CustomPlotBox(pos_arr=position_arr, x=xyxy, img=im0, label=label, 
                                  box_color=self.colors[int(cls)], 
                                  line_thickness=2)
        return (im0, position_arr)
    
    def NightProcess(self, img):
        return