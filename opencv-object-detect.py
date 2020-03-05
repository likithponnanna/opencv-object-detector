import numpy as np
import cv2
import matplotlib.pyplot as plt
import sys
import math
import os
import random as rnd

def equalize_hist(img):
    for c in range(0, 2):
       img[:,:,c] = cv2.equalizeHist(img[:,:,c])
    return img


def auto_canny(image, sigma=0.33):
    v = np.median(image)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)

    return edged


def find_key(input_dict, value):
    return [k for k, v in input_dict.items() if v == value]

def image_prep(image_path):
    color_img = cv2.imread(image_path)
    gray = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    auto = auto_canny(blurred)
    line_removed = auto
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,2))
    line_removed = cv2.dilate(line_removed,kernel,iterations = 1) 
    
    return line_removed, color_img, auto



def alternative_strategy_contour(cnt_num, color_line_removed, cnt_num_per, fin_dict_cnt):
    if len(cnt_num) ==1:
        new_cnt= fin_dict_cnt[cnt_num[0]]
        cv2.drawContours(color_line_removed,[new_cnt],0,255,-1)
        M = cv2.moments(new_cnt)
    else:
        new_cnt= fin_dict_cnt[cnt_num_per[0]]
        cv2.drawContours(color_line_removed,[new_cnt],0,255,-1)
        M = cv2.moments(new_cnt)
    return M

def calculate_centroid(M):
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    else:
        cX = 0
        cY = 0
    return (cX, cY)
        

def main(argv, arc): 
    image_path = argv[1]
    if image_path.endswith('.jpg') or image_path.endswith('.png'):
        line_removed, color_img, auto = image_prep(image_path)
        color_line_removed =  cv2.cvtColor(line_removed, cv2.COLOR_GRAY2BGR)   
        contours, hierarchy = cv2.findContours(line_removed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 

        fin_dict_cnt, fin_dict_perimeter, fin_dict_approx = {}, {}, {}
        perimeter_list,approx_list,cnt_list = list(),list(),list()
                
        for i,cnt in enumerate(contours):    
            approx = cv2.approxPolyDP(cnt,0.02*cv2.arcLength(cnt,True),True)

            x,y,w,h = cv2.boundingRect(cnt)
            cont_area = cv2.contourArea(cnt)
            rect_min_area = cv2.minAreaRect(cnt)
            cor_width = rect_min_area[1][0]
            cor_height = rect_min_area[1][1]
            perimeter = cv2.arcLength(cnt,True)
            
            if len(approx)>= 4 and len(approx)< 10 and cont_area >= 168 and cont_area<970 \
                and h<50 and w<50 and perimeter<155 and cor_width<70 and cor_height<70 and \
                    cor_width>8 and cor_height>5:
                fin_dict_cnt[i] = cnt
                fin_dict_perimeter[i] = perimeter
                fin_dict_approx[i] = len(approx)
                
                perimeter_list.append(perimeter)
                approx_list.append(len(approx))
                cnt_list.append(cnt)
                    
            
        M = None        
        if len(cnt_list) == 1:
            M = cv2.moments(cnt_list[0])
            cv2.drawContours(color_line_removed,[cnt_list[0]],0,255,-1)
        else:
            min_approx = min(approx_list)
            min_perimeter = min(perimeter_list)
            cnt_num = find_key(fin_dict_approx, min_approx)
            cnt_num_per = find_key(fin_dict_perimeter, min_perimeter)
        
            M = alternative_strategy_contour(cnt_num, color_line_removed, cnt_num_per, fin_dict_cnt)
            
        center = calculate_centroid(M)
        normX = center[0] / (auto.shape[1])
        normY = center[1] /(auto.shape[0])
        print("{:.4f} {:.4f}".format(normX,normY))
    else:
        print("Input file not an image!!")

    

if __name__ == '__main__':
    main(sys.argv, len(sys.argv))
    