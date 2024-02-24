import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
print('pid: {}     GPU: {}'.format(os.getpid(), os.environ['CUDA_VISIBLE_DEVICES']))
import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel
import numpy as np

##head pose +
import torchlm
from torchlm.tools import faceboxesv2
from torchlm.models import pipnet
torchlm.runtime.bind(faceboxesv2(device="cuda"))  # set device="cuda" if you want to run with CUDA
# set map_location="cuda" if you want to run with CUDA
torchlm.runtime.bind(
  pipnet(backbone="resnet18", pretrained=True,  
         num_nb=10, num_lms=98, net_stride=32, input_size=256,
         meanface_type="wflw", map_location="cuda", checkpoint=None) 
) # will auto download pretrained weights from latest release if pretrained=True


Look_Up=0
Look_Down=0
Look_Center=0
Look_left=0
Look_right=0
Look_Center2=0
last_status=''
last_status2=''
def headpose(img,face_points):
    global Look_Up
    global Look_Down
    global Look_Center
    global Look_left
    global Look_right
    global Look_Center2
    global last_status
    global last_status2


    #使用OpenCV的solvePnP函數來計算人臉的旋轉與位移。
    # 3維模型的座標點 (使用一般的3D人臉模型的座標點)
    model_points = np.array([
                                (0.0, 0.0, 0.0),             # Nose tip
                             (0.0, -330.0, -65.0),        # Chin
                             (-225.0, 170.0, -135.0),     # Left eye left corner
                             (225.0, 170.0, -135.0),      # Right eye right corne
                                (-150.0, -150.0, -125.0),    # Left Mouth corner
                                (150.0, -150.0, -125.0)      # Right mouth corner                         
                            ])

    # 焦距
    size = img.shape
    focal_length = size[1] 
    print("Cameria [focal_length]: ", focal_length)

    # 照像機內部成像的中心點(w, h)
    center = (size[1]/2, size[0]/2)

    # 照像機參數 (Camera internals )
    camera_matrix = np.array(
                         [[focal_length, 0, center[0]],
                         [0, focal_length, center[1]],
                         [0, 0, 1]], dtype = "double"
                         )
 
    print("Camera Matrix :\n {0}".format(camera_matrix))
    # 扭曲係數
    dist_coeffs = np.zeros((4,1)) # 假設沒有鏡頭的成像扭曲 (no lens distortion)

    # 使用OpenCV的solvePnP函數來計算人臉的旋轉與位移
    #(success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix
    #                                                              , dist_coeffs, flags=cv2.CV_ITERATIVE)
    # 參數:
    #   model_points 3維模型的座標點
    #   image_points 2維圖像的座標點
    #   camera_matrix 照像機矩陣
    #   dist_coeffs 照像機扭曲係數
    #   flags: cv2.SOLVEPNP_ITERATIVE
    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, face_points, camera_matrix
                                                              , dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

    print("Rotation Vector:\n {0}".format(rotation_vector)) # 旋轉向量
    print("Translation Vector:\n {0}".format(translation_vector)) # 位移向量

    # 計算歐拉角
    rvec_matrix = cv2.Rodrigues(rotation_vector)[0]
    proj_matrix = np.hstack((rvec_matrix, translation_vector))
    eulerAngles = -cv2.decomposeProjectionMatrix(proj_matrix)[6]

    yaw   = eulerAngles[1]
    pitch = eulerAngles[0]
    roll  = eulerAngles[2]

    if pitch > 0:
        pitch = 180 - pitch
    elif pitch < 0:
        pitch = -180 - pitch
    yaw = -yaw

    print("抬頭(+)/低頭(-) [pitch]: ", pitch) # 抬頭(+)/低頭(-)
    print("右轉(+)/左轉(-) [yaw]  : ", yaw)   # 右轉(+)/左轉(-)
    print("右傾(+)/左傾(-) [roll] : ", roll)  # 右傾(+)/左傾(-)


    if yaw<-5 :
        Look_left+=1
    elif yaw>5 :
        Look_right+=1
    else:
        Look_Center2+=1

    if pitch<0 :
        Look_Up+=1
    elif pitch> 75 :
        Look_Down+=1
    elif pitch> 0 and pitch< 75:
        Look_Center+=1

    max_index=99
    if Look_Up+Look_Down+Look_Center > 10 :
        marks = (Look_Up,Look_Down,Look_Center)
        max_index = marks.index(max(marks))
        print("max_index=",max_index)
        Look_Up=0
        Look_Down=0
        Look_Center=0

    max_index2=99
    if Look_left+Look_right+Look_Center2 > 10 :
        marks2 = (Look_left,Look_right,Look_Center2)
        max_index2 = marks2.index(max(marks2))
        print("max_index2=",max_index2)
        Look_left=0
        Look_right=0
        Look_Center2=0

    
    if max_index==0 :
        cv2.putText(img, "Look Up", (0, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        last_status="Look Up"
    elif max_index==1 :
        cv2.putText(img, "Look Down", (0, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        last_status="Look Down"
    elif max_index==2:
        cv2.putText(img, "Look Center", (0, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        last_status="Look Center"
    else:
        cv2.putText(img, last_status, (0, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


    if max_index2==0 :
        cv2.putText(img, "Look Left", (0, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        last_status2="Look Left"
    elif max_index2==1 :
        cv2.putText(img, "Look Right", (0, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        last_status2="Look Right"
    elif max_index2==2:
        cv2.putText(img, "Look Center", (0, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        last_status2="Look Center"
    else:
        cv2.putText(img, last_status2, (0, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)






    # 投射一個3D的點 (100.0, 0, 0)到2D圖像的座標上
    (x_end_point2D, jacobian) = cv2.projectPoints(np.array([(100.0, 0.0, 0.0)]), rotation_vector
                                                 , translation_vector, camera_matrix, dist_coeffs)

    # 投射一個3D的點 (0, 100.0, 0)到2D圖像的座標上
    (y_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 100.0, 0.0)]), rotation_vector
                                                 , translation_vector, camera_matrix, dist_coeffs)

    # 投射一個3D的點 (0, 0, 100.0)到2D圖像的座標上
    (z_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 100.0)]), rotation_vector
                                           , translation_vector, camera_matrix, dist_coeffs)


    # 以 Nose tip為中心點畫出x, y, z的軸線
    p_nose = (int(face_points[0][0]), int(face_points[0][1]))

    p_x = (int(x_end_point2D[0][0][0]), int(x_end_point2D[0][0][1]))

    p_y = (int(y_end_point2D[0][0][0]), int(y_end_point2D[0][0][1]))

    p_z = (int(z_end_point2D[0][0][0]), int(z_end_point2D[0][0][1]))

    cv2.line(img, p_nose, p_x, (0,0,255), 3)  # X軸 (紅色)
    cv2.line(img, p_nose, p_y, (0,255,0), 3)  # Y軸 (綠色)
    cv2.line(img, p_nose, p_z, (255,0,0), 3)  # Z軸 (藍色)






##head pose -


#from RetinaFaceMaster.test import predict
#from mtcnn.detect_face import MTCNN
#from model2 import MobileNetV2, BlazeLandMark

def plot_one_box2(x, img, color=None, label=None, line_thickness=3,x1=0,y1=0,pre_landmark=None):
    # Plots one bounding box on image img

    


    
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
        if "face" in label:


            
            # 取得單1人臉的98個人臉關鍵點的座標
            # 鼻尖 Nose tip: 57
            nose_tip = pre_landmark[0].astype(np.int32)[57:58]
            # 下巴 Chin: 16
            chin = pre_landmark[0].astype(np.int32)[16:17]
            # 左眼左角 Left eye left corner: 60
            left_eye_corner = pre_landmark[0].astype(np.int32)[60:61]
            # 右眼右角 Right eye right corner: 72
            right_eye_corner = pre_landmark[0].astype(np.int32)[72:73]
            # 嘴巴左角 Left Mouth corner: 76
            left_mouth_corner = pre_landmark[0].astype(np.int32)[76:77]
            # 嘴巴右角 Right Mouth corner: 82
            right_mouth_corner = pre_landmark[0].astype(np.int32)[82:83]


            # 把相關的6個座標串接起來
            face_points = np.concatenate((nose_tip, chin, left_eye_corner, right_eye_corner, left_mouth_corner, right_mouth_corner))
            face_points = face_points.astype(np.double)

            #print('face_points=',face_points)
            #print('face_points=',face_points.dtype)
            #print(type(face_points))
            #i=0
            for (x, y) in face_points.astype(np.int32):
            #for (x, y) in face_points_2:
                #cv2.circle(img, ( x,  y), 1, (0, 0, 255), 2)
                #print(x,  y)
                cv2.putText(img, 'o', ( x,  y), cv2.FONT_HERSHEY_SIMPLEX,  0.2, (0, 255, 255), 1, cv2.LINE_AA)
                #i+=1

            headpose(img,face_points)
            


            #---------eye 1-----------------------------
            # 取得單1人臉的98個人臉關鍵點的座標
            # 鼻尖 Nose tip: 57
            nose_tip = pre_landmark[0].astype(np.int32)[96:97]
            # 下巴 Chin: 16
            chin = pre_landmark[0].astype(np.int32)[66:67]
            # 左眼左角 Left eye left corner: 60
            left_eye_corner = pre_landmark[0].astype(np.int32)[61:62]
            # 右眼右角 Right eye right corner: 72
            right_eye_corner = pre_landmark[0].astype(np.int32)[63:64]
            # 嘴巴左角 Left Mouth corner: 76
            left_mouth_corner = pre_landmark[0].astype(np.int32)[60:61]
            # 嘴巴右角 Right Mouth corner: 82
            right_mouth_corner = pre_landmark[0].astype(np.int32)[64:65]



             # 把相關的6個座標串接起來
            face_points_eye1 = np.concatenate((nose_tip, chin, left_eye_corner, right_eye_corner, left_mouth_corner, right_mouth_corner))
            face_points_eye1 = face_points_eye1.astype(np.double)

            #print('face_points_eye1=',faface_points_eye1ce_points)
            #print('face_points_eye1=',face_points_eye1.dtype)
            #print(type(face_points_eye1))
            #i=0
            for (x, y) in face_points_eye1.astype(np.int32):
                #cv2.circle(img, ( x,  y), 1, (0, 0, 255), 2)
                #print(x,  y)
                cv2.putText(img, 'o', ( x,  y), cv2.FONT_HERSHEY_SIMPLEX,  0.2, (0, 255, 255), 1, cv2.LINE_AA)
                #i+=1

            headpose(img,face_points_eye1)



            #---------eye 2-----------------------------
            # 取得單1人臉的98個人臉關鍵點的座標
            # 鼻尖 Nose tip: 57
            nose_tip = pre_landmark[0].astype(np.int32)[97:98]
            # 下巴 Chin: 16
            chin = pre_landmark[0].astype(np.int32)[74:75]
            # 左眼左角 Left eye left corner: 60
            left_eye_corner = pre_landmark[0].astype(np.int32)[69:70]
            # 右眼右角 Right eye right corner: 72
            right_eye_corner = pre_landmark[0].astype(np.int32)[71:72]
            # 嘴巴左角 Left Mouth corner: 76
            left_mouth_corner = pre_landmark[0].astype(np.int32)[68:69]
            # 嘴巴右角 Right Mouth corner: 82
            right_mouth_corner = pre_landmark[0].astype(np.int32)[72:73]


            # 把相關的6個座標串接起來
            face_points_eye2 = np.concatenate((nose_tip, chin, left_eye_corner, right_eye_corner, left_mouth_corner, right_mouth_corner))
            face_points_eye2 = face_points_eye2.astype(np.double)

            #print('face_points_eye2=',face_points_eye2)
            #print('face_points_eye2=',face_points_eye2.dtype)
            #print(type(face_points_eye2))
            #i=0
            for (x, y) in face_points_eye2.astype(np.int32):
                #cv2.circle(img, ( x,  y), 1, (0, 0, 255), 2)
                #print(x,  y)
                cv2.putText(img, 'o', ( x,  y), cv2.FONT_HERSHEY_SIMPLEX,  0.2, (0, 255, 255), 1, cv2.LINE_AA)
                #i+=1

            headpose(img,face_points_eye2)




            #---------eye 1/eye 2---mouthh--------------------------
            # 取得單1人臉的98個人臉關鍵點的座標

            #  右眼up Left eye left corner: 62
            right_eye_up = pre_landmark[0].astype(np.int32)[62:63]
            # 右眼down Right eye right corner: 66
            right_eye_down = pre_landmark[0].astype(np.int32)[66:67]
            # 左眼up Left Mouth corner: 70
            left_eye_up = pre_landmark[0].astype(np.int32)[70:71]
            # 左眼down Right Mouth corner: 74
            left_eye_down = pre_landmark[0].astype(np.int32)[74:75]
            # mouth_up Left Mouth corner: 90
            mouth_up = pre_landmark[0].astype(np.int32)[90:91]
            # mouth_down Right Mouth corner: 94
            mouth_down = pre_landmark[0].astype(np.int32)[94:95]

         

            print("right_eye :" ,right_eye_down[0][1]-right_eye_up[0][1])
            print("left_eye :" ,left_eye_down[0][1]-left_eye_up[0][1])
            print("mouth :" ,mouth_down[0][1]-mouth_up[0][1])




            if right_eye_down[0][1]-right_eye_up[0][1]<=5 :
                cv2.putText(img, "eyes: Close", (0, 160), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            elif left_eye_down[0][1]-left_eye_up[0][1] <=5 :
                cv2.putText(img, "eyes: Close", (0, 160), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                cv2.putText(img, "eyes: Open", (0, 160), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


            if mouth_down[0][1]-mouth_up[0][1]>=30:
                cv2.putText(img, "mouth: Open", (0, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
            else:
                cv2.putText(img, "mouth: Close", (0, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)








def detect_landmarks(img0):

    landmarks, bboxes = torchlm.runtime.forward(img0)
    #print("bboxes=",bboxes)
    #print("landmarks=",landmarks[0])

    return landmarks





def detect(save_img=False):
    source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))
    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    #ckpt_file = './pretrained_model/blazelandmark.pth'
    #model_landmark = attempt_load(ckpt_file, map_location=device)  # load FP32 model
    #model_landmark = BlazeLandMark(nums_class=136)
    #model_landmark = torch.load(ckpt_file)
    #model_landmark.eval()
    #model_landmark.half()  # to FP16






    if trace:
        model = TracedModel(model, device, opt.img_size)

    if half:
        model.half()  # to FP16
        #model_landmark.half()

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=opt.augment)[0]

        # Inference
        fps = 0.0
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t3 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            #im0 = cv2.flip(im0,1)     #flip


            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string


                # Write results
                for *xyxy, conf, cls in reversed(det):

                    print(names[int(cls)])
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or view_img:  # Add bbox to image
                        label = f'{names[int(cls)]} {conf:.2f}'

                        #run face landmark model
                        if names[int(c)]=='face' :
                            face_box=[int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])]
                            #print("face_box=",face_box)
                            pre_landmark=detect_landmarks(im0)

                        plot_one_box2(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1,x1=face_box[0],y1=face_box[1],pre_landmark=pre_landmark)

            # Print time (inference + NMS)
            print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

            fps  = ( fps + (1./(time.time()-t1)) ) / 2
            #print("fps= %.2f"%(fps))
            frame = cv2.putText(im0, "fps= %.2f"%(fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Stream results
            if view_img:
                cv2.namedWindow(str(p), cv2.WINDOW_NORMAL)    # Create window with freedom of dimensions
                #cv2.resizeWindow(str(p), 1280, 1280)
                cv2.imshow(str(p), im0)
                c= cv2.waitKey(1) & 0xff
            if c==27:
                vid_cap.release()
                break

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                    print(f" The image with the result is saved in: {save_path}")
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        #print(f"Results saved to {save_dir}{s}")

    print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='./runs/train/yolov7_dms_face/weights/best.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='0', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.45, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.35, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', default=True, action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', default=False,action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    opt = parser.parse_args()
    print(opt)
    #check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov7.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
