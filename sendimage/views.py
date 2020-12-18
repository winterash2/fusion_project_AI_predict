from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from django.core.files.storage import default_storage

import cv2
import numpy as np


def get_image(request):
    filename = ''
    
    for files in request.FILES:
        file_obj = request.FILES[files]
        filename = str(file_obj)
        with default_storage.open(filename, 'wb+') as destination:
            for chunk in file_obj.chunks():
                destination.write(chunk)
    # 판별
    result = predict(filename)

    # dict 형태로 리턴
    result = {'result':result}

    return JsonResponse(result)


#판별기
def predict(filename):
    return_label = ''

    net = cv2.dnn.readNet('yolov3-custom_42000.weights','custom.cfg')

    classes = []
    with open("classes.txt", "r") as f:
        classes = f.read().splitlines()

    # cap = cv2.VideoCapture('test1.mp4')  # 영상 사용할때

    font = cv2.FONT_HERSHEY_PLAIN
    colors = np.random.uniform(0, 255, size=(100, 3))

    img = cv2.imread(filename)
    height, width, _ = img.shape

    blob = cv2.dnn.blobFromImage(img, 1/255, (608, 608), (0,0,0), swapRB=True, crop=False)
    net.setInput(blob)
    output_layers_names = net.getUnconnectedOutLayersNames()
    layerOutputs = net.forward(output_layers_names)

    boxes = []
    confidences = []
    class_ids = []

    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.2:
                center_x = int(detection[0]*width)
                center_y = int(detection[1]*height)
                w = int(detection[2]*width)
                h = int(detection[3]*height)

                x = int(center_x - w/2)
                y = int(center_y - h/2)

                boxes.append([x, y, w, h])
                confidences.append((float(confidence)))
                class_ids.append(class_id)

                return_label = str(classes[class_ids[0]])
                return return_label

    # indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.2, 0.4)

    # if len(indexes)>0:
    #     for i in indexes.flatten():
    #         x, y, w, h = boxes[i]
    #         label = str(classes[class_ids[i]])
    #         confidence = str(round(confidences[i],2))
    #         color = colors[i]
    #         cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
    #         cv2.putText(img, label + " " + confidence, (x, y+20), font, 2, (255,255,255), 2)
    #         return_label += label

#         cv2.imshow('Image', img)
#         key = cv2.waitKey(1)
#         if key==27:
#             break
    if return_label == '':
        return_label = 'Find Nothing'
    return return_label
