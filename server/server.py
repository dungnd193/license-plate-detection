import os
from flask import Flask, request, flash
from werkzeug.utils import secure_filename
import torch
from matplotlib import pyplot as plt
import numpy as np
import cv2
import easyocr
import pytesseract

UPLOAD_FOLDER = '/Users/dungnd/Desktop/Workspace/AI/license-plate-detection/server/file_upload'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

EASY_OCR = easyocr.Reader(['en']) ### initiating easyocr
region_threshold = 0.2
img_path = ''
ONE_LINE = 0
TWO_LINE = 1

model = torch.hub.load('ultralytics/yolov5', 'custom', path='/Users/dungnd/Desktop/Workspace/AI/license-plate-detection/yolov5/runs/train/exp2/weights/best.pt', force_reload=True)

def pre_processing_img(crop_img):
    # grayscale region within bounding box
    gray = cv2.cvtColor(crop_img, cv2.COLOR_RGB2GRAY)

    # resize image to three times as large as original for better readability
    gray = cv2.resize(gray, None, fx = 3, fy = 3, interpolation = cv2.INTER_CUBIC)

    # perform gaussian blur to smoothen image
    blur = cv2.GaussianBlur(gray, (5,5), 0)

    # threshold the image using Otsus method to preprocess for tesseract
    ret, binary = cv2.threshold(blur, 127, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # perform another blur on character region
    result = cv2.medianBlur(binary, 5)

    return result

def delete_all_cropped():
    folder_path = '/Users/dungnd/Desktop/Workspace/AI/license-plate-detection/server/crop_images'

    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path):
            os.remove(file_path)

def save_cropped_image(img, results):
    # Save cropped image
    delete_all_cropped()
    lp_list = results.pandas().xywh[0].values.tolist()
    for idx, lp in enumerate(lp_list):
        x,y,w,h =  lp[0], lp[1], lp[2], lp[3]
        crop_img = img[int(y-h//2-5):int(y+h//2+5) , int(x-w//2-5):int(x+w//2+5)]
        crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
        result = crop_img
        cv2.imwrite("/Users/dungnd/Desktop/Workspace/AI/license-plate-detection/server/crop_images/plate_" + str(idx+1) + ".jpeg", result)
        print("Save cropped image - crop_images/plate_" + str(idx+1) + ".jpeg")
      
def read_plate(plate_type):
    folder_path = '/Users/dungnd/Desktop/Workspace/AI/license-plate-detection/server/crop_images'
    image_names = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    plates = []
    for img_path in image_names:
        img_path = os.path.join(folder_path, img_path)
        img = cv2.imread(img_path)
        if (plate_type == ONE_LINE): 
            ocr_result = EASY_OCR.readtext(img)
            text = filter_text(img, ocr_result)
        elif (plate_type == TWO_LINE): 
            img = pre_processing_img(img)
            text = pytesseract.image_to_string(img, lang="eng", config="--psm 6")
        
        if text != "":
            plates.append(text)
            
        
    return plates
    
def filter_text(region, ocr_result, region_threshold=0.2):
    rectangle_size = region.shape[0]*region.shape[1]
    plate = []
    for result in ocr_result:
        length = np.sum(np.subtract(result[0][1], result[0][0]))
        height = np.sum(np.subtract(result[0][2], result[0][1]))
        if length*height / rectangle_size > region_threshold:
            plate.append(result[1])
    return " ".join(plate)

def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS



@app.route('/api/image', methods=['POST'])
def upload_image():
    # check if the post request has the file part
    if 'file' not in request.files:
        flash('No file part')
        return {"message": "Error"}
    file = request.files['file']
    # If the user does not select a file, the browser submits an empty file without a filename.
    if file.filename == '':
        flash('No selected file')
        return {"message": "Error"}
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        img_path = os.path.join('/Users/dungnd/Desktop/Workspace/AI/license-plate-detection/server/file_upload', filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = model(img)
        
        save_cropped_image(img, results)
        
        
        plate_type = request.args.get('plate_type')
        plates = read_plate(int(plate_type))
        
        return {"plates": plates}


if __name__ == '__main__':
    app.run()
