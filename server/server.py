import os
import uuid
import statistics
from flask import Flask, request, flash, send_from_directory
from werkzeug.utils import secure_filename
import torch
from matplotlib import pyplot as plt
import numpy as np
import cv2
import easyocr
import pytesseract

UPLOAD_FOLDER = '/Users/dungnd/Desktop/Workspace/AI/license-plate-detection/server/file_upload'
BOUNDING_BOX_FOLDER = '/Users/dungnd/Desktop/Workspace/AI/license-plate-detection/server/bounding_box'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['BOUNDING_BOX_FOLDER'] = BOUNDING_BOX_FOLDER

EASY_OCR = easyocr.Reader(['en'])  # initiating easyocr
region_threshold = 0.2
img_path = ''
ONE_LINE = 0
TWO_LINE = 1

model = torch.hub.load('ultralytics/yolov5', 'custom',
                       path='/Users/dungnd/Desktop/Workspace/AI/license-plate-detection/yolov5/runs/train/exp2/weights/best.pt', force_reload=True)


def pre_processing_img(crop_img):
    # grayscale region within bounding box
    gray = cv2.cvtColor(crop_img, cv2.COLOR_RGB2GRAY)

    # resize image to three times as large as original for better readability
    gray = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)

    # perform gaussian blur to smoothen image
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # threshold the image using Otsus method to preprocess for tesseract
    ret, binary = cv2.threshold(
        blur, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # perform another blur on character region
    result = cv2.medianBlur(binary, 5)

    return result

def delete_all_cropped_and_bd():
    folder_path = '/Users/dungnd/Desktop/Workspace/AI/license-plate-detection/server/crop_images'
    folder_path2 = '/Users/dungnd/Desktop/Workspace/AI/license-plate-detection/server/bounding_box'

    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path):
            os.remove(file_path)
    
    for file_name in os.listdir(folder_path2):
        file_path = os.path.join(folder_path2, file_name)
        if os.path.isfile(file_path):
            os.remove(file_path)

def save_cropped_image(img, results, bd_id):
    # Save cropped image
    delete_all_cropped_and_bd()
    lp_list = results.pandas().xywh[0].values.tolist()
    for idx, lp in enumerate(lp_list):
        x, y, w, h = lp[0], lp[1], lp[2], lp[3]
        x1 = int(x - w / 2)
        y1 = int(y - h / 2)
        x2 = int(x + w / 2)
        y2 = int(y + h / 2)
        crop_img = img[int(y-h//2-5):int(y+h//2+5),
                       int(x-w//2-5):int(x+w//2+5)]
        crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
        result = crop_img
        cv2.imwrite("/Users/dungnd/Desktop/Workspace/AI/license-plate-detection/server/crop_images/plate_" +
                    str(idx+1) + ".jpeg", result)
        print("Save cropped image - crop_images/plate_" + str(idx+1) + ".jpeg")
        
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 4)
    cv2.imwrite("/Users/dungnd/Desktop/Workspace/AI/license-plate-detection/server/bounding_box/" + bd_id +".jpeg", cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

def read_plate(plate_type):
    folder_path = '/Users/dungnd/Desktop/Workspace/AI/license-plate-detection/server/crop_images'
    image_names = [f for f in os.listdir(
        folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    plates = []
    for img_path in image_names:
        img_path = os.path.join(folder_path, img_path)
        img = cv2.imread(img_path)
        if (plate_type == ONE_LINE):
            img = pre_processing_img(img)
            # texts = pytesseract.image_to_string(
            #     img, lang="eng", config="--psm 6")
            df = pytesseract.image_to_data(img, output_type='data.frame')
            df = df[df.conf != -1]
            lines = df.groupby('block_num')['text'].apply(list)
            conf = df.groupby(['block_num'])['conf'].mean()
            
            plate_text = lines.iloc[0][0]
            ocr_confidence = conf[1]
            plates.append((plate_text, ocr_confidence))
        elif (plate_type == TWO_LINE):
            ocr_result = EASY_OCR.readtext(img)
            texts = filter_text(img, ocr_result)
            plates.append(texts)
    return plates

def filter_text(region, ocr_result, region_threshold=0.2):
    rectangle_size = region.shape[0]*region.shape[1]
    plate = []
    for result in ocr_result:
        length = np.sum(np.subtract(result[0][1], result[0][0]))
        height = np.sum(np.subtract(result[0][2], result[0][1]))
        if length*height / rectangle_size > region_threshold:
            plate.append(result)
    return plate

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
        img_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = model(img)

        bd_id = str(uuid.uuid4())
        save_cropped_image(img, results, bd_id)

        plate_type = request.args.get('plate_type')
        plates = read_plate(int(plate_type))
        output = []
        for (idx, plate) in enumerate(plates):
            plate_text = " ".join([x[1] for x in plate]) if int(plate_type) == TWO_LINE else plate[0]
            ocr_confidence = statistics.mean([x[2] for x in plate]) if int(plate_type) == TWO_LINE else plate[1]
            output.append({"plate_text": plate_text,
                           "ocr_confidence": ocr_confidence,
                           "object_detection_confidence": results.pandas().xywh[0]['confidence'][idx]
                        })

    return {"plates": output, "image": bd_id + ".jpeg"}

@app.route('/image/<name>')
def view_image(name):
    return send_from_directory(app.config['BOUNDING_BOX_FOLDER'], name)

if __name__ == '__main__':
    app.run()
