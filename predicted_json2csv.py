import json
import cv2 as cv
import os
import csv


'''
    This script is used to get the pre-detected CSV required in slowfast
'''

json_path = r"E:\ava\via_project_test.json"
keyframes_path = 'E:\\ava\\org_img\\'
# The image is loaded to get the total keyframes under the file and the image H W to normalize coordinates

img_dir = os.listdir(keyframes_path)
img_nums = len(img_dir)   # Get the number of pictures
img_0 = img_dir[0]   # The H and W of the same video frame are the same
name = img_0.split('_')[0]   # Get picture name
img = cv.imread(keyframes_path+img_0)

height = img.shape[0]
width = img.shape[1]

start_time = 901  # In order to write out the timestamp that you want

# Create CSV
csvFile = open("E:\\ava\\ann_csv\\predicted_ann.csv", "w+", encoding="gbk")
CSVwriter = csv.writer(csvFile)

with open(json_path) as f:
    data = json.load(f)
    img_data = data['metadata']  # The desired coordinates and action ID are under the dictionary
    # print(img_data)
    # print(len(img_data))
    # print(img_data.items())


    for i, (k, v) in enumerate(img_data.items()):
        if i in range(0, len(img_data)-img_nums):   # len(img_data) - img_numbers。 Remove useless data
            # print(k, v)

            bboxes = v['xy']    # obtain x1, y1, x2, y2,
                                # The coordinates in the JSON file are the upper-left coordinates and the width and height of the box
            vid = int(v['vid'])  # To write out the timestamp automatically
            # print(vid)
            score = v['score'][0]
            # print(score)
            t = start_time + vid
            x1 = round(bboxes[1]/width, 3)
            y1 = round(bboxes[2]/height, 3)
            x2 = round((bboxes[3] + x1)/width, 3)
            y2 = round((bboxes[4] + y1)/height, 3)
            action_id = v['av'].items()   # Get action ID
            for (_, id) in action_id:
                # id = id
                id = id.replace(',', '')   # Handling multiple labels
                # print(id)
                for _, id in enumerate(id):
                    CSVwriter.writerow([name, t, x1, y1, x2, y2, id, score])
                    # print(name, t, x1, y1, x2, y2, id)

csvFile.close()
