Installation：

mmcv: pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/{cu_version}/{torch_version}/index.html
      Please replace {cu_version} and {torch_version} in the url to your desired one.
      
      For example: pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.9.0/index.html

mmdet: pip install mmdet 

Use:

Step 1: python detectron2_outvia3.py path/to/your/faster_rcnn_r50_fpn_2x_coco.py /
        path/to/your/faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_20200504_210434-a5d8aa15.pth /
        --input path/to/your/keyframes/*/*.jpg /
        --gen_via3
        --output path/to/your/annotations_proposal
        --score-thr 0.5 --show
        
        For example:
        python detectron2_outvia3.py E:\ava\faster_rcnn_r50_fpn_2x_coco.py 
        E:\ava\faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_20200504_210434-a5d8aa15.pth
        --input E:\ava\org_img\*.jpg
        --gen_via3
        --output E:\ava\annotations_proposal
        --score-thr 0.5
        --show
        
        faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_20200504_210434-a5d8aa15.pth can be download here:
        https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_2x_coco/faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_20200504_210434-a5d8aa15.pth
        
You can get a JSON file containing the coordinate box and score. For example: self-made_ava_dataset_tool/annotations_proposal/org_img_proposal.json

Step2: Make annotation files in AVA dataset V2.1 format.    For example: self-made_ava_dataset_tool/ann_csv/ground_truth.csv
       Via3 can be download here: https://www.robots.ox.ac.uk/~vgg/software/via/.
       Open the via3/via_image_annotator.html.Then,
       upload the image and the JSON file generated in step 1.
       Then You can then adjust candidate boxes and delete unwanted candidates while tagging.
       Save ground truth json,and run gt_json2csv.py.
       The aciton kinds you can change in detectron2_outvia3.py.

Step3(optional)：slowfast needs predicted boxes. You can run predicted_json2csv.py to get it.
                In this step,You just delete the proposals you don't need, and you don't have to resize or reposition them.
                For example: self-made_ava_dataset_tool/ann_csv/predicted_ann.csv






        
        
