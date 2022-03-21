# VITON HD

NOTE: copy datasets/* and checkpoints/* to the working directories from the below google drive link:
https://drive.google.com/drive/folders/0B8kXrnobEVh9fnJHX3lCZzEtd20yUVAtTk5HdWk2OVV0RGl6YXc0NWhMOTlvb1FKX3Z1OUk?resourcekey=0-OIXHrDwCX8ChjypUbJo4fQ

To test:
1. update the person and cloth image pair in datasets/test_pairs.txt. NOTE: pick person image from datasets/test/image/ and cloth image from  datasets/test/cloth/
2. for new person image, segmentation image should be present in image-parse and pose image should be present in datasets/test/openpose-img/ and pose body key should be present in datasets/test/openpose-json/
3. for new cloth image, cloth-mask should be present in datasets/test/cloth-mask/
NOTE: We are keeping the clothes collection as given for simplicity.

Command to run:
python3 test.py --name res
The result person images with new clothes should appear in results/res/ directory



