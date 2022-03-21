# Body Segmentation

Our Body segmentation model is based on Self-Correction-Human-Parsing:
https://github.com/PeikeLi/Self-Correction-Human-Parsing

There are three pre-trained models which may be used depending upon the number of classes each have: LIP has 20 classes, ATR has 18 and Pascal-Person-Part has 7 classes.

We are using LIP model. The weight file can be downloaded from: https://drive.google.com/file/d/1k4dllHpu0bdx38J7H28rVVLpU-kOHmnH/view

NOTE: Rename the file to final.pth and place it under checkpoints/ directory

How to Run:
1. make sure the weight file of the model is added in checkpoints directory
2. place the image inside inputs/ directory
3. Run the command:
python3 simple_extractor.py --dataset 'lip' --model-restore 'checkpoints/final.pth' --input-dir 'inputs' --output-dir 'outputs'

The result - segmentation image will be created under output/ dir.

