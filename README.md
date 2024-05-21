# GroundingDINO setup
1. Modify config/dataset_OD.json, config/label_map.json and config/test.json
2. Download [BERT](https://www.kaggle.com/datasets/virajjayant/bertbaseuncased) & [Grounding Dino weights](https://huggingface.co/alexgenovese/background-workflow/blob/1cbf8c24aa8a2e8d5ca6871800442b35ff6f9d48/groundingdino_swint_ogc.pth)
3. Commands to run & inference using SwinT model
   - Fine tune: ```bash train_dist.sh 1 config/cfg_gdinoT.py config/dataset_OD.json logs```
   - Inference: ```python tools/inference_on_a_image.py   -c tools/GroundingDINO_SwinT_OGC.py   -p logs/checkpoint.pth   -i <IMG_PATH>.jpg   -t "<CLASS_NAMES>"   -o output```


# Installation
1. pip install -r requirements.txt 
2. cd models/GroundingDINO/ops
3. python3 setup.py build install --user
4. git clone https://github.com/IDEA-Research/GroundingDINO.git
5. cd GroundingDINO/
6. pip install --no-build-isolation -e . 
