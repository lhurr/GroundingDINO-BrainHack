# GroundingDINO setup
1. Modify config/dataset_OD.json, config/label_map.json and config/test.json


# Installation
1. pip install -r requirements.txt 
2. cd models/GroundingDINO/ops
3. python3 setup.py build install --user
4. git clone https://github.com/IDEA-Research/GroundingDINO.git
5. cd GroundingDINO/
6. pip install --no-build-isolation -e . 
