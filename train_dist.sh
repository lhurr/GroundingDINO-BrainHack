GPU_NUM=$1
CFG=$2
DATASETS=$3
OUTPUT_DIR=$4
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}


python3 main.py \
        --output_dir ${OUTPUT_DIR} \
        -c ${CFG} \
        --datasets ${DATASETS}  \
        --pretrain_model_path groundingdino_swint_ogc.pth \
        --options text_encoder_type=bert-base-uncased
# --resume logs/checkpoint0001.pth
        
        



# python3 tools/inference_on_a_image.py   -c tools/GroundingDINO_SwinT_OGC.py   -p logs/checkpoint0000.pth   -i ../til-ai-24-advanced/images/image_0.jpg   -t "grey missile . red, white, and blue light aircraft . green and black missile . white and red helicopter ."   -o output

# bash train_dist.sh 1 config/cfg_gdinoT.py config/dataset_OD.json logs 