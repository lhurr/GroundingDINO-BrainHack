data_aug_scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
data_aug_max_size = 1333
data_aug_scales2_resize = [400, 500, 600]
data_aug_scales2_crop = [384, 600]
data_aug_scale_overlap = None
batch_size = 1
modelname = "groundingdino"
backbone = "swin_B_384_22k"
position_embedding = "sine"
pe_temperatureH = 20
pe_temperatureW = 20
return_interm_indices = [1, 2, 3]
backbone_freeze_keywords = None
enc_layers = 6
dec_layers = 6
pre_norm = False
dim_feedforward = 2048
hidden_dim = 256
dropout = 0.0
nheads = 8
num_queries = 900
query_dim = 4
num_patterns = 0
num_feature_levels = 4
enc_n_points = 4
dec_n_points = 4
two_stage_type = "standard"
two_stage_bbox_embed_share = False
two_stage_class_embed_share = False
transformer_activation = "relu"
dec_pred_bbox_embed_share = True
dn_box_noise_scale = 1.0
dn_label_noise_ratio = 0.5
dn_label_coef = 1.0
dn_bbox_coef = 1.0
embed_init_tgt = True
dn_labelbook_size = 2000
max_text_len = 256
text_encoder_type = "bert-base-uncased"
use_text_enhancer = True
use_fusion_layer = True
use_checkpoint = True
use_transformer_ckpt = True
use_text_cross_attention = True
text_dropout = 0.0
fusion_dropout = 0.0
fusion_droppath = 0.1
sub_sentence_present = True

max_labels = 50                               # pos + neg

lr = 0.0003                                   # base learning rate

backbone_freeze_keywords = None               # only for gdino backbone

freeze_keywords = ['bert']                    # for whole model, e.g. ['backbone.0', 'bert'] for freeze visual encoder and text encoder

lr_backbone = 1e-05                           # specific learning rate

lr_backbone_names = ['backbone.0', 'bert']

lr_linear_proj_mult = 1e-05

lr_linear_proj_names = ['ref_point_head', 'sampling_offsets']

weight_decay = 0.0001

param_dict_type = 'ddetr_in_mmdet'

ddetr_lr_param = False

epochs = 15

lr_drop = 4

save_checkpoint_interval = 1

clip_max_norm = 0.1

onecyclelr = False

multi_step_lr = False

lr_drop_list = [4, 8]

frozen_weights = None

dilation = False

pdetr3_bbox_embed_diff_each_layer = False

pdetr3_refHW = -1

random_refpoints_xy = False

fix_refpoints_hw = -1

dabdetr_yolo_like_anchor_update = False

dabdetr_deformable_encoder = False

dabdetr_deformable_decoder = False

use_deformable_box_attn = False

box_attn_type = 'roi_align'

dec_layer_number = None

decoder_layer_noise = False

dln_xy_noise = 0.2

dln_hw_noise = 0.2

add_channel_attention = False

add_pos_value = False

two_stage_pat_embed = 0

two_stage_add_query_num = 0

two_stage_learn_wh = False

two_stage_default_hw = 0.05

two_stage_keep_all_tokens = False

num_select = 300

batch_norm_type = 'FrozenBatchNorm2d'

masks = False

aux_loss = True

set_cost_class = 1.0

set_cost_bbox = 5.0

set_cost_giou = 2.0

cls_loss_coef = 2.0

bbox_loss_coef = 5.0

giou_loss_coef = 2.0

enc_loss_coef = 1.0

interm_loss_coef = 1.0

no_interm_box_loss = False

mask_loss_coef = 1.0

dice_loss_coef = 1.0

focal_alpha = 0.25

focal_gamma = 2.0

decoder_sa_type = 'sa'

matcher_type = 'HungarianMatcher'

decoder_module_seq = ['sa', 'ca', 'ffn']

nms_iou_threshold = -1

dec_pred_class_embed_share = True





match_unstable_error = True

use_ema = False

ema_decay = 0.9997

ema_epoch = 0

use_detached_boxes_dec_out = False

use_coco_eval = False

label_list = ['black and brown camouflage helicopter', 'black and orange drone', 'black and white cargo aircraft', 'black and white commercial aircraft', 'black and white missile', 'black and yellow drone', 'black and yellow missile', 'black camouflage fighter jet', 'black cargo aircraft', 'black drone', 'black fighter jet', 'black fighter plane', 'black helicopter', 'blue and green fighter plane', 'blue and grey fighter jet', 'blue and red commercial aircraft', 'blue and red light aircraft', 'blue and white commercial aircraft', 'blue and white helicopter', 'blue and white light aircraft', 'blue and white missile', 'blue and yellow fighter jet', 'blue and yellow helicopter', 'blue camouflage fighter jet', 'blue commercial aircraft', 'blue helicopter', 'blue missile', 'blue, yellow, and black helicopter', 'blue, yellow, and green fighter plane', 'blue, yellow, and white cargo aircraft', 'green and black camouflage helicopter', 'green and black missile', 'green and brown camouflage fighter jet', 'green and brown camouflage fighter plane', 'green and brown camouflage helicopter', 'green and grey helicopter', 'green and white fighter plane', 'green and yellow fighter plane', 'green camouflage helicopter', 'green fighter plane', 'green helicopter', 'green light aircraft', 'green missile', 'grey and black fighter plane', 'grey and black helicopter', 'grey and green cargo aircraft', 'grey and red commercial aircraft', 'grey and red fighter jet', 'grey and red missile', 'grey and white fighter plane', 'grey and white light aircraft', 'grey and yellow fighter plane', 'grey camouflage fighter jet', 'grey cargo aircraft', 'grey commercial aircraft', 'grey drone', 'grey fighter jet', 'grey fighter plane', 'grey helicopter', 'grey light aircraft', 'grey missile', 'grey, red, and blue commercial aircraft', 'orange and black fighter jet', 'orange light aircraft', 'red and black drone', 'red and grey missile', 'red and white fighter jet', 'red and white fighter plane', 'red and white helicopter', 'red and white light aircraft', 'red and white missile', 'red fighter jet', 'red fighter plane', 'red helicopter', 'red light aircraft', 'red, white, and blue fighter jet', 'red, white, and blue light aircraft', 'silver and blue fighter plane', 'silver fighter plane', 'white and black cargo aircraft', 'white and black drone', 'white and black fighter jet', 'white and black fighter plane', 'white and black helicopter', 'white and black light aircraft', 'white and blue cargo aircraft', 'white and blue commercial aircraft', 'white and blue fighter jet', 'white and blue fighter plane', 'white and blue helicopter', 'white and blue light aircraft', 'white and grey helicopter', 'white and orange commercial aircraft', 'white and orange light aircraft', 'white and red commercial aircraft', 'white and red fighter jet', 'white and red fighter plane', 'white and red helicopter', 'white and red light aircraft', 'white and red missile', 'white and yellow commercial aircraft', 'white cargo aircraft', 'white commercial aircraft', 'white drone', 'white fighter jet', 'white fighter plane', 'white helicopter', 'white light aircraft', 'white missile', 'white, black, and grey missile', 'white, black, and red drone', 'white, blue, and red commercial aircraft', 'white, red, and blue commercial aircraft', 'white, red, and green fighter plane', 'yellow and black fighter plane', 'yellow and green helicopter', 'yellow and red light aircraft', 'yellow commercial aircraft', 'yellow fighter jet', 'yellow fighter plane', 'yellow helicopter', 'yellow light aircraft', 'yellow missile', 'yellow, black, and red helicopter', 'yellow, red, and blue fighter plane', 'yellow, red, and grey helicopter']
dn_scalar = 100