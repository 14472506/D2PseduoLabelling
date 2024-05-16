"""
Detials
"""
# Baseline Detectron2 Imports
import detectron2.utils.comm as comm # What does this do 
from detectron2.checkpoint import DetectionCheckpointer # is the method for saving models and data around the model
from detectron2.config import get_cfg # is the yacs config structure
from detectron2.engine import default_argument_parser, default_setup, launch # are the defaul args parser for detectron2, the default setup out of the box and a function that handles the launching of the training process 
from detectron2 import model_zoo

# For further model development other stuff goes here
# get the traininer
from baseline_dev.engine.trainer import MyTrainer
from baseline_dev.config import add_my_config

# This is the hacky way to register architectures that was talked about in ubteacher
from baseline_dev.modeling.my_rcnn import MyGeneralizedRCNN


def setup(args):
    cfg = get_cfg()
    add_my_config(cfg)
    # adding this for time being, FIND A WAY AROUND
    #cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    # any additional stuff to add to config goes here in an add_this_config(cfg) function call
    cfg.merge_from_file(args.config_file) # merges baseline and added files together
    cfg.merge_from_list(args.opts) # merges additional changes from args together
    cfg.freeze() # fixes config at current state
    default_setup(cfg, args) 
    return cfg

def main(args):
    cfg = setup(args) # get the config in running order based on setup function
    # if cfg.SOMETHINGNET.Trainer == "Something":
    #   Trainer = Somethingtrainer
    # else:
    Trainer = MyTrainer

    # if evaluating model based on cfg
    if args.eval_only:
    #   if cfg.SOMETHINGNET.Trainer == "Something"
    #       model = Trainer.build_model(cfg)
    #       other_stuff = other_setup_eval_stuff()
    # Then same as below
    # else:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        return res
    
    # otherwise training model based on cfg
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()

if __name__ == "__main__":
    # getting args
    args = default_argument_parser().parse_args()
    # printing args
    print("Command Line Args:", args)
    # structure for launch
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )

"""
example call:
python baseline_dev_train_net.py \
      --num-gpus 1 \
      --config configs/test.yaml
"""
