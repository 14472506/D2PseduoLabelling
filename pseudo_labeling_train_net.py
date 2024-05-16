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
from pseudo_labeling.engine.trainer import PseudoTrainer
from pseudo_labeling.config import add_pseudo_config

# This is the hacky way to register architectures that was talked about in ubteacher
from pseudo_labeling.modelling.my_rcnn import MyGeneralizedRCNN

# functions
def setup(args):
    cfg = get_cfg()
    add_pseudo_config(cfg)
    cfg.merge_from_file(args.config_file) # merges baseline and added files together
    cfg.merge_from_list(args.opts) # merges additional changes from args together
    cfg.freeze() # fixes config at current state
    default_setup(cfg, args) 
    return cfg

def main(args):
    # get config
    cfg = setup(args) # get the config in running order based on setup function
    # get trainer
    Trainer = PseudoTrainer
    # if evaluating model based on cfg
    if args.eval_only:
        # get model
        model = Trainer.build_model(cfg)
        # get checkpoint
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        # get results
        res = Trainer.test(cfg, model)
        # return results
        return res
    
    # otherwise training model based on cfg
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()

# execution
if __name__ == "__main__":
    
    # getting args currently fine, may need additional options see Detectron2 for more
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