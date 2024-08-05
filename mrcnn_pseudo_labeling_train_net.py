"""
        Title: pseudo_labeling_train_net.py

  Description: Script uses Detectron2 to train carry logit symmetry, mask based pseduo labeling for 
               instance segmentation architectures.

        Notes:  - Currently under development. 
                - only contrains mask r-cnn as instance sementation architecuture. more need to be added

  Last Edited: Bradley Hurst (12/06/2024)
"""
# === imports === #
# base
import os
import argparse
# third party
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import default_argument_parser, default_setup, launch
from detectron2 import model_zoo
# local
from pseudo_labeling.engine.trainer import PseudoTrainer
from pseudo_labeling.config import add_pseudo_config
from pseudo_labeling.modelling.my_rcnn import MyGeneralizedRCNN
from pseudo_labeling.modelling import mask_head 
from pseudo_labeling.modelling import custom_roi



# === functions === #
def parse_args():
    """ 
    Argument parser adding gpu specification functionality to Detectron2's 
    defualt_argument_parse function
    """
    parser = default_argument_parser()
    parser.add_argument('--use_gpu', type=str, default='0', help='GPU id to use')
    return parser.parse_args()

def set_gpu(gpu_id):
    """ Specifies GPU for use """
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
   
def setup(args):
    """ Initialise config and ammend based on command line arguments """
    cfg = get_cfg()
    # adding argments to base config to accomodate pseudo labeling
    add_pseudo_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg

def main(args):
    """ 
    Execute training with logit mask volume based pseudo labeling based of
    the provided config file and command line arguments
    """
    # get config and trainer
    cfg = setup(args)
    Trainer = PseudoTrainer

    # if eval only carry out model evaluation based on config - no training
    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        return res

    # train model based of config and command line arguments
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()

# === Execution === #
if __name__ == "__main__":
    """ Execution of logit mask volume based pseudo labeling training or evaluation """
    # get command line arguments, setup gpu allocation and report command line arguments
    args = parse_args()
    set_gpu(args.use_gpu)
    print("Command Line Args:", args)

    # use detectron2 launcher to begin training or evaluation
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )