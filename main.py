from loguru import logger

from opt import get_opts
from trainer.stage_1 import Trainer_1
from trainer.stage_2 import Trainer_2
from trainer.stage_3 import Trainer_3
from trainer.stage_4 import Trainer_4
                

def main(args):
    logger.info('stage 1 begin...')
    trainer_1 = Trainer_1(args)
    trainer_1.run()
    logger.info('stage 2 begin...')
    trainer_2 = Trainer_2(args)
    trainer_2.run()
    logger.info('stage 3 begin...')
    trainer_3 = Trainer_3(args)
    trainer_3.run()
    logger.info('stage 4 begin...')
    trainer_4 = Trainer_4(args)
    trainer_4.run()
    logger.info('train end')

if __name__ == "__main__":
    logger.info('starting job...')
    args = get_opts()
    main(args=args)