from conf import get_conf
from trainer import Trainer

if __name__ == "__main__":
    conf_cmd = get_conf()
    main_trainer = Trainer(conf_cmd)

    main_trainer.train()