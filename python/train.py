from imagenet_dataset.tiny_imagenet import TinyImageNet
from model.loss_functions import PixelReconstructionLoss
from model.mae import MAE
from utils.misc import get_logger, load_config
from utils.trainer import Trainer


def train():
    config = load_config("config/mae_config.yaml")
    print(config)
    logger = get_logger(config["LOG_DIR"])
    trainer = Trainer(config, logger)

    model = MAE(
        patch_kernel_size=config["MODEL"]["patch_kernel_size"],
        img_size=config["MODEL"]["img_size"],
        embed_size=config["MODEL"]["embed_size"],
        mask_fraction=config["MODEL"]["mask_fraction"],
    )
    trainer.set_model(model)

    train_dataset = TinyImageNet(root=config["DATA"]["root"], split="train")
    val_dataset = TinyImageNet(root=config["DATA"]["root"], split="val")
    trainer.set_dataset(train_dataset, val_dataset, data_config=config["DATA"])
    trainer.set_optimizer(optim_config=config["OPTIM"])
    trainer.set_loss_function(loss_fn=PixelReconstructionLoss())
    trainer.save_checkpoint()
    trainer.load_latest_checkpoint()
    trainer.train()


if __name__ == "__main__":
    train()
