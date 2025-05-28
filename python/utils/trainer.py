import torch
from pathlib import Path
from torch.utils.data import DataLoader
from utils.misc import get_device, save_images
from tqdm import tqdm


class Trainer:
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.epoch = 0

        self.ckpt_dir = Path(config["CKPT_DIR"])
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.device = get_device()

    def set_model(self, model):
        self.model = model
        self.model.to(self.device)
        self.logger.info("Model:")
        self.logger.info(self.model)

    def save_checkpoint(self):
        model_path = Path(self.ckpt_dir) / f"ckpt_{str(self.epoch).zfill(4)}.pt"
        torch.save(
            {
                "epoch": self.epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            model_path,
        )
        self.logger.info(f"Saved checkpoint in: {model_path}")

    def load_latest_checkpoint(self):
        if not self.ckpt_dir.exists():
            self.logger.info("No checkpoint directory found.")
            return None

        ckpt_files = sorted(self.ckpt_dir.glob("ckpt_*.pt"))
        if not ckpt_files:
            self.logger.info("No checkpoints found.")
            return None

        latest_ckpt = max(ckpt_files, key=lambda x: int(x.stem.split("_")[1]))
        self.logger.info(f"Loading checkpoint: {latest_ckpt}")

        checkpoint = torch.load(latest_ckpt, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.epoch = checkpoint.get("epoch", 0)
        return latest_ckpt

    def set_dataset(self, train_dataset, val_dataset, data_config):
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.data_config = data_config

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=data_config["batch_size"],
            shuffle=True,
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=1,
            shuffle=False,
        )
        self.logger.info(f"Train Dataset: {self.train_dataset}")
        self.logger.info(f"Val Dataset: {self.val_dataset}")

    def set_optimizer(self, optim_config):
        self.optim_config = optim_config
        if self.optim_config["optimizer"] == "adam":
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), lr=self.optim_config["lr"]
            )
        else:
            raise ValueError
        self.logger.info(f"Optimizer: {self.optimizer}")

    def set_loss_function(self, loss_fn):
        self.loss_fn = loss_fn.to(self.device)
        self.logger.info(f"Loss function {self.loss_fn}")

    def train(self):
        for curr_epoch in range(self.optim_config["num_epochs"]):
            self.epoch = curr_epoch
            self.train_one_epoch()
            self.evaluate_model()

    def train_one_epoch(self, eval_every_iter=500):
        self.model.train()
        with tqdm(enumerate(self.train_loader), desc=f"Epoch {self.epoch}") as pbar:
            for n_iter, (data, labels) in pbar:
                self.optimizer.zero_grad()
                data = data.to(self.device)
                output_dict = self.model(data)
                loss = self.loss_fn(
                    output_dict["pixel_preds"],
                    output_dict["x_patch"],
                    output_dict["pred_token_mask"],
                )
                loss.backward()
                self.optimizer.step()
                pbar.set_postfix({"loss": loss.item()})
                if n_iter % eval_every_iter == 0:
                    self.evaluate_model(n_iter=n_iter)
        self.save_checkpoint()

    def evaluate_model(self, max_num_imgs=5, n_iter=None):
        self.model.eval()
        all_pixel_preds = []
        all_x_patch = []
        for idx, (data, labels) in enumerate(self.val_loader):
            data = data.to(self.device)
            output_dict = self.model(data)
            pixel_preds = output_dict["pixel_preds"]
            x_patch = output_dict["x_patch"]
            all_pixel_preds.append(self.model.patch_image.fold(pixel_preds, (64, 64)))
            all_x_patch.append(self.model.patch_image.fold(x_patch, (64, 64)))
            if idx>max_num_imgs:
                break

        save_images(
            all_pixel_preds,
            all_x_patch,
            save_dir=self.config["IMG_OUT_DIR"],
            idx=f"{str(self.epoch).zfill(3)}_{str(n_iter).zfill(5)}",
        )
        self.model.train()

    def train_on_single_batch(self, num_epochs=1000):
        """
        Train the model on a single batch for a number of epochs to check if it can overfit.
        """
        self.model.train()
        # Get a single batch
        data_iter = iter(self.train_loader)
        data, labels = next(data_iter)
        data = data.to(self.device)
        labels = labels.to(self.device) if hasattr(labels, 'to') else labels
        for epoch in range(num_epochs):
            self.optimizer.zero_grad()
            output_dict = self.model(data)
            loss = self.loss_fn(
                output_dict["pixel_preds"], output_dict["x_patch"], output_dict["pred_token_mask"]
            )
            loss.backward()
            self.optimizer.step()
            print(f"[Single Batch] Epoch {epoch+1}/{num_epochs} - Loss: {loss.item():.6f}")
            # Optionally evaluate on the same batch
            self.model.eval()
            if epoch % 10 == 0:
                with torch.no_grad():
                        output_dict = self.model(data)
                        pixel_preds = output_dict["pixel_preds"]
                        x_patch = output_dict["x_patch"]
                        save_images(
                            self.model.patch_image.fold(pixel_preds, (64, 64)),
                            self.model.patch_image.fold(x_patch, (64, 64)),
                            save_dir=self.config["IMG_OUT_DIR"],
                            idx=f"single_batch_{str(self.epoch).zfill(3)}",
                        )
            self.model.train()
