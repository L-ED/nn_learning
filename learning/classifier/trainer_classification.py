import torch
import numpy as np
from ..trainer import Trainer

class Trainer_classification(Trainer):

    def calculate_metric(self, predict, target):
        pred = predict.argmax(dim=1)
        pred = pred.detach().cpu().numpy()
        trgt = target.detach().cpu().numpy()
        return super().calculate_metric(pred, trgt)



def train_(trainer):

    trainer.model.train()

    with trainer.profiler as profiler:

        for batch in trainer.trainloader:
            imgs, target = trainer.prepare_batch(batch)

            pred = trainer.model(imgs)

            loss= trainer.loss_fn(pred, target)

            trainer.train_metrics[trainer.metric](
                trainer.calculate_metric(pred, target)
            )
            trainer.train_metrics["loss"](loss.item())

            trainer.optimizer_step(loss)
        
        if trainer.scheduler is not None:
            trainer.scheduler.step(
                np.mean(
                    trainer.train_metrics["loss"].epoch_metric
                )
            )

        profiler.step()

    trainer.finalize_epoch("train")


def validate_(trainer):

    trainer.model.eval()

    with torch.no_grad():
        for batch in trainer.valloader:
            imgs, target = trainer.prepare_batch(batch)

            pred = trainer.model(imgs)
            loss = trainer.loss_fn(pred, target)

            trainer.val_metrics[trainer.metric](
                trainer.calculate_metric(pred, target)
            )
            trainer.val_metrics["loss"](loss.item())

        trainer.finalize_epoch("validate")
    