import torch
import numpy as np

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