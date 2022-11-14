import torch

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