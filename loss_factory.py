from datasets.cubicasa5k.losses.uncertainty_loss import UncertaintyLoss


def factory(cfg):
    if cfg.model.name == 'CubiCasa':
        if cfg.model.loss == 'UncertaintyLoss':
            return UncertaintyLoss(cfg.model.input_slice, use_mmd=cfg.model.use_mmd, batch_size=cfg.train.batch_size)

    # If no model is found, raise an error
    raise NotImplementedError(f"Model {cfg.model.name} with loss {cfg.model.loss}")
