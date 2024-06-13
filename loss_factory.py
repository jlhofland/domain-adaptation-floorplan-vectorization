from datasets.cubicasa5k.losses.uncertainty_loss import UncertaintyLoss


def factory(cfg):
    if cfg.model.loss == 'UncertaintyLoss':
        return UncertaintyLoss(cfg.model.input_slice, use_mmd=cfg.model.use_mmd)
    else:
        raise NotImplementedError(f"Model {cfg.model.name} with loss {cfg.model.loss}")
