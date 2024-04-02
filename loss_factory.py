from datasets.cubicasa5k.losses.uncertainty_loss import UncertaintyLoss


def factory(cfg):
    if cfg.model.name == 'cubicasa':
        if cfg.model.loss == 'uncertainty':
            return UncertaintyLoss(cfg.model.input_slice)
        else:
            raise NotImplementedError(f"Model {cfg.model.name} with loss {cfg.model.loss}")
    else:
        raise NotImplementedError(f"Model {cfg.model.name} with loss {cfg.model.loss}")
