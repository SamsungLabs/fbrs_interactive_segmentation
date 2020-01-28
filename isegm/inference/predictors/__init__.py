from .base import BasePredictor
from .brs import InputBRSPredictor, FeatureBRSPredictor
from .brs_functors import InputOptimizer, ScaleBiasOptimizer
from isegm.inference.transforms import ZoomIn


def get_predictor(net, brs_mode,
                  prob_thresh=0.49,
                  with_flip=True,
                  zoom_in_params=dict(),
                  predictor_params=None,
                  brs_opt_func_params=None,
                  lbfgs_params=None):
    lbfgs_params_ = {
        'm': 20,
        'factr': 0,
        'pgtol': 1e-8,
        'maxfun': 20,
    }

    predictor_params_ = {
        'num_max_points': 20,
        'optimize_after_n_clicks': 2
    }

    if zoom_in_params is not None:
        zoom_in = ZoomIn(**zoom_in_params)
    else:
        zoom_in = None

    if lbfgs_params is not None:
        lbfgs_params_.update(lbfgs_params)
    lbfgs_params_['maxiter'] = 2 * lbfgs_params_['maxfun']

    if brs_opt_func_params is None:
        brs_opt_func_params = dict()

    if brs_mode == 'NoBRS':
        net.collect_params().setattr('grad_req', 'null')
        if predictor_params is not None:
            predictor_params_.update(predictor_params)
        predictor = BasePredictor(net, zoom_in=zoom_in, with_flip=with_flip, **predictor_params_)
    elif brs_mode.startswith('f-BRS'):
        net.feature_extractor.collect_params().setattr('grad_req', 'null')

        predictor_params_.update({
            'net_clicks_limit': 8,
        })
        if predictor_params is not None:
            predictor_params_.update(predictor_params)

        insertion_mode = {
            'f-BRS-A': 'after_c4',
            'f-BRS-B': 'after_aspp',
            'f-BRS-C': 'after_deeplab'
        }[brs_mode]

        opt_functor = ScaleBiasOptimizer(prob_thresh=prob_thresh,
                                         with_flip=with_flip,
                                         optimizer_params=lbfgs_params_,
                                         **brs_opt_func_params)

        predictor = FeatureBRSPredictor(net,
                                        opt_functor=opt_functor,
                                        with_flip=with_flip,
                                        insertion_mode=insertion_mode,
                                        zoom_in=zoom_in,
                                        **predictor_params_)
    elif brs_mode == 'RGB-BRS' or brs_mode == 'DistMap-BRS':
        net.collect_params().setattr('grad_req', 'write')

        use_dmaps = brs_mode == 'DistMap-BRS'

        predictor_params_.update({
            'net_clicks_limit': 5,
        })
        if predictor_params is not None:
            predictor_params_.update(predictor_params)

        opt_functor = InputOptimizer(prob_thresh=prob_thresh,
                                     with_flip=with_flip,
                                     optimizer_params=lbfgs_params_,
                                     **brs_opt_func_params)

        predictor = InputBRSPredictor(net,
                                      optimize_target='dmaps' if use_dmaps else 'rgb',
                                      opt_functor=opt_functor,
                                      with_flip=with_flip,
                                      zoom_in=zoom_in,
                                      **predictor_params_)
    else:
        raise NotImplementedError

    return predictor
