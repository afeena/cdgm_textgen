import logging
import os
from argparse import Namespace
import numpy as np
import torch
from omegaconf import DictConfig
import argparse

from fairseq import distributed_utils, utils, checkpoint_utils
from fairseq.utils import reset_logging

from cdgm_textgen import options
from cdgm_textgen.dataclass.utils import convert_namespace_to_omegaconf

import faiss

logger = logging.getLogger("cdgm_textgen_cli.build_datastore")
torch.set_grad_enabled(False)

def calc_renyi_entr(features, alpha=0.5):
    K = features @ features.t()
    v = torch.log((torch.linalg.eigvals(K) / torch.trace(K)))
    renyi_entr = (1.0 / (1.0 - alpha)) * v.pow(alpha).sum()
    return renyi_entr

def calc_avg_cosine(features):
    normalized_features = features / features.norm(p=2, dim=-1, keepdim=True)
    cosine_sim_batch = normalized_features @ normalized_features.t()
    return cosine_sim_batch.mean()
def main(cfg: DictConfig, ntotal, override_args=None):
    if isinstance(cfg, Namespace):
        cfg = convert_namespace_to_omegaconf(cfg)

    utils.import_user_module(cfg.common)

    reset_logging()

    assert (
        cfg.dataset.max_tokens is not None or cfg.dataset.batch_size is not None
    ), "Must specify batch size either with --max-tokens or --batch-size"

    use_fp16 = cfg.common.fp16
    use_cuda = torch.cuda.is_available() and not cfg.common.cpu

    if use_cuda:
        torch.cuda.set_device(cfg.distributed_training.device_id)

    if cfg.distributed_training.distributed_world_size > 1:
        data_parallel_world_size = distributed_utils.get_data_parallel_world_size()
        data_parallel_rank = distributed_utils.get_data_parallel_rank()
    else:
        data_parallel_world_size = 1
        data_parallel_rank = 0

    if override_args is not None:
        overrides = vars(override_args)
        overrides.update(eval(getattr(override_args, "model_overrides", "{}")))
    else:
        overrides = None

    # Load ensemble
    logger.info("loading model(s) from {}".format(cfg.common_eval.path))
    models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task(
        [cfg.common_eval.path],
        arg_overrides=overrides,
        suffix=cfg.checkpoint.checkpoint_suffix,
    )
    model = models[0]

    # Move models to GPU
    for model in models:
        model.eval()
        if use_fp16:
            model.half()
        if use_cuda:
            model.cuda()

    for subset in cfg.dataset.valid_subset.split(","):
        try:
            task.load_dataset(subset, combine=False, epoch=1, task_cfg=saved_cfg.task)
            dataset = task.dataset(subset)
        except KeyError:
            raise Exception("Cannot find dataset: " + subset)

        # Initialize data iterator
        itr = task.get_batch_iterator(
            dataset=dataset,
            max_tokens=cfg.dataset.max_tokens,
            max_sentences=cfg.dataset.batch_size,
            max_positions=utils.resolve_max_positions(
                task.max_positions(),
                *[m.max_positions() for m in models],
            ),
            ignore_invalid_inputs=cfg.dataset.skip_invalid_size_inputs_valid_test,
            required_batch_size_multiple=cfg.dataset.required_batch_size_multiple,
            seed=cfg.common.seed,
            num_shards=data_parallel_world_size,
            shard_id=data_parallel_rank,
            num_workers=cfg.dataset.num_workers,
            data_buffer_size=cfg.dataset.data_buffer_size,
        ).next_epoch_itr(shuffle=False)

        all_target_token_features = []
        target_token_idx = 5

        data_folder = cfg.common_eval.results_path
        out_file = os.path.join(data_folder, f"{subset}.token{target_token_idx}.features.npy")

        for i, sample in enumerate(itr):
            sample = utils.move_to_cuda(sample) if use_cuda else sample
            target_token_mask = sample["target"].eq(target_token_idx)
            if not any(target_token_mask):
                continue
            decoder_out = model(**sample["net_input"])
            target_token_features = decoder_out[-1]["features"][target_token_mask]
            all_target_token_features.append(target_token_features)

        np.save(out_file, torch.cat(all_target_token_features).cpu().numpy())



if __name__ == "__main__":
    user_parser = argparse.ArgumentParser(description="Extract features from a model")
    user_parser.add_argument("--ntotal", type=int, help="Total number of tokens to extract")
    user_args, _ = user_parser.parse_known_args()

    parser = options.get_validation_parser()
    args, _ = options.parse_args_and_arch(parser, parse_known=True)


    # only override args that are explicitly given on the command line
    override_parser = options.get_validation_parser()
    override_args  = options.parse_args_and_arch(override_parser, suppress_defaults=True, parse_known=True)


    main(
        convert_namespace_to_omegaconf(args), user_args.ntotal, override_args=override_args
    )


