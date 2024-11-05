import logging
import os
from argparse import Namespace
import numpy as np
import torch
from omegaconf import DictConfig

from fairseq import distributed_utils, utils, checkpoint_utils
from fairseq.utils import reset_logging

from cdgm_textgen import options
from cdgm_textgen.dataclass.utils import convert_namespace_to_omegaconf

import faiss

logger = logging.getLogger("cdgm_textgen_cli.build_datastore")
torch.set_grad_enabled(False)

def main(cfg: DictConfig, override_args=None):
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
    index = faiss.IndexIDMap2(faiss.IndexFlatL2(model.decoder.output_embed_dim))

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

        data_folder = os.path.dirname(cfg.common_eval.path)
        datastore_file = os.path.join(data_folder, f"{subset}.datastore")

        for i, sample in enumerate(itr):
            sample = utils.move_to_cuda(sample) if use_cuda else sample
            #we are interested in the last layer only
            features, _ = model.extract_features(**sample["net_input"])
            target = model.get_targets(sample, features)
            sample_size = sample["target"].size(-1)
            for j in range(sample_size):
                store_token_idx = target[:,j].ne(task.target_dictionary.pad()).nonzero().squeeze(-1)
                store_target = torch.index_select(target[:,j], 0, store_token_idx)
                store_features = torch.index_select(features[:,j], 0, store_token_idx)
                # store token_idx and features
                index.add_with_ids(store_features.cpu(),store_token_idx.cpu())
                print(f"batch {i}, {len(store_features)} tokens is saved to a datastore")

        # save the index
        faiss.write_index(index, datastore_file)
if __name__ == "__main__":
    parser = options.get_validation_parser()
    args = options.parse_args_and_arch(parser)

    # only override args that are explicitly given on the command line
    override_parser = options.get_validation_parser()
    override_args = options.parse_args_and_arch(override_parser, suppress_defaults=True)

    distributed_utils.call_main(
        convert_namespace_to_omegaconf(args), main, override_args=override_args
    )
