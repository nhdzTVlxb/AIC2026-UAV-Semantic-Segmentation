import argparse
import datetime
import hashlib
import json
import math
import os
import re
import time
from scale_fusion.workspace_state import TEAM, CONFIG_PATH, PUBLIC_SHA, check_sources, digest, freeze_sources, load_own, local_path, write_json

def parameter_groups(model, phase, recipe):
    groups = {}
    alignment = phase['id'] == 'vehicle_alignment'
    for name, parameter in model.named_parameters():
        enabled = not alignment or 'conv_seg.vehicle_' in name
        parameter.requires_grad_(enabled)
        if not enabled:
            continue
        if alignment:
            key, rate = ('vehicle', phase['new_lr'])
        elif name.startswith('backbone.'):
            match = re.search('levels\\.(\\d+)(?:\\.blocks\\.(\\d+))?', name)
            depth = 0
            if match:
                stage = int(match[1])
                depths = [5, 5, 24, 5]
                depth = sum(depths[:stage]) + (int(match[2]) + 1 if match[2] else depths[stage])
            key, rate = (f'encoder{depth}', phase['backbone_lr'] * recipe['optimizer']['layer_decay'] ** (39 - depth))
        elif name.startswith('decode_head.scale_'):
            key, rate = ('scale_fusion', phase['new_lr'])
        else:
            key, rate = ('decoder', phase['decoder_lr'])
        decay = 0.0 if parameter.ndim == 1 or name.endswith('.bias') else recipe['optimizer']['weight_decay']
        entry = groups.setdefault((key, decay), {'name': key, 'params': [], 'lr': rate, 'initial_lr': rate, 'weight_decay': decay})
        entry['params'].append(parameter)
    return list(groups.values())

def state_hash(model, names=None):
    hasher = hashlib.sha256()
    for name, value in model.state_dict().items():
        if names is None or name in names:
            hasher.update(name.encode())
            hasher.update(value.detach().cpu().numpy().tobytes())
    return hasher.hexdigest()

def partition_samples(indices, seeds, steps, logical_workers, batch, rank, world):
    accumulation = logical_workers // world
    start = rank * accumulation
    shape = (steps, logical_workers, batch)
    return (indices.reshape(shape)[:, start:start + accumulation].reshape(-1), seeds.reshape(shape)[:, start:start + accumulation].reshape(-1))

def initial_random_streams(torch, seed, rank, world, logical_workers):
    streams = []
    for logical_rank in range(rank * (logical_workers // world), (rank + 1) * (logical_workers // world)):
        torch.manual_seed(seed + logical_rank * 7919)
        streams.append((torch.get_rng_state(), torch.cuda.get_rng_state()))
    return streams

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', required=True)
    parser.add_argument('--tag', required=True)
    parser.add_argument('--parent')
    parser.add_argument('--probe', action='store_true')
    parser.add_argument('--physical-gpus', nargs='+', type=int, required=True)
    args = parser.parse_args()
    import numpy as np
    import torch
    import torch.distributed as dist
    from torch import nn
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.utils.data import DataLoader
    from scale_fusion.pyramid.segmenting_graph import ScaleAdaptiveSegmenter
    from scale_fusion.supervision.pyramid_objective import loss_terms
    from scale_fusion.sampling.tile_passes import OfficialTiles, plan
    rank = int(os.environ['LOCAL_RANK'])
    world = int(os.environ['WORLD_SIZE'])
    recipe = json.loads(CONFIG_PATH.read_text())
    logical_workers = recipe['execution']['logical_partitions']
    if world != len(args.physical_gpus) or world not in (1, 2) or logical_workers != 2:
        raise ValueError('Use one device with two accumulations or two synchronized devices')
    accumulation = logical_workers // world
    phase = next((p for p in recipe['stages'] if p['id'] == args.stage))
    if phase['global_batch'] != 16:
        raise ValueError('Expected an effective global batch of 16')
    batch = phase['global_batch'] // logical_workers
    torch.set_num_threads(4)
    torch.cuda.set_device(rank)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision('high')
    use_checkpointing = os.environ.get('AIC3_GRADIENT_CHECKPOINTING', '0') == '1'
    dist.init_process_group('nccl', timeout=datetime.timedelta(seconds=300))
    folder = local_path('runs/' + args.tag)
    record = {'owner_team': TEAM.name, 'status': 'initializing', 'pid': os.getpid(), 'tag': args.tag, 'stage': args.stage, 'phase': phase, 'world_size': world, 'physical_gpus': args.physical_gpus, 'logical_partitions': logical_workers, 'partition_batch': batch, 'accumulation_steps': accumulation, 'effective_global_batch': batch * world * accumulation, 'single_model': True, 'checkpoints_used': 1, 'model_averaging': False, 'public_checkpoint_sha256': PUBLIC_SHA, 'optimizer_steps': 0}
    if rank == 0:
        folder.mkdir(parents=True, exist_ok=False)
        write_json(folder / 'training.json', record)
    dist.barrier(device_ids=[rank])
    try:
        torch.manual_seed(phase['seed'])
        if args.parent:
            model, parent = load_own(args.parent, checkpointing=use_checkpointing)
            parent_steps = parent['lineage_steps']
            initialization = {'own_parent': args.parent, 'own_parent_sha256': parent['checkpoint_sha256']}
        else:
            if args.stage != 'vehicle_alignment' and (not args.probe):
                raise ValueError('Joint training requires completed vehicle alignment')
            model = ScaleAdaptiveSegmenter(checkpointing=use_checkpointing)
            initialization = model.initialize_public()
            parent_steps = 0
        lineage = parent_steps + (0 if args.probe else phase['steps'])
        if lineage > 20000:
            raise ValueError('Lineage exceeds 20000 optimization steps')
        record.update(lineage_steps=lineage, initialization=initialization)
        alignment = args.stage == 'vehicle_alignment'
        groups = parameter_groups(model, phase, recipe)
        model.cuda(rank)
        model.eval() if alignment else model.train()
        for unit in model.modules():
            if isinstance(unit, nn.modules.batchnorm._BatchNorm):
                unit.eval()
        frozen_names = {n for n, p in model.named_parameters() if not p.requires_grad}
        frozen_names.update((n for n, _ in model.named_buffers()))
        frozen_before = state_hash(model, frozen_names)
        probe_before = state_hash(model) if args.probe else None
        network = DDP(model, device_ids=[rank], output_device=rank, broadcast_buffers=False, gradient_as_bucket_view=True, static_graph=True) if world > 1 else model
        torch.manual_seed(phase['seed'] + rank * 7919)
        steps = 3 if args.probe else phase['steps']
        total_count = steps * phase['global_batch']
        indices, seeds = plan(total_count, phase['seed'])
        local_indices, local_seeds = partition_samples(indices, seeds, steps, logical_workers, batch, rank, world)
        data = OfficialTiles(phase['crop'], local_indices, local_seeds)
        loader = DataLoader(data, batch_size=batch, shuffle=False, num_workers=3, pin_memory=True, persistent_workers=True, multiprocessing_context='spawn')
        optimizer = None if args.probe else torch.optim.AdamW(groups, fused=True)
        mean = torch.tensor([123.675, 116.28, 103.53], device=rank)[None, :, None, None]
        std = torch.tensor([58.395, 57.12, 57.375], device=rank)[None, :, None, None]
        if rank == 0:
            record.update(freeze_sources(folder))
            np.savez_compressed(folder / 'sampling.npz', indices=indices, augmentation_seeds=seeds)
            record.update(status='backward_probe' if args.probe else 'training', sampling_sha256=digest(folder / 'sampling.npz'), official_train_pairs=6996, sampled_images=total_count, unique_sampled_images=len(np.unique(indices)))
            write_json(folder / 'training.json', record)
        iterator = iter(loader)
        streams = initial_random_streams(torch, phase['seed'], rank, world, logical_workers)
        started = time.monotonic()
        for step in range(1, steps + 1):
            network.zero_grad(set_to_none=True)
            progress = max(0.0, (step - phase['warmup']) / max(1, phase['steps'] - phase['warmup']))
            multiplier = min(step / phase['warmup'], 1.0) * 0.5 * (1 + math.cos(math.pi * progress))
            if optimizer:
                for group in groups:
                    group['lr'] = group['initial_lr'] * multiplier
            step_values = None
            for micro in range(accumulation):
                torch.set_rng_state(streams[micro][0])
                torch.cuda.set_rng_state(streams[micro][1])
                rgb, labels = next(iterator)
                images = (rgb.cuda(rank, non_blocking=True).float() - mean) / std
                labels = labels.cuda(rank, non_blocking=True)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    logits, auxiliary = network(images)
                loss, parts = loss_terms(logits, auxiliary, labels, alignment)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Non-finite training objective')
                (loss / accumulation).backward()
                values = torch.stack([v.detach().float() for v in parts.values()])
                step_values = values if step_values is None else step_values + values
                streams[micro] = (torch.get_rng_state(), torch.cuda.get_rng_state())
                del images, labels, logits, auxiliary, loss, parts, values, rgb
            norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], recipe['optimizer']['gradient_clip'], error_if_nonfinite=True)
            if optimizer:
                optimizer.step()
            if step == 1 or step % 25 == 0 or step == steps:
                dist.all_reduce(step_values)
                if rank == 0:
                    record.update(optimizer_steps=0 if args.probe else step, backward_steps=step * accumulation, losses=dict(zip(('ce', 'lovasz', 'scale_ce'), (step_values / (world * accumulation)).tolist())), gradient_norm=float(norm), seconds=time.monotonic() - started, peak_allocated_gib=torch.cuda.max_memory_allocated() / 2 ** 30, lr_multiplier=multiplier)
                    write_json(folder / 'training.json', record)
                    print(TEAM.name, args.tag, step, '/', steps, record['losses'], 'seconds', round(record['seconds'], 1), flush=True)
        if state_hash(model, frozen_names) != frozen_before:
            raise ValueError('Frozen parameters or BN statistics changed')
        final_hash = state_hash(model)
        hashes = [None] * world
        dist.all_gather_object(hashes, final_hash)
        if len(set(hashes)) != 1:
            raise ValueError('Synchronized model parameters diverged')
        if args.probe and probe_before != final_hash:
            raise ValueError('Zero-step backward probe changed model state')
        if rank == 0:
            check_sources(record)
            record.update(synchronized_state_sha256=final_hash, all_ranks_exact=True, frozen_parameters_and_buffers_exact=True)
            if args.probe:
                record.update(status='real_backward_passed', checkpoint=None, optimizer_steps=0)
            else:
                target = folder / 'weights.pth'
                torch.save({'model': {n: v.detach().cpu() for n, v in model.state_dict().items()}, 'owner_team': TEAM.name, 'lineage_steps': lineage}, target)
                record.update(status='trained', checkpoint=str(target.relative_to(TEAM)), checkpoint_sha256=digest(target))
            write_json(folder / 'training.json', record)
        dist.barrier(device_ids=[rank])
    except BaseException as error:
        if rank == 0:
            record.update(status='failed', error_type=type(error).__name__, error=str(error))
            write_json(folder / 'training.json', record)
        raise
    finally:
        dist.destroy_process_group()
if __name__ == '__main__':
    main()
