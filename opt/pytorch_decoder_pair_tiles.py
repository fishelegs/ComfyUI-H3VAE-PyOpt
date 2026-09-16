"""Opt-in paired spatial tiles for batch-one MiniMax-H3 inference."""
import torch


def enable_paired_decoder_tiles(model):
    """Install batch-two decoder tile scheduling; return the old scheduler.

    Enable before warmup of a whole-compiled decoder. Do not wrap this in the
    existing batch-one static CUDA Graph. Tile overlap/blending stays original.
    The change is for inference; different GEMM batch shapes introduce FP16
    rounding differences. Encoder calls retain the original scheduler.
    """
    core = getattr(model, 'model', model)
    state_fn = core.tiled_decode.__func__.__globals__.get('get_parallel_state')
    state = state_fn() if state_fn is not None else {}
    if core.training or state.get('sp_size', 1) != 1 or state.get('tp_size', 1) != 1:
        raise ValueError('paired tiles require eval mode and single-device execution')
    original = core._run_tile_tasks

    def paired(tiles, indices, forward_fn, stack_tiling, cls_agg=None):
        if (cls_agg is not None or not tiles or tiles[0].shape[1] != 24
                or tiles[0].shape[0] != 1 or torch.is_grad_enabled()):
            return original(tiles, indices, forward_fn, stack_tiling, cls_agg)
        outputs = []
        for start in range(0, len(indices), 2):
            selected = indices[start:start+2]
            pixels = forward_fn(torch.cat([tiles[i] for i in selected], dim=0))
            outputs.extend(pixels.split(1, dim=0))
        return outputs

    core._run_tile_tasks = paired
    return original
