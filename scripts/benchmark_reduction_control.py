"""Isolated benchmark-only control: bind generated norm globals to reference objects.

No CUDA operation occurs on import. Capture a production decode, then execute
one candidate-graph quality decode under bind_candidate(). Subsequent quality,
warmup, and timing calls require no CachingAutotuner.run monkeypatch. Keep the
bindings alive through those calls and restore them in finally. Every new input
shape must first be exercised by capture_reference(), then bind_candidate().

This controls Inductor norm scheduling; it does not assert default independent
compilations are bitwise identical. No model or production runtime is modified.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import sys
from unittest.mock import patch


class ReferenceReductionBindings:
    def __init__(self, autotuner_type=None):
        if autotuner_type is None:
            from torch._inductor.runtime.triton_heuristics import CachingAutotuner
            autotuner_type = CachingAutotuner
        self.autotuner_type = autotuner_type
        self.reference = {}
        self._restores = []
        self.bindings = []

    @staticmethod
    def _selected(kernel):
        name = kernel.inductor_meta.get('kernel_name', '')
        return (getattr(kernel.heuristic_type, 'name', '') == 'REDUCTION'
                and ('rsqrt' in name or 'native_layer_norm' in name))

    @staticmethod
    def _key(kernel):
        return hashlib.sha256(kernel.fn.src.encode()).hexdigest()

    @staticmethod
    def _schedule(kernel):
        if len(kernel.launchers) != 1:
            raise RuntimeError('reference norm does not have one final launcher')
        cfg = kernel.launchers[0].config
        if not getattr(cfg, 'found_by_coordesc', False):
            raise RuntimeError('reference norm coordinate-descent selection is not frozen')
        return {'kwargs': dict(cfg.kwargs), 'num_warps': cfg.num_warps,
                'num_stages': cfg.num_stages}

    @staticmethod
    def _compatible(kernel, reference):
        if kernel.fn.src != reference.fn.src:
            raise RuntimeError('norm source differs')
        # fn.src excludes decorators: compare their dispatch/compiler contracts too.
        for field in ('triton_meta', 'device_props', 'size_hints', 'heuristic_type',
                      'mutated_arg_names', 'reset_to_zero_arg_names'):
            if getattr(kernel, field, None) != getattr(reference, field, None):
                raise RuntimeError(f'norm {field} differs')
        for field in ('grid_type', 'fixed_grid', 'extra_launcher_args', 'no_x_dim',
                      'persistent_reduction', 'backend_hash'):
            if kernel.inductor_meta.get(field) != reference.inductor_meta.get(field):
                raise RuntimeError(f'norm grid/backend contract differs: {field}')

    @staticmethod
    def _arguments(args, kwargs):
        import torch
        descriptors = []
        tensors = []
        for value in args:
            if isinstance(value, torch.Tensor):
                pointer = value.data_ptr()
                descriptors.append(('tensor', str(value.dtype), str(value.device),
                                    tuple(value.shape), tuple(value.stride()), pointer % 16))
                tensors.append(pointer)
            elif isinstance(value, (int, float, bool, str, type(None))):
                descriptors.append((type(value).__name__, value))
            else:
                raise RuntimeError(f'unsupported norm argument type: {type(value).__name__}')
        # Do not specialize a stream handle; the caller continues to supply it.
        if set(kwargs) != {'stream'}:
            raise RuntimeError(f'unexpected norm run keywords: {tuple(kwargs)}')
        aliases = tuple((i, j) for i in range(len(tensors))
                        for j in range(i) if tensors[i] == tensors[j])
        return tuple(descriptors), aliases

    @contextmanager
    def capture_reference(self):
        """Collect actual old-graph norm objects after their final launch choice."""
        original = self.autotuner_type.run

        def observed(kernel, *args, **kwargs):
            result = original(kernel, *args, **kwargs)
            if self._selected(kernel):
                key = self._key(kernel)
                schedule = self._schedule(kernel)
                contract = self._arguments(args, kwargs)
                if key in self.reference:
                    previous = self.reference[key]
                    self._compatible(kernel, previous['kernel'])
                    if previous['schedule'] != schedule or previous['arguments'] != contract:
                        raise RuntimeError('production norm source has inconsistent schedule/arguments')
                self.reference[key] = {'kernel': kernel, 'schedule': schedule,
                                       'arguments': contract}
            return result

        with patch.object(self.autotuner_type, 'run', observed):
            yield

    @contextmanager
    def bind_candidate(self):
        """Route initial quality calls and permanently rebind their module globals."""
        original = self.autotuner_type.run

        def observed(kernel, *args, **kwargs):
            if not self._selected(kernel):
                return original(kernel, *args, **kwargs)
            key = self._key(kernel)
            if key not in self.reference:
                raise RuntimeError('no production norm for this candidate source/shape')
            entry = self.reference[key]
            reference = entry['kernel']
            self._compatible(kernel, reference)
            if self._arguments(args, kwargs) != entry['arguments']:
                raise RuntimeError('candidate norm arguments differ from production')
            if self._schedule(reference) != entry['schedule']:
                raise RuntimeError('production norm launcher changed')
            if kernel is not reference:
                # Current Inductor Runner.call performs module-global kernel.run(...).
                # Fail closed if a future generated wrapper changes that convention.
                namespace = sys._getframe(1).f_globals
                symbol = kernel.inductor_meta['kernel_name']
                if namespace.get(symbol) is not kernel:
                    raise RuntimeError('candidate norm is not the expected caller global')
                self._restores.append((namespace, symbol, kernel, reference))
                namespace[symbol] = reference
                self.bindings.append({'kernel': symbol, 'source_sha256': key,
                                      'schedule': dict(entry['schedule'])})
            return original(reference, *args, **kwargs)

        with patch.object(self.autotuner_type, 'run', observed):
            yield

    def assert_bound(self):
        """Call outside timing to fail if any graph binding or launcher changed."""
        if not self.bindings:
            raise RuntimeError('no candidate norm globals were rebound')
        # A given global may be rebound more than once after new input compilation.
        seen = set()
        for namespace, symbol, _, reference in reversed(self._restores):
            marker = (id(namespace), symbol)
            if marker in seen:
                continue
            seen.add(marker)
            if namespace.get(symbol) is not reference:
                raise RuntimeError('candidate norm binding changed')
            key = self._key(reference)
            if self._schedule(reference) != self.reference[key]['schedule']:
                raise RuntimeError('bound norm launcher changed')

    def restore(self):
        for namespace, symbol, original, _ in reversed(self._restores):
            namespace[symbol] = original
        self._restores.clear()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.restore()
