"""CPU-only structural tests of the isolated module-global control."""
import copy
from types import SimpleNamespace
import unittest
import torch
from scripts.benchmark_reduction_control import ReferenceReductionBindings


class FakeAutotuner:
    def __init__(self, marker):
        self.marker = marker
        self.calls = 0
        self.fn = SimpleNamespace(src='def norm(x, out, n): return x')
        self.triton_meta = {'signature': {'x': '*fp16', 'out': '*fp16', 'n': 'i32'},
                           'constants': {}, 'enable_fp_fusion': True}
        self.inductor_meta = {'kernel_name': 'triton_red_rsqrt_4', 'grid_type': 'Grid1D'}
        self.heuristic_type = SimpleNamespace(name='REDUCTION')
        self.device_props = 'test_cpu'
        self.size_hints = {'x': 4, 'r0_': 8}
        self.mutated_arg_names = []
        self.reset_to_zero_arg_names = []
        self.launchers = [SimpleNamespace(config=SimpleNamespace(
            kwargs={'XBLOCK': 1, 'R0_BLOCK': 8}, num_warps=4, num_stages=1,
            found_by_coordesc=True))]

    def run(self, *args, **kwargs):
        self.calls += 1
        return self.marker


def graph(kernel):
    namespace = {'triton_red_rsqrt_4': kernel}
    exec('def call(x, out):\n    return triton_red_rsqrt_4.run(x, out, 8, stream=0)', namespace)
    return namespace


class BindingTests(unittest.TestCase):
    def setUp(self):
        self.x = torch.zeros(4, 8, dtype=torch.float16)
        self.out = torch.empty_like(self.x)
        self.old, self.new = FakeAutotuner('old'), FakeAutotuner('new')
        self.og, self.ng = graph(self.old), graph(self.new)

    def reference(self, control):
        with control.capture_reference():
            self.assertEqual(self.og['call'](self.x, self.out), 'old')

    def test_bind_restore_and_unhooked_execution(self):
        original_run = FakeAutotuner.run
        with ReferenceReductionBindings(FakeAutotuner) as control:
            self.reference(control)
            with control.bind_candidate():
                self.assertEqual(self.ng['call'](self.x, self.out), 'old')
            self.assertIs(FakeAutotuner.run, original_run)
            self.assertIs(self.ng['triton_red_rsqrt_4'], self.old)
            control.assert_bound()
            self.assertEqual(self.ng['call'](self.x, self.out), 'old')
            self.assertEqual(self.new.calls, 0)
        self.assertIs(self.ng['triton_red_rsqrt_4'], self.new)
        self.assertEqual(self.ng['call'](self.x, self.out), 'new')

    def test_mismatched_compiler_flags_fail_before_run(self):
        with ReferenceReductionBindings(FakeAutotuner) as control:
            self.reference(control)
            self.new.triton_meta['enable_fp_fusion'] = False
            with control.bind_candidate(), self.assertRaisesRegex(RuntimeError, 'triton_meta'):
                self.ng['call'](self.x, self.out)
            self.assertEqual(self.new.calls, 0)

    def test_unseen_source_and_alias_contract_fail(self):
        with ReferenceReductionBindings(FakeAutotuner) as control:
            self.reference(control)
            with control.bind_candidate(), self.assertRaisesRegex(RuntimeError, 'arguments'):
                self.ng['call'](self.x, self.x)
            self.new.fn = SimpleNamespace(src='a different kernel')
            with control.bind_candidate(), self.assertRaisesRegex(RuntimeError, 'no production'):
                self.ng['call'](self.x, self.out)

    def test_graph_rebinding_is_checked_and_restored_on_exception(self):
        with self.assertRaisesRegex(RuntimeError, 'binding changed'):
            with ReferenceReductionBindings(FakeAutotuner) as control:
                self.reference(control)
                with control.bind_candidate():
                    self.ng['call'](self.x, self.out)
                self.ng['triton_red_rsqrt_4'] = copy.copy(self.old)
                control.assert_bound()
        self.assertIs(self.ng['triton_red_rsqrt_4'], self.new)


if __name__ == '__main__':
    unittest.main()
