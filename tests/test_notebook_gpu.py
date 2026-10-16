"""Device and migration checks for the Colab continuation notebook."""
import ast
import random
from pathlib import Path
import tempfile
import time
import unittest

import nbformat
import numpy as np
import pandas as pd
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
GPU = nbformat.read(ROOT / 'musical_genre_classification.ipynb', as_version=4)
CPU = nbformat.read(ROOT / 'musical_genre_paper_reproduction.ipynb', as_version=4)


def nodes(notebook):
    return [node for c in notebook.cells if c.cell_type == 'code'
            for node in ast.parse(c.source).body]


def namespace(device):
    ns = dict(np=np, pd=pd, torch=torch, nn=nn, random=random, time=time,
              DROPOUT=.1, N_CLASSES=10, SEED=0, DEVICE=torch.device(device),
              LEARNING_RATE=1e-4, BATCH_SIZE=4)
    wanted = {'seed_everything', 'xavier', 'TwoLayerNN', 'FixedDeepSoftmaxAE', 'predict', 'train',
              'cross_entropy', 'gather', 'batched', 'dsae_loss', 'standardizer', 'fit_model'}
    for node in nodes(GPU):
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in wanted:
            exec(compile(ast.Module(body=[node], type_ignores=[]), '<gpu-notebook>', 'exec'), ns)
    return ns


class GPUContinuationTests(unittest.TestCase):
    def test_architectures_are_unchanged(self):
        original = {n.name: ast.dump(n) for n in nodes(CPU) if isinstance(n, ast.ClassDef)}
        gpu = {n.name: ast.dump(n) for n in nodes(GPU) if isinstance(n, ast.ClassDef)}
        self.assertEqual(original, gpu)

    def exercise_device(self, device):
        ns = namespace(device)
        ns['song_label'] = np.array([0, 9, 5, 7])
        data = np.random.default_rng(0).normal(size=(4, 4, 500)).astype(np.float32)
        parts = {'train': np.array([(k, j) for k in range(4) for j in [0, 1]]),
                 'dev': np.array([(k, 2) for k in range(4)])}
        model, norm, history, best_epoch = ns['fit_model'](
            ns['FixedDeepSoftmaxAE'](), data, parts, ns['dsae_loss'](.1),
            max_epochs=1, lr=1e-4, batch=4, patience=1, seed=0)
        self.assertEqual(next(model.parameters()).device.type, device)
        self.assertEqual(best_epoch, 1)
        self.assertTrue(np.isfinite(history.train_loss).all())
        inputs = norm(ns['gather'](data, parts['dev']))
        result = ns['batched'](model, inputs, size=2)
        self.assertEqual(result['logits'].shape, (4, 10))
        self.assertTrue(all(v.device.type == 'cpu' for v in result.values()))
        self.assertEqual(set(ns['batched'](model, inputs, keys=('logits',))), {'logits'})
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'weights.pt'
            torch.save({k: v.cpu() for k, v in model.state_dict().items()}, p)
            restored = ns['FixedDeepSoftmaxAE']().to(device)
            restored.load_state_dict(torch.load(p, map_location=device, weights_only=True))
            torch.testing.assert_close(ns['batched'](restored, inputs)['logits'], result['logits'])

    def test_training_and_checkpoint_portability_on_cpu(self):
        self.exercise_device('cpu')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA runtime required')
    def test_training_and_checkpoint_portability_on_cuda(self):
        self.exercise_device('cuda')

    def test_completed_part1_training_is_reused(self):
        ns = namespace('cpu')
        ns.update(X_t=torch.randn(12, 500), y_t=torch.arange(12) % 10,
                  split={'train': np.arange(8), 'dev': np.arange(8, 12)}, REUSE_COMPLETED=True)
        with tempfile.TemporaryDirectory() as d:
            ns['OUT'] = Path(d)
            model, history = ns['train']('baseline', ns['TwoLayerNN'], ns['cross_entropy'], 1)
            p = Path(d) / 'baseline.pt'
            saved_time = p.stat().st_mtime_ns
            restored, reused = ns['train']('baseline', ns['TwoLayerNN'], ns['cross_entropy'], 1)
            self.assertEqual(p.stat().st_mtime_ns, saved_time)
            pd.testing.assert_frame_equal(history, reused)
            for key, value in model.state_dict().items():
                torch.testing.assert_close(value, restored.state_dict()[key])
            self.assertEqual(ns['predict'](restored, np.arange(12))['logits'].shape, (12, 10))


if __name__ == '__main__':
    unittest.main()
