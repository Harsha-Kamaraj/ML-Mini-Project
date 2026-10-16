"""Regression checks for the ten-genre notebook, without running its long sweeps."""
import ast
import copy
from pathlib import Path
import unittest

import nbformat
import numpy as np
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = nbformat.read(ROOT / "musical_genre_paper_reproduction.ipynb", as_version=4)


def definitions(namespace, names):
    for cell in NOTEBOOK.cells:
        if cell.cell_type != "code":
            continue
        nodes = [node for node in ast.parse(cell.source).body
                 if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
        if nodes:
            exec(compile(ast.Module(body=nodes, type_ignores=[]), "<notebook>", "exec"), namespace)


class TenGenreNotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ns = {"np": np, "torch": torch, "nn": nn, "DROPOUT": .1, "SEED": 0,
                  "GAMMA": .9, "N_CLASSES": 10, "SPLIT_SIZES": (15000, 2500, 2500)}
        definitions(cls.ns, {"xavier", "TwoLayerNN", "Encoder", "Decoder", "VanillaAutoencoder",
                             "DeepSoftmaxAutoencoder", "FixedDeepSoftmaxAE", "conv_block",
                             "deconv_block", "ConvDeepSoftmaxAE", "conv2d_block", "deconv2d_block",
                             "MelDeepSoftmaxAE", "cross_entropy", "reconstruction", "deep_softmax_loss",
                             "dsae_loss", "random_split", "song_partition", "make_split", "cache_key"})

    def test_all_classifiers_accept_tenth_label_and_backpropagate(self):
        specs = [("TwoLayerNN", (), (3, 500)), ("DeepSoftmaxAutoencoder", (), (3, 500)),
                 ("FixedDeepSoftmaxAE", (500,), (3, 500)),
                 ("FixedDeepSoftmaxAE", (2000,), (3, 2000)),
                 ("ConvDeepSoftmaxAE", (2500,), (3, 2500)),
                 ("ConvDeepSoftmaxAE", (5000,), (3, 5000)),
                 ("MelDeepSoftmaxAE", ((64, 40),), (3, 64, 40))]
        for name, args, shape in specs:
            with self.subTest(model=name, input_shape=shape):
                model = self.ns[name](*args)
                x = torch.randn(shape)
                out = model(x)
                self.assertEqual(out["logits"].shape, (3, 10))
                if "encoding" in out:
                    self.assertEqual(out["encoding"].shape, (3, 64))
                    self.assertEqual(out["reconstruction"].shape, x.shape)
                loss = (self.ns["cross_entropy"] if name == "TwoLayerNN" else self.ns["dsae_loss"](.1))(
                    out, x, torch.tensor([0, 5, 9]))
                self.assertTrue(torch.isfinite(loss))
                loss.backward()
                self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))

    def test_hidden_architectures_are_independent_of_class_count(self):
        specs = [("TwoLayerNN", ()), ("VanillaAutoencoder", ()), ("DeepSoftmaxAutoencoder", ()),
                 ("FixedDeepSoftmaxAE", ()), ("ConvDeepSoftmaxAE", (5000,)),
                 ("MelDeepSoftmaxAE", ((64, 40),))]
        for name, args in specs:
            with self.subTest(model=name):
                self.ns["N_CLASSES"] = 4
                old = self.ns[name](*args).state_dict()
                self.ns["N_CLASSES"] = 10
                new = self.ns[name](*args).state_dict()
                self.assertEqual(old.keys(), new.keys())
                changed = [k for k in old if old[k].shape != new[k].shape]
                if name == "VanillaAutoencoder":
                    self.assertEqual(changed, [])
                else:
                    self.assertEqual(len(changed), 2)
                    for key in changed:
                        self.assertEqual(old[key].shape[0], 4)
                        self.assertEqual(new[key].shape[0], 10)
                        self.assertEqual(old[key].shape[1:], new[key].shape[1:])

    def test_splits_cover_full_dataset_and_keep_cross_genre_duplicates_together(self):
        ns = self.ns
        ns.update(X=np.zeros((20000, 1), dtype=np.float32), N_SONGS=1000,
                  PAPER_SEGS=20, ALL_SEGS=33, song_label=np.repeat(np.arange(10), 100))
        digests = np.array([str(i) for i in range(1000)], dtype=object)
        for a, b in [(698, 694), (634, 695), (758, 960), (339, 345)]:
            digests[b] = digests[a]
        ns["audio_digest"] = digests
        for seed in range(5):
            with self.subTest(seed=seed):
                clips = ns["random_split"](seed)
                self.assertEqual([len(v) for v in clips.values()], [15000, 2500, 2500])
                self.assertEqual(len(np.unique(np.concatenate(list(clips.values())))), 20000)
                code = ns["random_split"](seed, (15000, 5000, 0))
                self.assertEqual([len(v) for v in code.values()], [15000, 5000, 0])
                parts = ns["make_split"]("song", seed, extra_segments=True)
                song_sets = {p: set(v[:, 0]) for p, v in parts.items()}
                self.assertEqual(set.union(*song_sets.values()), set(range(1000)))
                for p in parts:
                    self.assertEqual(set(ns["song_label"][list(song_sets[p])]), set(range(10)))
                    self.assertEqual(set(parts[p][:, 1]), set(range(33 if p == "train" else 20)))
                    for q in parts:
                        if p != q:
                            self.assertFalse(song_sets[p] & song_sets[q])
                            self.assertFalse(set(digests[list(song_sets[p])]) & set(digests[list(song_sets[q])]))
                self.assertTrue(np.array_equal(ns["song_partition"](seed), ns["song_partition"](seed)))
        with self.assertRaises(AssertionError):
            ns["random_split"](0, (6000, 1000, 1000))
        with self.assertRaises(ValueError):
            ns["make_split"]("typo", 0)

    def test_cache_identity_includes_label_order_and_dataset(self):
        import hashlib
        import json
        self.ns.update(hashlib=hashlib, json=json)
        self.ns["CACHE_CONTEXT"] = {"genres": [str(i) for i in range(10)], "audio_sha256": "first"}
        cfg = {"name": "step3", "seed": 0}
        first = self.ns["cache_key"](cfg)
        self.assertEqual(first, self.ns["cache_key"](copy.deepcopy(cfg)))
        self.ns["CACHE_CONTEXT"]["genres"].reverse()
        self.assertNotEqual(first, self.ns["cache_key"](cfg))
        self.ns["CACHE_CONTEXT"]["genres"].reverse()
        self.ns["CACHE_CONTEXT"]["audio_sha256"] = "second"
        self.assertNotEqual(first, self.ns["cache_key"](cfg))


if __name__ == "__main__":
    unittest.main()
