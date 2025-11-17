import unittest

import torch

from rethink import HiddenStateCache, RethinkEngine
from rethink.adapters import RethinkLlamaConfig, RethinkLlamaForCausalLM


class HiddenStateCacheTest(unittest.TestCase):
    def test_record_and_latest(self):
        cache = HiddenStateCache(max_steps=2)
        tensor = torch.randn(1, 1, 4)
        cache.record(layer=0, hidden_state=tensor, step=0)
        cache.record(layer=0, hidden_state=tensor + 1.0, step=1)
        latest = cache.latest(0)
        self.assertIsNotNone(latest)
        self.assertEqual(latest.step, 1)
        self.assertAlmostEqual(float(latest.hidden_state.mean()), float((tensor + 1.0).mean()))


class RethinkModelTest(unittest.TestCase):
    def setUp(self):
        self.config = RethinkLlamaConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            max_position_embeddings=32,
            rethink_layers=[0, 1],
        )
        self.model = RethinkLlamaForCausalLM(self.config)
        self.model.eval()

    def test_forward_populates_cache(self):
        cache = HiddenStateCache(max_steps=4)
        input_ids = torch.randint(low=0, high=self.config.vocab_size, size=(1, 5))
        with torch.no_grad():
            self.model(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                rethink_cache=cache,
                return_dict=True,
            )
        self.assertGreater(len(cache), 0)
        latest = cache.latest(layer=0)
        self.assertIsNotNone(latest)
        self.assertEqual(latest.hidden_state.shape[-1], self.config.hidden_size)

    def test_engine_wrapper(self):
        cache = HiddenStateCache(max_steps=4)
        engine = RethinkEngine(self.model, cache=cache)
        input_ids = torch.randint(low=0, high=self.config.vocab_size, size=(1, 4))
        outputs, cache, scores, action = engine.generate(
            input_ids,
            max_rethink_loops=1,
            max_new_tokens=2,
        )
        self.assertIsNotNone(outputs.sequences)
        self.assertGreaterEqual(len(scores), 0)
        self.assertIsNotNone(action)


if __name__ == "__main__":
    unittest.main()
