import math
from collections import defaultdict, deque
from collections.abc import Hashable, Iterable, Sized

import numpy as np
import torch
from torch.utils.data.sampler import Sampler


# Inspired by:
# https://raw.githubusercontent.com/KevinMusgrave/pytorch-metric-learning/refs/heads/master/src/pytorch_metric_learning/samplers/m_per_class_sampler.py
class MPerClassSampler(Sampler):
    def __init__(
            self, 
            labels : torch.Tensor | np.ndarray | Iterable[Hashable], 
            m : int,
            batch_size : int,
            biased : bool=True
        ):
        """Sample ``m`` elements from ``c`` classes such that:
        ```
        m * c == batch_size
        ```

        Args:
            labels: A tensor/array/list of class indices from 0 to n.
            m: Number of elements from each class to use in a batch.
            batch_size: Batch size. Must be a multiple of ``m``.
            biased: Continue creating batches when only a subset of classes can be included.
        """
        if isinstance(labels, torch.Tensor):
            self.labels = labels
        elif isinstance(labels, np.ndarray):
            self.labels = torch.from_numpy(labels)
        else:
            labels = list(labels)
            l2idx = {k : i for k, i in enumerate(sorted(list(set(labels))))}
            labels = [l2idx[k] for k in labels]
            self.labels = torch.tensor(labels)
        self.labels = self.labels.clone().long()
        assert torch.all(self.labels.unique().sort().values == torch.arange(self.labels.max()+1))
        
        self.m = m
        self.batch_size = batch_size
        self.c = self.batch_size // self.m
        assert self.m * self.c == self.batch_size
        self.biased = biased

        self.ncls = int(self.labels.max().item() + 1)
        self.class_mask : torch.Tensor = torch.ones((self.ncls,), dtype=torch.bool)
        self.item_set : list[torch.Tensor] = []
        self.item_perm : list[torch.Tensor] = []
        self.item_off : list[int] = []

        for cls in range(self.ncls):
            idxs = torch.nonzero(self.labels == cls).flatten()
            self.item_set.append(idxs) 
            self.item_perm.append(idxs[torch.randperm(len(idxs))])
            self.item_off.append(0)

        self._len = len(self.labels) // self.batch_size

    def __len__(self):
        return self._len

    def _prepare(self):
        self.class_mask |= True
        for cls in range(self.ncls):
            self.item_perm[cls] = self.item_set[cls][torch.randperm(len(self.item_set[cls]))]
            self.item_off[cls] = 0
    
    def __iter__(self):
        self._prepare()
        while torch.any(self.class_mask):
            batch = []
            while len(batch) < self.batch_size and torch.any(self.class_mask):
                acls = self.class_mask.sum().item()
                i = torch.randperm(acls)[:min(self.c, acls)]
                cls = torch.nonzero(self.class_mask).flatten()[i]
                for c in cls: 
                    oidx = self.item_off[c]
                    aidx = len(self.item_set[c]) - oidx
                    nidx = self.m
                    if aidx < self.m:
                        self.class_mask[c] = False
                        nidx = aidx
                        if not self.biased:
                            return
                    batch.extend(self.item_perm[c][oidx:(oidx+nidx)].tolist())
                    self.item_off[c] = oidx + nidx
            yield batch

class LogBucketQueue:
    def __init__(
            self, 
            width : float=2.0,
            starvation_rate : float=0.05, 
            seed=None
        ):
        """A probabilistic priority queue.

        Instead of deterministically popping the element with the highest priority,
        when popping an element, an element is sampled with probability:
        ```
        prob ~ priority / sum(priority)
        ```
        
        To avoid starvation a rate for how often 
        
        Args:
            width: The maximum error factor on priority when sampling.
            starvation_rate: How often a completely random element is chosen when popping to prevent starvation.
                The default (0.05) makes the minimum rotation time (how often an element will be seen if we
                repeatedly pop and insert with the same priority) 20x the size of the queue.
            seed: Seed for RNG in sampling.
        """
        self.width = width
        self.buckets = defaultdict(list)
        self.weights = defaultdict(float)
        self.total = 0.0
        self.starvation_rate = starvation_rate
        self._inv = 1 / math.log(self.width)
        self._generator = np.random.default_rng(seed=seed)
        self._len = 0

    def __len__(self):
        return self._len

    def insert(self, item, priority : float):
        if priority < 0:
            raise ValueError(f'Negative priority `{priority}` is invalid.')
        elif priority == 0:
            idx = float('-inf')
        else:
            idx = int(math.log(priority) * self._inv)
        self.buckets[idx].append((item, priority))
        self.weights[idx] += priority
        self.total += priority
        self._len += 1

    def pop(self):
        assert len(self) > 0
        sample_uniform = self.total == 0 or self._generator.random() < self.starvation_rate
        if sample_uniform:
            target = self._generator.integers(0, len(self))
            for idx, items in self.buckets.items():
                weight = len(items)
                if target < weight:
                    break
                target -= weight
        else:
            target = self._generator.uniform(0, self.total)
            for idx, weight in self.weights.items():
                if target < weight:
                    break
                target -= weight
        bucket = self.buckets[idx]
        upper = self.width ** (idx + 1)
        while True:
            jdx = self._generator.integers(0, len(bucket))
            item, priority = bucket[jdx]
            if sample_uniform or priority > self._generator.random() * upper:
                break
        bucket[jdx] = bucket[-1]
        bucket.pop()
        self.weights[idx] -= priority
        self.total -= priority
        self._len -= 1
        if len(bucket) == 0:
            self.buckets.pop(idx)
            self.weights.pop(idx)
        return item, priority
    
    def state_dict(self):
        return {
            'width': self.width,
            'starvation_rate': self.starvation_rate,
            'buckets': dict(self.buckets),
            'weights': dict(self.weights),
            'total': self.total,
            '_len': self._len,
            'rng_state': self._generator.bit_generator.state,
        }

    def load_state_dict(self, state_dict):
        self.width = state_dict['width']
        self.starvation_rate = state_dict['starvation_rate']
        self._inv = 1.0 / math.log(self.width)
        self.buckets = defaultdict(list, state_dict['buckets'])
        self.weights = defaultdict(float, state_dict['weights'])
        self.total = state_dict['total']
        self._len = state_dict['_len']
        self._generator.bit_generator.state = state_dict['rng_state']


class InfiniteOnlineHardExampleSampler(Sampler[int]):
    def __init__(
        self,
        data_source : Sized,
        seed=None
    ):
        self.data_source = data_source
        self.queue = LogBucketQueue(width=1.5, starvation_rate=0.0, seed=seed)
        self.out = deque()
        # Populate queue with practically infinite priority at start
        for i in range(len(self.data_source)):
            self.queue.insert(i, 1e12)

    def __len__(self):
        return len(self.data_source)
    
    def __iter__(self):
        while len(self.out) > 0:
            self.queue.insert(*self.out.popleft())
        for _ in range(len(self)):
            retval, old_priority = self.queue.pop()
            # Store old priority to prevent desync due to "drop_last" at end-of-epoch
            self.out.append((retval, old_priority))
            yield retval

    @classmethod
    def _normalize_priorities(cls, priorities) -> list[float]:
        # torch.Tensor or np.ndarray
        if hasattr(priorities, 'ndim'):
            if priorities.ndim > 1:
                priorities = priorities.sum(axis=-1)
            return priorities.flatten().tolist()

        priorities = list(priorities)
        if not priorities: 
            return []        
        head = priorities[0]

        # [Tensor(N), Tensor(N)] -> Tensor(N) -> cls._normalize_priorities -> List[float]
        if hasattr(head, 'ndim'):
            return cls._normalize_priorities(sum(priorities))    
        if isinstance(head, (list, tuple)):
            return [sum(terms) for terms in zip(*priorities)]
        return [float(p) for p in priorities]

    @torch.no_grad()
    def update(self, priorities : list[float] | list[list[float]] | torch.Tensor | np.ndarray):
        priorities = self._normalize_priorities(priorities)
        if len(priorities) > len(self.out):
            raise RuntimeError(
                'Attempt to update InfiniteOnlineHardExampleSampler with too many priorities:\n'
                f'\t{len(priorities)} > {len(self.out)}'
            )
        for p in priorities:
            el, _ = self.out.popleft()
            self.queue.insert(el, (1-math.exp(-p))**4)

    def state_dict(self):
        return {
            'queue': self.queue.state_dict(),
            'out': list(self.out)
        }

    def load_state_dict(self, state_dict):
        self.queue.load_state_dict(state_dict['queue'])
        self.out = deque(state_dict['out'])
