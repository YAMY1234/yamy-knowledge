---
id: caching-strategies
sidebar_position: 4
---

# 缓存策略

## KV缓存优化

### 缓存设计
```python
class KVCache:
    def __init__(self, max_size):
        self.cache = {}
        self.max_size = max_size
        
    def get(self, key):
        if key in self.cache:
            return self.cache[key]
        return None
        
    def put(self, key, value):
        if len(self.cache) >= self.max_size:
            # LRU淘汰
            self._evict()
        self.cache[key] = value
```

### 内存管理
```python
class MemoryManager:
    def __init__(self):
        self.allocated = {}
        self.free_blocks = []
        
    def allocate(self, size):
        # 分配内存块
        if self.free_blocks:
            block = self._find_best_fit(size)
            if block:
                return block
        return self._allocate_new(size)
        
    def free(self, ptr):
        # 释放内存块
        if ptr in self.allocated:
            block = self.allocated.pop(ptr)
            self.free_blocks.append(block)
```

### 缓存更新策略
1. LRU (最近最少使用)
2. LFU (最不经常使用)
3. FIFO (先进先出)

## 结果缓存

### 相似查询识别
```python
def compute_similarity(query1, query2):
    """
    计算查询相似度
    """
    # 编辑距离
    distance = levenshtein_distance(query1, query2)
    # 语义相似度
    semantic_sim = semantic_similarity(query1, query2)
    return (distance, semantic_sim)

def find_similar_cached(query, cache):
    """
    查找相似的缓存结果
    """
    for cached_query in cache:
        sim = compute_similarity(query, cached_query)
        if sim > SIMILARITY_THRESHOLD:
            return cache[cached_query]
    return None
```

### 缓存失效策略
```python
class CacheManager:
    def __init__(self):
        self.cache = {}
        self.ttl = {}
        
    def set_with_ttl(self, key, value, ttl):
        self.cache[key] = value
        self.ttl[key] = time.time() + ttl
        
    def get(self, key):
        if key in self.cache:
            if time.time() < self.ttl[key]:
                return self.cache[key]
            else:
                # TTL过期，删除缓存
                del self.cache[key]
                del self.ttl[key]
        return None
```

### 分布式缓存
```python
from redis import Redis

class DistributedCache:
    def __init__(self):
        self.redis = Redis(host='localhost', port=6379)
        
    async def get_or_compute(self, key, compute_func):
        # 尝试从缓存获取
        result = await self.redis.get(key)
        if result:
            return result
            
        # 计算结果
        result = await compute_func()
        
        # 存入缓存
        await self.redis.set(key, result, ex=3600)
        return result
```

## 预计算与缓存

### 常用计算结果预存储
```python
class PrecomputeCache:
    def __init__(self):
        self.cache = {}
        
    def precompute(self, inputs):
        """
        预计算并存储结果
        """
        for input in inputs:
            result = self.compute_expensive_function(input)
            self.cache[input] = result
            
    def get_result(self, input):
        return self.cache.get(input)
```

### 模型中间结果缓存
```python
class ModelCache:
    def __init__(self, model):
        self.model = model
        self.layer_outputs = {}
        
    def forward_with_cache(self, x):
        """
        前向传播时缓存中间结果
        """
        for layer in self.model.layers:
            x = layer(x)
            self.layer_outputs[layer.name] = x
        return x
        
    def get_intermediate(self, layer_name):
        return self.layer_outputs.get(layer_name)
```

### 增量计算优化
```python
class IncrementalCompute:
    def __init__(self):
        self.last_state = None
        self.last_result = None
        
    def compute(self, new_input):
        """
        增量计算，只计算变化部分
        """
        if self.last_state is None:
            result = self.full_compute(new_input)
        else:
            diff = self.compute_diff(new_input, self.last_state)
            result = self.update_result(self.last_result, diff)
            
        self.last_state = new_input
        self.last_result = result
        return result
```

## 最佳实践

### 缓存容量规划
1. 内存使用评估
```python
def estimate_memory_usage(cache_size, item_size):
    """
    估算缓存内存使用
    """
    total_memory = cache_size * item_size
    overhead = total_memory * 0.1  # 额外开销
    return total_memory + overhead
```

2. 容量限制设置
```python
def set_cache_limits():
    """
    设置缓存限制
    """
    max_memory = get_available_memory() * 0.8
    max_items = max_memory // average_item_size
    return max_items
```

### 命中率优化
1. 监控指标
```python
class CacheMetrics:
    def __init__(self):
        self.hits = 0
        self.misses = 0
        
    def record_access(self, hit):
        if hit:
            self.hits += 1
        else:
            self.misses += 1
            
    def hit_rate(self):
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0
```

2. 优化策略
- 预热缓存
- 动态TTL
- 智能预取

### 成本收益分析
| 缓存类型 | 内存占用 | 访问延迟 | 更新成本 | 适用场景 |
|---------|---------|---------|---------|---------|
| 本地缓存 | 中 | 低 | 低 | 单机部署 |
| 分布式缓存 | 高 | 中 | 高 | 集群部署 |
| KV缓存 | 低 | 极低 | 中 | 推理加速 |
| 结果缓存 | 高 | 低 | 高 | 查询优化 |

## 性能优化

### 1. 缓存预热
```python
async def warmup_cache():
    """
    系统启动时预热缓存
    """
    common_queries = load_common_queries()
    for query in common_queries:
        result = await compute_result(query)
        cache.set(query, result)
```

### 2. 并发访问优化
```python
from asyncio import Lock

class ThreadSafeCache:
    def __init__(self):
        self.cache = {}
        self.locks = {}
        
    async def get_or_compute(self, key, compute_func):
        if key not in self.locks:
            self.locks[key] = Lock()
            
        async with self.locks[key]:
            if key in self.cache:
                return self.cache[key]
                
            result = await compute_func()
            self.cache[key] = result
            return result
```

### 3. 内存优化
```python
class MemoryOptimizedCache:
    def __init__(self, max_memory_mb):
        self.max_memory = max_memory_mb * 1024 * 1024
        self.current_memory = 0
        self.cache = {}
        
    def add_to_cache(self, key, value):
        value_size = sys.getsizeof(value)
        if self.current_memory + value_size > self.max_memory:
            self._evict_until_fit(value_size)
        self.cache[key] = value
        self.current_memory += value_size
``` 