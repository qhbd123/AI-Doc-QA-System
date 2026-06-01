# Redis 缓存管理模块（支持统计、动态前缀、批量操作）
import redis
import os
import time
from typing import Optional, List, Dict
from dotenv import load_dotenv

# 从项目根目录的 .env 文件中读取环境变量，并将其加载到 Python 的 os.environ 字典中,后续代码就可以通过 os.getenv() 来获取配置信息
load_dotenv()


class CacheClient:
    """封装 Redis 缓存客户端，提供统计、动态前缀和批量操作"""

    def __init__(self):  # 构造函数/初始化方法
        # 配置
        self.prefix = os.getenv("CACHE_PREFIX", "doc_qa:")  # 动态前缀
        self.expire = int(os.getenv("CACHE_EXPIRE", 3600))  # 缓存过期时间
        # 统计信息
        self.stats = {
            "get_total": 0,
            "get_hit": 0,
            "get_miss": 0,
            "get_total_time_ms": 0,
            "set_total": 0,
            "set_total_time_ms": 0,
        }
        # 连接 Redis
        self.client = None  # 防御性编程，先声明属性，根据是否连接成功再赋值，后续方法也根据此属性判断是否直接返回或跳过，避免报错
        try:
            # 客户端初始化
            self.client = redis.Redis(
                host=os.getenv("REDIS_HOST"),
                port=int(os.getenv("REDIS_PORT")),
                db=int(os.getenv("REDIS_DB")),
                decode_responses=True,  # 指示 Redis 客户端自动将返回的二进制数据（bytes）解码为字符串（str）
                socket_timeout=5  # 套接字超时时间
            )
            # 向 Redis 服务器发送一个 PING 命令，用于测试连接是否正常
            self.client.ping()
            print("[cache] Redis 连接成功")
        except Exception as e:
            print(f"[cache] Redis 连接失败: {e}")

    def _make_key(self, key: str) -> str:
        """为传入的原始 key 添加一个统一的前缀（self.prefix），生成最终存储在 Redis 中的完整 key，让缓存 key 的生成规则统一且易于维护"""
        return f"{self.prefix}{key}"

    def get(self, key: str) -> Optional[str]:
        """获取单个缓存，自动更新统计"""
        if not self.client:  # 防御性检查：检查 self.client 是否为 None，即 Redis 连接是否成功建立
            return None
        start = time.perf_counter()  # 高精度的时间戳
        self.stats["get_total"] += 1
        try:
            full_key = self._make_key(key)  # 对原始键调用 _make_key(key) 加上前缀，生成完整键名
            val = self.client.get(full_key)  # 执行 Redis 的 GET 命令，根据完整键名获取值
            if val is not None:
                self.stats["get_hit"] += 1
            else:
                self.stats["get_miss"] += 1
            elapsed_ms = (time.perf_counter() - start) * 1000
            self.stats["get_total_time_ms"] += elapsed_ms  # 总耗时
            print(f"[cache] {'命中' if val else '未命中'} | {key[:30]}... | {elapsed_ms:.2f}ms")
            return val
        except Exception as e:
            print(f"[cache] 读取异常: {e}")
            return None

    def set(self, key: str, value: str) -> None:
        """写入单个缓存，自动过期"""
        if not self.client:  # 防御性检查：Redis 连接是否成功建立
            return
        start = time.perf_counter()
        self.stats["set_total"] += 1
        try:
            full_key = self._make_key(key)  # 对原始键调用 _make_key(key) 加上前缀，生成完整键名
            self.client.setex(full_key, self.expire, value)  # 执行 Redis 的 SETEX 命令，设置键值对并指定过期时间，单位秒
            elapsed_ms = (time.perf_counter() - start) * 1000
            self.stats["set_total_time_ms"] += elapsed_ms  # 总耗时
            print(f"[cache] 写入成功 | {key[:30]}... | {elapsed_ms:.2f}ms")
        except Exception as e:
            print(f"[cache] 写入异常: {e}")

    def get_many(self, keys: List[str]) -> Dict[str, Optional[str]]:
        """批量获取缓存，返回字典 {key: value}，不存在的 key 对应 None"""
        if not self.client or not keys:  # 防御性检查：Redis 连接是否成功建立 + key是否为空列表
            return {k: None for k in keys}
        start = time.perf_counter()
        full_keys = [self._make_key(k) for k in keys]  # 对原始键调用 _make_key(key) 加上前缀，生成完整键名
        try:
            values = self.client.mget(full_keys)  # 执行 Redis 的 MGET 命令，一次获取多个键的值
            result = {keys[i]: values[i] for i in range(len(keys))}  # 构造返回字典，原始键key对应value
            elapsed_ms = (time.perf_counter() - start) * 1000
            self.stats["get_total"] += len(keys)
            hit_cnt = sum(1 for v in values if v is not None)  # 计算命中次数
            self.stats["get_hit"] += hit_cnt
            self.stats["get_miss"] += (len(keys) - hit_cnt)
            self.stats["get_total_time_ms"] += elapsed_ms  # 总耗时
            print(f"[cache] 批量读取 {len(keys)} 个 key，命中 {hit_cnt} 个，耗时 {elapsed_ms:.2f}ms")
            return result
        except Exception as e:
            print(f"[cache] 批量读取异常: {e}")
            return {k: None for k in keys}

    def set_many(self, mapping: Dict[str, str]) -> None:
        """批量写入缓存（每个 key 单独设置过期时间，使用 pipeline 提高效率）"""
        if not self.client or not mapping:  # 防御性检查：Redis 连接是否成功建立 + mapping是否为空字典
            return
        start = time.perf_counter()
        try:
            # 创建一个 Redis 管道（Pipeline）对象：将多个命令缓存在客户端，最后一次性发送到 Redis 服务器，减少网络往返
            pipe = self.client.pipeline()
            for key, value in mapping.items():
                full_key = self._make_key(key)  # 对原始键调用 _make_key(key) 加上前缀，生成完整键名
                pipe.setex(full_key, self.expire, value)  # 管道中添加一个 SETEX 命令，设置键值对并指定过期时间，单位秒
            pipe.execute()  # 将管道中累积的所有命令一次性发送到 Redis 服务器执行
            elapsed_ms = (time.perf_counter() - start) * 1000
            self.stats["set_total"] += len(mapping)
            self.stats["set_total_time_ms"] += elapsed_ms  # 总耗时
            print(f"[cache] 批量写入 {len(mapping)} 个 key，耗时 {elapsed_ms:.2f}ms")
        except Exception as e:
            print(f"[cache] 批量写入异常: {e}")

    def get_stats(self) -> Dict:
        """获取缓存统计信息"""
        total_get = self.stats["get_total"]
        hit_rate = self.stats["get_hit"] / total_get if total_get > 0 else 0  # 计算命中率
        avg_get_time = self.stats["get_total_time_ms"] / total_get if total_get > 0 else 0  # get操作的平均用时
        avg_set_time = self.stats["set_total_time_ms"] / self.stats["set_total"] if self.stats[
                                                                                        "set_total"] > 0 else 0  # set操作的平均用时
        return {
            "get_requests": total_get,
            "get_hits": self.stats["get_hit"],
            "get_misses": self.stats["get_miss"],
            "hit_rate": round(hit_rate, 4),
            "avg_get_latency_ms": round(avg_get_time, 2),
            "set_requests": self.stats["set_total"],
            "avg_set_latency_ms": round(avg_set_time, 2),
        }

    # ========== 清除所有问答缓存 ==========
    def clear_qa_cache(self) -> int:
        """清除所有问答缓存（即所有带此前缀的 key）"""
        if not self.client:  # # 防御性检查：Redis 连接是否成功建立
            return 0
        count = 0
        pattern = f"{self.prefix}*"  # 构造扫描匹配模式：匹配所有以该前缀开头的键
        # 通过 Redis 的 SCAN 命令，分批（每批约100个键）遍历并返回所有匹配 pattern 模式的键，避免阻塞 Redis 服务
        for key in self.client.scan_iter(match=pattern, count=100):
            self.client.delete(key)  # Redis 的 DEL 命令，删除键
            count += 1
        print(f"[cache] 已清除 {count} 个问答缓存")
        return count


# 全局单例：1.只初始化一次 Redis 连接；2.统计信息（self.stats）在所有调用中共享，正确累计;3.封装实现细节
# 创建一个 CacheClient 类的实例对象，并将其赋值给模块级的私有变量 _cache_client
_cache_client = CacheClient()


# 为保持向后兼容，保留原有的函数接口
# 包装 _cache_client.get 方法，对外提供 get_cache(key) 函数
# 调用者只需传入原始 key，无需关心前缀和统计细节
def get_cache(key: str) -> Optional[str]:
    return _cache_client.get(key)


def set_cache(key: str, value: str) -> None:
    _cache_client.set(key, value)


def get_many_cache(keys: List[str]) -> Dict[str, Optional[str]]:
    return _cache_client.get_many(keys)


def set_many_cache(mapping: Dict[str, str]) -> None:
    _cache_client.set_many(mapping)


def get_cache_stats() -> Dict:
    return _cache_client.get_stats()


def clear_qa_cache() -> int:
    return _cache_client.clear_qa_cache()


# 可选：导出 redis_client 保持原有兼容
# 将 CacheClient 实例内部的原始 Redis 客户端对象（即 _cache_client.client）导出为模块级变量 redis_client
# 可以直接使用 redis_client 执行原生 Redis 命令，而无需重新创建连接
redis_client = _cache_client.client
