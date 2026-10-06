"""
Persistence, one small class per concern. Each store holds a synchronous SQLAlchemy engine and
runs in a worker thread (asyncio.to_thread or an executor), where a blocking driver is correct.
A caller depends only on the stores it uses; none of them knows about HTTP or about workers.
"""
