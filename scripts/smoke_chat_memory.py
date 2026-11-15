"""Manual check of chat memory against REAL Redis.

Run from the project root:  python -m scripts.smoke_chat_memory
"""

from redis import Redis

from app.core.config import get_settings
from app.services.chat_memory import ChatMemory, new_conversation_id


def main() -> None:
    settings = get_settings()
    redis_client = Redis.from_url(settings.redis_url, decode_responses=True)
    # A short TTL so this script cleans up after itself.
    memory = ChatMemory(redis_client, max_messages=6, ttl_seconds=600)

    first, second = new_conversation_id(), new_conversation_id()
    for n in range(1, 5):
        memory.append_exchange(first, f"question {n}", f"answer {n}")
    memory.append_exchange(second, "hello from another conversation", "hi")

    print(f"conversation 1: {first}  (4 exchanges stored, limit is 6 messages)")
    for message in memory.get_history(first):
        print(f"  {message.role:9} {message.content}")
    print(f"conversation 2 has {len(memory.get_history(second))} messages (isolated)")

    key = ChatMemory.key(first)
    print(f"\nRedis key: {key}")
    print(
        f"type={redis_client.type(key)} length={redis_client.llen(key)} "
        f"ttl={redis_client.ttl(key)}s"
    )
    print(f"\nInspect it yourself:\n  docker compose exec redis redis-cli lrange {key} 0 -1")


if __name__ == "__main__":
    main()