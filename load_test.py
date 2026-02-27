"""
Load test for the RAG API.

Usage:
    pip install aiohttp
    python load_test.py --url http://206.189.137.221 --users 50 --duration 60

Measures:
    - Total requests
    - Success / Failure count
    - P50, P95, P99 latency
    - Requests per second
"""

import asyncio
import aiohttp
import argparse
import time
import statistics
import json


async def send_health_check(session, base_url, results):
    """Single health check request with latency tracking."""
    start = time.perf_counter()
    try:
        async with session.get(f"{base_url}/health", timeout=aiohttp.ClientTimeout(total=10)) as resp:
            elapsed = (time.perf_counter() - start) * 1000  # ms
            results.append({
                "status": resp.status,
                "latency_ms": round(elapsed, 2),
                "success": resp.status == 200,
            })
    except Exception as e:
        elapsed = (time.perf_counter() - start) * 1000
        results.append({
            "status": 0,
            "latency_ms": round(elapsed, 2),
            "success": False,
            "error": str(e),
        })


async def send_ask_request(session, base_url, results):
    """Single /ask request with latency tracking."""
    payload = {"question": "What is the document about?", "session_id": "loadtest"}
    start = time.perf_counter()
    try:
        async with session.post(
            f"{base_url}/ask",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=30),
        ) as resp:
            await resp.read()
            elapsed = (time.perf_counter() - start) * 1000
            results.append({
                "status": resp.status,
                "latency_ms": round(elapsed, 2),
                "success": resp.status == 200,
            })
    except Exception as e:
        elapsed = (time.perf_counter() - start) * 1000
        results.append({
            "status": 0,
            "latency_ms": round(elapsed, 2),
            "success": False,
            "error": str(e),
        })


async def worker(session, base_url, results, duration, endpoint):
    """Continuously send requests until duration expires."""
    end_time = time.time() + duration
    fn = send_health_check if endpoint == "health" else send_ask_request
    while time.time() < end_time:
        await fn(session, base_url, results)


async def run_load_test(base_url, num_users, duration, endpoint):
    results = []
    connector = aiohttp.TCPConnector(limit=num_users)

    print(f"\n{'='*60}")
    print(f"  LOAD TEST: {endpoint.upper()} endpoint")
    print(f"  Target:    {base_url}/{endpoint}")
    print(f"  Users:     {num_users} concurrent")
    print(f"  Duration:  {duration}s")
    print(f"{'='*60}\n")

    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [
            worker(session, base_url, results, duration, endpoint)
            for _ in range(num_users)
        ]
        await asyncio.gather(*tasks)

    # Calculate metrics
    total = len(results)
    successes = sum(1 for r in results if r["success"])
    failures = total - successes
    latencies = sorted([r["latency_ms"] for r in results])

    if latencies:
        p50 = latencies[int(len(latencies) * 0.50)]
        p95 = latencies[int(len(latencies) * 0.95)]
        p99 = latencies[int(min(len(latencies) * 0.99, len(latencies) - 1))]
        avg = statistics.mean(latencies)
        rps = total / duration
    else:
        p50 = p95 = p99 = avg = rps = 0

    report = {
        "endpoint": endpoint,
        "concurrent_users": num_users,
        "duration_seconds": duration,
        "total_requests": total,
        "successes": successes,
        "failures": failures,
        "requests_per_second": round(rps, 2),
        "latency_ms": {
            "avg": round(avg, 2),
            "p50": round(p50, 2),
            "p95": round(p95, 2),
            "p99": round(p99, 2),
            "min": round(min(latencies), 2) if latencies else 0,
            "max": round(max(latencies), 2) if latencies else 0,
        },
    }

    print(f"\n{'='*60}")
    print(f"  RESULTS")
    print(f"{'='*60}")
    print(f"  Total requests:      {total}")
    print(f"  Successes:           {successes}")
    print(f"  Failures:            {failures}")
    print(f"  Requests/sec:        {report['requests_per_second']}")
    print(f"  Avg latency:         {report['latency_ms']['avg']}ms")
    print(f"  P50 latency:         {report['latency_ms']['p50']}ms")
    print(f"  P95 latency:         {report['latency_ms']['p95']}ms")
    print(f"  P99 latency:         {report['latency_ms']['p99']}ms")
    print(f"  Min latency:         {report['latency_ms']['min']}ms")
    print(f"  Max latency:         {report['latency_ms']['max']}ms")
    print(f"{'='*60}\n")

    # Save report
    with open("load_test_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Report saved to load_test_report.json\n")

    return report


def main():
    parser = argparse.ArgumentParser(description="RAG API Load Tester")
    parser.add_argument("--url", default="http://localhost:8000", help="Base URL")
    parser.add_argument("--users", type=int, default=50, help="Concurrent users")
    parser.add_argument("--duration", type=int, default=60, help="Duration in seconds")
    parser.add_argument("--endpoint", default="health", choices=["health", "ask"],
                        help="Endpoint to test (health for infra, ask for full RAG)")
    args = parser.parse_args()

    asyncio.run(run_load_test(args.url, args.users, args.duration, args.endpoint))


if __name__ == "__main__":
    main()
