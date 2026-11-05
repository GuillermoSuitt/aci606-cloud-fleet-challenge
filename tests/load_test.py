"""Prueba de carga: envía solicitudes concurrentes a POST /predict y mide la latencia
vista desde el cliente (como lo hará el evaluador).

Uso:
    python tests/load_test.py --api https://TU-URL.onrender.com --requests 200 --concurrency 10
"""
from __future__ import annotations

import argparse
import random
import statistics
import time
from concurrent.futures import ThreadPoolExecutor

import requests


def muestra(i: int) -> dict:
    """Genera una lectura de sensores plausible y distinta en cada solicitud."""
    return {
        "robot_id": f"LOAD-{i % 20:02d}",
        "timestamp": time.time(),
        "x": random.uniform(0, 10),
        "y": random.uniform(0, 10),
        "speed": random.uniform(0.3, 1.2),
        "battery": random.uniform(45, 100),
        "motor_temp": random.uniform(40, 82),
        "vibration": random.uniform(0.1, 1.9),
        "motor_current": random.uniform(3, 7),
        "obstacle_distance": random.uniform(0.05, 4),
    }


def percentil(valores: list[float], p: float) -> float:
    ordenados = sorted(valores)
    return ordenados[min(len(ordenados) - 1, int(p * len(ordenados)))]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", required=True)
    parser.add_argument("--requests", type=int, default=200)
    parser.add_argument("--concurrency", type=int, default=10)
    args = parser.parse_args()
    url = args.api.rstrip("/") + "/predict"

    print("Despertando el servicio (GET /health)...")
    requests.get(args.api.rstrip("/") + "/health", timeout=120)

    sesion = requests.Session()

    def una(i: int) -> tuple[int, float]:
        inicio = time.perf_counter()
        try:
            r = sesion.post(url, json=muestra(i), timeout=30)
            codigo = r.status_code
        except requests.RequestException:
            codigo = 0
        return codigo, (time.perf_counter() - inicio) * 1000

    print(f"Enviando {args.requests} solicitudes a {url} con {args.concurrency} en paralelo...")
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        resultados = list(pool.map(una, range(args.requests)))
    duracion = time.perf_counter() - t0

    latencias = [ms for codigo, ms in resultados if codigo == 200]
    errores = [codigo for codigo, _ in resultados if codigo != 200]

    print("\n=== Resultado ===")
    print(f"Exitosas:        {len(latencias)} / {args.requests}")
    print(f"Errores:         {len(errores)} {sorted(set(errores)) if errores else ''}")
    print(f"Duración total:  {duracion:.1f} s  ({args.requests / duracion:.1f} solicitudes/s)")
    if latencias:
        print(f"Latencia p50:    {percentil(latencias, 0.50):.0f} ms")
        print(f"Latencia p95:    {percentil(latencias, 0.95):.0f} ms")
        print(f"Latencia máx.:   {max(latencias):.0f} ms")
        print(f"Latencia media:  {statistics.mean(latencias):.0f} ms")


if __name__ == "__main__":
    main()
