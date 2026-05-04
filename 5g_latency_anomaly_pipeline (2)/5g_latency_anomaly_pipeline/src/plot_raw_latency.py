import argparse
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import DOMAINS, RAW_DATA_DIR


class DatasetGenerator:
    def __init__(self, strategy=1, days=10, seed=42):
        self.strategy = strategy
        self.days = days
        self.seed = seed
        self.start_date = datetime(2026, 1, 1)
        self.total_timestamps = days * 24 * 60
        self.rng = np.random.default_rng(seed)

    def generate(self):
        if self.strategy == 1:
            return self._strategy1()
        if self.strategy == 2:
            return self._strategy2()
        if self.strategy == 3:
            return self._strategy3()
        raise ValueError("Invalid strategy. Choose 1, 2, or 3.")

    def save(self, output_path):
        df = self.generate()
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Saved: {output_path}")
        return df

    # ===================================================================
    # STRATEGY 1: Uniform + Normal noise (unchanged)
    # ===================================================================
    def _strategy1(self):
        rows = []
        current_time = self.start_date

        for _ in range(self.total_timestamps):
            row = {"timestamp": current_time, "slice_id": "slice_1"}
            total_latency = 0.0

            for domain in DOMAINS:
                cpu = self.rng.uniform(25, 75)
                memory = self.rng.uniform(20, 70)
                disk = self.rng.uniform(30, 80)
                bandwidth = self.rng.uniform(20, 80)

                cpu += self.rng.normal(0, 3.0)
                memory += self.rng.normal(0, 2.5)
                disk += self.rng.normal(0, 2.0)
                bandwidth += self.rng.normal(0, 4.0)

                cpu = np.clip(cpu, 20, 100)
                memory = np.clip(memory, 10, 100)
                disk = np.clip(disk, 30, 100)
                bandwidth = np.clip(bandwidth, 10, 100)

                latency = (cpu + memory) / 8000 + bandwidth / 1000 + disk / 15000

                self._add_domain_metrics(row, domain, cpu, memory, disk, bandwidth)
                total_latency += latency

            row["end_to_end_latency"] = total_latency
            row["is_anomaly"] = 0
            row["anomaly_domain"] = "none"
            rows.append(row)
            current_time += timedelta(minutes=1)

        return pd.DataFrame(rows)

    # ===================================================================
    # STRATEGY 2: HIGHLY VOLATILE + SPIKY (matches your new plot)
    # ===================================================================
    def _strategy2(self):
        """Highly volatile and spiky latency - matches the blue line in your latest plot."""
        rows = []
        current_time = self.start_date

        # Initial state for each domain
        state = {
            domain: {
                "cpu": self.rng.uniform(30, 60),
                "memory": self.rng.uniform(25, 55),
                "disk": self.rng.uniform(35, 65),
                "bandwidth": self.rng.uniform(30, 70),
            }
            for domain in DOMAINS
        }

        for step in range(self.total_timestamps):
            row = {"timestamp": current_time, "slice_id": "slice_1"}
            total_latency = 0.0
            anomaly_domains = []

            for domain in DOMAINS:
                prev = state[domain]

                # Strong volatility + random spikes (this creates the spiky pattern you want)
                cpu = prev["cpu"] + self.rng.normal(0, 6.0)
                memory = prev["memory"] + self.rng.normal(0, 5.0)
                disk = prev["disk"] + self.rng.normal(0, 7.0)
                bandwidth = prev["bandwidth"] + self.rng.normal(0, 8.0)

                # Occasional large random spikes (causes sharp latency jumps)
                if self.rng.random() < 0.08:   # 8% chance per timestep per domain
                    disk += self.rng.uniform(20, 45)
                    bandwidth -= self.rng.uniform(15, 35)

                cpu = np.clip(cpu, 20, 100)
                memory = np.clip(memory, 10, 100)
                disk = np.clip(disk, 30, 100)
                bandwidth = np.clip(bandwidth, 5, 100)

                # Short anomalies
                if self.rng.random() < 0.001:
                    cpu = self.rng.uniform(88, 98)
                    memory = self.rng.uniform(82, 95)
                    bandwidth = self.rng.uniform(8, 25)
                    anomaly_domains.append(domain)

                latency = (cpu + memory) / 8000 + bandwidth / 1000 + disk / 15000

                self._add_domain_metrics(row, domain, cpu, memory, disk, bandwidth)
                total_latency += latency

                state[domain] = {"cpu": cpu, "memory": memory, "disk": disk, "bandwidth": bandwidth}

            row["end_to_end_latency"] = total_latency + self.rng.normal(0, 0.8)
            row["is_anomaly"] = int(bool(anomaly_domains))
            row["anomaly_domain"] = "|".join(anomaly_domains) if anomaly_domains else "none"
            rows.append(row)
            current_time += timedelta(minutes=1)

        return pd.DataFrame(rows)

    # Strategy 3 remains unchanged
    def _strategy3(self):
        def queue_delay(utilization):
            rho = min(utilization / 100, 0.98)
            return 1 / (1 - rho)

        state = {domain: {"cpu": self.rng.uniform(30, 50),
                          "memory": self.rng.uniform(30, 50),
                          "disk": self.rng.uniform(40, 60),
                          "bandwidth": self.rng.uniform(30, 50)}
                 for domain in DOMAINS}
        anomaly_counter = {domain: 0 for domain in DOMAINS}

        rows = []
        current_time = self.start_date

        for _ in range(self.total_timestamps):
            row = {"timestamp": current_time, "slice_id": "slice_1"}
            anomaly_domains = []

            for domain in DOMAINS:
                prev = state[domain]

                if anomaly_counter[domain] > 0:
                    is_anomaly = True
                    anomaly_counter[domain] -= 1
                elif self.rng.random() < 0.0005:
                    is_anomaly = True
                    anomaly_counter[domain] = 10
                else:
                    is_anomaly = False

                cpu = prev["cpu"] + self.rng.normal(0, 2)
                memory = 0.7 * cpu + 0.3 * prev["memory"] + self.rng.normal(0, 3)
                disk = prev["disk"] + self.rng.normal(0, 0.5)
                bandwidth = 100 - cpu + self.rng.normal(0, 5)

                if is_anomaly:
                    cpu = self.rng.uniform(90, 100)
                    memory = self.rng.uniform(85, 100)
                    bandwidth = self.rng.uniform(5, 20)
                    anomaly_domains.append(domain)

                cpu, memory = np.clip(cpu, 20, 100), np.clip(memory, 10, 100)
                disk, bandwidth = np.clip(disk, 30, 100), np.clip(bandwidth, 5, 100)
                state[domain] = {"cpu": cpu, "memory": memory, "disk": disk, "bandwidth": bandwidth}
                self._add_domain_metrics(row, domain, cpu, memory, disk, bandwidth)

            row["current_latency"] = sum(queue_delay(state[d]["cpu"]) for d in DOMAINS)
            row["is_anomaly"] = int(bool(anomaly_domains))
            row["anomaly_domain"] = "|".join(anomaly_domains) if anomaly_domains else "none"
            rows.append(row)
            current_time += timedelta(minutes=1)

        df = pd.DataFrame(rows)
        df["end_to_end_latency"] = df["current_latency"].shift(-1)
        return df.dropna().reset_index(drop=True)

    @staticmethod
    def _add_domain_metrics(row, domain, cpu, memory, disk, bandwidth):
        row[f"{domain}_cpu"] = cpu
        row[f"{domain}_memory"] = memory
        row[f"{domain}_disk"] = disk
        row[f"{domain}_bandwidth"] = bandwidth


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic 5G slice latency data.")
    parser.add_argument("--strategy", type=int, choices=[1, 2, 3], default=2)
    parser.add_argument("--days", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    output = args.output or RAW_DATA_DIR / f"strategy{args.strategy}_data.csv"
    DatasetGenerator(args.strategy, args.days, args.seed).save(output)


if __name__ == "__main__":
    main()