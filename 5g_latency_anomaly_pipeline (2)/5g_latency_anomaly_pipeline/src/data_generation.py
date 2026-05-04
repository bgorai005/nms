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
    # STRATEGY 1: Uniform base + Normal noise (smooth variation)
    # ===================================================================
    def _strategy1(self):
        """Strategy 1: Uniform distribution with normal noise - smooth variation."""
        rows = []
        current_time = self.start_date

        for _ in range(self.total_timestamps):
            row = {"timestamp": current_time, "slice_id": "slice_1"}
            total_latency = 0.0

            for domain in DOMAINS:
                # Uniform base values
                cpu = self.rng.uniform(25, 75)
                memory = self.rng.uniform(20, 70)
                disk = self.rng.uniform(30, 80)
                bandwidth = self.rng.uniform(20, 80)

                # Add realistic normal noise
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
    # STRATEGY 2: Sharp Step-wise Function (Long flat plateaus + sudden jumps)
    # ===================================================================
    def _strategy2(self):
        """Strategy 2: Sharp step-wise regimes with long flat plateaus + sudden jumps.
        This matches the blue 'Actual' line in your plot."""
        rows = []
        current_time = self.start_date
        block_size = 180  # 3 hours per regime (change to 120/240/360 for different plateau lengths)

        # Initial state
        state = {
            domain: {
                "cpu": self.rng.uniform(30, 55),
                "memory": self.rng.uniform(25, 50),
                "disk": self.rng.uniform(35, 65),
                "bandwidth": self.rng.uniform(35, 70),
            }
            for domain in DOMAINS
        }

        for step in range(self.total_timestamps):
            row = {"timestamp": current_time, "slice_id": "slice_1"}
            total_latency = 0.0
            anomaly_domains = []

            # === SHARP REGIME CHANGE (Step function) ===
            if step % block_size == 0:
                profile = self._strategy2_step_profile(step, block_size)
                for domain in DOMAINS:
                    # Strong sharp jump to new regime level
                    state[domain]["cpu"] = self._safe_poisson(45 * profile["cpu"])
                    state[domain]["memory"] = self._safe_poisson(40 * profile["memory"])
                    state[domain]["disk"] = self._safe_poisson(55 * profile["disk"])
                    state[domain]["bandwidth"] = self._safe_poisson(60 * profile["bandwidth"])

            for domain in DOMAINS:
                prev = state[domain]

                # VERY SMALL noise → creates long flat plateaus
                cpu = prev["cpu"] + self.rng.normal(0, 0.4)
                memory = prev["memory"] + self.rng.normal(0, 0.35)
                disk = prev["disk"] + self.rng.normal(0, 0.3)
                bandwidth = prev["bandwidth"] + self.rng.normal(0, 0.6)

                cpu, memory = np.clip(cpu, 20, 100), np.clip(memory, 10, 100)
                disk, bandwidth = np.clip(disk, 30, 100), np.clip(bandwidth, 5, 100)

                # Occasional short anomaly
                if self.rng.random() < 0.0008:
                    cpu = self.rng.uniform(88, 98)
                    memory = self.rng.uniform(82, 95)
                    bandwidth = self.rng.uniform(8, 25)
                    anomaly_domains.append(domain)

                latency = (cpu + memory) / 8000 + bandwidth / 1000 + disk / 15000

                self._add_domain_metrics(row, domain, cpu, memory, disk, bandwidth)
                total_latency += latency

                # Update state
                state[domain] = {"cpu": cpu, "memory": memory, "disk": disk, "bandwidth": bandwidth}

            row["end_to_end_latency"] = total_latency + self.rng.normal(0, 0.5)
            row["is_anomaly"] = int(bool(anomaly_domains))
            row["anomaly_domain"] = "|".join(anomaly_domains) if anomaly_domains else "none"
            rows.append(row)
            current_time += timedelta(minutes=1)

        return pd.DataFrame(rows)

    def _strategy2_step_profile(self, step_idx, block_size):
        """Define different traffic regimes for sharp step changes."""
        minute_of_day = step_idx % (24 * 60)
        block_of_day = minute_of_day // block_size
        profiles = [
            {"cpu": 0.75, "memory": 0.80, "disk": 0.90, "bandwidth": 1.20},  # Low load
            {"cpu": 1.00, "memory": 1.00, "disk": 1.00, "bandwidth": 1.00},  # Normal
            {"cpu": 1.35, "memory": 1.25, "disk": 2.50, "bandwidth": 0.85},  # Peak
            {"cpu": 1.10, "memory": 1.15, "disk": 1.80, "bandwidth": 0.95},  # Evening
        ]
        return profiles[block_of_day % len(profiles)]

    def _safe_poisson(self, lam):
        return float(self.rng.poisson(max(lam, 1.0)))

    # ===================================================================
    # STRATEGY 3 (Unchanged - already realistic)
    # ===================================================================
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
    parser.add_argument("--strategy", type=int, choices=[1, 2, 3], default=3)
    parser.add_argument("--days", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    output = args.output or RAW_DATA_DIR / f"strategy{args.strategy}_data.csv"
    DatasetGenerator(args.strategy, args.days, args.seed).save(output)


if __name__ == "__main__":
    main()