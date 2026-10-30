"""Bi-directional threshold controller with recovery hysteresis (SPEC 3.5)."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from .config import CascadeConfig


def p95(values) -> float:
    vals = sorted(values)
    if not vals:
        return 0.0
    k = max(0, min(len(vals) - 1, int(round(0.95 * (len(vals) - 1)))))
    return vals[k]


@dataclass
class AdaptationController:
    cfg: CascadeConfig = field(default_factory=CascadeConfig)
    enabled: bool = True

    def __post_init__(self):
        self.tau = self.cfg.tau_base
        self.tier1_passes: deque[bool] = deque(maxlen=self.cfg.window)
        self.audits: deque[bool] = deque(maxlen=self.cfg.audit_window)  # shadow-audit agreement samples
        self.latencies: deque[float] = deque(maxlen=self.cfg.window)
        self.healthy_streak = 0
        self.steps_since_tighten = self.cfg.tighten_cooldown
        self.history: list[dict] = []

    @property
    def pass_rate(self) -> float | None:
        """Estimated Tier-1 success = P(verifier pass) x P(audit agrees | verifier pass).

        Verifier failures are certain errors; shadow audits estimate the confident-but-wrong share of
        verifier passes. With fewer than `min_audit_samples` audits the second factor is taken as 1.
        """
        if len(self.tier1_passes) < self.cfg.min_window_samples:
            return None
        verifier_pass = sum(self.tier1_passes) / len(self.tier1_passes)
        if len(self.audits) >= self.cfg.min_audit_samples:
            return verifier_pass * sum(self.audits) / len(self.audits)
        return verifier_pass

    def observe(self, latency_ms: float, tier1_passed: bool | None, audit_agree: bool | None = None) -> float:
        """Record one completed request (tier1_passed=None if Tier-1 was not attempted; audit_agree=None if
        no shadow audit ran); returns the new tau."""
        self.latencies.append(latency_ms)
        if tier1_passed is not None:
            self.tier1_passes.append(tier1_passed)
        if audit_agree is not None:
            self.audits.append(audit_agree)
        self.steps_since_tighten += 1
        action = "hold"
        if self.enabled:
            rate, lat95, slo = self.pass_rate, p95(self.latencies), self.cfg.latency_slo_ms
            overloaded = len(self.latencies) >= self.cfg.min_window_samples and lat95 > slo
            failing = rate is not None and rate < self.cfg.tighten_pass_rate
            healthy = rate is not None and rate >= self.cfg.recover_pass_rate and \
                lat95 < self.cfg.recover_latency_frac * slo
            if (failing or overloaded) and self.steps_since_tighten >= self.cfg.tighten_cooldown:
                self.tau = max(self.cfg.tau_min, self.tau - self.cfg.tighten_step)
                self.steps_since_tighten = 0
                self.healthy_streak = 0
                self.tier1_passes.clear()  # next decision must rest on fresh post-tighten evidence
                self.audits.clear()
                action = "tighten"
            elif healthy:
                self.healthy_streak += 1
                if self.healthy_streak >= self.cfg.recover_consecutive and self.tau < self.cfg.tau_base:
                    self.tau = min(self.cfg.tau_base, self.tau + self.cfg.recover_step)
                    action = "recover"
            else:
                self.healthy_streak = 0
        self.tau = round(min(self.cfg.tau_max, max(self.cfg.tau_min, self.tau)), 4)
        self.history.append({"tau": self.tau, "action": action, "pass_rate": self.pass_rate,
                             "p95_ms": round(p95(self.latencies), 2)})
        return self.tau
