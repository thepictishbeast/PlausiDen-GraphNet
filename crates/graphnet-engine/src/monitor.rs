//! Resource monitoring + cost estimation — Phase 8.
//!
//! Owner direction: "RAM, GPU, and CPU usage also and display how much each
//! costs in compute". Engine-side ships:
//!
//! - [`ResourceSample`] — point-in-time snapshot of CPU + RAM (host) +
//!   process-resident metrics. GPU metrics come via the Python `pynvml`
//!   adapter (Phase 8 Python side); kept out of the Rust core to avoid an
//!   nvml dependency.
//! - [`ResourceMonitor`] — periodic sampling helper backed by `sysinfo`.
//! - [`flop_estimate`] — analytical FLOP counter for Stack forward.
//! - [`CostEstimate`] + [`estimate_cost`] — kWh-per-hour × electricity rate
//!   × wall-time → $ + CO₂-equivalent.

use serde::{Deserialize, Serialize};
use sysinfo::{Pid, System};

/// Point-in-time CPU + RAM observation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceSample {
    /// ISO-8601 capture timestamp (UTC).
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// Host-wide CPU utilisation, 0.0–1.0 (averaged across all cores).
    pub host_cpu_load: f64,
    /// Host total RAM in bytes.
    pub host_ram_total: u64,
    /// Host used RAM in bytes.
    pub host_ram_used: u64,
    /// Current process resident memory in bytes (zero if unobservable).
    pub proc_ram_used: u64,
    /// Number of logical CPUs visible to the host.
    pub cpu_count: usize,
}

impl ResourceSample {
    /// Host-wide RAM utilisation as a fraction in 0.0–1.0.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn host_ram_fraction(&self) -> f64 {
        if self.host_ram_total == 0 {
            return 0.0;
        }
        self.host_ram_used as f64 / self.host_ram_total as f64
    }

    /// Process RAM utilisation as a fraction of host total in 0.0–1.0.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn proc_ram_fraction(&self) -> f64 {
        if self.host_ram_total == 0 {
            return 0.0;
        }
        self.proc_ram_used as f64 / self.host_ram_total as f64
    }
}

/// Lightweight wrapper around `sysinfo::System` that publishes
/// [`ResourceSample`] observations on demand.
pub struct ResourceMonitor {
    system: System,
    self_pid: Pid,
}

impl ResourceMonitor {
    /// Construct a new monitor.
    #[must_use]
    pub fn new() -> Self {
        let mut system = System::new_all();
        system.refresh_all();
        let self_pid = sysinfo::get_current_pid().unwrap_or_else(|_| Pid::from(0));
        Self { system, self_pid }
    }

    /// Refresh + capture a single sample.
    pub fn sample(&mut self) -> ResourceSample {
        self.system.refresh_cpu_all();
        self.system.refresh_memory();
        self.system
            .refresh_processes(sysinfo::ProcessesToUpdate::All, true);

        let cpus = self.system.cpus();
        let cpu_count = cpus.len();
        let cpu_load = if cpu_count == 0 {
            0.0
        } else {
            let sum: f64 = cpus.iter().map(|c| f64::from(c.cpu_usage())).sum();
            #[allow(clippy::cast_precision_loss)]
            let avg = sum / cpu_count as f64;
            (avg / 100.0).clamp(0.0, 1.0)
        };

        let proc_ram_used = self
            .system
            .process(self.self_pid)
            .map(sysinfo::Process::memory)
            .unwrap_or(0);

        ResourceSample {
            timestamp: chrono::Utc::now(),
            host_cpu_load: cpu_load,
            host_ram_total: self.system.total_memory(),
            host_ram_used: self.system.used_memory(),
            proc_ram_used,
            cpu_count,
        }
    }
}

impl Default for ResourceMonitor {
    fn default() -> Self {
        Self::new()
    }
}

/// Analytical FLOP estimate for one Stack forward at dimensionality `D`
/// with `n_ops` operations.
///
/// Counts:
/// - Each Dense / HrrBind / Identity contributes O(D) operations (one
///   multiply-add per dim).
/// - Bundle of `n_ops` outputs contributes O(n_ops × D) (sum + sign).
///
/// This is a back-of-envelope number for cost dashboards; the actual cost
/// of the rustfft HRR path is higher by a log(D) factor that's amortised
/// over the host's FFT planner.
#[must_use]
pub fn flop_estimate(dim: usize, n_ops: usize) -> u64 {
    let per_op = u64::try_from(dim).unwrap_or(u64::MAX);
    let n = u64::try_from(n_ops).unwrap_or(u64::MAX);
    per_op
        .saturating_mul(n)
        .saturating_add(per_op.saturating_mul(n))
}

/// Energy + cost estimate for a body of work.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostEstimate {
    /// Wall-clock seconds elapsed.
    pub wall_secs: f64,
    /// Assumed host TDP in watts (or measured if pynvml present).
    pub watts: f64,
    /// Joules consumed (= watts × wall_secs).
    pub joules: f64,
    /// kWh consumed (= joules / 3_600_000).
    pub kwh: f64,
    /// USD electricity cost (= kwh × cents_per_kwh / 100).
    pub usd: f64,
    /// Grams of CO₂-equivalent (kwh × g_co2_per_kwh).
    pub g_co2: f64,
}

/// Compute a cost estimate given wall-time, host TDP, and grid intensity.
///
/// Defaults a typical operator should know:
/// - desktop with one mid-range GPU under load: ~300 W
/// - US average electricity: ~16 cents / kWh (2026 rough estimate)
/// - US grid carbon intensity: ~400 g CO₂e / kWh
#[must_use]
pub fn estimate_cost(
    wall_secs: f64,
    watts: f64,
    cents_per_kwh: f64,
    g_co2_per_kwh: f64,
) -> CostEstimate {
    let joules = wall_secs * watts;
    let kwh = joules / 3_600_000.0;
    let usd = kwh * cents_per_kwh / 100.0;
    let g_co2 = kwh * g_co2_per_kwh;
    CostEstimate {
        wall_secs,
        watts,
        joules,
        kwh,
        usd,
        g_co2,
    }
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn sample_returns_plausible_values() {
        let mut m = ResourceMonitor::new();
        let s = m.sample();
        assert!(s.host_ram_total > 0);
        // Host CPU load may be 0.0 on a quiet system; just check bounds.
        assert!((0.0..=1.0).contains(&s.host_cpu_load));
        assert!(s.cpu_count >= 1);
    }

    #[test]
    fn host_ram_fraction_is_bounded() {
        let mut m = ResourceMonitor::new();
        let s = m.sample();
        let f = s.host_ram_fraction();
        assert!((0.0..=1.0).contains(&f), "fraction {f} out of range");
    }

    #[test]
    fn flop_estimate_scales_linearly_in_dim() {
        let small = flop_estimate(1_000, 3);
        let large = flop_estimate(10_000, 3);
        assert_eq!(large, small * 10);
    }

    #[test]
    fn flop_estimate_scales_with_ops() {
        let one_op = flop_estimate(1_000, 1);
        let four_ops = flop_estimate(1_000, 4);
        assert_eq!(four_ops, one_op * 4);
    }

    #[test]
    fn cost_estimate_round_trip() {
        // 1 hour at 300 W on 16¢/kWh, 400 g/kWh
        let c = estimate_cost(3600.0, 300.0, 16.0, 400.0);
        assert!((c.joules - 1_080_000.0).abs() < 1.0);
        assert!((c.kwh - 0.3).abs() < 1e-6);
        assert!((c.usd - 0.048).abs() < 1e-6);
        assert!((c.g_co2 - 120.0).abs() < 1e-6);
    }

    #[test]
    fn cost_zero_wall_secs_returns_zero() {
        let c = estimate_cost(0.0, 300.0, 16.0, 400.0);
        assert_eq!(c.joules, 0.0);
        assert_eq!(c.kwh, 0.0);
        assert_eq!(c.usd, 0.0);
    }
}
