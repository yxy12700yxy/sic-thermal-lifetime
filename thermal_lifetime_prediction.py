"""
================================================================================
SiC Power Module Thermal Lifetime Prediction - AQG 324 Compliant
================================================================================

Part of Master's Thesis: Lifetime Modeling of SiC Power Modules 
in Automotive Traction Inverters

Developed in collaboration with Ferrari S.p.A. and 
ALMA MATER STUDIORUM - UNIVERSITÀ DI BOLOGNA

AUTHOR: Matteo Totaro
EMAIL: tmatteos@gmail.com
VERSION: 1.1.0
LICENSE: MIT

================================================================================
MATHEMATICAL FORMULATION
================================================================================

1. POWER LOSS CALCULATION
   P_total = P_sw_on + P_sw_off + P_cond + P_diode
   
   Where:
   - P_sw_on = E_on(T_j) × f_sw × (V_DC/V_ref)
   - P_sw_off = E_off(T_j) × f_sw × (V_DC/V_ref)
   - P_cond = I_rms² × R_DS_on(T_j)
   - P_diode = V_F × I_avg + R_F × I_rms²

2. THERMAL NETWORK (Explicit Euler Integration)
   For each layer j:
   C_th,j × dT_j/dt = Q_in,j - Q_out,j
   
   Where:
   - Q_in,j = (T_j-1 - T_j) / R_th,j-1
   - Q_out,j = (T_j - T_j+1) / R_th,j
   
   Euler update: T_j(t+Δt) = T_j(t) + Δt × dT_j/dt
   Stability: Δt < 2 × min(R_th × C_th)

3. BAYERER MODEL (CIPS08) - POWER CYCLING LIFETIME
   N_f = K × (ΔT_j)^β₁ × exp(β₂/T_j_max) × t_on^β₃
   
   Parameters (from Bayerer et al., 2008):
   - β₁ = -3.483  (Coffin-Manson: thermal strain)
   - β₂ = 1917 K  (Arrhenius: temperature activation)
   - β₃ = -0.438  (Time-dependent creep/diffusion)
   
   Note: The original model includes β₄ (current per bond wire), but for
   fixed module technology, this is absorbed into constant K.

4. MINER'S RULE (Cumulative Damage)
   D_total = Σ(n_i / N_f_i)
   
   Failure predicted when D_total ≥ 1.0 (100% life consumed)

5. RAINFLOW CYCLE COUNTING (ASTM E1049-85)
   Extracts fatigue-relevant thermal cycles from temperature history
   using the three-point algorithm

REFERENCES:
[1] ECPE AQG 324 - Qualification of Automotive Electronic Components
[2] Infineon FS03MR12A6MA1B Datasheet Rev. 2.0
[3] ASTM E1049-85 - Rainflow Cycle Counting Standard
[4] Bayerer, R., Herrmann, T., Licht, T., Lutz, J., & Feller, M. (2008).
    "Model for Power Cycling lifetime of IGBT Modules - various factors 
    influencing lifetime", 5th International Conference on Integrated 
    Power Electronics Systems (CIPS), Nuremberg, Germany, pp. 1-6
[5] Miner (1945) - Cumulative Damage in Fatigue
[6] IEC 60749-34 - Power Cycling Test Standards

================================================================================
"""

import os
import sys
import warnings
from typing import Tuple, List, Dict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec

# Optional: progress bar
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

# Beautiful matplotlib styling
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'axes.titleweight': 'bold',
    'axes.labelweight': 'bold',
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'legend.framealpha': 0.95,
    'figure.titlesize': 14,
    'figure.dpi': 100,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.25,
    'grid.linestyle': '--',
    'grid.linewidth': 0.8,
    'lines.linewidth': 2.0,
    'axes.facecolor': '#fafafa',
    'figure.facecolor': 'white',
    'axes.edgecolor': '#333333',
    'axes.linewidth': 1.2
})

warnings.filterwarnings('ignore', category=RuntimeWarning)

# ================================================================================
# SECTION 1: MODULE CONFIGURATION
# ================================================================================

@dataclass
class InfineonModuleFS03MR12A6MA1B:
    """Infineon FS03MR12A6MA1B - 1200V 310A Dual-Side Cooled SiC MOSFET"""
    part_number: str = "FS03MR12A6MA1B"
    voltage_rating: float = 1200.0
    current_rating: float = 310.0
    
    R_th_jc: float = 0.1
    R_th_jc_max: float = 0.108
    
    T_j_points: np.ndarray = field(default_factory=lambda: np.array([25, 125, 150]))
    E_on_data: np.ndarray = field(default_factory=lambda: np.array([19.48, 19.85, 20.16]))
    E_off_data: np.ndarray = field(default_factory=lambda: np.array([17.61, 17.95, 18.21]))
    R_DS_on_data: np.ndarray = field(default_factory=lambda: np.array([2.75, 4.0, 4.55]) * 1e-3)
    
    V_DC_ref: float = 800.0
    I_ref: float = 310.0
    V_F: float = 1.3
    
    PC_min_temp: float = 50.0
    PC_max_temp: float = 150.0
    delta_T_test: float = 100.0
    t_on_test: float = 1.0
    N_f_test: float = 1000.0
    
    beta_1: float = -3.483
    beta_2: float = 1917.0
    beta_3: float = -0.438
    
    def get_E_on(self, T_j: float) -> float:
        return float(np.interp(T_j, self.T_j_points, self.E_on_data))
    
    def get_E_off(self, T_j: float) -> float:
        return float(np.interp(T_j, self.T_j_points, self.E_off_data))
    
    def get_R_DS_on(self, T_j: float) -> float:
        return float(np.interp(T_j, self.T_j_points, self.R_DS_on_data))

module = InfineonModuleFS03MR12A6MA1B()

scaling_factor = module.R_th_jc / 0.19
THERMAL_NETWORK = {
    'R_th': np.array([0.03, 0.04, 0.06, 0.06]) * scaling_factor,
    'C_th': np.array([0.01, 0.05, 0.4, 1.5])
}

PROFILES_DIR = Path("profiles")

# ================================================================================
# SECTION 2: MISSION PROFILE GENERATION (DETERMINISTIC)
# ================================================================================

def generate_mission_profile(scenario: str = 'urban',
                            duration_sec: int = 10,
                            dt: float = 0.0001) -> pd.DataFrame:
    """
    Generate deterministic mission profiles (no randomness for reproducibility)
    
    Intensity progression: Urban < Highway < Performance
    """
    print(f"\n{'='*70}")
    print(f"  Generating {scenario.upper()} Profile (Deterministic)")
    print(f"{'='*70}")
    
    time = np.arange(0, duration_sec, dt)
    n = len(time)
    
    if scenario == 'urban':
        # Urban: Lowest intensity, reduced torque by 5x
        speed = 4000 + 3000*np.sin(2*np.pi*0.008*time) + 2000*np.sin(2*np.pi*0.03*time)
        speed = np.clip(speed, 0, 8000)
        
        speed_norm = speed / 15000
        base_torque = 40 * (1 - 0.4*speed_norm)
        torque = np.clip(base_torque + 16*np.sin(2*np.pi*0.02*time), 10, 50)
        
    elif scenario == 'highway':
        # Highway: Medium intensity
        speed = 9000 + 2500*np.sin(2*np.pi*0.003*time) + 1500*np.sin(2*np.pi*0.01*time)
        speed = np.clip(speed, 5000, 14000)
        
        speed_norm = speed / 15000
        base_torque = 280 * (1 - 0.6*speed_norm)
        torque_boost = np.where(speed < 10000, 50, 0)
        torque = np.clip(base_torque + torque_boost + 60*np.sin(2*np.pi*0.005*time), 20, 320)
        
    else:  # performance
        # Performance: Highest intensity
        speed = 7500 + 6000*np.sin(2*np.pi*0.015*time) + 3000*np.sin(2*np.pi*0.05*time)
        speed = np.clip(speed, 0, 15000)
        
        speed_norm = speed / 15000
        base_torque = 350 * (1 - 0.5*speed_norm)
        torque = np.clip(base_torque + 150*np.sin(2*np.pi*0.025*time), 20, 450)
    
    V_DC = 800 + 30*np.sin(2*np.pi*0.001*time)
    V_DC = np.clip(V_DC, 750, 850)
    
    profile = pd.DataFrame({
        'Time_s': time,
        'Speed_RPM': speed,
        'Torque_Nm': torque,
        'V_DC_V': V_DC
    })
    
    profile['I_load_A'] = profile.apply(
        lambda row: calculate_phase_current(row['Torque_Nm'], row['Speed_RPM'], row['V_DC_V']),
        axis=1
    )
    profile['f_sw_Hz'] = profile['Speed_RPM'].apply(calculate_switching_frequency)
    
    print(f"  Duration: {duration_sec}s | Points: {len(profile):,} | dt: {dt*1000:.2f}ms")
    print(f"  Speed: {speed.min():.0f}-{speed.max():.0f} RPM")
    print(f"  Torque: {torque.min():.0f}-{torque.max():.0f} Nm")
    print(f"  Current: {profile['I_load_A'].min():.0f}-{profile['I_load_A'].max():.0f} A")
    
    return profile

def save_profile_to_csv(profile: pd.DataFrame, scenario: str):
    """Save mission profile to CSV file"""
    PROFILES_DIR.mkdir(exist_ok=True)
    filepath = PROFILES_DIR / f"{scenario}_profile.csv"
    profile.to_csv(filepath, index=False)
    print(f"  ✓ Saved to: {filepath}")

def load_profile_from_csv(scenario: str) -> pd.DataFrame:
    """Load mission profile from CSV file"""
    filepath = PROFILES_DIR / f"{scenario}_profile.csv"
    if not filepath.exists():
        raise FileNotFoundError(f"Profile file not found: {filepath}")
    profile = pd.read_csv(filepath)
    print(f"  ✓ Loaded from: {filepath} ({len(profile):,} points)")
    return profile

def ensure_profiles_exist():
    """Generate and save all profiles if they don't exist"""
    scenarios = ['urban', 'highway', 'performance']
    for scenario in scenarios:
        filepath = PROFILES_DIR / f"{scenario}_profile.csv"
        if not filepath.exists():
            print(f"\n[Generating missing profile: {scenario}]")
            profile = generate_mission_profile(scenario, duration_sec=10, dt=0.0001)
            save_profile_to_csv(profile, scenario)

# ================================================================================
# SECTION 3: POWER LOSS & THERMAL CALCULATIONS
# ================================================================================

def calculate_phase_current(torque_Nm: float, speed_RPM: float, V_DC: float) -> float:
    """Calculate phase current from torque and speed"""
    k_t = 0.8
    efficiency = 0.95
    if k_t == 0:
        return 0.0
    return float(np.clip(torque_Nm / (k_t * efficiency), 0, 500))

def calculate_switching_frequency(speed_RPM: float) -> float:
    """Calculate switching frequency based on motor speed"""
    f_min, f_max = 10e3, 30e3
    speed_max = 20000
    f_sw = f_min + (f_max - f_min) * (speed_RPM / speed_max)
    return float(np.clip(f_sw, f_min, f_max))

def calculate_power_losses(I_phase: float, V_DC: float, f_sw: float, T_j: float) -> float:
    """Calculate total power losses at junction temperature T_j"""
    E_on = module.get_E_on(T_j)
    E_off = module.get_E_off(T_j)
    R_DS_on = module.get_R_DS_on(T_j)
    
    voltage_factor = V_DC / module.V_DC_ref
    
    E_on_scaled = E_on * voltage_factor
    E_off_scaled = E_off * voltage_factor
    
    P_sw_on = E_on_scaled * 1e-3 * f_sw
    P_sw_off = E_off_scaled * 1e-3 * f_sw
    
    I_rms = I_phase / np.sqrt(2) * np.sqrt(0.5)
    P_cond = I_rms**2 * R_DS_on
    
    diode_duty = 0.1
    I_diode_avg = I_phase * diode_duty
    R_F = 0.01
    P_diode = module.V_F * I_diode_avg + R_F * (1.2 * I_diode_avg)**2
    
    return P_sw_on + P_sw_off + P_cond + P_diode

def solve_coupled_thermal_electrical(profile: pd.DataFrame, T_ambient: float = 25.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solve coupled electro-thermal system with feedback loop:
    T_j(i-1) → P_loss(i) → Thermal Network → T_j(i)
    """
    dt = float(profile['Time_s'].iloc[1] - profile['Time_s'].iloc[0])
    n_points = len(profile)
    
    R_th = THERMAL_NETWORK['R_th']
    C_th = THERMAL_NETWORK['C_th']
    n_layers = len(R_th)
    
    tau_min = float(np.min(R_th * C_th))
    if dt > 2 * tau_min:
        print(f"  ⚠ WARNING: dt={dt:.6f}s > stability ({2*tau_min:.6f}s)")
    
    print(f"  Solving coupled system (dt={dt*1000:.3f}ms)...")
    
    T_layers = np.zeros((n_points, n_layers))
    T_layers[0, :] = T_ambient
    
    T_j = np.zeros(n_points)
    T_j[0] = T_ambient
    
    P_loss = np.zeros(n_points)
    
    iterator = range(1, n_points)
    if HAS_TQDM:
        iterator = tqdm(iterator, desc="  Thermal simulation", unit="steps")
    
    for i in iterator:
        I = profile['I_load_A'].iloc[i]
        f_sw = profile['f_sw_Hz'].iloc[i]
        V_DC = profile['V_DC_V'].iloc[i]
        
        # Feedback loop: use previous temperature for current losses
        P_loss[i] = calculate_power_losses(I, V_DC, f_sw, T_j[i-1])
        
        for j in range(n_layers):
            T_current = T_layers[i-1, j]
            
            if j == 0:
                Q_in = P_loss[i]
            else:
                T_prev = T_layers[i-1, j-1]
                Q_in = (T_prev - T_current) / R_th[j-1]
            
            if j == n_layers - 1:
                T_next = T_ambient
            else:
                T_next = T_layers[i-1, j+1]
            
            Q_out = (T_current - T_next) / R_th[j]
            
            dT_dt = (Q_in - Q_out) / C_th[j]
            T_layers[i, j] = T_current + dt * dT_dt
        
        T_j[i] = T_layers[i, 0]
    
    print(f"  ✓ T_j: {T_j.min():.1f}-{T_j.max():.1f}°C | P: {P_loss.mean():.0f}W avg")
    
    return T_j, P_loss

# ================================================================================
# SECTION 4: RAINFLOW & LIFETIME CALCULATION
# ================================================================================

def rainflow_count(signal: np.ndarray, dt: float) -> Tuple[List[Tuple], np.ndarray, np.ndarray]:
    """Rainflow cycle counting algorithm (ASTM E1049-85)"""
    extrema, extrema_idx = [], []
    extrema.append(signal[0])
    extrema_idx.append(0)
    
    for i in range(1, len(signal)-1):
        if ((signal[i] > signal[i-1] and signal[i] > signal[i+1]) or
            (signal[i] < signal[i-1] and signal[i] < signal[i+1])):
            extrema.append(signal[i])
            extrema_idx.append(i)
    
    extrema.append(signal[-1])
    extrema_idx.append(len(signal)-1)
    
    extrema_values = np.array(extrema)
    extrema_indices = np.array(extrema_idx)
    
    cycles = []
    stack, stack_idx = [], []
    
    for val, idx in zip(extrema, extrema_idx):
        stack.append(val)
        stack_idx.append(idx)
        
        while len(stack) >= 3:
            Y, X, W = stack[-1], stack[-2], stack[-3]
            Y_idx, X_idx = stack_idx[-1], stack_idx[-2]
            
            range_XY = abs(Y - X)
            range_WX = abs(X - W)
            
            if len(stack) >= 4 and range_XY >= range_WX:
                cycles.append((range_XY, (X+Y)/2, max(X,Y), abs(Y_idx-X_idx)*dt))
                stack.pop()
                stack.pop()
                stack_idx.pop()
                stack_idx.pop()
            else:
                break
    
    while len(stack) >= 2:
        Y, X = stack.pop(), stack.pop()
        Y_idx, X_idx = stack_idx.pop(), stack_idx.pop()
        cycles.append((abs(Y-X), (X+Y)/2, max(X,Y), abs(Y_idx-X_idx)*dt))
    
    return cycles, extrema_values, extrema_indices

def calculate_cycles_to_failure(delta_T: float, T_max: float, t_on: float) -> float:
    """
    Calculate cycles to failure using Bayerer model (CIPS08)
    
    N_f = K × (ΔT_j)^β₁ × exp(β₂/T_j_max) × t_on^β₃
    """
    T_max_K = T_max + 273.15
    T_test_K = module.PC_max_temp + 273.15
    
    AF_CM = (delta_T / module.delta_T_test) ** module.beta_1
    AF_Time = (t_on / module.t_on_test) ** module.beta_3
    AF_Arr = np.exp(module.beta_2 * (1/T_max_K - 1/T_test_K))
    
    return max(module.N_f_test * AF_CM * AF_Time * AF_Arr, 1.0)

def calculate_damage(cycles: List[Tuple]) -> Tuple[float, pd.DataFrame]:
    """Calculate cumulative damage using Miner's rule"""
    damage_data = []
    total_damage = 0.0
    
    for delta_T, T_mean, T_max, t_on in cycles:
        if delta_T < 5.0:
            continue
        
        N_f = calculate_cycles_to_failure(delta_T, T_max, t_on)
        damage = 1.0 / N_f
        total_damage += damage
        
        damage_data.append({
            'DeltaT_C': delta_T,
            'T_max_C': T_max,
            't_on_s': t_on,
            'N_f': N_f,
            'Damage': damage
        })
    
    return total_damage, pd.DataFrame(damage_data).sort_values('Damage', ascending=False)

# ================================================================================
# SECTION 5: VISUALIZATION
# ================================================================================

def plot_comparison_all_scenarios(results_dict: Dict):
    """Create comprehensive visualization comparing all scenarios"""
    scenarios_order = ['urban', 'highway', 'performance']
    colors = {
        'urban': '#0077B6',
        'highway': '#9D4EDD',
        'performance': '#F72585'
    }
    
    fig = plt.figure(figsize=(22, 16))
    gs = gridspec.GridSpec(5, 3, figure=fig, hspace=0.4, wspace=0.3,
                          height_ratios=[1, 1, 1, 0.1, 1.2])
    
    fig.suptitle('SiC Power Module Thermal Lifetime Analysis - AQG 324 Comparison',
                 fontsize=18, fontweight='bold', y=0.995)
    
    for idx, scenario in enumerate(scenarios_order):
        if scenario not in results_dict:
            continue
        
        profile, T_j, P_loss, cycles, damage_df, total_damage = results_dict[scenario]
        
        downsample = max(1, len(profile) // 5000)
        profile_plot = profile.iloc[::downsample]
        T_j_plot = T_j[::downsample]
        time_min = profile_plot['Time_s'] / 60
        
        # Speed
        ax_speed = fig.add_subplot(gs[0, idx])
        ax_speed.plot(time_min, profile_plot['Speed_RPM'],
                     color=colors[scenario], lw=2.5, alpha=0.85)
        ax_speed.fill_between(time_min, 0, profile_plot['Speed_RPM'],
                             color=colors[scenario], alpha=0.15)
        ax_speed.set_title(f'{scenario.upper()} - Motor Speed',
                          fontweight='bold', fontsize=13)
        ax_speed.set_ylabel('Speed (RPM)', fontweight='bold')
        ax_speed.set_xlabel('Time (min)')
        ax_speed.grid(True, alpha=0.2, linewidth=0.8)
        
        # Torque
        ax_torque = fig.add_subplot(gs[1, idx])
        ax_torque.plot(time_min, profile_plot['Torque_Nm'],
                      color=colors[scenario], lw=2.5, alpha=0.85)
        ax_torque.fill_between(time_min, 0, profile_plot['Torque_Nm'],
                              color=colors[scenario], alpha=0.15)
        ax_torque.set_title(f'{scenario.upper()} - Motor Torque',
                           fontweight='bold', fontsize=13)
        ax_torque.set_ylabel('Torque (Nm)', fontweight='bold')
        ax_torque.set_xlabel('Time (min)')
        ax_torque.grid(True, alpha=0.2, linewidth=0.8)
        
        # Junction Temperature
        ax_tj = fig.add_subplot(gs[2, idx])
        ax_tj.plot(time_min, T_j_plot, color='#DC2626', lw=2.5,
                  label='T_junction', zorder=3)
        ax_tj.axhline(25, color='#0284C7', ls='--', lw=2.5, alpha=0.8,
                     label='T_coolant (25°C)', zorder=2)
        ax_tj.axhline(module.PC_max_temp, color='#EF4444', ls=':', lw=2.5,
                     alpha=0.7, label=f'AQG T_max ({module.PC_max_temp:.0f}°C)', zorder=1)
        ax_tj.fill_between(time_min, 25, T_j_plot,
                          where=(T_j_plot >= 25), color='#DC2626', alpha=0.15)
        ax_tj.set_title(f'{scenario.upper()} - Junction Temperature',
                       fontweight='bold', fontsize=13)
        ax_tj.set_ylabel('Temperature (°C)', fontweight='bold')
        ax_tj.set_xlabel('Time (min)')
        ax_tj.legend(loc='upper right', framealpha=0.98, edgecolor='#333', fancybox=True)
        ax_tj.grid(True, alpha=0.2, linewidth=0.8)
        ax_tj.set_ylim([20, max(T_j_plot.max()*1.05, module.PC_max_temp+10)])
    
    # Summary table
    ax_summary = fig.add_subplot(gs[4, :])
    ax_summary.axis('off')
    
    summary_lines = []
    summary_lines.append("="*140)
    summary_lines.append(" "*50 + "COMPARATIVE ANALYSIS SUMMARY")
    summary_lines.append("="*140)
    summary_lines.append("")
    
    header = f"{'Scenario':<15} {'Speed (RPM)':<18} {'Torque (Nm)':<18} {'T_j (°C)':<16} {'P_avg (W)':<12} {'Cycles':<12} {'Damage (%)':<14} {'Lifetime':<20} {'AQG 324':<10}"
    summary_lines.append(header)
    summary_lines.append("-"*140)
    
    for scenario in scenarios_order:
        if scenario not in results_dict:
            continue
        
        profile, T_j, P_loss, cycles, damage_df, total_damage = results_dict[scenario]
        
        delta_T_pc = module.PC_max_temp - module.PC_min_temp
        N_f_pc = calculate_cycles_to_failure(delta_T_pc, module.PC_max_temp, module.t_on_test)
        equivalent_pc = total_damage / (1.0/N_f_pc) if total_damage > 0 else 0
        
        hours = profile['Time_s'].iloc[-1]/3600/total_damage if total_damage > 0 else float('inf')
        years = hours / 8760
        
        status = "✓ PASS" if equivalent_pc < module.N_f_test else "✗ FAIL"
        
        speed_str = f"{profile['Speed_RPM'].min():.0f}-{profile['Speed_RPM'].max():.0f}"
        torque_str = f"{profile['Torque_Nm'].min():.0f}-{profile['Torque_Nm'].max():.0f}"
        tj_str = f"{T_j.min():.1f}-{T_j.max():.1f}"
        cycles_str = f"{len(cycles)} ({len(damage_df)})"
        lifetime_str = f"{hours:.0f}h ({years:.1f}y)"
        
        line = f"{scenario.upper():<15} {speed_str:<18} {torque_str:<18} {tj_str:<16} {P_loss.mean():<12.0f} {cycles_str:<12} {total_damage*100:<14.6f} {lifetime_str:<20} {status:<10}"
        summary_lines.append(line)
    
    summary_lines.append("-"*140)
    summary_lines.append(f"Module: {module.part_number} | {module.voltage_rating:.0f}V / {module.current_rating:.0f}A | R_th_jc: {module.R_th_jc:.3f} K/W")
    summary_lines.append(f"AQG 324 Reference: ΔT={module.delta_T_test:.0f}°C, t_on={module.t_on_test:.0f}s, T_max={module.PC_max_temp:.0f}°C, N_f={module.N_f_test:.0f} cycles")
    summary_lines.append("="*140)
    
    summary_text = "\n".join(summary_lines)
    
    ax_summary.text(0.5, 0.5, summary_text, transform=ax_summary.transAxes,
                   fontsize=10, family='monospace', va='center', ha='center',
                   bbox=dict(boxstyle='round,pad=1.5', facecolor='#E3F2FD',
                            edgecolor='#1976D2', linewidth=2.5, alpha=0.95))
    
    plt.savefig('thermal_lifetime_all_scenarios_comparison.png', dpi=300, bbox_inches='tight')
    print("\n✓ Comparison plot saved: thermal_lifetime_all_scenarios_comparison.png")
    plt.show()

# ================================================================================
# MAIN EXECUTION
# ================================================================================

def run_lifetime_test_all_scenarios():
    """Run complete analysis for all scenarios"""
    
    print("\n" + "="*80)
    print(" "*15 + "SiC POWER MODULE THERMAL LIFETIME PREDICTION")
    print(" "*25 + "AQG 324 Compliant")
    print("="*80)
    print(f"\nModule: {module.part_number} | {module.voltage_rating:.0f}V / {module.current_rating:.0f}A")
    print(f"R_th_jc: {module.R_th_jc:.3f} K/W | Coolant: 25°C")
    
    # Ensure all profile files exist
    print("\n[Checking mission profiles...]")
    ensure_profiles_exist()
    
    scenarios = ['urban', 'highway', 'performance']
    results_dict = {}
    
    for scenario in scenarios:
        print(f"\n{'#'*80}")
        print(f"  SCENARIO: {scenario.upper()}")
        print(f"{'#'*80}")
        
        print(f"\n[LOADING PROFILE]")
        profile = load_profile_from_csv(scenario)
        
        print(f"\n[THERMAL-ELECTRICAL COUPLING]")
        T_j, P_loss = solve_coupled_thermal_electrical(profile, T_ambient=25.0)
        
        print(f"\n[RAINFLOW CYCLE EXTRACTION]")
        dt_actual = float(profile['Time_s'].iloc[1] - profile['Time_s'].iloc[0])
        cycles, extrema_vals, extrema_idx = rainflow_count(T_j, dt_actual)
        print(f"  ✓ Extracted {len(cycles)} thermal cycles")
        
        print(f"\n[LIFETIME CALCULATION]")
        total_damage, damage_df = calculate_damage(cycles)
        
        delta_T_pc = module.PC_max_temp - module.PC_min_temp
        N_f_pc = calculate_cycles_to_failure(delta_T_pc, module.PC_max_temp, module.t_on_test)
        equivalent_pc = total_damage / (1.0/N_f_pc) if total_damage > 0 else 0
        
        print(f"\n  {'='*70}")
        print(f"  LIFETIME CONSUMPTION: {total_damage*100:.6f}%")
        print(f"  {'='*70}")
        
        if total_damage > 0:
            hours = profile['Time_s'].iloc[-1]/3600/total_damage
            years = hours/8760
            print(f"  Extrapolated: {hours:.0f} hours ({years:.1f} years)")
        else:
            print(f"  Extrapolated: Infinite")
        
        print(f"  AQG 324: {equivalent_pc:.2f} / {module.N_f_test:.0f} PC cycles")
        
        if equivalent_pc < module.N_f_test:
            margin = module.N_f_test - equivalent_pc
            print(f"  ✓ PASS - Margin: {margin:.0f} cycles ({margin/module.N_f_test*100:.1f}%)")
        else:
            shortage = equivalent_pc - module.N_f_test
            print(f"  ✗ FAIL - Shortage: {shortage:.0f} cycles")
        
        results_dict[scenario] = (profile, T_j, P_loss, cycles, damage_df, total_damage)
    
    print(f"\n{'='*80}")
    print(f"  GENERATING VISUALIZATION")
    print(f"{'='*80}")
    
    plot_comparison_all_scenarios(results_dict)
    
    print("\n" + "="*80)
    print(" "*30 + "TEST COMPLETE")
    print("="*80)
    print("\n✓ All scenarios analyzed successfully")
    print("✓ Comparative plot generated")
    
    return results_dict

if __name__ == "__main__":
    try:
        print("\n" + "="*80)
        print(" "*15 + "SiC THERMAL LIFETIME PREDICTION TOOL")
        print(" "*22 + "Master's Thesis - Ferrari S.p.A.")
        print(" "*30 + "Version 1.1.0")
        print("="*80)
        
        results = run_lifetime_test_all_scenarios()
        
        print("\n" + "="*80)
        print(" "*25 + "SIMULATION COMPLETE ✓")
        print("="*80)
        print("\nResults available in dictionary:")
        print("  results['urban'] = (profile, T_j, P_loss, cycles, damage_df, total_damage)")
        print("  results['highway'] = (profile, T_j, P_loss, cycles, damage_df, total_damage)")
        print("  results['performance'] = (profile, T_j, P_loss, cycles, damage_df, total_damage)")
        
    except KeyboardInterrupt:
        print("\n\n⚠ Simulation interrupted by user")
        sys.exit(1)
        
    except Exception as e:
        print(f"\n\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)