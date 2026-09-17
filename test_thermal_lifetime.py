"""
================================================================================
Unit Tests for SiC Power Module Thermal Lifetime Prediction
================================================================================

AUTHOR: Matteo Totaro
EMAIL: tmatteos@gmail.com
VERSION: 1.0.0
LICENSE: MIT

Test Coverage:
- Module configuration
- Profile generation and CSV operations
- Power loss calculations
- Thermal network simulation
- Rainflow cycle counting
- Lifetime prediction models
- Data integrity and physical constraints
================================================================================
"""

import unittest
import os
import shutil
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd

# Import the main module
from thermal_lifetime_prediction import (
    InfineonModuleFS03MR12A6MA1B,
    generate_mission_profile,
    save_profile_to_csv,
    load_profile_from_csv,
    calculate_phase_current,
    calculate_switching_frequency,
    calculate_power_losses,
    solve_coupled_thermal_electrical,
    rainflow_count,
    calculate_cycles_to_failure,
    calculate_damage,
    THERMAL_NETWORK,
    module,
    PROFILES_DIR
)


class TestModuleConfiguration(unittest.TestCase):
    """Test the SiC module configuration and parameters"""
    
    def test_module_initialization(self):
        """Test that module is properly initialized"""
        self.assertEqual(module.part_number, "FS03MR12A6MA1B")
        self.assertEqual(module.voltage_rating, 1200.0)
        self.assertEqual(module.current_rating, 310.0)
        self.assertEqual(module.R_th_jc, 0.1)
    
    def test_temperature_interpolation(self):
        """Test temperature-dependent parameter interpolation"""
        # Test at known points
        E_on_25 = module.get_E_on(25)
        E_on_150 = module.get_E_on(150)
        
        self.assertAlmostEqual(E_on_25, 19.48, places=2)
        self.assertAlmostEqual(E_on_150, 20.16, places=2)
        
        # Test interpolation
        E_on_100 = module.get_E_on(100)
        self.assertTrue(E_on_25 < E_on_100 < E_on_150)
    
    def test_lifetime_model_parameters(self):
        """Test Bayerer model parameters"""
        self.assertEqual(module.beta_1, -3.483)
        self.assertEqual(module.beta_2, 1917.0)
        self.assertEqual(module.beta_3, -0.438)
        self.assertEqual(module.N_f_test, 1000.0)
    
    def test_thermal_network(self):
        """Test thermal network configuration"""
        self.assertEqual(len(THERMAL_NETWORK['R_th']), 4)
        self.assertEqual(len(THERMAL_NETWORK['C_th']), 4)
        self.assertTrue(all(THERMAL_NETWORK['R_th'] > 0))
        self.assertTrue(all(THERMAL_NETWORK['C_th'] > 0))


class TestProfileGeneration(unittest.TestCase):
    """Test mission profile generation and CSV operations"""
    
    def setUp(self):
        """Create temporary directory for test profiles"""
        self.test_dir = Path(tempfile.mkdtemp())
        self.original_profiles_dir = PROFILES_DIR
        
    def tearDown(self):
        """Clean up temporary directory"""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
    
    def test_urban_profile_generation(self):
        """Test urban profile generation"""
        profile = generate_mission_profile('urban', duration_sec=50, dt=0.01)
        
        self.assertIsInstance(profile, pd.DataFrame)
        self.assertIn('Time_s', profile.columns)
        self.assertIn('Speed_RPM', profile.columns)
        self.assertIn('Torque_Nm', profile.columns)
        self.assertIn('V_DC_V', profile.columns)
        self.assertIn('I_load_A', profile.columns)
        self.assertIn('f_sw_Hz', profile.columns)
        
        # Check value ranges
        self.assertTrue(profile['Speed_RPM'].min() >= 0)
        self.assertTrue(profile['Speed_RPM'].max() <= 8000)
        self.assertTrue(profile['Torque_Nm'].min() >= 10)
        self.assertTrue(profile['Torque_Nm'].max() <= 50)
    
    def test_highway_profile_generation(self):
        """Test highway profile generation"""
        profile = generate_mission_profile('highway', duration_sec=10, dt=0.01)
        
        self.assertTrue(profile['Speed_RPM'].min() >= 5000)
        self.assertTrue(profile['Speed_RPM'].max() <= 14000)
        self.assertTrue(profile['Torque_Nm'].min() >= 20)
        self.assertTrue(profile['Torque_Nm'].max() <= 320)
    
    def test_performance_profile_generation(self):
        """Test performance profile generation"""
        profile = generate_mission_profile('performance', duration_sec=10, dt=0.01)
        
        self.assertTrue(profile['Speed_RPM'].min() >= 0)
        self.assertTrue(profile['Speed_RPM'].max() <= 15000)
        self.assertTrue(profile['Torque_Nm'].min() >= 20)
        self.assertTrue(profile['Torque_Nm'].max() <= 450)
    
    def test_profile_intensity_ordering(self):
        """Test that scenarios have correct intensity ordering"""
        urban = generate_mission_profile('urban', duration_sec=100, dt=0.01)
        highway = generate_mission_profile('highway', duration_sec=100, dt=0.01)
        performance = generate_mission_profile('performance', duration_sec=100, dt=0.01)
        
        # Urban should have lowest average torque
        self.assertTrue(urban['Torque_Nm'].mean() < highway['Torque_Nm'].mean())
        self.assertTrue(highway['Torque_Nm'].mean() < performance['Torque_Nm'].mean())
    
    def test_csv_save_load(self):
        """Test CSV save and load operations"""
        # Generate a small profile
        original_profile = generate_mission_profile('urban', duration_sec=5, dt=0.1)
        
        # Save to temporary location
        test_file = self.test_dir / "test_profile.csv"
        original_profile.to_csv(test_file, index=False)
        
        # Load back
        loaded_profile = pd.read_csv(test_file)
        
        # Verify columns match
        self.assertEqual(list(original_profile.columns), list(loaded_profile.columns))
        
        # Verify data integrity
        pd.testing.assert_frame_equal(original_profile, loaded_profile, check_dtype=False)


class TestPowerLossCalculations(unittest.TestCase):
    """Test power loss calculation functions"""
    
    def test_phase_current_calculation(self):
        """Test phase current calculation from torque"""
        current = calculate_phase_current(100, 5000, 800)
        
        self.assertGreater(current, 0)
        self.assertLessEqual(current, 500)  # Clipped at 500A
    
    def test_switching_frequency_calculation(self):
        """Test switching frequency calculation"""
        f_sw_low = calculate_switching_frequency(0)
        f_sw_high = calculate_switching_frequency(20000)
        
        self.assertAlmostEqual(f_sw_low, 10e3, delta=100)
        self.assertAlmostEqual(f_sw_high, 30e3, delta=100)
        
        # Mid-range should be between min and max
        f_sw_mid = calculate_switching_frequency(10000)
        self.assertTrue(10e3 < f_sw_mid < 30e3)
    
    def test_power_losses_positive(self):
        """Test that power losses are always positive"""
        for T_j in [25, 75, 125, 150]:
            P_loss = calculate_power_losses(100, 800, 20e3, T_j)
            self.assertGreater(P_loss, 0)
    
    def test_power_losses_temperature_dependency(self):
        """Test that power losses increase with temperature"""
        P_loss_25 = calculate_power_losses(200, 800, 20e3, 25)
        P_loss_150 = calculate_power_losses(200, 800, 20e3, 150)
        
        # Losses should increase with temperature due to R_DS_on increase
        self.assertGreater(P_loss_150, P_loss_25)


class TestThermalSimulation(unittest.TestCase):
    """Test thermal network simulation"""
    
    def test_thermal_simulation_stability(self):
        """Test that thermal simulation remains stable"""
        profile = generate_mission_profile('urban', duration_sec=50, dt=0.0001)  # Changed from 0.001 to 0.0001
        T_j, P_loss = solve_coupled_thermal_electrical(profile, T_ambient=25.0)
        
        # Check for NaN or Inf
        self.assertFalse(np.any(np.isnan(T_j)))
        self.assertFalse(np.any(np.isinf(T_j)))
        self.assertFalse(np.any(np.isnan(P_loss)))
        self.assertFalse(np.any(np.isinf(P_loss)))
    
    def test_thermal_simulation_bounds(self):
        """Test that temperatures stay within physical bounds"""
        profile = generate_mission_profile('performance', duration_sec=10, dt=0.0001)  # Changed from 0.001 to 0.0001
        T_j, P_loss = solve_coupled_thermal_electrical(profile, T_ambient=25.0)
        
        # Temperature should be above ambient
        self.assertTrue(np.all(T_j >= 25))
        
        # Temperature should not exceed extreme limits
        self.assertTrue(np.all(T_j < 250))
        
        # Power should be positive
        self.assertTrue(np.all(P_loss >= 0))
    
    def test_thermal_feedback_loop(self):
        """Test that thermal-electrical coupling works"""
        profile = generate_mission_profile('highway', duration_sec=5, dt=0.0001)  # Changed from 0.001 to 0.0001
        T_j, P_loss = solve_coupled_thermal_electrical(profile, T_ambient=25.0)
        
        # Temperature should eventually rise above ambient
        self.assertGreater(np.max(T_j), 25)
        
        # Power losses should vary with temperature
        self.assertGreater(np.std(P_loss), 0)


class TestRainflowCounting(unittest.TestCase):
    """Test rainflow cycle counting algorithm"""
    
    def test_rainflow_simple_cycle(self):
        """Test rainflow on simple triangular signal"""
        # Simple up-down cycle
        signal = np.array([0, 10, 0, 10, 0])
        dt = 1.0
        
        cycles, extrema_vals, extrema_idx = rainflow_count(signal, dt)
        
        self.assertGreater(len(cycles), 0)
        
        # Check that cycles have correct structure
        for delta_T, T_mean, T_max, t_on in cycles:
            self.assertGreater(delta_T, 0)
            self.assertGreater(t_on, 0)
    
    def test_rainflow_realistic_temperature(self):
        """Test rainflow on realistic temperature profile"""
        profile = generate_mission_profile('urban', duration_sec=10, dt=0.0001)  # Changed from 0.01 to 0.0001
        T_j, P_loss = solve_coupled_thermal_electrical(profile, T_ambient=25.0)
        
        dt = float(profile['Time_s'].iloc[1] - profile['Time_s'].iloc[0])
        cycles, extrema_vals, extrema_idx = rainflow_count(T_j, dt)
        
        self.assertGreater(len(cycles), 0)
        
        # All cycles should have positive values
        for delta_T, T_mean, T_max, t_on in cycles:
            self.assertGreaterEqual(delta_T, 0)
            self.assertGreater(T_max, 0)
            self.assertGreater(t_on, 0)


class TestLifetimePrediction(unittest.TestCase):
    """Test lifetime prediction models"""
    
    def test_cycles_to_failure_bayerer_model(self):
        """Test Bayerer model calculation"""
        # Test case matching AQG 324 reference
        N_f = calculate_cycles_to_failure(
            delta_T=100,
            T_max=150,
            t_on=1.0
        )
        
        # Should match reference condition
        self.assertAlmostEqual(N_f, 1000.0, delta=1)
    
    def test_cycles_to_failure_temperature_effect(self):
        """Test that higher temperature reduces lifetime"""
        N_f_low = calculate_cycles_to_failure(100, 100, 1.0)
        N_f_high = calculate_cycles_to_failure(100, 150, 1.0)
        
        # Higher temperature should reduce lifetime
        self.assertGreater(N_f_low, N_f_high)
    
    def test_cycles_to_failure_delta_T_effect(self):
        """Test that larger ΔT reduces lifetime"""
        N_f_small = calculate_cycles_to_failure(50, 150, 1.0)
        N_f_large = calculate_cycles_to_failure(100, 150, 1.0)
        
        # Larger ΔT should reduce lifetime
        self.assertGreater(N_f_small, N_f_large)
    
    def test_damage_accumulation(self):
        """Test Miner's rule damage accumulation"""
        # Create some dummy cycles
        cycles = [
            (50, 100, 125, 1.0),
            (75, 110, 140, 2.0),
            (100, 120, 150, 1.5)
        ]
        
        total_damage, damage_df = calculate_damage(cycles)
        
        # Total damage should be sum of individual damages
        self.assertGreater(total_damage, 0)
        self.assertLessEqual(total_damage, 1.0)  # Should be less than 100% for these cycles
        
        # Dataframe should have correct columns
        self.assertIn('DeltaT_C', damage_df.columns)
        self.assertIn('T_max_C', damage_df.columns)
        self.assertIn('t_on_s', damage_df.columns)
        self.assertIn('N_f', damage_df.columns)
        self.assertIn('Damage', damage_df.columns)


class TestScenarioComparison(unittest.TestCase):
    """Test scenario intensity progression"""
    
    def test_damage_progression(self):
        """Test that damage increases: urban < highway < performance"""
        scenarios = ['urban', 'highway', 'performance']
        damages = {}
        
        for scenario in scenarios:
            profile = generate_mission_profile(scenario, duration_sec=50, dt=0.0001)  # Changed from 0.001 to 0.0001
            T_j, P_loss = solve_coupled_thermal_electrical(profile, T_ambient=25.0)
            dt = float(profile['Time_s'].iloc[1] - profile['Time_s'].iloc[0])
            cycles, _, _ = rainflow_count(T_j, dt)
            total_damage, _ = calculate_damage(cycles)
            damages[scenario] = total_damage
        
        # Verify progression
        self.assertLess(damages['urban'], damages['highway'])
        self.assertLess(damages['highway'], damages['performance'])
    
    def test_temperature_progression(self):
        """Test that max temperature increases across scenarios"""
        scenarios = ['urban', 'highway', 'performance']
        max_temps = {}
        
        for scenario in scenarios:
            profile = generate_mission_profile(scenario, duration_sec=50, dt=0.0001)  # Changed from 0.001 to 0.0001
            T_j, P_loss = solve_coupled_thermal_electrical(profile, T_ambient=25.0)
            max_temps[scenario] = T_j.max()
        
        # Verify progression
        self.assertLess(max_temps['urban'], max_temps['highway'])
        self.assertLess(max_temps['highway'], max_temps['performance'])


def run_tests():
    """Run all unit tests"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestModuleConfiguration))
    suite.addTests(loader.loadTestsFromTestCase(TestProfileGeneration))
    suite.addTests(loader.loadTestsFromTestCase(TestPowerLossCalculations))
    suite.addTests(loader.loadTestsFromTestCase(TestThermalSimulation))
    suite.addTests(loader.loadTestsFromTestCase(TestRainflowCounting))
    suite.addTests(loader.loadTestsFromTestCase(TestLifetimePrediction))
    suite.addTests(loader.loadTestsFromTestCase(TestScenarioComparison))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Return exit code
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    import sys
    sys.exit(run_tests())