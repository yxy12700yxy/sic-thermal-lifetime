"""
================================================================================
Mission Profile Generator - CSV Creation Script
================================================================================

This script generates the three mission profiles (urban, highway, performance)
and saves them as CSV files in the profiles/ directory.

Run this script once to create the profile files, then they can be reused
by the main thermal_lifetime_prediction.py script.

AUTHOR: Matteo Totaro
EMAIL: tmatteos@gmail.com
================================================================================
"""

from pathlib import Path
from thermal_lifetime_prediction import (
    generate_mission_profile,
    save_profile_to_csv,
    PROFILES_DIR
)


def main():
    """Generate all mission profiles and save to CSV"""
    print("\n" + "="*80)
    print(" "*20 + "MISSION PROFILE GENERATOR")
    print("="*80)
    
    # Create profiles directory if it doesn't exist
    PROFILES_DIR.mkdir(exist_ok=True)
    print(f"\nProfiles directory: {PROFILES_DIR.absolute()}")
    
    scenarios = ['urban', 'highway', 'performance']
    duration_sec = 600  # 10 minutes
    dt = 0.0001  # 0.1 ms time step
    
    for scenario in scenarios:
        print(f"\n{'='*80}")
        print(f"Generating {scenario.upper()} profile...")
        print(f"{'='*80}")
        
        profile = generate_mission_profile(scenario, duration_sec, dt)
        save_profile_to_csv(profile, scenario)
        
        # Display statistics
        print(f"\nProfile Statistics:")
        print(f"  Total points: {len(profile):,}")
        print(f"  Duration: {profile['Time_s'].iloc[-1]:.1f} seconds")
        print(f"  Time step: {dt*1000:.2f} ms")
        print(f"  File size: {(PROFILES_DIR / f'{scenario}_profile.csv').stat().st_size / 1024:.1f} KB")
    
    print("\n" + "="*80)
    print(" "*25 + "GENERATION COMPLETE ✓")
    print("="*80)
    print(f"\nAll profiles saved to: {PROFILES_DIR.absolute()}")
    print("\nFiles created:")
    for scenario in scenarios:
        filepath = PROFILES_DIR / f"{scenario}_profile.csv"
        if filepath.exists():
            print(f"  ✓ {filepath.name}")
    
    print("\nYou can now run: python thermal_lifetime_prediction.py")


if __name__ == "__main__":
    main()