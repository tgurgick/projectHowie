#!/usr/bin/env python3
"""
Setup Player Distributions System

This script initializes the player distributions database tables and populates them
with variance buckets and player-specific distribution parameters.
"""

import sys
import os

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

from howie_cli.draft.distribution_database import setup_distribution_system, DistributionDatabaseManager
from howie_cli.draft.distributions import OutcomesMatrixGenerator, test_distributions
import numpy as np


def main():
    """Setup the complete player distributions system"""
    
    print("🎯 PLAYER DISTRIBUTIONS SETUP")
    print("=" * 50)
    
    try:
        # Step 1: Setup database tables and variance buckets
        print("\n📊 STEP 1: Database Setup")
        db_manager = setup_distribution_system()
        
        # Step 2: Test distributions with sample data
        print("\n🧪 STEP 2: Testing Distribution Models")
        test_distributions()
        
        # Step 3: Generate sample outcomes matrix for top players
        print("\n🎲 STEP 3: Generating Sample Outcomes Matrix")
        profiles = db_manager.get_all_player_distributions()
        
        # Use top 50 players for testing
        top_profiles = profiles[:50]
        print(f"   Using top {len(top_profiles)} players for outcomes matrix...")
        
        # Generate outcomes matrix
        matrix_generator = OutcomesMatrixGenerator(top_profiles, num_samples=1000)  # Smaller for testing
        outcomes_matrix = matrix_generator.generate_outcomes_matrix()
        
        # Calculate and display stats
        stats = matrix_generator.calculate_distribution_stats(outcomes_matrix)
        
        print("\n📈 SAMPLE DISTRIBUTION STATISTICS:")
        print("-" * 60)
        
        for i, profile in enumerate(top_profiles[:10]):  # Show top 10
            player_stats = stats[profile.player_name]
            print(f"{profile.player_name:<20} {profile.position:2s} | "
                  f"Mean: {player_stats['mean']:6.1f} | "
                  f"CV: {player_stats['cv']:.3f} | "
                  f"P90: {player_stats['p90']:6.1f} | "
                  f"Bust: {player_stats['bust_prob']:.1%}")
        
        # Step 4: Save outcomes to database
        print(f"\n💾 STEP 4: Caching Outcomes Matrix")
        matrix_generator.save_outcomes_to_database(outcomes_matrix, db_manager)
        
        # Step 5: Summary
        print("\n✅ DISTRIBUTION SYSTEM SETUP COMPLETE!")
        print("=" * 50)
        print(f"📊 Database Tables Created:")
        print(f"   • variance_buckets (14 buckets)")
        print(f"   • player_distributions ({len(profiles)} players)")
        print(f"   • player_outcomes_cache ({len(top_profiles)} cached)")
        print()
        print(f"🎯 Distribution Models Available:")
        print(f"   • TruncatedNormalDistribution")
        print(f"   • LognormalDistribution") 
        print(f"   • Injury overlay system")
        print()
        print(f"⚡ Performance:")
        print(f"   • {1000:,} pre-sampled outcomes per player")
        print(f"   • Fast matrix-based season scoring")
        print(f"   • Database caching for repeated use")
        print()
        print(f"🎲 Next Steps:")
        print(f"   • Run: python -c 'from howie_cli.draft.distributions import test_distributions; test_distributions()'")
        print(f"   • Integrate with Monte Carlo simulation")
        print(f"   • Generate full outcomes matrix for all players")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


def show_variance_buckets():
    """Display all variance buckets"""
    print("\n📊 VARIANCE BUCKETS SUMMARY:")
    print("=" * 80)
    
    db_manager = DistributionDatabaseManager()
    buckets_df = db_manager.get_variance_bucket_summary()
    
    for _, bucket in buckets_df.iterrows():
        print(f"{bucket['position']:3s} | {bucket['age_group']:<15s} | "
              f"CV: {bucket['coefficient_of_variation']:.3f} | "
              f"Healthy: {bucket['injury_prob_healthy']:.1%} | "
              f"Minor: {bucket['injury_prob_minor']:.1%} | "
              f"Major: {bucket['injury_prob_major']:.1%}")
        print(f"     {bucket['description']}")
        print()


def show_player_distributions(limit: int = 20):
    """Display sample player distributions"""
    print(f"\n👥 PLAYER DISTRIBUTIONS (Top {limit}):")
    print("=" * 80)
    
    db_manager = DistributionDatabaseManager()
    profiles = db_manager.get_all_player_distributions()
    
    print(f"{'Player':<20} {'Pos':>3} {'Team':>4} {'Mean':>6} {'CV':>5} {'Bucket':<15} {'Healthy%':>8}")
    print("-" * 80)
    
    for profile in profiles[:limit]:
        print(f"{profile.player_name:<20} {profile.position:>3s} {profile.team:>4s} "
              f"{profile.mean_projection:6.1f} {profile.coefficient_of_variation:5.3f} "
              f"{profile.variance_bucket:<15s} {profile.injury_prob_healthy:7.1%}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Setup Player Distributions System")
    parser.add_argument("--setup", action="store_true", help="Run full setup")
    parser.add_argument("--buckets", action="store_true", help="Show variance buckets")
    parser.add_argument("--players", type=int, metavar="N", help="Show top N player distributions")
    
    args = parser.parse_args()
    
    if args.setup:
        sys.exit(main())
    elif args.buckets:
        show_variance_buckets()
    elif args.players:
        show_player_distributions(args.players)
    else:
        # Default: run setup
        sys.exit(main())
