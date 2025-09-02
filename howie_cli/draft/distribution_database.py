"""
Player Distributions Database Management

This module manages the player_distributions table that stores variance buckets,
injury probabilities, and distribution parameters for each player.
"""

import sqlite3
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
from datetime import datetime


@dataclass
class VarianceBucket:
    """Variance bucket definition for position × age/tenure groups"""
    position: str
    age_group: str  # 'rookie', 'peak_24_27', 'decline_28_plus', etc.
    coefficient_of_variation: float  # CV for normal distribution
    injury_prob_healthy: float  # P(0 games missed)
    injury_prob_minor: float   # P(1-3 games missed)
    injury_prob_major: float   # P(4+ games missed)
    description: str


@dataclass
class PlayerDistributionProfile:
    """Complete distribution profile for a player"""
    player_name: str
    position: str
    team: str
    season: int
    
    # Player-specific attributes
    age: Optional[int] = None
    years_experience: Optional[int] = None
    
    # Distribution parameters
    mean_projection: float = 0.0
    coefficient_of_variation: float = 0.25
    distribution_type: str = 'truncated_normal'  # 'truncated_normal', 'lognormal'
    
    # Injury probabilities
    injury_prob_healthy: float = 0.8  # P(0 games missed)
    injury_prob_minor: float = 0.15   # P(1-3 games missed) 
    injury_prob_major: float = 0.05   # P(4+ games missed)
    
    # Variance bucket assignment
    variance_bucket: Optional[str] = None
    
    # Metadata
    last_updated: Optional[datetime] = None
    confidence_score: Optional[float] = None


class DistributionDatabaseManager:
    """Manage player distributions in the database"""
    
    def __init__(self, db_path: str = None):
        if db_path:
            self.db_path = db_path
        else:
            self.db_path = self._get_database_path()
    
    def _get_database_path(self) -> str:
        """Get database path using ProjectHowie's path resolution"""
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(script_dir))
        db_path = os.path.join(project_root, "data", "fantasy_ppr.db")
        
        if os.path.exists(db_path):
            return db_path
            
        # Fallback to relative path
        fallback_path = "data/fantasy_ppr.db"
        if os.path.exists(fallback_path):
            return fallback_path
            
        raise FileNotFoundError(f"Database not found: {db_path}")
    
    def create_distribution_tables(self):
        """Create tables for player distributions and variance buckets"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Variance Buckets table - position × age/tenure definitions
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS variance_buckets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                position TEXT NOT NULL,
                age_group TEXT NOT NULL,
                coefficient_of_variation REAL NOT NULL,
                injury_prob_healthy REAL NOT NULL,
                injury_prob_minor REAL NOT NULL,
                injury_prob_major REAL NOT NULL,
                description TEXT,
                
                created_date DATETIME DEFAULT CURRENT_TIMESTAMP,
                
                UNIQUE(position, age_group)
            )
        """)
        
        # Player Distributions table - individual player parameters
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS player_distributions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                player_name TEXT NOT NULL,
                position TEXT NOT NULL,
                team TEXT NOT NULL,
                season INTEGER NOT NULL DEFAULT 2025,
                
                -- Player attributes
                age INTEGER,
                years_experience INTEGER,
                
                -- Distribution parameters
                mean_projection REAL NOT NULL,
                coefficient_of_variation REAL NOT NULL DEFAULT 0.25,
                distribution_type TEXT NOT NULL DEFAULT 'truncated_normal',
                
                -- Injury probabilities
                injury_prob_healthy REAL NOT NULL DEFAULT 0.8,
                injury_prob_minor REAL NOT NULL DEFAULT 0.15,
                injury_prob_major REAL NOT NULL DEFAULT 0.05,
                
                -- Variance bucket assignment
                variance_bucket TEXT,
                
                -- Metadata
                last_updated DATETIME DEFAULT CURRENT_TIMESTAMP,
                confidence_score REAL,
                data_source TEXT DEFAULT 'computed',
                
                UNIQUE(player_name, team, season),
                
                FOREIGN KEY (variance_bucket) REFERENCES variance_buckets(id)
            )
        """)
        
        # Pre-sampled Outcomes table - for caching outcomes matrix
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS player_outcomes_cache (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                player_name TEXT NOT NULL,
                team TEXT NOT NULL,
                season INTEGER NOT NULL DEFAULT 2025,
                
                -- Pre-sampled outcomes (JSON array)
                outcomes_data TEXT NOT NULL,  -- JSON array of 10k+ outcomes
                num_samples INTEGER NOT NULL,
                sample_method TEXT NOT NULL DEFAULT 'monte_carlo',
                
                -- Cache metadata
                cache_key TEXT NOT NULL,  -- Hash of distribution parameters
                created_date DATETIME DEFAULT CURRENT_TIMESTAMP,
                expires_date DATETIME,
                
                UNIQUE(player_name, team, season, cache_key)
            )
        """)
        
        conn.commit()
        conn.close()
    
    def initialize_default_variance_buckets(self):
        """Initialize default variance buckets for each position"""
        
        default_buckets = [
            # QB Variance Buckets
            VarianceBucket('QB', 'rookie', 0.35, 0.75, 0.20, 0.05, 'Rookie QBs - high variance, moderate injury risk'),
            VarianceBucket('QB', 'peak_24_30', 0.18, 0.85, 0.12, 0.03, 'Peak QBs (24-30) - low variance, low injury risk'),
            VarianceBucket('QB', 'veteran_31_plus', 0.25, 0.80, 0.15, 0.05, 'Veteran QBs (31+) - moderate variance, slightly higher injury risk'),
            
            # RB Variance Buckets  
            VarianceBucket('RB', 'rookie', 0.32, 0.72, 0.18, 0.10, 'Rookie RBs - high variance, high injury risk'),
            VarianceBucket('RB', 'peak_24_27', 0.20, 0.82, 0.12, 0.06, 'Peak RBs (24-27) - moderate variance, moderate injury risk'),
            VarianceBucket('RB', 'decline_28_plus', 0.28, 0.77, 0.15, 0.08, 'Veteran RBs (28+) - higher variance, higher injury risk'),
            
            # WR Variance Buckets
            VarianceBucket('WR', 'rookie', 0.28, 0.78, 0.14, 0.08, 'Rookie WRs - moderate-high variance, moderate injury risk'),
            VarianceBucket('WR', 'peak_24_28', 0.18, 0.85, 0.10, 0.05, 'Peak WRs (24-28) - low variance, low injury risk'),
            VarianceBucket('WR', 'decline_29_plus', 0.22, 0.82, 0.12, 0.06, 'Veteran WRs (29+) - moderate variance, moderate injury risk'),
            
            # TE Variance Buckets
            VarianceBucket('TE', 'rookie', 0.30, 0.76, 0.16, 0.08, 'Rookie TEs - high variance, moderate injury risk'),
            VarianceBucket('TE', 'peak_25_29', 0.20, 0.83, 0.12, 0.05, 'Peak TEs (25-29) - moderate variance, low injury risk'),
            VarianceBucket('TE', 'veteran_30_plus', 0.25, 0.80, 0.14, 0.06, 'Veteran TEs (30+) - moderate variance, moderate injury risk'),
            
            # K/DEF Variance Buckets (lower variance, different injury patterns)
            VarianceBucket('K', 'all_ages', 0.15, 0.90, 0.08, 0.02, 'Kickers - low variance, very low injury risk'),
            VarianceBucket('DEF', 'all_ages', 0.22, 0.88, 0.10, 0.02, 'Defense/ST - moderate variance, low injury risk'),
        ]
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for bucket in default_buckets:
            cursor.execute("""
                INSERT OR REPLACE INTO variance_buckets (
                    position, age_group, coefficient_of_variation,
                    injury_prob_healthy, injury_prob_minor, injury_prob_major,
                    description
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                bucket.position, bucket.age_group, bucket.coefficient_of_variation,
                bucket.injury_prob_healthy, bucket.injury_prob_minor, bucket.injury_prob_major,
                bucket.description
            ))
        
        conn.commit()
        conn.close()
        
        print(f"✅ Initialized {len(default_buckets)} variance buckets")
    
    def assign_variance_buckets_to_players(self, season: int = 2025):
        """Assign variance buckets to all players based on position and age"""
        conn = sqlite3.connect(self.db_path)
        
        # Get all players with projections
        players_df = pd.read_sql_query("""
            SELECT 
                player_name, position, team_name, fantasy_points,
                bye_week
            FROM player_projections 
            WHERE season = ? AND projection_type = ?
            ORDER BY fantasy_points DESC
        """, conn, params=[season, 'preseason'])
        
        # Get variance buckets
        buckets_df = pd.read_sql_query("""
            SELECT position, age_group, coefficient_of_variation,
                   injury_prob_healthy, injury_prob_minor, injury_prob_major
            FROM variance_buckets
        """, conn)
        
        profiles = []
        
        for i, (_, player) in enumerate(players_df.iterrows()):
            # Normalize position and assign age group
            player_position = player['position'].upper()
            
            # Handle DST/DEF mismatch - DST players use DEF variance bucket
            if player_position in ['DST', 'DEF']:
                bucket_position = 'DEF'
            else:
                bucket_position = player_position
                
            age_group = self._assign_age_group(player['player_name'], player['position'])
            variance_bucket = f"{bucket_position}_{age_group}"
            
            # Find matching bucket using the correct bucket position
            bucket_match = buckets_df[
                (buckets_df['position'] == bucket_position) & 
                (buckets_df['age_group'] == age_group)
            ]
            
            if not bucket_match.empty:
                bucket = bucket_match.iloc[0]
                
                profile = PlayerDistributionProfile(
                    player_name=player['player_name'],
                    position=player['position'],
                    team=player['team_name'],
                    season=season,
                    mean_projection=player['fantasy_points'],
                    coefficient_of_variation=bucket['coefficient_of_variation'],
                    injury_prob_healthy=bucket['injury_prob_healthy'],
                    injury_prob_minor=bucket['injury_prob_minor'],
                    injury_prob_major=bucket['injury_prob_major'],
                    variance_bucket=variance_bucket,
                    last_updated=datetime.now()
                )
                
                profiles.append(profile)
        
        # Save to database
        cursor = conn.cursor()
        for profile in profiles:
            cursor.execute("""
                INSERT OR REPLACE INTO player_distributions (
                    player_name, position, team, season,
                    mean_projection, coefficient_of_variation,
                    injury_prob_healthy, injury_prob_minor, injury_prob_major,
                    variance_bucket, last_updated
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                profile.player_name, profile.position, profile.team, profile.season,
                profile.mean_projection, profile.coefficient_of_variation,
                profile.injury_prob_healthy, profile.injury_prob_minor, profile.injury_prob_major,
                profile.variance_bucket, profile.last_updated
            ))
        
        conn.commit()
        conn.close()
        
        print(f"✅ Assigned variance buckets to {len(profiles)} players")
        return profiles
    
    def _assign_age_group(self, player_name: str, position: str) -> str:
        """Assign age group based on player name and position (simplified)"""
        # This is a simplified implementation
        # In production, you'd use real player ages or draft years
        
        # Normalize position for case-insensitive matching
        position = position.upper()
        name_lower = player_name.lower()
        
        # Known veterans (add more as needed)
        veteran_indicators = [
            # QBs (31+)
            'aaron rodgers', 'russell wilson', 'derek carr', 'kirk cousins',
            'ryan tannehill', 'matthew stafford', 'jimmy garoppolo',
            # RBs (28+) 
            'derrick henry', 'aaron jones', 'alvin kamara', 'dalvin cook',
            'nick chubb', 'joe mixon', 'david montgomery', 'james conner',
            'ezekiel elliott', 'leonard fournette', 'james robinson',
            # WRs (29+)
            'julio jones', 'mike evans', 'davante adams', 'stefon diggs', 
            'tyreek hill', 'keenan allen', 'adam thielen', 'jarvis landry',
            'allen robinson', 'mike williams', 'robert woods', 'cooper kupp',
            # TEs (30+)
            'travis kelce', 'george kittle', 'zach ertz', 'tyler higbee',
            'logan thomas', 'hunter henry'
        ]
        if any(vet in name_lower for vet in veteran_indicators):
            if position == 'QB':
                return 'veteran_31_plus'
            elif position == 'RB':
                return 'decline_28_plus'
            elif position == 'WR':
                return 'decline_29_plus'
            elif position == 'TE':
                return 'veteran_30_plus'
        
        # Known rookies/young players (2024-2025 draft class)
        rookie_indicators = [
            # QBs
            'caleb williams', 'jayden daniels', 'drake maye', 'bo nix',
            'michael penix', 'joe milton', 'spencer rattler',
            # RBs  
            'ashton jeanty', 'omarion hampton', 'rj harvey', 'quinshon judkins',
            'trevyon henderson', 'treveyon henderson', 'dylan sampson', 'cam skattebo',
            'braelon allen', 'kaleb johnson', 'kyle monangai', 'jaydon blue',
            'bhayshul tuten', 'trey benson',
            # WRs
            'rome odunze', 'malik nabers', 'marvin harrison', 'brian thomas',
            'keon coleman', 'ladd mcconkey', 'malik washington', 'xavier worthy',
            'adonai mitchell', 'troy franklin', 'ricky pearsall', 'jalyn polk',
            # TEs
            'brock bowers', 'jatavion sanders', 'cade stover', 'ben sinnott'
        ]
        if any(rookie in name_lower for rookie in rookie_indicators):
            return 'rookie'
        
        # Default to peak years for most players
        if position == 'QB':
            return 'peak_24_30'
        elif position == 'RB':
            return 'peak_24_27'
        elif position == 'WR':
            return 'peak_24_28'
        elif position == 'TE':
            return 'peak_25_29'
        else:  # K, DEF, DST
            return 'all_ages'
    
    def get_player_distribution(self, player_name: str, team: str, season: int = 2025) -> Optional[PlayerDistributionProfile]:
        """Get distribution profile for a specific player"""
        conn = sqlite3.connect(self.db_path)
        
        result = pd.read_sql_query("""
            SELECT * FROM player_distributions
            WHERE player_name = ? AND team = ? AND season = ?
        """, conn, params=[player_name, team, season])
        
        conn.close()
        
        if result.empty:
            return None
        
        row = result.iloc[0]
        return PlayerDistributionProfile(
            player_name=row['player_name'],
            position=row['position'],
            team=row['team'],
            season=row['season'],
            age=row['age'],
            years_experience=row['years_experience'],
            mean_projection=row['mean_projection'],
            coefficient_of_variation=row['coefficient_of_variation'],
            distribution_type=row['distribution_type'],
            injury_prob_healthy=row['injury_prob_healthy'],
            injury_prob_minor=row['injury_prob_minor'],
            injury_prob_major=row['injury_prob_major'],
            variance_bucket=row['variance_bucket'],
            last_updated=row['last_updated'],
            confidence_score=row['confidence_score']
        )
    
    def get_all_player_distributions(self, season: int = 2025) -> List[PlayerDistributionProfile]:
        """Get all player distribution profiles for a season"""
        conn = sqlite3.connect(self.db_path)
        
        result = pd.read_sql_query("""
            SELECT * FROM player_distributions
            WHERE season = ?
            ORDER BY mean_projection DESC
        """, conn, params=[season])
        
        conn.close()
        
        profiles = []
        for _, row in result.iterrows():
            profiles.append(PlayerDistributionProfile(
                player_name=row['player_name'],
                position=row['position'],
                team=row['team'],
                season=row['season'],
                age=row['age'],
                years_experience=row['years_experience'],
                mean_projection=row['mean_projection'],
                coefficient_of_variation=row['coefficient_of_variation'],
                distribution_type=row['distribution_type'],
                injury_prob_healthy=row['injury_prob_healthy'],
                injury_prob_minor=row['injury_prob_minor'],
                injury_prob_major=row['injury_prob_major'],
                variance_bucket=row['variance_bucket'],
                last_updated=row['last_updated'],
                confidence_score=row['confidence_score']
            ))
        
        return profiles
    
    def update_variance_bucket(self, position: str, age_group: str, **kwargs):
        """Update a variance bucket's parameters"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Build update query dynamically
        update_fields = []
        values = []
        
        for field, value in kwargs.items():
            if field in ['coefficient_of_variation', 'injury_prob_healthy', 'injury_prob_minor', 'injury_prob_major', 'description']:
                update_fields.append(f"{field} = ?")
                values.append(value)
        
        if update_fields:
            query = f"""
                UPDATE variance_buckets 
                SET {', '.join(update_fields)}
                WHERE position = ? AND age_group = ?
            """
            values.extend([position, age_group])
            
            cursor.execute(query, values)
            conn.commit()
        
        conn.close()
    
    def get_variance_bucket_summary(self) -> pd.DataFrame:
        """Get summary of all variance buckets"""
        conn = sqlite3.connect(self.db_path)
        
        df = pd.read_sql_query("""
            SELECT 
                position, age_group, coefficient_of_variation,
                injury_prob_healthy, injury_prob_minor, injury_prob_major,
                description
            FROM variance_buckets
            ORDER BY position, age_group
        """, conn)
        
        conn.close()
        return df


def setup_distribution_system():
    """Initialize the complete distribution system"""
    print("🎯 Setting up Player Distribution System...")
    
    # Initialize database manager
    db_manager = DistributionDatabaseManager()
    
    # Create tables
    print("📊 Creating distribution tables...")
    db_manager.create_distribution_tables()
    
    # Initialize variance buckets
    print("🏗️  Initializing variance buckets...")
    db_manager.initialize_default_variance_buckets()
    
    # Assign buckets to players
    print("👥 Assigning variance buckets to players...")
    profiles = db_manager.assign_variance_buckets_to_players()
    
    print(f"✅ Distribution system setup complete!")
    print(f"   • Created 3 new database tables")
    print(f"   • Initialized 14 variance buckets")
    print(f"   • Assigned distributions to {len(profiles)} players")
    
    return db_manager


if __name__ == "__main__":
    # Test setup
    setup_distribution_system()
