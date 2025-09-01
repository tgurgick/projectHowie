# Draft Simulation Implementation Guide
## Detailed Technical Specification for ProjectHowie Integration

## 🎯 System Overview

This draft simulation system integrates directly with ProjectHowie's existing database (`fantasy_ppr.db`) to provide intelligent, data-driven draft recommendations. The system uses real projections and ADP data to simulate thousands of draft scenarios and recommend optimal picks for each round.

## 🔧 Configuration System

### **League Configuration**
```python
@dataclass
class LeagueConfig:
    """Complete league configuration"""
    
    # Basic Settings
    num_teams: int = 12
    roster_size: int = 16
    scoring_type: str = "ppr"  # ppr, half_ppr, standard
    
    # Roster Requirements
    qb_slots: int = 1
    rb_slots: int = 2
    wr_slots: int = 2
    te_slots: int = 1
    flex_slots: int = 1  # RB/WR/TE
    superflex_slots: int = 0  # QB/RB/WR/TE
    k_slots: int = 1
    def_slots: int = 1
    bench_slots: int = 6
    
    # Draft Settings
    draft_type: str = "snake"  # snake, linear, auction
    draft_position: int = 6  # Your pick position (1-based)
    
    # Keeper Configuration
    keepers_enabled: bool = True
    keeper_slots: int = 1
    keeper_rules: str = "round_based"  # "first_round", "round_based", "auction_value"

@dataclass
class KeeperPlayer:
    """Individual keeper configuration"""
    player_name: str
    team_name: str  # Team that owns the keeper
    keeper_round: int  # Round where keeper is "drafted"
    original_round: int  # Where they went last year (for round_based)
    keeper_cost: int = None  # For auction leagues

@dataclass
class Player:
    """Enhanced player model with intelligence data"""
    name: str
    position: str
    team: str
    projection: float
    adp: float
    adp_position: int
    bye_week: int
    sos_rank: float = None
    sos_playoff: float = None
    
    # Enhanced Intelligence Fields
    is_projected_starter: bool = None
    starter_confidence: float = None  # 0.0 to 1.0
    injury_risk_level: str = None  # 'LOW', 'MEDIUM', 'HIGH'
    injury_details: str = None
```

### **Database Integration**
```python
class DatabaseConnector:
    """Connect to ProjectHowie's existing database"""
    
    def __init__(self):
        # Use ProjectHowie's database path resolution
        self.db_path = self._get_database_path()
        
    def _get_database_path(self) -> str:
        """Use same path resolution as ProjectHowie"""
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        db_path = os.path.join(project_root, "data", "fantasy_ppr.db")
        
        if os.path.exists(db_path):
            return db_path
        
        # Fallback to relative path
        fallback_path = "data/fantasy_ppr.db"
        if os.path.exists(fallback_path):
            return fallback_path
            
        raise FileNotFoundError(f"Database not found: {db_path}")
    
    def get_player_projections(self, season: int = 2025) -> pd.DataFrame:
        """Get projections from existing database"""
        conn = sqlite3.connect(self.db_path)
        
        query = """
        SELECT 
            player_name,
            position,
            team_name,
            fantasy_points,
            games,
            pass_att, pass_cmp, pass_yds, pass_tds, pass_ints,
            rush_att, rush_yds, rush_tds,
            targets, receptions, rec_yds, rec_tds,
            bye_week
        FROM player_projections 
        WHERE season = ? AND projection_type = 'preseason'
        ORDER BY fantasy_points DESC
        """
        
        df = pd.read_sql_query(query, conn, params=[season])
        conn.close()
        return df
    
    def get_adp_data(self, season: int = 2025) -> pd.DataFrame:
        """Get ADP data from existing database"""
        conn = sqlite3.connect(self.db_path)
        
        query = """
        SELECT 
            player_name,
            position,
            team,
            adp_overall,
            adp_position,
            adp_source
        FROM adp_data 
        WHERE season = ?
        ORDER BY adp_overall
        """
        
        df = pd.read_sql_query(query, conn, params=[season])
        conn.close()
        return df
    
    def get_combined_player_data(self, season: int = 2025) -> pd.DataFrame:
        """Combine projections with ADP, SoS, and intelligence data for complete player profiles"""
        conn = sqlite3.connect(self.db_path)
        
        query = """
        SELECT 
            pp.player_name,
            pp.position,
            pp.team_name,
            pp.fantasy_points,
            pp.games,
            pp.bye_week,
            ad.adp_overall,
            ad.adp_position,
            sos.season_rank as sos_rank,
            sos.playoff_rank as sos_playoff,
            pdi.is_projected_starter,
            pdi.starter_confidence,
            pdi.injury_risk_level,
            pdi.injury_details,
            pdi.last_updated as intel_updated
        FROM player_projections pp
        LEFT JOIN adp_data ad ON LOWER(pp.player_name) = LOWER(ad.player_name) 
            AND ad.season = pp.season
        LEFT JOIN strength_of_schedule sos ON pp.team_name = sos.team 
            AND pp.position = sos.position AND sos.season = pp.season
        LEFT JOIN player_draft_intelligence pdi ON LOWER(pp.player_name) = LOWER(pdi.player_name)
            AND pp.team_name = pdi.team AND pdi.season = pp.season
        WHERE pp.season = ? AND pp.projection_type = 'preseason'
        ORDER BY pp.fantasy_points DESC
        """
        
        df = pd.read_sql_query(query, conn, params=[season])
        conn.close()
        return df
    
    def create_intelligence_tables(self):
        """Create tables to store draft intelligence data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Player Draft Intelligence table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS player_draft_intelligence (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                player_name TEXT NOT NULL,
                team TEXT NOT NULL,
                position TEXT NOT NULL,
                season INTEGER NOT NULL DEFAULT 2025,
                
                -- Starter Status
                is_projected_starter BOOLEAN DEFAULT NULL,
                starter_confidence REAL DEFAULT NULL,  -- 0.0 to 1.0
                depth_chart_position INTEGER DEFAULT NULL,  -- 1=starter, 2=backup, etc.
                
                -- Injury Risk Assessment  
                injury_risk_level TEXT DEFAULT NULL,  -- 'LOW', 'MEDIUM', 'HIGH'
                injury_details TEXT DEFAULT NULL,
                current_injury_status TEXT DEFAULT NULL,  -- 'HEALTHY', 'QUESTIONABLE', 'INJURED'
                
                -- Intelligence Metadata
                last_updated DATETIME DEFAULT CURRENT_TIMESTAMP,
                confidence_score REAL DEFAULT NULL,
                source_summary TEXT DEFAULT NULL,
                
                UNIQUE(player_name, team, season)
            )
        """)
        
        # Enhanced Strength of Schedule with positional breakdown
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS enhanced_strength_of_schedule (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                team TEXT NOT NULL,
                position TEXT NOT NULL,
                season INTEGER NOT NULL DEFAULT 2025,
                
                -- Overall Rankings (1=easiest, 32=hardest)
                season_rank INTEGER DEFAULT NULL,
                playoff_rank INTEGER DEFAULT NULL,
                
                -- Detailed Metrics
                avg_points_allowed REAL DEFAULT NULL,
                fantasy_points_vs_position REAL DEFAULT NULL,
                strength_rating REAL DEFAULT NULL,  -- Normalized 0-1 score
                
                -- Situational Factors
                home_games INTEGER DEFAULT NULL,
                division_strength REAL DEFAULT NULL,
                
                last_updated DATETIME DEFAULT CURRENT_TIMESTAMP,
                
                UNIQUE(team, position, season)
            )
        """)
        
        conn.commit()
        conn.close()
```

## 🎯 Draft Simulation Engine

### **Core Simulator with Keeper Support**
```python
class DraftSimulator:
    """Advanced draft simulator with keeper integration"""
    
    def __init__(self, league_config: LeagueConfig, keepers: List[KeeperPlayer] = None):
        self.config = league_config
        self.keepers = keepers or []
        self.db = DatabaseConnector()
        
        # Load player data
        self.player_data = self.db.get_combined_player_data()
        self.available_players = self._initialize_player_pool()
        
        # Initialize draft board
        self.draft_board = self._create_draft_board()
        
    def _initialize_player_pool(self) -> List[Player]:
        """Create available player pool accounting for keepers"""
        all_players = []
        
        for _, row in self.player_data.iterrows():
            player = Player(
                name=row['player_name'],
                position=row['position'],
                team=row['team_name'],
                projection=row['fantasy_points'],
                adp=row['adp_overall'] or 999,
                bye_week=row['bye_week'],
                sos_rank=row['sos_rank'],
                sos_playoff=row['sos_playoff'],
                is_projected_starter=row['is_projected_starter'],
                starter_confidence=row['starter_confidence'],
                injury_risk_level=row['injury_risk_level'],
                injury_details=row['injury_details']
            )
            all_players.append(player)
        
        # Remove keepers from available pool
        kept_names = {keeper.player_name.lower() for keeper in self.keepers}
        available = [p for p in all_players if p.name.lower() not in kept_names]
        
        return available
    
    def _create_draft_board(self) -> DraftBoard:
        """Create draft board with keeper slots filled"""
        total_picks = self.config.num_teams * self.config.roster_size
        board = DraftBoard(total_picks, self.config.num_teams)
        
        # Fill keeper slots
        for keeper in self.keepers:
            pick_number = self._calculate_keeper_pick(keeper)
            board.make_pick(pick_number, keeper.player_name, keeper.team_name)
        
        return board
    
    def _calculate_keeper_pick(self, keeper: KeeperPlayer) -> int:
        """Calculate which overall pick the keeper occupies"""
        if self.config.keeper_rules == "first_round":
            # All keepers go in first round
            team_index = self._get_team_index(keeper.team_name)
            return team_index + 1
            
        elif self.config.keeper_rules == "round_based":
            # Keeper goes in same round as last year
            team_index = self._get_team_index(keeper.team_name)
            round_start = (keeper.keeper_round - 1) * self.config.num_teams
            
            # Account for snake draft order
            if keeper.keeper_round % 2 == 0:  # Even rounds reverse order
                pick_in_round = self.config.num_teams - team_index
            else:  # Odd rounds normal order
                pick_in_round = team_index + 1
                
            return round_start + pick_in_round
        
        return 1  # Fallback
```

### **Pick Recommendation Engine**
```python
@dataclass
class PickRecommendation:
    player: Player
    overall_score: float
    vorp: float
    vona: float
    scarcity_score: float
    tier_info: Dict[str, Any]
    roster_fit: float
    opportunity_cost: float
    
    # Enhanced Evaluation Factors
    sos_advantage: float  # Strength of Schedule benefit (higher = easier SoS)
    starter_status_score: float  # Projected starter confidence
    injury_risk_score: float  # Injury risk assessment (higher = lower risk)
    
    primary_reason: str
    risk_factors: List[str]
    confidence: float
    enhanced_factors: Dict[str, str]  # Human-readable factor descriptions

class PickRecommendationEngine:
    """Generate optimal pick recommendations for each round"""
    
    def __init__(self, simulator: DraftSimulator):
        self.simulator = simulator
        self.value_calculator = ValueCalculator()
        self.scarcity_analyzer = ScarcityAnalyzer()
        
    def generate_round_recommendations(
        self, 
        round_number: int, 
        current_roster: Roster
    ) -> List[PickRecommendation]:
        """Generate top 10 recommendations for a specific round"""
        
        # Get available players at this round
        pick_number = self._calculate_pick_number(round_number)
        available = self.simulator.get_available_players(pick_number)
        
        recommendations = []
        
        # Evaluate top candidates (more than 10 to account for runs)
        candidates = available[:25]
        
        for player in candidates:
            # Calculate comprehensive metrics
            value_metrics = self.value_calculator.calculate_value(
                player, current_roster, available
            )
            
            scarcity_metrics = self.scarcity_analyzer.analyze_scarcity(
                player, pick_number, available
            )
            
            # Calculate enhanced evaluation factors
            sos_advantage = self._calculate_sos_advantage(player)
            starter_status_score = self._calculate_starter_status_score(player)
            injury_risk_score = self._calculate_injury_risk_score(player)
            
            # Project roster after this pick
            projected_roster = current_roster.add_player(player)
            roster_strength = self._evaluate_roster_strength(projected_roster)
            
            # Calculate opportunity cost
            opportunity_cost = self._calculate_opportunity_cost(
                player, available, current_roster
            )
            
            # Generate enhanced factors description
            enhanced_factors = self._generate_enhanced_factors_description(
                player, sos_advantage, starter_status_score, injury_risk_score
            )
            
            recommendation = PickRecommendation(
                player=player,
                overall_score=self._calculate_overall_score(
                    value_metrics, scarcity_metrics, roster_strength,
                    sos_advantage, starter_status_score, injury_risk_score
                ),
                vorp=value_metrics.vorp,
                vona=value_metrics.vona,
                scarcity_score=scarcity_metrics.scarcity_score,
                tier_info=scarcity_metrics.tier_info,
                roster_fit=roster_strength,
                opportunity_cost=opportunity_cost,
                sos_advantage=sos_advantage,
                starter_status_score=starter_status_score,
                injury_risk_score=injury_risk_score,
                primary_reason=self._generate_primary_reason(
                    player, value_metrics, scarcity_metrics, enhanced_factors
                ),
                risk_factors=self._identify_risk_factors(player, enhanced_factors),
                confidence=self._calculate_confidence(player, enhanced_factors),
                enhanced_factors=enhanced_factors
            )
            
            recommendations.append(recommendation)
        
        # Sort by overall score and return top 10
        recommendations.sort(key=lambda x: x.overall_score, reverse=True)
        return recommendations[:10]
    
    def _calculate_sos_advantage(self, player: Player) -> float:
        """Calculate Strength of Schedule advantage (higher = easier SoS)"""
        if not player.sos_rank:
            return 0.5  # Neutral if no data
        
        # Convert rank to advantage score (1=easiest becomes highest score)
        # Rank 1-32, convert to 0.9-0.1 scale
        max_rank = 32
        sos_advantage = 1.0 - ((player.sos_rank - 1) / (max_rank - 1))
        
        # Ensure bounds
        return max(0.1, min(0.9, sos_advantage))
    
    def _calculate_starter_status_score(self, player: Player) -> float:
        """Calculate starter status confidence score"""
        if player.is_projected_starter is None:
            return 0.6  # Neutral/unknown
        
        if not player.is_projected_starter:
            return 0.2  # Backup/uncertain role
        
        # Use confidence if available, otherwise high score for confirmed starters
        if player.starter_confidence is not None:
            return player.starter_confidence
        
        return 0.8  # Default high confidence for projected starters
    
    def _calculate_injury_risk_score(self, player: Player) -> float:
        """Calculate injury risk score (higher = lower risk)"""
        if not player.injury_risk_level:
            return 0.7  # Neutral if no data
        
        risk_mapping = {
            'LOW': 0.9,
            'MEDIUM': 0.6,
            'HIGH': 0.3
        }
        
        return risk_mapping.get(player.injury_risk_level.upper(), 0.7)
    
    def _calculate_overall_score(
        self, 
        value_metrics, 
        scarcity_metrics, 
        roster_strength,
        sos_advantage: float,
        starter_status_score: float,
        injury_risk_score: float
    ) -> float:
        """Calculate weighted overall score including enhanced factors"""
        
        # Base scoring weights
        base_score = (
            value_metrics.vorp * 0.25 +
            scarcity_metrics.scarcity_score * 0.20 +
            roster_strength * 0.15
        )
        
        # Enhanced factor weights (total 40% of score)
        enhanced_score = (
            sos_advantage * 0.15 +        # 15% - Schedule matters significantly
            starter_status_score * 0.20 +  # 20% - Starting role is crucial
            injury_risk_score * 0.05       # 5% - Injury risk is important but not dominant
        )
        
        return base_score + enhanced_score
    
    def _generate_enhanced_factors_description(
        self,
        player: Player,
        sos_advantage: float,
        starter_status_score: float, 
        injury_risk_score: float
    ) -> Dict[str, str]:
        """Generate human-readable descriptions of enhanced factors"""
        
        factors = {}
        
        # Strength of Schedule
        if player.sos_rank:
            if player.sos_rank <= 10:
                factors['sos'] = f"🟢 Favorable SoS (Rank {player.sos_rank})"
            elif player.sos_rank <= 22:
                factors['sos'] = f"🟡 Average SoS (Rank {player.sos_rank})" 
            else:
                factors['sos'] = f"🔴 Tough SoS (Rank {player.sos_rank})"
        else:
            factors['sos'] = "❓ SoS Unknown"
        
        # Starter Status
        if player.is_projected_starter:
            confidence_pct = int((player.starter_confidence or 0.8) * 100)
            factors['starter'] = f"✅ Projected Starter ({confidence_pct}% confidence)"
        elif player.is_projected_starter is False:
            factors['starter'] = "⚠️  Backup/Committee Role"
        else:
            factors['starter'] = "❓ Role Uncertain"
        
        # Injury Risk
        if player.injury_risk_level:
            risk_emojis = {'LOW': '💪', 'MEDIUM': '⚠️ ', 'HIGH': '🚑'}
            emoji = risk_emojis.get(player.injury_risk_level.upper(), '❓')
            factors['injury'] = f"{emoji} {player.injury_risk_level.title()} Injury Risk"
            
            if player.injury_details:
                factors['injury'] += f" ({player.injury_details[:50]}...)"
        else:
            factors['injury'] = "❓ Injury Status Unknown"
        
        return factors
```

### **Intelligence Data Management**
```python
class IntelligenceDataManager:
    """Manage updates to draft intelligence data"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
    
    def update_player_intelligence(
        self,
        player_name: str,
        team: str,
        position: str,
        is_projected_starter: bool = None,
        starter_confidence: float = None,
        injury_risk_level: str = None,
        injury_details: str = None,
        season: int = 2025
    ):
        """Update intelligence data for a player"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Insert or update player intelligence
        cursor.execute("""
            INSERT OR REPLACE INTO player_draft_intelligence (
                player_name, team, position, season,
                is_projected_starter, starter_confidence,
                injury_risk_level, injury_details,
                last_updated
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        """, (
            player_name, team, position, season,
            is_projected_starter, starter_confidence,
            injury_risk_level, injury_details
        ))
        
        conn.commit()
        conn.close()
    
    def bulk_update_from_intel_system(self):
        """Update intelligence data from ProjectHowie's /intel system"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get all team intelligence data
        cursor.execute("""
            SELECT team, position, intelligence_summary, injury_updates, confidence_score
            FROM team_position_intelligence 
            WHERE season = 2025
        """)
        
        intel_data = cursor.fetchall()
        
        for team, position, summary, injury_updates, confidence in intel_data:
            # Parse intelligence summary for starter information
            starters = self._parse_starters_from_intelligence(summary, position)
            
            # Parse injury information
            injury_info = self._parse_injury_info(injury_updates)
            
            # Update each identified player
            for starter_info in starters:
                self.update_player_intelligence(
                    player_name=starter_info['name'],
                    team=team,
                    position=position,
                    is_projected_starter=starter_info['is_starter'],
                    starter_confidence=starter_info['confidence'],
                    injury_risk_level=injury_info.get(starter_info['name'], {}).get('risk_level'),
                    injury_details=injury_info.get(starter_info['name'], {}).get('details')
                )
        
        conn.close()
    
    def _parse_starters_from_intelligence(self, summary: str, position: str) -> List[Dict]:
        """Parse starter information from intelligence summary"""
        starters = []
        
        if not summary:
            return starters
        
        # Simple parsing logic - can be enhanced with NLP
        summary_lower = summary.lower()
        
        # Look for starter indicators
        starter_patterns = [
            r'(\w+\s+\w+)\s+(?:is|will be|expected to be)\s+(?:the\s+)?starter',
            r'starter:\s*(\w+\s+\w+)',
            r'(\w+\s+\w+)\s+leads?\s+the\s+depth\s+chart'
        ]
        
        for pattern in starter_patterns:
            matches = re.findall(pattern, summary_lower)
            for match in matches:
                starters.append({
                    'name': match.title(),
                    'is_starter': True,
                    'confidence': 0.8  # Default high confidence
                })
        
        return starters
    
    def _parse_injury_info(self, injury_updates: str) -> Dict[str, Dict]:
        """Parse injury information from updates"""
        injury_info = {}
        
        if not injury_updates:
            return injury_info
        
        # Parse injury risk levels
        lines = injury_updates.split('\n')
        for line in lines:
            if any(word in line.lower() for word in ['questionable', 'doubtful', 'injured']):
                # Extract player name and risk level
                # This is simplified - real implementation would be more sophisticated
                words = line.split()
                if len(words) >= 2:
                    player_name = f"{words[0]} {words[1]}"
                    
                    if 'questionable' in line.lower():
                        risk_level = 'MEDIUM'
                    elif 'doubtful' in line.lower() or 'injured' in line.lower():
                        risk_level = 'HIGH'
                    else:
                        risk_level = 'LOW'
                    
                    injury_info[player_name] = {
                        'risk_level': risk_level,
                        'details': line[:100]  # First 100 chars
                    }
        
        return injury_info
```

## 🌳 Tree Search Optimization

### **Strategy Tree Search**
```python
class StrategyTreeSearch:
    """Tree search to find optimal round-by-round strategy"""
    
    def __init__(self, simulator: DraftSimulator, max_depth: int = 5):
        self.simulator = simulator
        self.max_depth = max_depth
        self.cache = {}  # Memoization for repeated states
        
    def find_optimal_strategy(
        self, 
        starting_round: int = 1
    ) -> OptimalStrategy:
        """Find optimal strategy using tree search"""
        
        initial_state = DraftState(
            round=starting_round,
            roster=Roster(),
            available_players=self.simulator.available_players.copy()
        )
        
        # Run tree search
        best_path, expected_value = self._search_recursive(
            state=initial_state,
            depth=0,
            alpha=float('-inf'),
            beta=float('inf')
        )
        
        return OptimalStrategy(
            round_by_round_targets=best_path,
            expected_roster_value=expected_value,
            confidence_intervals=self._calculate_confidence(best_path)
        )
    
    def _search_recursive(
        self, 
        state: DraftState, 
        depth: int,
        alpha: float,
        beta: float
    ) -> Tuple[List[PositionTarget], float]:
        """Recursive tree search with alpha-beta pruning"""
        
        # Base case: max depth reached
        if depth >= self.max_depth or state.round > 16:
            return [], self._evaluate_final_roster(state.roster)
        
        # Check cache
        state_key = self._get_state_key(state)
        if state_key in self.cache:
            return self.cache[state_key]
        
        best_path = []
        best_value = float('-inf')
        
        # Get realistic position targets for this round
        position_targets = self._get_position_targets(state)
        
        for target in position_targets:
            # Simulate picking best available at target position
            best_at_position = self._get_best_available(
                state.available_players, 
                target.position
            )
            
            if not best_at_position:
                continue
                
            # Create new state after this pick
            new_state = self._simulate_pick(state, best_at_position, target)
            
            # Recursively search remaining rounds
            remaining_path, future_value = self._search_recursive(
                new_state, depth + 1, alpha, beta
            )
            
            # Calculate total value
            immediate_value = self._calculate_immediate_value(
                best_at_position, state.roster
            )
            total_value = immediate_value + future_value
            
            # Update best if this path is better
            if total_value > best_value:
                best_value = total_value
                best_path = [target] + remaining_path
                
            # Alpha-beta pruning
            alpha = max(alpha, total_value)
            if beta <= alpha:
                break
        
        # Cache result
        self.cache[state_key] = (best_path, best_value)
        return best_path, best_value
    
    def _get_position_targets(self, state: DraftState) -> List[PositionTarget]:
        """Get realistic position targets for this round"""
        targets = []
        
        # Analyze roster needs
        needs = state.roster.get_needs()
        
        # Get players likely available at this pick
        pick_number = self._calculate_pick_number(state.round)
        likely_available = self._predict_available_players(pick_number)
        
        # Generate position targets based on:
        # 1. Roster needs
        # 2. Value available
        # 3. Positional scarcity
        for position in ['QB', 'RB', 'WR', 'TE']:
            best_at_pos = self._get_best_available(likely_available, position)
            
            if best_at_pos:
                scarcity = self._calculate_position_scarcity(position, likely_available)
                need_score = needs.get(position, 0)
                value_score = best_at_pos.projection
                
                target = PositionTarget(
                    position=position,
                    target_player=best_at_pos,
                    scarcity_score=scarcity,
                    need_score=need_score,
                    value_score=value_score,
                    composite_score=self._calculate_composite_score(
                        scarcity, need_score, value_score
                    )
                )
                targets.append(target)
        
        # Sort by composite score
        targets.sort(key=lambda x: x.composite_score, reverse=True)
        return targets[:4]  # Return top 4 realistic targets
```

## 📊 Output Generation

### **Round-by-Round Analysis**
```python
class DraftAnalysisGenerator:
    """Generate comprehensive draft analysis output"""
    
    def generate_pre_draft_analysis(
        self, 
        league_config: LeagueConfig,
        keepers: List[KeeperPlayer] = None
    ) -> PreDraftAnalysis:
        """Generate complete pre-draft analysis"""
        
        # Initialize simulator
        simulator = DraftSimulator(league_config, keepers)
        
        # Run tree search optimization
        tree_search = StrategyTreeSearch(simulator)
        optimal_strategy = tree_search.find_optimal_strategy()
        
        # Generate round-by-round recommendations
        round_analyses = []
        current_roster = Roster()
        
        for round_num in range(1, 17):  # 16 rounds typical
            # Update roster based on optimal strategy
            if round_num <= len(optimal_strategy.round_by_round_targets):
                target = optimal_strategy.round_by_round_targets[round_num - 1]
                # Simulate adding best player at target position
                best_at_target = simulator.get_best_available_at_position(
                    target.position, round_num
                )
                if best_at_target:
                    current_roster.add_player(best_at_target)
            
            # Generate recommendations for this round
            rec_engine = PickRecommendationEngine(simulator)
            recommendations = rec_engine.generate_round_recommendations(
                round_num, current_roster
            )
            
            round_analysis = RoundAnalysis(
                round_number=round_num,
                your_pick_number=self._calculate_your_pick(round_num, league_config),
                optimal_target=optimal_strategy.round_by_round_targets[round_num - 1] 
                    if round_num <= len(optimal_strategy.round_by_round_targets) else None,
                top_recommendations=recommendations,
                scarcity_alerts=self._generate_scarcity_alerts(round_num, simulator),
                roster_state=current_roster.copy()
            )
            
            round_analyses.append(round_analysis)
        
        return PreDraftAnalysis(
            league_config=league_config,
            keeper_summary=self._summarize_keepers(keepers),
            optimal_strategy=optimal_strategy,
            round_by_round=round_analyses,
            key_insights=self._generate_key_insights(optimal_strategy, simulator),
            contingency_plans=self._generate_contingency_plans(optimal_strategy)
        )
    
    def format_analysis_output(self, analysis: PreDraftAnalysis) -> str:
        """Format analysis for display"""
        
        output = []
        
        # Header
        output.append("=" * 80)
        output.append(f"DRAFT STRATEGY ANALYSIS - {analysis.league_config.num_teams} Team League")
        output.append(f"Your Draft Position: {analysis.league_config.draft_position}")
        output.append(f"Scoring: {analysis.league_config.scoring_type.upper()}")
        output.append("=" * 80)
        
        # Optimal Strategy Summary
        output.append("\n🎯 OPTIMAL STRATEGY OVERVIEW")
        output.append("-" * 50)
        for i, target in enumerate(analysis.optimal_strategy.round_by_round_targets[:8]):
            round_num = i + 1
            output.append(f"Round {round_num:2d}: {target.position:2s} - {target.target_player.name}")
            output.append(f"          Reasoning: {target.primary_reason}")
        
        # Round-by-Round Analysis (first 8 rounds detailed)
        for round_analysis in analysis.round_by_round[:8]:
            output.append(f"\n📍 ROUND {round_analysis.round_number} ANALYSIS")
            output.append(f"Your Pick: #{round_analysis.your_pick_number}")
            output.append("-" * 50)
            
            # Top 10 recommendations
            output.append("Top 10 Recommendations:")
            for i, rec in enumerate(round_analysis.top_recommendations, 1):
                output.append(
                    f"{i:2d}. {rec.player.name:<20} {rec.player.position:2s} "
                    f"({rec.player.projection:.1f} pts) - Score: {rec.overall_score:.1f}"
                )
                output.append(f"    💡 {rec.primary_reason}")
                
                # Show enhanced factors for top 5 picks
                if i <= 5:
                    factors = rec.enhanced_factors
                    output.append(f"    📅 {factors.get('sos', 'SoS Unknown')}")
                    output.append(f"    🏈 {factors.get('starter', 'Role Unknown')}")
                    output.append(f"    🏥 {factors.get('injury', 'Health Unknown')}")
                
                if rec.risk_factors:
                    output.append(f"    ⚠️  Risks: {', '.join(rec.risk_factors)}")
                
                if i <= 3:  # Show detailed scoring breakdown for top 3
                    output.append(
                        f"    📊 VORP: {rec.vorp:.1f} | "
                        f"SoS: {rec.sos_advantage:.2f} | "
                        f"Starter: {rec.starter_status_score:.2f} | "
                        f"Health: {rec.injury_risk_score:.2f}"
                    )
            
            # Scarcity alerts
            if round_analysis.scarcity_alerts:
                output.append("\n⚠️  Scarcity Alerts:")
                for alert in round_analysis.scarcity_alerts:
                    output.append(f"    • {alert}")
        
        # Key Insights
        output.append(f"\n💡 KEY INSIGHTS")
        output.append("-" * 50)
        for insight in analysis.key_insights:
            output.append(f"• {insight}")
        
        return "\n".join(output)
```

## 🗺️ Implementation Roadmap

### **Phase 1: Foundation (Week 1)**
```python
# Core Infrastructure
class Phase1Tasks:
    tasks = [
        "Create LeagueConfig dataclass with all settings",
        "Build DatabaseConnector using ProjectHowie's path resolution", 
        "Implement Player and Roster classes",
        "Create basic DraftBoard with keeper slot support",
        "Test database connectivity and data loading"
    ]
    
    deliverables = [
        "league_config.py - Configuration management",
        "database_connector.py - Database integration", 
        "draft_models.py - Core data models",
        "test_foundation.py - Unit tests"
    ]
```

### **Phase 2: Value Calculation (Week 2)** 
```python
class Phase2Tasks:
    tasks = [
        "Implement ValueCalculator using projections + ADP",
        "Create ScarcityAnalyzer for positional scarcity",
        "Build MarginalValueCalculator for VORP/VONA",
        "Implement OpportunityCostAnalyzer",
        "Test value calculations against known scenarios"
    ]
    
    deliverables = [
        "value_calculator.py - Core value metrics",
        "scarcity_analyzer.py - Positional scarcity",
        "test_value_calculations.py - Validation tests"
    ]
```

### **Phase 3: Pick Recommendations (Week 3)**
```python  
class Phase3Tasks:
    tasks = [
        "Create PickRecommendationEngine",
        "Implement round-by-round analysis generation",
        "Build recommendation scoring algorithm",
        "Create risk assessment for each pick",
        "Test with various roster states and scenarios"
    ]
    
    deliverables = [
        "recommendation_engine.py - Pick recommendations",
        "analysis_generator.py - Output formatting", 
        "test_recommendations.py - Validation"
    ]
```

### **Phase 4: Tree Search Optimization (Week 4)**
```python
class Phase4Tasks:
    tasks = [
        "Implement StrategyTreeSearch with alpha-beta pruning",
        "Create state caching for performance",
        "Build position target generation logic", 
        "Implement optimal strategy calculation",
        "Performance optimization and testing"
    ]
    
    deliverables = [
        "tree_search.py - Strategy optimization",
        "optimization_models.py - Supporting classes",
        "test_tree_search.py - Algorithm validation"
    ]
```

### **Phase 5: Integration & CLI (Week 5)**
```python
class Phase5Tasks:
    tasks = [
        "Integrate with ProjectHowie CLI (/draft command)",
        "Create interactive configuration wizard",
        "Implement analysis export (markdown/PDF)",
        "Add keeper management interface", 
        "Final testing and documentation"
    ]
    
    deliverables = [
        "draft_cli.py - CLI integration",
        "config_wizard.py - Interactive setup",
        "export_manager.py - Output formats",
        "documentation.md - User guide"
    ]
```

## 🚀 Quick Start Implementation

### **Immediate Next Steps:**
1. **Create core data models** in `howie_cli/draft/models.py`
2. **Set up database integration** using existing ProjectHowie patterns
3. **Build basic configuration system** with keeper support
4. **Implement simple value calculator** using projection + ADP data
5. **Create CLI command** `/draft config` to start interactive setup

### **Success Metrics:**
- Can load all player data from existing database ✅
- Can configure league with keepers ✅ 
- Can generate top 10 picks for Round 1 ✅
- Tree search completes in <30 seconds ✅
- Analysis output is actionable and clear ✅

This implementation leverages ProjectHowie's existing infrastructure while adding sophisticated draft analysis capabilities that provide clear, actionable recommendations for each round of your draft.
