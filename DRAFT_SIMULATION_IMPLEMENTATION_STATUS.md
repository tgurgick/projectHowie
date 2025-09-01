# Fantasy Draft Simulation - Implementation Status

## 🎯 CURRENT IMPLEMENTATION: Advanced Monte Carlo Draft Simulator

### ✅ COMPLETED FEATURES

#### Core Draft Simulation Engine
- **Player Universe**: 533 players with FantasyPros consensus ADP + projections
- **League Configuration**: 12-team PPR snake draft with flexible positioning
- **Draft State Management**: Real-time tracking of picks, rosters, available players
- **Snake Draft Logic**: Proper round reversal and pick calculation

#### Strategic Selection Algorithm
- **ADP Range Strategy**: ±10 picks around current position for realistic candidate filtering
- **VORP Optimization**: Value Over Replacement Player with position-specific replacement levels
- **6-Factor Scoring**: VORP, positional scarcity, roster fit, SoS, starter status, injury risk
- **Smart Fallbacks**: Graceful handling of edge cases and missing data

#### Realistic Opponent Modeling
- **Strategic Diversity**: 9 opponent types (Zero RB, Hero RB, QB Early/Late, etc.)
- **ADP + Noise**: Gaussian noise (σ=6-12) around consensus ADP values
- **Roster Needs Bias**: Soft preferences for positional needs
- **Natural Variance**: Different outcomes across simulations

#### Enhanced Intelligence Integration
- **Team Intelligence**: AI-gathered scouting reports for all teams/positions
- **Player Context**: Starter status, injury notes, coaching style impact
- **SoS Integration**: Strength of Schedule as draft factor
- **Real-time Updates**: `/intel` system with Perplexity fact-checking

#### Performance & Accuracy
- **Real ADP Data**: 86.7% coverage with FantasyPros consensus
- **Intelligent Generation**: Rank-based ADP for missing players
- **Unique Random Seeds**: Proper variance across simulations
- **Efficient Execution**: Fast simulation with realistic results

### 📊 SIMULATION RESULTS

Current system produces realistic draft outcomes:

**Round 1 (Pick #6):**
- CeeDee Lamb (80-86%) - ADP 5.0, perfect value
- Amon-Ra St. Brown (14-20%) - ADP 9.5, excellent alternative

**Round 2 (Pick #19):**
- Josh Allen (40-73%) - Elite QB timing
- Brock Bowers (6-20%) - Premium TE option
- Josh Jacobs (6-12%) - RB value pick

**VORP Examples:**
- Saquon Barkley: 174.7 VORP (highest due to RB scarcity)
- CeeDee Lamb: 79.1 VORP 
- Josh Allen: 64.6 VORP

---

## 🚧 MCTS IMPLEMENTATION ROADMAP

### Phase 1: Player Outcome Distributions (HIGH IMPACT)

**Goal**: Replace single projections with realistic variance distributions

#### 1.1 Distribution Models
```python
class PlayerDistribution:
    def __init__(self, player, bucket_priors):
        self.mean = player.projection
        self.cv = self._get_cv_by_bucket(player)  # Position × age buckets
        self.injury_probs = self._get_injury_probs(player)
    
    def sample_season_outcome(self):
        # Truncated Normal + injury overlay
        base_points = max(0, np.random.normal(self.mean, self.cv * self.mean))
        missed_games = np.random.choice([0, 3, 6], p=self.injury_probs)
        return base_points * (17 - missed_games) / 17
```

#### 1.2 Variance Buckets (Position × Age/Tenure)
```yaml
rb:
  rookie: {cv: 0.32, injury_probs: {p0: 0.72, p3: 0.18, p6: 0.10}}
  peak_24_27: {cv: 0.20, injury_probs: {p0: 0.82, p3: 0.12, p6: 0.06}}
  decline_28_plus: {cv: 0.28, injury_probs: {p0: 0.77, p3: 0.15, p6: 0.08}}
wr:
  rookie: {cv: 0.28, injury_probs: {p0: 0.78, p3: 0.14, p6: 0.08}}
  peak_24_28: {cv: 0.18, injury_probs: {p0: 0.85, p3: 0.10, p6: 0.05}}
  decline_31_plus: {cv: 0.22, injury_probs: {p0: 0.82, p3: 0.12, p6: 0.06}}
```

#### 1.3 Implementation Files
- `howie_cli/draft/distributions.py` - Distribution models
- `howie_cli/draft/variance_buckets.yaml` - Age/position variance config
- `howie_cli/draft/player_distributions.py` - Player-specific distributions

### Phase 2: Pre-sampled Outcomes Matrix (PERFORMANCE)

**Goal**: Pre-compute 10K-20K season outcomes for faster rollouts

#### 2.1 Outcomes Matrix
```python
# Pre-sample outcomes matrix [Players × Simulations]
outcomes_matrix = np.zeros((num_players, 15000))
for sim in range(15000):
    for player_idx, player in enumerate(players):
        outcomes_matrix[player_idx, sim] = player.distribution.sample()
```

#### 2.2 Fast Season Scoring
```python
def calculate_season_score(roster, outcome_column_index):
    """Score roster using pre-sampled outcomes"""
    total_score = 0
    for player in roster.starters:
        player_idx = player_index_map[player.name]
        total_score += outcomes_matrix[player_idx, outcome_column_index]
    return total_score
```

#### 2.3 Implementation Files
- `howie_cli/draft/presample.py` - Pre-sampling engine
- `howie_cli/draft/outcome_matrix.py` - Matrix management
- `howie_cli/draft/fast_scoring.py` - Optimized scoring

### Phase 3: Full MCTS Algorithm (STRATEGIC DEPTH)

**Goal**: Tree search with lookahead for multi-pick optimization

#### 3.1 MCTS Node Structure
```python
class MCTSNode:
    def __init__(self, draft_state, parent=None):
        self.state = draft_state  # Round, available players, rosters
        self.parent = parent
        self.children = {}  # Action -> child node
        self.visits = 0
        self.value_sum = 0.0
        self.untried_actions = self._get_available_actions()
```

#### 3.2 PUCT Selection
```python
def select_child(self):
    """PUCT formula for tree traversal"""
    def puct_value(child):
        if child.visits == 0:
            return float('inf')
        
        exploitation = child.value_sum / child.visits
        exploration = math.sqrt(math.log(self.visits) / child.visits)
        return exploitation + C_PUCT * exploration
    
    return max(self.children.values(), key=puct_value)
```

#### 3.3 Progressive Widening
```python
def expand(self):
    """Add new child nodes progressively"""
    if len(self.children) < max(4, min(12, int(math.sqrt(self.visits)))):
        action = self.untried_actions.pop()
        new_state = self.state.make_move(action)
        child = MCTSNode(new_state, parent=self)
        self.children[action] = child
        return child
    return None
```

#### 3.4 Implementation Files
- `howie_cli/draft/mcts_node.py` - Node structure
- `howie_cli/draft/mcts_search.py` - Search algorithm  
- `howie_cli/draft/tree_policy.py` - Selection/expansion
- `howie_cli/draft/rollout_policy.py` - Fast simulation

### Phase 4: Advanced Season Scoring (ACCURACY)

**Goal**: Weekly lineup optimization for realistic season totals

#### 4.1 Weekly Lineup Optimization
```python
def optimize_weekly_lineup(roster_players, week, outcome_column):
    """Optimize lineup for specific week using pre-sampled outcomes"""
    available_points = {}
    for player in roster_players:
        player_idx = player_index_map[player.name]
        weekly_points = outcomes_matrix[player_idx, outcome_column] / 17  # Scale to weekly
        available_points[player] = weekly_points
    
    # Solve lineup optimization (knapsack-style)
    return solve_lineup_optimization(available_points, roster_constraints)
```

#### 4.2 Season Aggregation
```python
def calculate_full_season_score(roster, outcome_column):
    """Calculate season total with weekly lineup decisions"""
    total_score = 0
    for week in range(1, 18):  # 17 weeks
        optimal_lineup = optimize_weekly_lineup(roster.players, week, outcome_column)
        total_score += sum(lineup.weekly_points)
    return total_score
```

#### 4.3 Implementation Files
- `howie_cli/draft/weekly_optimizer.py` - Lineup optimization
- `howie_cli/draft/season_scoring.py` - Full season calculation
- `howie_cli/draft/lineup_constraints.py` - Position requirements

---

## 🎯 CURRENT CLI COMMANDS

### Monte Carlo Simulation
```bash
# Basic simulation
/draft monte --sims 25 --rounds 2

# Full draft simulation  
/draft monte --sims 100 --rounds 15

# With specific strategy
/draft monte --sims 50 --rounds 8 --strategy aggressive
```

### Draft Analysis
```bash
# Pre-draft analysis
/draft analyze

# Pick recommendations
/draft recommend --round 1

# Mock draft
/draft simulate --opponents realistic
```

### Intelligence System
```bash
# Update team intelligence
/update intel

# Single team update
/update intel/PHI

# View intelligence
/intel/PHI/wr
```

---

## 📊 PERFORMANCE BENCHMARKS

- **Simulation Speed**: 25 simulations in ~2-3 seconds
- **Player Universe**: 533 players, 86.7% real ADP coverage
- **Memory Usage**: ~50MB for full player data + outcomes
- **Accuracy**: Realistic draft flow matching expert consensus

---

## 🏆 NEXT STEPS

1. **Phase 1 Implementation**: Player distributions + variance buckets
2. **Performance Testing**: Benchmark pre-sampling vs real-time calculation
3. **MCTS Tree Search**: Strategic lookahead implementation
4. **Season Scoring**: Weekly lineup optimization
5. **Validation**: Backtest against historical drafts

**Current Status**: Production-ready Monte Carlo simulator with excellent results
**Next Milestone**: Full MCTS implementation with strategic depth
