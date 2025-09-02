# ProjectHowie Draft System Architecture

## 🎯 System Overview

ProjectHowie's draft analysis system uses a sophisticated three-layer approach to provide optimal fantasy football draft recommendations:

1. **Pick Recommendation Engine** - Generates top 10 picks for any round
2. **Strategy Tree Search** - Finds optimal round-by-round draft strategy 
3. **Monte Carlo Simulation** - Evaluates strategy performance against realistic opponents

## 🏗️ Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    DRAFT ANALYSIS SYSTEM                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────┐  │
│  │ RECOMMENDATION  │    │   TREE SEARCH   │    │ MONTE CARLO │  │
│  │     ENGINE      │    │    STRATEGY     │    │ EVALUATION  │  │
│  │                 │    │                 │    │             │  │
│  │ • Round picks   │    │ • Unbiased      │    │ • Realistic │  │
│  │ • Value (VORP)  │    │ • ADP opponents │    │   opponents │  │
│  │ • Scarcity      │    │ • Tree search   │    │ • Multiple  │  │
│  │ • Enhanced      │    │ • 16 rounds     │    │   scenarios │  │
│  │   factors       │    │                 │    │             │  │
│  └─────────────────┘    └─────────────────┘    └─────────────┘  │
│           │                        │                     │       │
│           └────────────────────────┼─────────────────────┘       │
│                                    │                             │
│  ┌─────────────────────────────────┼─────────────────────────────┤
│  │                DATA LAYER       │                             │
│  ├─────────────────────────────────┼─────────────────────────────┤
│  │                                 │                             │
│  │ • 533 Players w/ Projections   │                             │
│  │ • FantasyPros ADP Data          │                             │
│  │ • Strength of Schedule          │                             │
│  │ • Injury Risk Assessment        │                             │
│  │ • Starter Status Intelligence   │                             │
│  │ • League & Keeper Configuration │                             │
│  │                                 │                             │
│  └─────────────────────────────────┘                             │
└─────────────────────────────────────────────────────────────────┘
```

## 🎮 Component Details

### 1. Pick Recommendation Engine

**Purpose**: Generate ranked pick recommendations for any round based on current roster state.

**Key Features**:
- **6-Factor Scoring**: VORP, positional scarcity, roster fit, SoS, starter status, injury risk
- **ADP Filtering**: Only considers realistically available players (±12 picks from ADP)
- **Roster Context**: Adapts recommendations based on existing players
- **Enhanced Intelligence**: Uses AI-gathered scouting data for context

**Algorithm**:
```python
for each_candidate_player:
    # Core value metrics
    vorp = calculate_value_over_replacement()
    scarcity = calculate_positional_scarcity()
    roster_fit = calculate_roster_construction_value()
    
    # Enhanced factors (40% of total score)
    sos_advantage = calculate_schedule_strength()
    starter_confidence = get_projected_starter_status()
    injury_risk = assess_injury_risk()
    
    # Weighted combination
    overall_score = combine_all_factors()
```

**Commands**:
- `/draft analyze` - Full round-by-round analysis
- `/draft quick` - Fast single-round picks

### 2. Strategy Tree Search

**Purpose**: Find optimal draft strategy using unbiased tree search across all 16 rounds.

**Key Design Principles**:
- ✅ **Unbiased Opponent Simulation**: Uses generic ADP-based opponent picks
- ✅ **No Monte Carlo Bias**: Strategy selection independent of MC variance
- ✅ **Roster Balance**: Enforces position limits (max 2 QBs, 5 RBs, 6 WRs, etc.)
- ✅ **Realistic Availability**: Simulates opponent picks between user turns

**Algorithm**:
```python
# Phase 1: Strategy Creation (Unbiased)
def find_optimal_strategy():
    root_node = create_initial_state()
    
    for iteration in range(max_iterations):
        # UCB1 selection - balance exploitation vs exploration
        leaf = select_promising_leaf(root_node)
        
        # Progressive expansion - add new position targets
        if leaf.visits > expansion_threshold:
            expanded_leaf = expand_node(leaf)
        
        # Fast rollout with unbiased opponents
        value = simulate_draft_completion(expanded_leaf)
        
        # Backpropagate value up the tree
        update_node_values(expanded_leaf, value)
    
    return extract_optimal_path(root_node)
```

**Opponent Simulation**:
- Uses **ADP + variance** (σ=6-12 picks) for realistic opponent behavior
- Simulates picks between user turns to ensure realistic player availability
- **No Monte Carlo data** used during strategy selection

**Commands**:
- `/draft strategy generate` - Create new optimal strategy
- `/draft strategy view` - View saved strategies

### 3. Monte Carlo Simulation & Evaluation

**Purpose**: Evaluate strategy performance against diverse opponent scenarios **after** strategy creation.

**Two-Phase Approach**:

#### Phase 1: Strategy Selection (Tree Search)
- Uses unbiased ADP-based opponents
- Pure value optimization
- Independent of Monte Carlo variance

#### Phase 2: Strategy Evaluation (Monte Carlo)
- Tests strategy against realistic opponent models
- Multiple drafting personalities (Zero RB, Robust RB, etc.)
- Statistical performance analysis

**Monte Carlo Features**:
- **Diverse Opponents**: 6+ drafting personalities with different biases
- **Realistic Variance**: Gaussian noise around ADP values
- **Player Distributions**: Variance-adjusted projections with upside/downside
- **Statistical Analysis**: Win probability, roster strength, consistency metrics

**Commands**:
- `/draft monte 25 8` - Run 25 simulations for 8 rounds
- `/draft view` - View saved Monte Carlo results

## 🔄 Integration Flow

```
1. User Request
   ↓
2. Load Configuration (League + Keepers)
   ↓
3. Pick Recommendations → Generate top 10 for current round
   ↓
4. Tree Search Strategy → Find optimal 16-round plan
   ↓
5. Monte Carlo Evaluation → Test strategy performance
   ↓
6. Save Results → Store for future reference
   ↓
7. Display Analysis → Rich formatted output
```

## 📊 Example Output

### Pick Recommendations
```
📍 ROUND 1 ANALYSIS - Pick #8

🎯 TOP 3 RECOMMENDATIONS:
1. CeeDee Lamb        WR  (310 pts) - Score: 8.7
   💡 Elite WR1 value with favorable schedule
   📅 🟢 Favorable SoS (Rank 8)
   🏈 ✅ Projected Starter (95% confidence)
   🏥 💪 Low Injury Risk

2. Amon-Ra St. Brown  WR  (298 pts) - Score: 8.4
   💡 Consistent WR1 with target volume
   📊 VORP: 89.1 | SoS: 0.72 | Starter: 0.95
```

### Strategy Overview
```
🎯 OPTIMAL STRATEGY OVERVIEW
Round  1: WR - Elite receiver value
Round  2: QB - Secure premium QB position  
Round  3: RB - Address scarcity concern
Round  4: WR - Build receiving depth
Round  5: TE - Premium tight end value
```

## 🎯 Key Benefits

### Unbiased Strategy Creation
- Tree search uses generic opponents, not specific Monte Carlo scenarios
- Strategies are fundamentally sound, not fitted to particular simulations
- Multiple strategies can be compared fairly

### Realistic Player Availability  
- Opponent simulation removes elite players appropriately
- No more "Garrett Wilson available in Round 4" scenarios
- True scarcity modeling based on realistic draft flow

### Comprehensive Analysis
- 533 players with enhanced intelligence data
- 6-factor scoring including advanced metrics
- Full 16-round strategy with contingency planning

### Performance Validation
- Post-hoc Monte Carlo evaluation measures true strategy performance
- Statistical analysis across multiple scenarios
- Win probability and consistency metrics

## 🛠️ Technical Implementation

**Core Files**:
- `howie_cli/draft/recommendation_engine.py` - Pick recommendations
- `howie_cli/draft/strategy_tree_search.py` - Tree search algorithm
- `howie_cli/draft/enhanced_monte_carlo.py` - Monte Carlo simulation
- `howie_cli/draft/draft_cli.py` - CLI integration

**Database Integration**:
- Uses existing ProjectHowie `fantasy_ppr.db`
- 533 players with FantasyPros ADP
- Enhanced with AI-gathered intelligence data
- League and keeper configuration system

**Performance**:
- Recommendations: <1 second
- Tree search: 10-15 seconds for 16 rounds  
- Monte Carlo: 2-3 seconds for 25 simulations

This architecture provides professional-grade draft analysis that balances speed, accuracy, and strategic depth while maintaining clear separation between strategy creation and performance evaluation.
