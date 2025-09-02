"""
Strategy Tree Search for Optimal Draft Planning
Uses tree search with variance-adjusted values to find optimal draft strategies
"""

import copy
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict

from .models import Player, Roster, LeagueConfig
from .recommendation_engine import PickRecommendationEngine
from .variance_adjusted_value import get_risk_tolerance_for_context
from .database import DraftDatabaseConnector


@dataclass
class PositionTarget:
    """Recommended position target for a specific round"""
    round_number: int
    position: str
    target_player: Optional[Player]
    confidence: float
    reasoning: str
    alternatives: List[str]  # Alternative positions
    value_score: float
    scarcity_urgency: float


@dataclass
class DraftStrategy:
    """Complete draft strategy with round-by-round targets"""
    league_config: LeagueConfig
    position_targets: List[PositionTarget]
    expected_value: float
    confidence_score: float
    strategy_summary: str
    key_insights: List[str]
    risk_tolerance_profile: Dict[int, float]  # round -> risk tolerance
    contingency_plans: Dict[int, List[str]]  # round -> backup plans
    generated_at: str


@dataclass 
class TreeNode:
    """Node in the strategy tree search"""
    round_number: int
    roster: Roster
    available_players: List[Player]
    position_picked: Optional[str]
    player_picked: Optional[Player]
    value: float
    visits: int
    children: Dict[str, 'TreeNode']
    parent: Optional['TreeNode']
    
    def is_leaf(self) -> bool:
        return len(self.children) == 0
    
    def get_path(self) -> List[str]:
        """Get the path of positions from root to this node"""
        path = []
        node = self
        while node.parent is not None:
            if node.position_picked:
                path.append(node.position_picked)
            node = node.parent
        return list(reversed(path))


class StrategyTreeSearch:
    """Tree search to find optimal draft strategy using Monte Carlo simulation data"""
    
    def __init__(self, league_config: LeagueConfig, player_universe: List[Player], use_monte_carlo: bool = True):
        self.config = league_config
        self.players = player_universe
        self.use_monte_carlo = use_monte_carlo
        self.rec_engine = PickRecommendationEngine(league_config, player_universe, use_variance_adjustment=True)
        
        # Monte Carlo data for realistic player availability
        self.monte_carlo_results = None
        self.availability_data = {}
        
        # Search parameters
        self.max_depth = 16  # Search all 16 rounds for complete roster
        self.iterations_per_round = 100  # Iterations per round
        self.expansion_threshold = 5  # Visits before expanding
        
    def find_optimal_strategy(self) -> DraftStrategy:
        """Find optimal draft strategy using tree search"""
        
        print("🌳 Starting strategy tree search...")
        print(f"📋 League: {self.config.num_teams}T {self.config.scoring_type.upper()}, position #{self.config.draft_position}")
        
        # NOTE: We do NOT use Monte Carlo data during tree search to avoid bias
        # Tree search uses unbiased ADP-based opponent simulation
        # Monte Carlo is only used AFTER to evaluate the final strategy
        print("🎯 Using unbiased ADP-based opponent simulation for strategy selection")
        
        start_time = time.time()
        
        # Initialize root node
        root = TreeNode(
            round_number=0,
            roster=Roster(self.config),
            available_players=copy.deepcopy(self.players),
            position_picked=None,
            player_picked=None,
            value=0.0,
            visits=0,
            children={},
            parent=None
        )
        
        # Run tree search
        for iteration in range(self.iterations_per_round * self.max_depth):
            if iteration % 50 == 0:
                print(f"   Tree search iteration {iteration}/{self.iterations_per_round * self.max_depth}")
            
            # Selection phase
            leaf_node = self._select_leaf(root)
            
            # Expansion phase
            if leaf_node.visits >= self.expansion_threshold and leaf_node.round_number < self.max_depth:
                self._expand_node(leaf_node)
            
            # Simulation phase
            value = self._simulate_from_node(leaf_node)
            
            # Backpropagation phase
            self._backpropagate(leaf_node, value)
        
        # Extract optimal path
        optimal_path = self._extract_optimal_path(root)
        
        # Generate strategy from path
        strategy = self._generate_strategy_from_path(optimal_path)
        
        elapsed = time.time() - start_time
        print(f"✅ Strategy tree search completed in {elapsed:.1f} seconds")
        
        return strategy
    
    def _select_leaf(self, node: TreeNode) -> TreeNode:
        """Select a leaf node using UCB1"""
        current = node
        
        while not current.is_leaf():
            if current.visits == 0:
                return current
            
            # UCB1 selection
            best_child = None
            best_ucb = float('-inf')
            
            for child in current.children.values():
                if child.visits == 0:
                    return child
                
                # UCB1 formula
                exploitation = child.value / child.visits
                exploration = (2 * np.log(current.visits) / child.visits) ** 0.5
                ucb = exploitation + exploration
                
                if ucb > best_ucb:
                    best_ucb = ucb
                    best_child = child
            
            current = best_child
        
        return current
    
    def _expand_node(self, node: TreeNode) -> None:
        """Expand node by adding children for each viable position"""
        if node.round_number >= self.max_depth:
            return
        
        round_number = node.round_number + 1
        
        # Get list of drafted players from roster
        drafted_players = [p for p in node.roster.players]
        
        # Simulate opponent picks that would happen before this round
        realistic_available = self._simulate_opponents_until_round(
            node.available_players, node.round_number, round_number
        )
        
        # Get recommendations for this round with realistic availability
        recommendations = self.rec_engine.generate_round_recommendations(
            round_number, node.roster, drafted_players
        )
        
        # Filter recommendations to only include realistically available players
        realistic_recommendations = [
            rec for rec in recommendations 
            if any(p.name == rec.player.name for p in realistic_available)
        ]
        
        # Group by position and select best from each (use realistic recommendations)
        position_candidates = defaultdict(list)
        for rec in realistic_recommendations[:15]:  # Top 15 realistic candidates
            position_candidates[rec.player.position].append(rec)
        
        # Create child nodes for each position with roster balance constraints
        for position, candidates in position_candidates.items():
            if not candidates:
                continue
            
            # Check if we should limit this position based on roster balance
            if not self._should_consider_position(position, node.roster, round_number):
                continue
            
            # Use best player from this position
            best_candidate = max(candidates, key=lambda x: x.overall_score)
            best_player = best_candidate.player
            
            # Double-check player isn't already on roster
            if any(p.name == best_player.name for p in node.roster.players):
                continue
            
            # Create new roster with this pick
            new_roster = node.roster.add_player(best_player)
            
            # Filter available players from realistic availability
            new_available = [p for p in realistic_available if p.name != best_player.name]
            
            # Create child node
            child = TreeNode(
                round_number=round_number,
                roster=new_roster,
                available_players=new_available,
                position_picked=position,
                player_picked=best_player,
                value=0.0,
                visits=0,
                children={},
                parent=node
            )
            
            node.children[position] = child
    
    def _should_consider_position(self, position: str, roster: Roster, round_number: int) -> bool:
        """Check if we should consider drafting this position based on roster balance"""
        # Count current positions
        position_counts = {}
        for player in roster.players:
            pos = player.position
            position_counts[pos] = position_counts.get(pos, 0) + 1
        
        current_count = position_counts.get(position, 0)
        
        # Position limits based on typical roster construction
        if position == 'QB':
            # Max 2 QBs, don't draft 2nd until mid/late rounds
            if current_count >= 2:
                return False
            if current_count >= 1 and round_number <= 8:
                return False
        elif position == 'RB':
            # Max 5 RBs total, but don't load up too early
            if current_count >= 5:
                return False
            if current_count >= 4 and round_number <= 10:
                return False
            if current_count >= 3 and round_number <= 6:
                return False
        elif position == 'WR':
            # Max 6 WRs total  
            if current_count >= 6:
                return False
            if current_count >= 4 and round_number <= 8:
                return False
        elif position == 'TE':
            # Max 2 TEs, don't draft 2nd until late
            if current_count >= 2:
                return False
            if current_count >= 1 and round_number <= 10:
                return False
        elif position in ['K', 'DEF']:
            # Only 1 kicker/defense, and only in final rounds
            if current_count >= 1:
                return False
            if round_number <= 14:
                return False
        
        return True
    
    def _simulate_opponents_until_round(
        self, 
        available_players: List[Player], 
        current_round: int, 
        target_round: int
    ) -> List[Player]:
        """Simulate opponent picks between current round and target round"""
        if target_round <= current_round:
            return available_players
        
        # Calculate picks that would happen between rounds
        user_position = self.config.draft_position
        remaining_players = available_players.copy()
        
        for round_num in range(current_round + 1, target_round + 1):
            # Calculate pick order for this round
            if round_num % 2 == 1:  # Odd rounds (1, 3, 5...)
                pick_order = list(range(1, self.config.num_teams + 1))
            else:  # Even rounds (snake draft)
                pick_order = list(range(self.config.num_teams, 0, -1))
            
            # Simulate picks until user's turn
            for pick_position in pick_order:
                if pick_position == user_position:
                    # This is the user's pick - stop simulating this round
                    break
                
                if not remaining_players:
                    break
                
                # Simulate opponent pick using ADP-based selection with some randomness
                picked_player = self._simulate_opponent_pick(remaining_players, round_num)
                if picked_player:
                    remaining_players = [p for p in remaining_players if p.name != picked_player.name]
        
        return remaining_players
    
    def _simulate_opponent_pick(self, available_players: List[Player], round_num: int) -> Player:
        """Simulate a single opponent pick using realistic selection logic"""
        if not available_players:
            return None
        
        # Calculate pick number for this round to determine ADP expectations
        round_start_pick = (round_num - 1) * self.config.num_teams + 1
        round_end_pick = round_num * self.config.num_teams
        avg_pick_in_round = (round_start_pick + round_end_pick) / 2
        
        # Filter to players with reasonable ADP for this round (±18 picks buffer)
        adp_buffer = 18
        reasonable_picks = []
        
        for player in available_players:
            if player.adp < 999:  # Has ADP data
                if abs(player.adp - avg_pick_in_round) <= adp_buffer:
                    reasonable_picks.append(player)
            else:
                # For players without ADP, use projection threshold
                min_projection = max(50, 200 - (round_num * 10))
                if player.projection >= min_projection:
                    reasonable_picks.append(player)
        
        # If no reasonable picks, fall back to best available
        if not reasonable_picks:
            reasonable_picks = available_players[:10]
        
        # Sort by a combination of ADP and projection (weighted toward ADP)
        def selection_score(player):
            if player.adp < 999:
                # Better ADP = lower number, so invert it
                adp_score = 300 - player.adp  # Invert so lower ADP = higher score
                projection_score = player.projection
                return adp_score * 0.7 + projection_score * 0.3
            else:
                return player.projection * 0.8  # No ADP penalty
        
        reasonable_picks.sort(key=selection_score, reverse=True)
        
        # Add some randomness - pick from top 5 options
        import random
        top_options = reasonable_picks[:5]
        weights = [5, 4, 3, 2, 1]  # Favor earlier picks but add variance
        
        if len(top_options) < len(weights):
            weights = weights[:len(top_options)]
        
        return random.choices(top_options, weights=weights)[0]
    
    def _simulate_from_node(self, node: TreeNode) -> float:
        """Simulate draft completion from this node"""
        if node.round_number >= self.max_depth:
            return self._evaluate_roster(node.roster)
        
        # Fast simulation using greedy selection
        current_roster = copy.deepcopy(node.roster)
        current_available = copy.deepcopy(node.available_players)
        
        for round_num in range(node.round_number + 1, self.max_depth + 1):
            if not current_available:
                break
            
            # Get risk tolerance for this round
            risk_tolerance = get_risk_tolerance_for_context(
                round_num, 
                {'strength_percentile': self._estimate_roster_strength(current_roster)}
            )
            
            # Simple greedy selection based on variance-adjusted value
            best_player = None
            best_score = float('-inf')
            
            # Check top candidates with roster balance constraints
            for player in current_available[:20]:
                # Skip if position is not viable for roster balance
                if not self._should_consider_position(player.position, current_roster, round_num):
                    continue
                
                base_vorp = self.rec_engine.value_calc.calculate_vorp(player)
                roster_fit = self._calculate_simple_roster_fit(player, current_roster)
                
                # Simple unbiased scoring - no Monte Carlo bias during strategy selection
                score = base_vorp * (1 + risk_tolerance * 0.2) + roster_fit * 10
                
                if score > best_score:
                    best_score = score
                    best_player = player
            
            if best_player:
                current_roster = current_roster.add_player(best_player)
                current_available = [p for p in current_available if p.name != best_player.name]
        
        return self._evaluate_roster(current_roster)
    
    def _backpropagate(self, node: TreeNode, value: float) -> None:
        """Backpropagate value up the tree"""
        current = node
        while current is not None:
            current.visits += 1
            current.value += value
            current = current.parent
    
    def _extract_optimal_path(self, root: TreeNode) -> List[TreeNode]:
        """Extract the optimal path from root to leaf"""
        path = []
        current = root
        
        while current.children:
            # Select child with highest average value
            best_child = None
            best_avg_value = float('-inf')
            
            for child in current.children.values():
                if child.visits > 0:
                    avg_value = child.value / child.visits
                    if avg_value > best_avg_value:
                        best_avg_value = avg_value
                        best_child = child
            
            if best_child is None:
                break
            
            path.append(best_child)
            current = best_child
        
        return path
    
    def _generate_strategy_from_path(self, path: List[TreeNode]) -> DraftStrategy:
        """Generate complete strategy from optimal path"""
        
        position_targets = []
        risk_tolerance_profile = {}
        
        for i, node in enumerate(path):
            round_number = i + 1
            
            # Get risk tolerance for this round
            risk_tolerance = get_risk_tolerance_for_context(
                round_number,
                {'strength_percentile': self._estimate_roster_strength(node.parent.roster if node.parent else Roster(self.config))}
            )
            risk_tolerance_profile[round_number] = risk_tolerance
            
            # Generate recommendations for this round to get reasoning
            recommendations = self.rec_engine.generate_round_recommendations(
                round_number, 
                node.parent.roster if node.parent else Roster(self.config)
            )
            
            # Find the recommendation for this position
            position_rec = None
            alternatives = []
            for rec in recommendations[:10]:
                if rec.player.position == node.position_picked:
                    if position_rec is None:
                        position_rec = rec
                else:
                    alternatives.append(rec.player.position)
            
            # Create position target
            target = PositionTarget(
                round_number=round_number,
                position=node.position_picked,
                target_player=node.player_picked,
                confidence=position_rec.confidence if position_rec else 0.7,
                reasoning=position_rec.primary_reason if position_rec else f"Optimal {node.position_picked} target",
                alternatives=list(set(alternatives[:3])),  # Top 3 alternative positions
                value_score=node.value / max(1, node.visits),
                scarcity_urgency=self._calculate_scarcity_urgency(node.position_picked, round_number)
            )
            
            position_targets.append(target)
        
        # Calculate strategy metrics
        expected_value = path[-1].value / max(1, path[-1].visits) if path else 0
        confidence_score = self._calculate_strategy_confidence(position_targets)
        
        # Generate insights and summary
        key_insights = self._generate_key_insights(position_targets, risk_tolerance_profile)
        strategy_summary = self._generate_strategy_summary(position_targets, key_insights)
        contingency_plans = self._generate_contingency_plans(position_targets)
        
        strategy = DraftStrategy(
            league_config=self.config,
            position_targets=position_targets,
            expected_value=expected_value,
            confidence_score=confidence_score,
            strategy_summary=strategy_summary,
            key_insights=key_insights,
            risk_tolerance_profile=risk_tolerance_profile,
            contingency_plans=contingency_plans,
            generated_at=time.strftime("%Y-%m-%d %H:%M:%S")
        )
        
        # EVALUATE strategy with Monte Carlo (unbiased post-hoc evaluation)
        if self.use_monte_carlo:
            print("🎲 Evaluating strategy performance with Monte Carlo...")
            evaluated_value = self._evaluate_strategy_with_monte_carlo(strategy)
            strategy.expected_value = evaluated_value
        
        return strategy
    
    def _evaluate_strategy_with_monte_carlo(self, strategy: DraftStrategy) -> float:
        """Evaluate strategy performance using Monte Carlo simulation (unbiased)"""
        try:
            from .enhanced_monte_carlo import EnhancedMonteCarloSimulator, OutcomeAwareStrategy
            from .monte_carlo_viewer import MonteCarloResultsViewer
            
            # Create strategy that follows the tree search plan
            class TreeSearchStrategy(OutcomeAwareStrategy):
                def __init__(self, position_targets):
                    super().__init__()
                    self.targets = {target.round_number: target.position for target in position_targets}
                
                def select_pick(self, draft_state, config):
                    current_round = draft_state.current_round
                    target_position = self.targets.get(current_round)
                    
                    if target_position:
                        # Try to get a player at the target position
                        candidates = [p for p in draft_state.available_players if p.position == target_position]
                        if candidates:
                            # Pick best available at target position
                            return max(candidates, key=lambda p: p.projection)
                    
                    # Fallback to best available
                    if draft_state.available_players:
                        return max(draft_state.available_players, key=lambda p: p.projection)
                    return None
            
            # Run simulation to evaluate strategy
            simulator = EnhancedMonteCarloSimulator(self.config)
            strategy_impl = TreeSearchStrategy(strategy.position_targets)
            
            results = simulator.simulate_draft_with_outcomes(
                strategy_impl,
                num_simulations=25,  # Quick evaluation
                rounds_to_simulate=min(8, len(strategy.position_targets))
            )
            
            # Save evaluation results
            viewer = MonteCarloResultsViewer()
            session_name = f"Strategy Evaluation ({self.config.num_teams}T {self.config.scoring_type.upper()} #{self.config.draft_position})"
            viewer.save_results(results, session_name)
            
            # Extract performance metric
            if 'average_score' in results:
                return results['average_score']
            elif 'roster_scores' in results:
                return sum(results['roster_scores']) / len(results['roster_scores'])
            else:
                return 1000.0  # Default fallback
                
        except Exception as e:
            print(f"⚠️  Monte Carlo evaluation failed: {e}")
            return 1000.0  # Fallback value
    
    def _evaluate_roster(self, roster: Roster) -> float:
        """Evaluate the quality of a completed roster"""
        if not roster.players:
            return 0.0
        
        # Sum VORP of all players
        total_vorp = sum(self.rec_engine.value_calc.calculate_vorp(player) for player in roster.players)
        
        # Add positional balance bonus
        position_counts = defaultdict(int)
        for player in roster.players:
            position_counts[player.position] += 1
        
        # Bonus for balanced roster
        balance_bonus = 0
        if position_counts.get('QB', 0) >= 1:
            balance_bonus += 10
        if position_counts.get('RB', 0) >= 2:
            balance_bonus += 15
        if position_counts.get('WR', 0) >= 2:
            balance_bonus += 15
        if position_counts.get('TE', 0) >= 1:
            balance_bonus += 8
        
        return total_vorp + balance_bonus
    
    def _estimate_roster_strength(self, roster: Roster) -> float:
        """Estimate roster strength percentile"""
        return self.rec_engine._estimate_roster_strength(roster)
    
    def _calculate_simple_roster_fit(self, player: Player, roster: Roster) -> float:
        """Simple roster fit calculation"""
        return self.rec_engine._calculate_roster_fit(player, roster)
    
    def _calculate_scarcity_urgency(self, position: str, round_number: int) -> float:
        """Calculate how urgent it is to draft this position"""
        # Simple scarcity model - can be enhanced
        scarcity_by_round = {
            'QB': {1: 0.3, 2: 0.4, 3: 0.6, 4: 0.8, 5: 0.9},
            'RB': {1: 0.9, 2: 0.8, 3: 0.7, 4: 0.6, 5: 0.5},
            'WR': {1: 0.8, 2: 0.7, 3: 0.6, 4: 0.5, 5: 0.4},
            'TE': {1: 0.4, 2: 0.5, 3: 0.7, 4: 0.8, 5: 0.9}
        }
        
        return scarcity_by_round.get(position, {}).get(round_number, 0.5)
    
    def _calculate_strategy_confidence(self, targets: List[PositionTarget]) -> float:
        """Calculate overall confidence in strategy"""
        if not targets:
            return 0.5
        
        avg_confidence = sum(target.confidence for target in targets) / len(targets)
        return min(1.0, max(0.0, avg_confidence))
    
    def _generate_key_insights(self, targets: List[PositionTarget], risk_profile: Dict[int, float]) -> List[str]:
        """Generate key strategic insights"""
        insights = []
        
        # Early round strategy
        early_positions = [t.position for t in targets[:3]]
        if early_positions.count('RB') >= 2:
            insights.append("🏃 Robust RB strategy - securing RB depth early")
        elif early_positions.count('WR') >= 2:
            insights.append("🎯 Zero RB approach - prioritizing WR value")
        elif 'QB' in early_positions:
            insights.append("🚀 Early QB strategy - securing elite quarterback")
        
        # Risk tolerance patterns
        avg_risk = sum(risk_profile.values()) / len(risk_profile)
        if avg_risk > 0.6:
            insights.append("📈 Aggressive approach - targeting high-upside players")
        elif avg_risk < 0.4:
            insights.append("🛡️ Conservative approach - prioritizing safe floors")
        
        # Position balance
        position_counts = defaultdict(int)
        for target in targets:
            position_counts[target.position] += 1
        
        if position_counts.get('TE', 0) >= 2:
            insights.append("💎 TE premium strategy - investing in tight end depth")
        
        return insights
    
    def _generate_strategy_summary(self, targets: List[PositionTarget], insights: List[str]) -> str:
        """Generate high-level strategy summary"""
        if not targets:
            return "No strategy generated"
        
        # Position sequence
        position_sequence = " → ".join([f"R{t.round_number}: {t.position}" for t in targets[:5]])
        
        # Primary strategy type
        early_positions = [t.position for t in targets[:3]]
        if early_positions.count('RB') >= 2:
            strategy_type = "Robust RB"
        elif early_positions.count('WR') >= 2:
            strategy_type = "Zero RB"
        elif 'QB' in early_positions[:2]:
            strategy_type = "Early QB"
        else:
            strategy_type = "Balanced"
        
        summary = f"{strategy_type} Strategy: {position_sequence}"
        
        if insights:
            summary += f"\n\nKey Focus: {insights[0].replace('🏃 ', '').replace('🎯 ', '').replace('🚀 ', '')}"
        
        return summary
    
    def _generate_contingency_plans(self, targets: List[PositionTarget]) -> Dict[int, List[str]]:
        """Generate contingency plans for each round"""
        contingencies = {}
        
        for target in targets:
            round_num = target.round_number
            plans = []
            
            # Add alternative positions
            for alt_pos in target.alternatives:
                plans.append(f"If {target.position} unavailable, consider {alt_pos}")
            
            # Add scarcity-based advice
            if target.scarcity_urgency > 0.7:
                plans.append(f"High urgency - avoid waiting on {target.position}")
            elif target.scarcity_urgency < 0.3:
                plans.append(f"Can afford to wait on {target.position}")
            
            contingencies[round_num] = plans
        
        return contingencies
    
    def _run_monte_carlo_analysis(self) -> None:
        """Run Monte Carlo simulations to get realistic player availability data"""
        from .monte_carlo_simulator import MonteCarloSimulator
        from .keeper_integration import KeeperAwareDraftSimulator
        from .keeper_system import KeeperManager
        
        print("🎲 Running Monte Carlo analysis for realistic player availability...")
        
        # Check if keepers are enabled
        if self.config.keepers_enabled:
            try:
                keeper_manager = KeeperManager()
                keeper_config = keeper_manager.load_keeper_configuration('data/keeper_config.json')
                
                if keeper_config:
                    print(f"   Using keeper-aware simulation with {len(keeper_config.keepers)} keepers")
                    print(f"   Keeper rules: {keeper_config.keeper_rules}")
                    
                    # Show first few keepers for verification
                    print("   Recent keepers loaded:")
                    for i, keeper in enumerate(keeper_config.keepers[:3], 1):
                        print(f"     {i}. {keeper.player_name} - {keeper.team_name} (Round {keeper.keeper_round})")
                    if len(keeper_config.keepers) > 3:
                        print(f"     ... and {len(keeper_config.keepers) - 3} more")
                    
                    simulator = KeeperAwareDraftSimulator(self.config, keeper_config)
                    
                    # Run keeper-aware simulation
                    from .enhanced_monte_carlo import OutcomeAwareStrategy
                    strategy = OutcomeAwareStrategy(risk_tolerance=0.5)
                    results = simulator.simulate_keeper_aware_draft(
                        strategy=strategy, 
                        num_simulations=50,  # Smaller number for tree search preprocessing
                        rounds_to_simulate=self.max_depth
                    )
                    
                    # Extract availability data from keeper-aware results
                    if 'availability_by_round' in results:
                        self.availability_data = results['availability_by_round']
                    elif 'player_availability_by_round' in results:
                        self.availability_data = results['player_availability_by_round']
                    else:
                        # Extract from aggregated results if available
                        self.availability_data = {}
                        
                    self.monte_carlo_results = results
                    print(f"   ✅ Completed keeper-aware Monte Carlo analysis")
                    return
                    
            except Exception as e:
                print(f"   ⚠️  Keeper simulation failed, using regular Monte Carlo: {e}")
        
        # Regular Monte Carlo simulation
        print("   Using regular Monte Carlo simulation")
        simulator = MonteCarloSimulator(self.config, self.players, use_realistic_opponents=True)
        
        results = simulator.run_simulation(
            num_simulations=50,  # Smaller number for preprocessing
            rounds_to_simulate=self.max_depth
        )
        
        # Extract availability data
        self.availability_data = results.availability_rates
        self.monte_carlo_results = results
        print(f"   ✅ Completed Monte Carlo analysis")
    
    def _get_realistic_availability_score(self, player: Player, round_number: int) -> float:
        """Get realistic availability score based on Monte Carlo data"""
        if not self.availability_data:
            return 1.0  # No data, assume always available
        
        player_availability = self.availability_data.get(player.name, {})
        availability_rate = player_availability.get(round_number, 0.0)
        
        # Return availability rate (0.0 = never available, 1.0 = always available)
        return availability_rate


# Add numpy import for UCB1 calculation
import numpy as np
