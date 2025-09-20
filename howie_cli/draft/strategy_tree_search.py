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
    
    def __init__(self, league_config: LeagueConfig, player_universe: List[Player], use_monte_carlo: bool = True, fast_mode: bool = False):
        self.config = league_config
        self.original_players = player_universe
        self.use_monte_carlo = use_monte_carlo
        
        # Remove kept players from available pool and load keeper config
        self.players = self._remove_kept_players(player_universe)
        self.keeper_config = self._load_keeper_config()
        self.rec_engine = PickRecommendationEngine(league_config, player_universe, use_variance_adjustment=True)
        
        # Monte Carlo data for realistic player availability
        self.monte_carlo_results = None
        self.availability_data = {}
        
        # Search parameters - optimized for TUI performance
        self.max_depth = 16  # Search all 16 rounds for complete roster
        if fast_mode:
            self.iterations_per_round = 15   # Even faster for TUI quick generation
            self.expansion_threshold = 2     # Very quick expansion
        else:
            self.iterations_per_round = 30   # Reduced from 100 for faster TUI performance
            self.expansion_threshold = 3     # Reduced from 5 for faster expansion
        
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
        total_iterations = self.iterations_per_round * self.max_depth
        print(f"🔍 Running {total_iterations} iterations ({self.iterations_per_round} per round × {self.max_depth} rounds)")
        
        for iteration in range(total_iterations):
            if iteration % 30 == 0:  # More frequent progress updates
                progress = (iteration / total_iterations) * 100
                print(f"   Progress: {iteration}/{total_iterations} ({progress:.1f}%)")
            
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
        total_players = len(roster.players)
        
        # ESSENTIAL POSITION REQUIREMENTS - must draft these eventually
        # Force drafting essential positions if we're getting late and don't have them
        if round_number >= 13:  # Rounds 13-16
            if position == 'QB' and position_counts.get('QB', 0) == 0:
                return True  # MUST have a QB
            if position == 'TE' and position_counts.get('TE', 0) == 0:
                return True  # MUST have a TE
            if position == 'K' and position_counts.get('K', 0) == 0:
                return True  # MUST have a K
            if position == 'DEF' and position_counts.get('DEF', 0) == 0:
                return True  # MUST have a DEF
        
        # Position limits based on realistic roster construction
        if position == 'QB':
            # Need at least 1 QB, max 2 QBs
            if current_count >= 2:
                return False
            # Encourage QB by round 10 if we don't have one
            if current_count == 0 and round_number >= 10:
                return True  # Priority for QB
            if current_count >= 1 and round_number <= 6:
                return False  # Don't draft 2nd QB too early
        elif position == 'RB':
            # Need 2-3 startable RBs, max 4-5 total
            if current_count >= 5:
                return False
            if current_count >= 3 and round_number <= 8:
                return False  # Don't overdraft RBs early
        elif position == 'WR':
            # Need 3-4 startable WRs, max 6 total
            if current_count >= 6:
                return False
            if current_count >= 4 and round_number <= 10:
                return False
        elif position == 'TE':
            # Need at least 1 TE, max 2 TEs
            if current_count >= 2:
                return False
            # Encourage TE by round 12 if we don't have one
            if current_count == 0 and round_number >= 12:
                return True  # Priority for TE
            if current_count >= 1 and round_number <= 8:
                return False  # Don't draft 2nd TE too early
        elif position in ['K', 'DEF']:
            # Need exactly 1 of each, only in final rounds
            if current_count >= 1:
                return False
            if round_number <= 13:
                return False  # Only draft K/DEF in rounds 14-16
        
        return True
    
    def _simulate_opponents_until_round(
        self, 
        available_players: List[Player], 
        current_round: int, 
        target_round: int
    ) -> List[Player]:
        """Simulate keeper-aware opponent picks between current round and target round"""
        if target_round <= current_round:
            return available_players
        
        # Calculate keeper picks and which draft slots they occupy
        keeper_picks = self._calculate_keeper_picks()
        user_position = self.config.draft_position
        remaining_players = available_players.copy()
        
        # Simulate each round between current and target
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
                
                # Calculate overall pick number
                if round_num % 2 == 1:  # Odd rounds
                    overall_pick = (round_num - 1) * self.config.num_teams + pick_position
                else:  # Even rounds (snake)
                    overall_pick = round_num * self.config.num_teams - (pick_position - 1)
                
                # Check if this pick is a keeper pick (skip if so)
                if overall_pick in keeper_picks:
                    # This pick is used by a keeper - no simulation needed
                    continue
                
                # Simulate opponent pick with keeper awareness
                picked_player = self._simulate_keeper_aware_opponent_pick(
                    remaining_players, round_num, pick_position
                )
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
    
    def _simulate_keeper_aware_opponent_pick(self, available_players: List[Player], round_num: int, team_position: int) -> Player:
        """Simulate opponent pick with keeper awareness and positional constraints"""
        if not available_players:
            return None
        
        # Get what this team already has (based on keepers)
        team_keepers = []
        if self.keeper_config:
            team_keepers = [k for k in self.keeper_config.keepers if k.draft_position == team_position]
        
        # Count positions already on this team
        position_counts = {}
        for keeper in team_keepers:
            # Get position from kept player
            kept_player = next((p for p in self.original_players if p.name == keeper.player_name), None)
            if kept_player:
                pos = kept_player.position
                position_counts[pos] = position_counts.get(pos, 0) + 1
        
        # Filter candidates based on positional needs and ADP
        candidates = self._filter_keeper_aware_candidates(available_players, round_num, position_counts)
        
        if not candidates:
            # Fallback to regular opponent pick
            return self._simulate_opponent_pick(available_players, round_num)
        
        # Use weighted selection based on value and positional need
        return self._select_best_candidate_for_team(candidates, position_counts, round_num)
    
    def _filter_keeper_aware_candidates(self, available_players: List[Player], round_num: int, position_counts: dict) -> List[Player]:
        """Filter candidates based on keeper-aware positional constraints"""
        candidates = []
        
        for player in available_players:
            # Basic ADP filtering (same as before)
            round_start_pick = (round_num - 1) * self.config.num_teams + 1
            round_end_pick = round_num * self.config.num_teams
            avg_pick_in_round = (round_start_pick + round_end_pick) / 2
            
            # Reasonable ADP range
            if player.adp < 999 and abs(player.adp - avg_pick_in_round) > 24:
                continue
            
            # Positional constraints based on what team already has
            pos = player.position.upper()
            current_count = position_counts.get(pos, 0)
            
            # Apply realistic positional limits
            if pos == 'QB' and current_count >= 1 and round_num <= 10:
                continue  # Don't draft 2nd QB early
            elif pos == 'RB' and current_count >= 3 and round_num <= 6:
                continue  # Don't hoard RBs too early
            elif pos == 'WR' and current_count >= 4 and round_num <= 8:
                continue  # Don't hoard WRs too early
            elif pos == 'TE' and current_count >= 1 and round_num <= 8:
                continue  # Don't draft 2nd TE early
            elif pos in ['K', 'DEF'] and round_num <= 12:
                continue  # Never draft K/DEF early
            
            candidates.append(player)
        
        return candidates
    
    def _select_best_candidate_for_team(self, candidates: List[Player], position_counts: dict, round_num: int) -> Player:
        """Select best candidate considering team needs"""
        if not candidates:
            return None
        
        # Score candidates based on value + positional need
        scored_candidates = []
        
        for player in candidates:
            pos = player.position.upper()
            current_count = position_counts.get(pos, 0)
            
            # Base score from projection
            base_score = player.projection
            
            # Positional need multiplier
            need_multiplier = 1.0
            if round_num <= 6:  # Early rounds - prioritize RB/WR
                if pos in ['RB', 'WR'] and current_count < 2:
                    need_multiplier = 1.3
                elif pos == 'QB' and current_count == 0:
                    need_multiplier = 1.1
            elif round_num <= 10:  # Mid rounds - fill gaps
                if pos == 'QB' and current_count == 0:
                    need_multiplier = 1.4
                elif pos == 'TE' and current_count == 0:
                    need_multiplier = 1.2
            else:  # Late rounds - essential positions
                if pos in ['QB', 'TE'] and current_count == 0:
                    need_multiplier = 1.5  # Must have these
                elif pos in ['K', 'DEF'] and current_count == 0:
                    need_multiplier = 1.3
            
            final_score = base_score * need_multiplier
            scored_candidates.append((player, final_score))
        
        # Sort and add some randomness
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Select from top 3 with weighted randomness
        import random
        top_candidates = scored_candidates[:3]
        weights = [score for _, score in top_candidates]
        
        if sum(weights) > 0:
            return random.choices([player for player, _ in top_candidates], weights=weights)[0]
        else:
            return top_candidates[0][0]
    
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
    
    def _load_most_recent_strategy(self) -> Optional['DraftStrategy']:
        """Load the most recent tree search strategy for this league configuration"""
        try:
            from .strategy_manager import StrategyManager
            manager = StrategyManager()
            strategies = manager.list_strategies()
            
            # Find strategy matching current configuration
            for session in strategies:
                strategy = session.strategy
                if (strategy.league_config.num_teams == self.config.num_teams and
                    strategy.league_config.scoring_type == self.config.scoring_type and
                    strategy.league_config.draft_position == self.config.draft_position):
                    return strategy
            
            return None
        except Exception as e:
            print(f"⚠️  Could not load strategy: {e}")
            return None
    
    def _get_strategy_position_target(self, tree_strategy: 'DraftStrategy', round_num: int) -> Optional[str]:
        """Get the position target from tree search strategy for a specific round"""
        if not tree_strategy or not tree_strategy.position_targets:
            return None
        
        for target in tree_strategy.position_targets:
            if target.round_number == round_num:
                return target.position
        
        return None
    
    def _calculate_strategy_alignment_score(self, player: 'Player', target_position: Optional[str], round_num: int) -> float:
        """Calculate how well a player aligns with the tree search strategy"""
        if not target_position:
            return 0.5  # Neutral if no strategy target
        
        # Perfect match for target position
        if player.position.upper() == target_position.upper():
            return 1.0
        
        # Partial credit for flex-eligible positions
        if target_position.upper() == 'FLEX':
            if player.position.upper() in ['RB', 'WR', 'TE']:
                return 0.8
        
        # Position priority adjustments based on round
        if round_num <= 6:  # Early rounds: premium positions
            if player.position.upper() in ['RB', 'WR'] and target_position.upper() in ['RB', 'WR']:
                return 0.6  # Cross-position flexibility early
        
        return 0.2  # Low alignment for off-target positions
    
    def generate_round_by_round_recommendations(self) -> str:
        """Generate 5-7 recommendations for each of 16 rounds using tree search strategy"""
        output = []
        current_roster = Roster(self.config)
        drafted_players = []
        
        # Try to load the most recent tree search strategy
        tree_strategy = self._load_most_recent_strategy()
        
        output.append("🎯 STRATEGIC ROUND-BY-ROUND DRAFT RECOMMENDATIONS")
        output.append("=" * 80)
        output.append(f"League: {self.config.num_teams}T {self.config.scoring_type.upper()}")
        output.append(f"Your Draft Position: #{self.config.draft_position}")
        
        if tree_strategy:
            output.append(f"🌳 Strategy Base: Tree Search Strategy (Generated: {tree_strategy.generated_at})")
            output.append(f"📊 Expected Strategy Value: {tree_strategy.expected_value:.1f} pts")
        else:
            output.append("⚠️  No tree search strategy found - generating fresh recommendations")
            
        output.append("=" * 80)
        
        for round_num in range(1, 17):  # All 16 rounds
            pick_number = self._calculate_pick_number(round_num)
            
            output.append(f"\n📍 ROUND {round_num} - Pick #{pick_number}")
            output.append("-" * 60)
            
            # SIMULATE OPPONENTS before your pick
            # For round 1, simulate picks 1-7 before your pick #8
            # For later rounds, simulate all picks between last round and your current pick
            
            available_players_before_simulation = [p for p in self.players if p not in drafted_players]
            
            if round_num == 1:
                # Round 1: Simulate keeper-aware picks 1 through (your_position - 1)
                keeper_picks = self._calculate_keeper_picks()
                picks_before_you = self.config.draft_position - 1
                opponent_picks = []
                
                for pick_num in range(1, picks_before_you + 1):
                    # Check if this pick is a keeper pick
                    if pick_num in keeper_picks:
                        # This pick is used by a keeper - skip simulation
                        continue
                    
                    if available_players_before_simulation:
                        # Simulate keeper-aware opponent picking
                        picked_player = self._simulate_keeper_aware_opponent_pick(
                            available_players_before_simulation, round_num, pick_num
                        )
                        if picked_player:
                            opponent_picks.append(picked_player)
                            available_players_before_simulation.remove(picked_player)
                
                drafted_players.extend(opponent_picks)
                
                # Show simulation results
                keeper_count = len([p for p in range(1, picks_before_you + 1) if p in keeper_picks])
                if opponent_picks or keeper_count > 0:
                    total_removed = len(opponent_picks) + keeper_count
                    output.append(f"🤖 Round 1 pre-picks: {total_removed} players/picks before #{self.config.draft_position}")
                    if keeper_count > 0:
                        output.append(f"    🏆 {keeper_count} keeper picks")
                    if opponent_picks:
                        output.append(f"    📊 {len(opponent_picks)} opponent picks: {', '.join([p.name for p in opponent_picks[:3]])}{'...' if len(opponent_picks) > 3 else ''}")
            
            elif round_num > 1:
                # Later rounds: Simulate picks between last round and this round
                available_after_opponents = self._simulate_opponents_until_round(
                    available_players_before_simulation, 
                    round_num - 1, 
                    round_num
                )
                # Update drafted players list to include simulated opponent picks
                opponent_picks = [p for p in available_players_before_simulation if p not in available_after_opponents]
                drafted_players.extend(opponent_picks)
                
                if opponent_picks:
                    output.append(f"🤖 Simulated opponent picks: {len(opponent_picks)} players removed")
            
            # Get tree search strategy target for this round
            target_position = self._get_strategy_position_target(tree_strategy, round_num)
            
            # Get top recommendations for this round (with opponents simulated)
            recommendations = self.rec_engine.generate_round_recommendations(
                round_num, current_roster, drafted_players
            )
            
            # ENHANCE recommendations with strategy alignment scoring
            enhanced_recommendations = []
            for rec in recommendations:
                if self._should_consider_position(rec.player.position, current_roster, round_num):
                    # Calculate strategy alignment score
                    strategy_alignment = self._calculate_strategy_alignment_score(
                        rec.player, target_position, round_num
                    )
                    
                    # Create enhanced recommendation with strategy scoring
                    enhanced_rec = type('EnhancedRecommendation', (), {
                        'player': rec.player,
                        'overall_score': rec.overall_score,
                        'vorp': rec.vorp,
                        'scarcity_score': rec.scarcity_score,
                        'roster_fit': rec.roster_fit,
                        'explanation': rec.primary_reason,
                        'details': {
                            'vorp': rec.vorp,
                            'scarcity': rec.scarcity_score,
                            'roster_fit': rec.roster_fit
                        },
                        'strategy_alignment': strategy_alignment,
                        'is_strategy_target': (rec.player.position.upper() == (target_position or '').upper()),
                        'combined_score': rec.overall_score * (1 + strategy_alignment * 0.3)  # Boost strategy-aligned picks
                    })()
                    
                    enhanced_recommendations.append(enhanced_rec)
            
            # Sort by combined score (value + strategy alignment)
            enhanced_recommendations.sort(key=lambda r: r.combined_score, reverse=True)
            
            # Take top 7 recommendations
            valid_recommendations = enhanced_recommendations[:7]
            
            if not valid_recommendations:
                output.append("⚠️  No valid recommendations (roster balance constraints)")
                continue
            
            # Show strategy target
            if target_position:
                strategy_emoji = "🎯" if tree_strategy else "🔍"
                output.append(f"{strategy_emoji} STRATEGY TARGET: {target_position}")
                output.append("")
            
            # Show recommendations with strategy alignment
            output.append("🎯 TOP RECOMMENDATIONS:")
            for i, rec in enumerate(valid_recommendations, 1):
                # Strategy indicators
                if rec.is_strategy_target:
                    strategy_indicator = "🌳"  # Tree search target
                elif rec.strategy_alignment > 0.6:
                    strategy_indicator = "🎯"  # High alignment
                elif rec.strategy_alignment > 0.4:
                    strategy_indicator = "📊"  # Moderate alignment
                else:
                    strategy_indicator = "🔍"  # Off-strategy
                
                output.append(
                    f"{i}. {rec.player.name:<20} {rec.player.position:2s} "
                    f"({rec.player.projection:.0f} pts) {strategy_indicator} Score: {rec.combined_score:.1f}"
                )
                
                # Enhanced explanation with strategy context
                explanation = rec.explanation
                if rec.is_strategy_target:
                    explanation += " - TREE SEARCH TARGET"
                elif rec.strategy_alignment > 0.6:
                    explanation += " - High strategy alignment"
                
                output.append(f"   💡 {explanation}")
                
                if i <= 3:  # Show detailed metrics for top 3
                    output.append(
                        f"   📊 VORP: {rec.details.get('vorp', 0):.1f} | "
                        f"Scarcity: {rec.details.get('scarcity', 0):.2f} | "
                        f"Fit: {rec.details.get('roster_fit', 0):.1f} | "
                        f"Strategy: {rec.strategy_alignment:.1f}"
                    )
                output.append("")
            
            # Show current roster composition
            position_counts = {}
            for player in current_roster.players:
                pos = player.position
                position_counts[pos] = position_counts.get(pos, 0) + 1
            
            if position_counts:
                roster_summary = " | ".join([f"{pos}: {count}" for pos, count in position_counts.items()])
                output.append(f"📋 Current roster: {roster_summary}")
            
            # Simulate taking the top recommendation for next round
            if valid_recommendations:
                top_pick = valid_recommendations[0].player
                current_roster = current_roster.add_player(top_pick)
                drafted_players.append(top_pick)
                output.append(f"➡️  Simulating pick: {top_pick.name} ({top_pick.position})")
        
        return "\n".join(output)
    
    def generate_positional_strategy_guide(self) -> str:
        """Generate contingency-based positional strategy with primary/backup plans"""
        output = []
        current_roster = Roster(self.config)
        drafted_players = []
        
        # Load tree search strategy
        tree_strategy = self._load_most_recent_strategy()
        
        output.append("🎯 POSITIONAL STRATEGY GUIDE - PRIMARY & BACKUP PLANS")
        output.append("=" * 80)
        output.append(f"League: {self.config.num_teams}T {self.config.scoring_type.upper()}")
        output.append(f"Your Draft Position: #{self.config.draft_position}")
        
        if tree_strategy:
            output.append(f"🌳 Based on: Tree Search Strategy (Generated: {tree_strategy.generated_at})")
        else:
            output.append("⚠️  No tree search strategy - using value-based guidance")
            
        output.append("=" * 80)
        
        for round_num in range(1, 17):
            pick_number = self._calculate_pick_number(round_num)
            
            output.append(f"\n📍 ROUND {round_num} - Pick #{pick_number}")
            output.append("-" * 60)
            
            # Simulate opponent picks to get realistic availability
            available_players_before_simulation = [p for p in self.players if p not in drafted_players]
            
            if round_num == 1:
                # Round 1: Simulate picks 1 through (your_position - 1)
                picks_before_you = self.config.draft_position - 1
                opponent_picks = []
                
                for pick_num in range(1, picks_before_you + 1):
                    if available_players_before_simulation:
                        picked_player = self._simulate_opponent_pick(available_players_before_simulation, round_num)
                        if picked_player:
                            opponent_picks.append(picked_player)
                            available_players_before_simulation.remove(picked_player)
                
                drafted_players.extend(opponent_picks)
            
            elif round_num > 1:
                # Later rounds: Simulate picks between rounds
                available_after_opponents = self._simulate_opponents_until_round(
                    available_players_before_simulation, 
                    round_num - 1, 
                    round_num
                )
                opponent_picks = [p for p in available_players_before_simulation if p not in available_after_opponents]
                drafted_players.extend(opponent_picks)
            
            # Get primary position target from tree search
            primary_position = self._get_strategy_position_target(tree_strategy, round_num)
            
            # Generate positional strategy for this round
            strategy_plan = self._generate_round_positional_strategy(
                round_num, current_roster, drafted_players, primary_position
            )
            
            # Display the strategy plan
            output.extend(strategy_plan)
            
            # Simulate taking the best available primary target for next round context
            if strategy_plan:
                available = [p for p in self.players if p not in drafted_players]
                
                # Get the actual primary options that were shown
                primary_candidates = [p for p in available if p.position.upper() == (primary_position or '').upper()]
                primary_candidates = [p for p in primary_candidates if self._should_consider_position(p.position, current_roster, round_num)]
                primary_candidates.sort(key=lambda p: p.projection, reverse=True)
                
                if primary_candidates:
                    # Take the #1 recommended player (highest projection + roster fit)
                    best_pick = primary_candidates[0]
                    current_roster = current_roster.add_player(best_pick)
                    drafted_players.append(best_pick)
                    output.append(f"     ➡️  Simulating pick: {best_pick.name} ({best_pick.position})")
                else:
                    # Try backup positions from the strategy plan
                    backup_positions = self._determine_backup_positions(primary_position or '', round_num, current_roster)
                    for backup_pos in backup_positions:
                        backup_candidates = [p for p in available if p.position.upper() == backup_pos.upper()]
                        backup_candidates = [p for p in backup_candidates if self._should_consider_position(p.position, current_roster, round_num)]
                        if backup_candidates:
                            best_backup = max(backup_candidates, key=lambda p: p.projection)
                            current_roster = current_roster.add_player(best_backup)
                            drafted_players.append(best_backup)
                            output.append(f"     ➡️  Simulating pick: {best_backup.name} ({best_backup.position}) [backup]")
                            break
        
        return "\n".join(output)
    
    def _generate_round_positional_strategy(self, round_num: int, roster: Roster, drafted_players: List[Player], primary_position: Optional[str]) -> List[str]:
        """Generate primary and backup positional strategy for a specific round"""
        output = []
        available = [p for p in self.players if p not in drafted_players]
        
        # Determine primary position (from tree search or smart default)
        if primary_position:
            output.append(f"🎯 PRIMARY TARGET: {primary_position}")
        else:
            # Smart default based on roster needs and round
            smart_position = self._determine_smart_position_target(round_num, roster)
            output.append(f"🔍 RECOMMENDED FOCUS: {smart_position}")
            primary_position = smart_position
        
        # Calculate pick number for availability context
        pick_number = self._calculate_pick_number(round_num)
        
        # Get top 5 players from primary position
        primary_candidates = [p for p in available if p.position.upper() == primary_position.upper()]
        primary_candidates = [p for p in primary_candidates if self._should_consider_position(p.position, roster, round_num)]
        primary_candidates.sort(key=lambda p: p.projection, reverse=True)
        
        if primary_candidates[:5]:
            output.append(f"✅ {primary_position} OPTIONS ({len(primary_candidates[:5])}):")
            for i, player in enumerate(primary_candidates[:5], 1):
                availability_indicator = "🔥" if player.adp <= pick_number * 0.8 else "⚡" if player.adp <= pick_number * 1.2 else "📊"
                output.append(f"   {i}. {player.name} ({player.projection:.0f} pts) {availability_indicator}")
        else:
            output.append(f"❌ No viable {primary_position} options available")
        
        # Generate backup positions
        backup_positions = self._determine_backup_positions(primary_position, round_num, roster)
        
        for backup_pos in backup_positions[:2]:  # Show top 2 backup positions
            backup_candidates = [p for p in available if p.position.upper() == backup_pos.upper()]
            backup_candidates = [p for p in backup_candidates if self._should_consider_position(p.position, roster, round_num)]
            backup_candidates.sort(key=lambda p: p.projection, reverse=True)
            
            if backup_candidates[:3]:  # Show top 3 for backups
                output.append(f"🔄 BACKUP PLAN: {backup_pos} ({len(backup_candidates[:3])} options)")
                for i, player in enumerate(backup_candidates[:3], 1):
                    availability_indicator = "🔥" if player.adp <= pick_number * 0.8 else "⚡" if player.adp <= pick_number * 1.2 else "📊"
                    output.append(f"   {i}. {player.name} ({player.projection:.0f} pts) {availability_indicator}")
        
        output.append("")
        
        return output
    
    def _determine_smart_position_target(self, round_num: int, roster: Roster) -> str:
        """Determine smart position target when no tree search strategy available"""
        position_counts = {}
        for player in roster.players:
            pos = player.position
            position_counts[pos] = position_counts.get(pos, 0) + 1
        
        # Early rounds: Premium positions
        if round_num <= 3:
            if position_counts.get('RB', 0) < 2:
                return 'RB'
            elif position_counts.get('WR', 0) < 2:
                return 'WR'
            else:
                return 'RB'  # Default to RB early
        
        # Mid rounds: Fill needs
        elif round_num <= 8:
            if position_counts.get('QB', 0) == 0:
                return 'QB'
            elif position_counts.get('TE', 0) == 0:
                return 'TE'
            elif position_counts.get('RB', 0) < 3:
                return 'RB'
            elif position_counts.get('WR', 0) < 4:
                return 'WR'
            else:
                return 'WR'
        
        # Late rounds: Fill remaining slots
        else:
            if position_counts.get('QB', 0) == 0:
                return 'QB'
            elif position_counts.get('TE', 0) == 0:
                return 'TE'
            elif position_counts.get('K', 0) == 0:
                return 'K'
            elif position_counts.get('DEF', 0) == 0:
                return 'DEF'
            else:
                return 'WR'  # Default depth
    
    def _determine_backup_positions(self, primary_position: str, round_num: int, roster: Roster) -> List[str]:
        """Determine backup positions if primary is not available"""
        position_counts = {}
        for player in roster.players:
            pos = player.position
            position_counts[pos] = position_counts.get(pos, 0) + 1
        
        backups = []
        
        # Position-specific backup logic
        if primary_position.upper() == 'RB':
            if position_counts.get('WR', 0) < 4:
                backups.append('WR')
            if position_counts.get('TE', 0) == 0 and round_num <= 10:
                backups.append('TE')
            if position_counts.get('QB', 0) == 0 and round_num <= 12:
                backups.append('QB')
        
        elif primary_position.upper() == 'WR':
            if position_counts.get('RB', 0) < 4:
                backups.append('RB')
            if position_counts.get('TE', 0) == 0 and round_num <= 10:
                backups.append('TE')
            if position_counts.get('QB', 0) == 0 and round_num <= 12:
                backups.append('QB')
        
        elif primary_position.upper() == 'QB':
            if position_counts.get('WR', 0) < 5:
                backups.append('WR')
            if position_counts.get('RB', 0) < 4:
                backups.append('RB')
            if position_counts.get('TE', 0) == 0:
                backups.append('TE')
        
        elif primary_position.upper() == 'TE':
            if position_counts.get('WR', 0) < 5:
                backups.append('WR')
            if position_counts.get('RB', 0) < 4:
                backups.append('RB')
            if position_counts.get('QB', 0) == 0:
                backups.append('QB')
        
        # Generic backups if none specific
        if not backups:
            if position_counts.get('WR', 0) < 6:
                backups.append('WR')
            if position_counts.get('RB', 0) < 5:
                backups.append('RB')
        
        return backups
    
    def _remove_kept_players(self, player_universe: List[Player]) -> List[Player]:
        """Remove kept players from the available player pool"""
        try:
            from .keeper_system import KeeperManager
            
            keeper_manager = KeeperManager()
            keeper_config = keeper_manager.load_keeper_configuration('data/keeper_config.json')
            
            if keeper_config and keeper_config.keepers:
                kept_player_names = set(keeper_config.get_kept_players())
                available_players = [p for p in player_universe if p.name not in kept_player_names]
                
                removed_count = len(player_universe) - len(available_players)
                print(f"🏆 Removed {removed_count} kept players from draft pool")
                
                return available_players
            else:
                print("📝 No keepers found - using full player pool")
                return player_universe
                
        except Exception as e:
            print(f"⚠️  Could not load keepers: {e}")
            return player_universe
    
    def _load_keeper_config(self):
        """Load keeper configuration for opponent simulation"""
        try:
            from .keeper_system import KeeperManager
            
            keeper_manager = KeeperManager()
            return keeper_manager.load_keeper_configuration('data/keeper_config.json')
        except Exception as e:
            print(f"⚠️  Could not load keeper config: {e}")
            return None
    
    def _calculate_keeper_picks(self) -> dict:
        """Calculate which overall pick numbers are used by keepers"""
        if not self.keeper_config:
            return {}
        
        keeper_picks = {}
        for keeper in self.keeper_config.keepers:
            # Calculate overall pick number for this keeper
            round_num = keeper.keeper_round
            team_position = keeper.draft_position
            
            if round_num % 2 == 1:  # Odd rounds
                overall_pick = (round_num - 1) * self.config.num_teams + team_position
            else:  # Even rounds (snake)
                overall_pick = round_num * self.config.num_teams - (team_position - 1)
            
            keeper_picks[overall_pick] = keeper.player_name
        
        return keeper_picks
    
    def _calculate_pick_number(self, round_number: int) -> int:
        """Calculate your pick number in this round"""
        if round_number % 2 == 1:  # Odd rounds
            return (round_number - 1) * self.config.num_teams + self.config.draft_position
        else:  # Even rounds (snake)
            return round_number * self.config.num_teams - (self.config.draft_position - 1)
    
    def _evaluate_strategy_with_monte_carlo(self, strategy: DraftStrategy) -> float:
        """Evaluate strategy performance using Monte Carlo simulation (unbiased)"""
        try:
            from .enhanced_monte_carlo import EnhancedMonteCarloSimulator, OutcomeAwareStrategy
            from .monte_carlo_viewer import MonteCarloResultsViewer
            
            # Create strategy that follows the tree search plan
            class TreeSearchStrategy(OutcomeAwareStrategy):
                def __init__(self, position_targets):
                    super().__init__(risk_tolerance=0.5)  # Balanced risk tolerance
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
        """Evaluate roster based on PURE STARTING LINEUP PROJECTION TOTAL"""
        if not roster.players:
            return 0.0
        
        # Get optimal starting lineup (this is what matters for weekly scoring)
        starting_lineup = self._get_optimal_starting_lineup_for_evaluation(roster.players)
        
        # PURE PROJECTION TOTAL - no bonuses, just raw weekly scoring power
        total_projection = sum(player.projection for player in starting_lineup)
        
        # Only apply penalties for incomplete lineups (can't field a full team)
        position_counts = defaultdict(int)
        for player in starting_lineup:
            pos = player.position.upper()
            if pos == 'DST':
                pos = 'DEF'  # Normalize
            position_counts[pos] += 1
        
        # Heavy penalties ONLY for missing required positions (can't start incomplete lineup)
        completeness_penalty = 0
        
        if self.config.qb_slots > 0 and position_counts.get('QB', 0) < self.config.qb_slots:
            completeness_penalty -= 1000 * (self.config.qb_slots - position_counts.get('QB', 0))  # Must have QB
        
        if self.config.rb_slots > 0 and position_counts.get('RB', 0) < self.config.rb_slots:
            completeness_penalty -= 1000 * (self.config.rb_slots - position_counts.get('RB', 0))  # Must have RBs
        
        if self.config.wr_slots > 0 and position_counts.get('WR', 0) < self.config.wr_slots:
            completeness_penalty -= 1000 * (self.config.wr_slots - position_counts.get('WR', 0))  # Must have WRs
        
        if self.config.te_slots > 0 and position_counts.get('TE', 0) < self.config.te_slots:
            completeness_penalty -= 1000 * (self.config.te_slots - position_counts.get('TE', 0))  # Must have TE
        
        if self.config.k_slots > 0 and position_counts.get('K', 0) < self.config.k_slots:
            completeness_penalty -= 1000 * (self.config.k_slots - position_counts.get('K', 0))  # Must have K
        
        def_count = position_counts.get('DEF', 0)
        if self.config.def_slots > 0 and def_count < self.config.def_slots:
            completeness_penalty -= 1000 * (self.config.def_slots - def_count)  # Must have DEF
        
        # Return total projected points minus any completeness penalties
        return total_projection + completeness_penalty
    
    def _get_optimal_starting_lineup_for_evaluation(self, roster_players: List[Player]) -> List[Player]:
        """Get optimal starting lineup based on league configuration"""
        if not roster_players:
            return []
        
        # Sort players by position and projection
        by_position = {}
        for player in roster_players:
            pos = player.position.upper()
            # Handle DST/DEF naming
            if pos == 'DST':
                pos = 'DEF'
            if pos not in by_position:
                by_position[pos] = []
            by_position[pos].append(player)
        
        # Sort each position by projection (highest first)
        for pos in by_position:
            by_position[pos].sort(key=lambda p: p.projection, reverse=True)
        
        starting_lineup = []
        
        # Fill starting positions based on league config
        # QB slots
        if 'QB' in by_position and self.config.qb_slots > 0:
            starting_lineup.extend(by_position['QB'][:self.config.qb_slots])
        
        # RB slots
        if 'RB' in by_position and self.config.rb_slots > 0:
            starting_lineup.extend(by_position['RB'][:self.config.rb_slots])
        
        # WR slots (this will now read from config!)
        if 'WR' in by_position and self.config.wr_slots > 0:
            starting_lineup.extend(by_position['WR'][:self.config.wr_slots])
        
        # TE slots
        if 'TE' in by_position and self.config.te_slots > 0:
            starting_lineup.extend(by_position['TE'][:self.config.te_slots])
        
        # FLEX slots (best remaining RB/WR/TE)
        for _ in range(self.config.flex_slots):
            flex_candidates = []
            if 'RB' in by_position and len(by_position['RB']) > self.config.rb_slots:
                flex_candidates.extend(by_position['RB'][self.config.rb_slots:])  # Remaining RBs
            if 'WR' in by_position and len(by_position['WR']) > self.config.wr_slots:
                flex_candidates.extend(by_position['WR'][self.config.wr_slots:])  # Remaining WRs
            if 'TE' in by_position and len(by_position['TE']) > self.config.te_slots:
                flex_candidates.extend(by_position['TE'][self.config.te_slots:])  # Remaining TEs
            
            # Remove already selected flex players
            flex_candidates = [p for p in flex_candidates if p not in starting_lineup]
            
            if flex_candidates:
                best_flex = max(flex_candidates, key=lambda p: p.projection)
                starting_lineup.append(best_flex)
        
        # SUPERFLEX slots (if any)
        for _ in range(self.config.superflex_slots):
            superflex_candidates = []
            if 'QB' in by_position:
                superflex_candidates.extend([p for p in by_position['QB'] if p not in starting_lineup])
            if 'RB' in by_position:
                superflex_candidates.extend([p for p in by_position['RB'] if p not in starting_lineup])
            if 'WR' in by_position:
                superflex_candidates.extend([p for p in by_position['WR'] if p not in starting_lineup])
            if 'TE' in by_position:
                superflex_candidates.extend([p for p in by_position['TE'] if p not in starting_lineup])
            
            if superflex_candidates:
                best_superflex = max(superflex_candidates, key=lambda p: p.projection)
                starting_lineup.append(best_superflex)
        
        # K slots
        if 'K' in by_position and self.config.k_slots > 0:
            starting_lineup.extend(by_position['K'][:self.config.k_slots])
        
        # DEF slots
        if 'DEF' in by_position and self.config.def_slots > 0:
            starting_lineup.extend(by_position['DEF'][:self.config.def_slots])
        
        return starting_lineup
    
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
