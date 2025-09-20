"""
Monte Carlo Draft Simulation Engine
Simulates thousands of draft scenarios with AI opponents to predict realistic outcomes
"""

import random
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict, Counter

from .models import Player, Roster, LeagueConfig
from .ai_opponents import AIDrafterPersonality, DrafterType
from .realistic_opponents import RealisticOpponentManager, generate_realistic_adp_for_players
from .draft_state import DraftState


@dataclass
class SimulationResult:
    """Results from a single draft simulation"""
    your_roster: Roster
    final_pick_number: int
    players_available_at_picks: Dict[int, List[Player]]  # What was available when you picked
    opponent_picks: List[Tuple[int, Player, str]]  # (pick_number, player, drafter_type)


@dataclass
class MonteCarloResults:
    """Aggregated results from Monte Carlo simulation"""
    total_simulations: int
    player_availability_rates: Dict[str, Dict[int, float]]  # player_name -> {round: availability_rate}
    average_roster_strength: float
    roster_outcomes: List[Roster]
    pick_recommendations: Dict[int, List[Tuple[Player, float]]]  # round -> [(player, selection_rate)]
    statistical_summary: Dict[str, any]


class MonteCarloSimulator:
    """Monte Carlo draft simulation engine"""
    
    def __init__(self, league_config: LeagueConfig, player_universe: List[Player], use_realistic_opponents: bool = True):
        self.config = league_config
        self.players = player_universe
        self.use_realistic_opponents = use_realistic_opponents
        
        if use_realistic_opponents:
            # Generate realistic ADP for players missing it
            self._enhance_player_adp()
            self.realistic_opponents = RealisticOpponentManager(league_config)
            self.ai_opponents = None  # Not used in realistic mode
        else:
            self.ai_opponents = self._initialize_ai_opponents()
            self.realistic_opponents = None
        
    def _initialize_ai_opponents(self) -> List[AIDrafterPersonality]:
        """Create AI opponents with different drafting personalities"""
        opponents = []
        
        # Create diverse opponent mix
        drafter_types = [
            DrafterType.VALUE_DRAFTER,
            DrafterType.NEED_BASED,
            DrafterType.SCARCE_HUNTER,
            DrafterType.TIER_BASED,
            DrafterType.ZERO_RB,
            DrafterType.ROBUST_RB,
            DrafterType.HERO_RB,
            DrafterType.QB_EARLY,
            DrafterType.QB_LATE,
            DrafterType.TE_PREMIUM,
            DrafterType.BEST_AVAILABLE
        ]
        
        # Assign personalities to other 11 teams
        for i in range(self.config.num_teams - 1):
            drafter_type = drafter_types[i % len(drafter_types)]
            opponents.append(AIDrafterPersonality(drafter_type, team_number=i+1))
            
        return opponents
    
    def _enhance_player_adp(self):
        """Generate realistic ADP for players missing it (keep existing real ADP)"""
        
        # Check how many players are missing ADP
        missing_adp = [p for p in self.players if p.adp >= 999]
        has_real_adp = [p for p in self.players if p.adp < 999]
        
        print(f"   Using real ADP data for {len(has_real_adp)} players ({len(has_real_adp)/len(self.players)*100:.1f}%)")
        
        if missing_adp:
            print(f"   Generating realistic ADP for {len(missing_adp)} missing players...")
            
            # Only generate ADP for missing players, keeping real ADP intact
            realistic_adp = self._generate_missing_adp_only(missing_adp)
            
            # Update only the missing ADP values
            for player in missing_adp:
                if player.name in realistic_adp:
                    player.adp = realistic_adp[player.name]
    
    def _generate_missing_adp_only(self, missing_players: List) -> Dict[str, float]:
        """Generate ADP only for players missing it, based on their projection rank among ALL players"""
        
        # Sort ALL players by projection to get overall ranking
        all_players_sorted = sorted(self.players, key=lambda p: p.projection, reverse=True)
        
        # Create rank lookup
        player_ranks = {player.name: rank for rank, player in enumerate(all_players_sorted, 1)}
        
        realistic_adp = {}
        
        # Position penalties
        position_penalties = {
            'QB': 25,   # QBs go ~25 picks later than projection rank
            'RB': -3,   # RBs go ~3 picks earlier  
            'WR': 0,    # WRs go about where projected
            'TE': 20,   # TEs go ~20 picks later
            'K': 120,   # Kickers go much later
            'DEF': 100, # Defense goes much later
            'DST': 100  # Defense goes much later
        }
        
        for player in missing_players:
            # Get overall rank among all players
            overall_rank = player_ranks.get(player.name, len(all_players_sorted))
            
            # Apply position penalty
            position = player.position.upper()
            penalty = position_penalties.get(position, 0)
            adjusted_adp = overall_rank + penalty
            
            # Add randomness
            final_adp = adjusted_adp + np.random.normal(0, 8)
            
            # Ensure reasonable bounds
            final_adp = max(1, min(400, final_adp))
            
            realistic_adp[player.name] = final_adp
        
        return realistic_adp
    
    def run_simulation(
        self, 
        num_simulations: int = 1000,
        rounds_to_simulate: int = 16,
        strategy_override: Optional[str] = None
    ) -> MonteCarloResults:
        """Run Monte Carlo simulation with multiple draft scenarios"""
        
        print(f"🎲 Running Monte Carlo simulation: {num_simulations} drafts...")
        
        simulation_results = []
        player_availability = defaultdict(lambda: defaultdict(int))
        pick_selections = defaultdict(lambda: defaultdict(int))
        
        for sim_num in range(num_simulations):
            if sim_num % 100 == 0:
                print(f"   Completed {sim_num}/{num_simulations} simulations...")
            
            # Set unique random seed for each simulation to ensure variance
            np.random.seed(sim_num * 12345)
            random.seed(sim_num * 54321)
            
            # Run single draft simulation
            result = self._simulate_single_draft(rounds_to_simulate, strategy_override)
            simulation_results.append(result)
            
            # Track player availability at each of your picks
            for pick_round, available_players in result.players_available_at_picks.items():
                for player in available_players:
                    player_availability[player.name][pick_round] += 1
            
            # Track what you actually picked
            for pick_round, player in enumerate(result.your_roster.players, 1):
                if pick_round <= rounds_to_simulate:
                    pick_selections[pick_round][player.name] += 1
        
        # Calculate availability rates
        availability_rates = {}
        for player_name, round_counts in player_availability.items():
            availability_rates[player_name] = {}
            for round_num, count in round_counts.items():
                availability_rates[player_name][round_num] = count / num_simulations
        
        # Calculate pick recommendations (most frequently selected)
        recommendations = {}
        for round_num, player_counts in pick_selections.items():
            round_recs = []
            total_picks = sum(player_counts.values())
            # Convert defaultdict to Counter for most_common method
            counter = Counter(player_counts)
            for player_name, count in counter.most_common(10):
                selection_rate = count / total_picks if total_picks > 0 else 0
                # Find player object
                player_obj = next((p for p in self.players if p.name == player_name), None)
                if player_obj:
                    round_recs.append((player_obj, selection_rate))
            recommendations[round_num] = round_recs
        
        # Calculate statistical summary
        roster_strengths = [self._calculate_roster_strength(r.your_roster) for r in simulation_results]
        
        statistical_summary = {
            'avg_roster_strength': np.mean(roster_strengths),
            'roster_strength_std': np.std(roster_strengths),
            'roster_strength_percentiles': {
                '10th': np.percentile(roster_strengths, 10),
                '25th': np.percentile(roster_strengths, 25),
                '50th': np.percentile(roster_strengths, 50),
                '75th': np.percentile(roster_strengths, 75),
                '90th': np.percentile(roster_strengths, 90)
            },
            'total_unique_outcomes': len(set(tuple(p.name for p in r.your_roster.players) for r in simulation_results))
        }
        
        return MonteCarloResults(
            total_simulations=num_simulations,
            player_availability_rates=availability_rates,
            average_roster_strength=statistical_summary['avg_roster_strength'],
            roster_outcomes=[r.your_roster for r in simulation_results],
            pick_recommendations=recommendations,
            statistical_summary=statistical_summary
        )
    
    def _simulate_single_draft(
        self, 
        rounds_to_simulate: int,
        strategy_override: Optional[str] = None
    ) -> SimulationResult:
        """Simulate a single complete draft"""
        
        # Initialize draft state
        draft_state = DraftState(self.config, self.players.copy())
        your_roster = Roster(self.config)
        players_available_at_picks = {}
        opponent_picks = []
        
        # Simulate draft round by round
        for round_num in range(1, rounds_to_simulate + 1):
            for pick_in_round in range(1, self.config.num_teams + 1):
                
                # Calculate absolute pick number
                if round_num % 2 == 1:  # Odd rounds (1, 3, 5...)
                    team_position = pick_in_round
                else:  # Even rounds (2, 4, 6...) - snake draft
                    team_position = self.config.num_teams - pick_in_round + 1
                
                absolute_pick = (round_num - 1) * self.config.num_teams + pick_in_round
                
                # Check if it's your turn
                is_your_pick = (team_position == self.config.draft_position)
                
                if is_your_pick:
                    # Your strategic pick
                    available_players = draft_state.get_available_players()
                    players_available_at_picks[round_num] = available_players.copy()
                    
                    # Use strategic selection (existing algorithm)
                    selected_player = self._make_strategic_pick(
                        available_players, your_roster, round_num, strategy_override
                    )
                    
                    your_roster = your_roster.add_player(selected_player)
                    draft_state.draft_player(selected_player)
                    
                else:
                    # Opponent pick (AI or realistic)
                    available_players = draft_state.get_available_players()
                    team_roster = draft_state.get_team_roster(team_position)
                    
                    if self.use_realistic_opponents:
                        # Use realistic opponent model
                        realistic_drafter = self.realistic_opponents.get_opponent_for_team(team_position)
                        if realistic_drafter:
                            selected_player = realistic_drafter.make_pick(
                                available_players, team_roster, round_num
                            )
                            drafter_name = f"{realistic_drafter.strategy_type}_{realistic_drafter.team_number}"
                        else:
                            # Fallback
                            selected_player = self._fallback_pick(available_players)
                            drafter_name = "Fallback"
                    else:
                        # Use AI personality system
                        opponent_index = (team_position - 1) if team_position < self.config.draft_position else (team_position - 2)
                        if opponent_index < len(self.ai_opponents):
                            ai_drafter = self.ai_opponents[opponent_index]
                            selected_player = ai_drafter.make_pick(
                                available_players, team_roster, round_num
                            )
                            drafter_name = ai_drafter.drafter_type.name
                        else:
                            selected_player = self._fallback_pick(available_players)
                            drafter_name = "Fallback"
                    
                    if selected_player:
                        draft_state.draft_player(selected_player)
                        opponent_picks.append((absolute_pick, selected_player, drafter_name))
        
        return SimulationResult(
            your_roster=your_roster,
            final_pick_number=absolute_pick,
            players_available_at_picks=players_available_at_picks,
            opponent_picks=opponent_picks
        )
    
    def _fallback_pick(self, available_players: List[Player]) -> Player:
        """Fallback pick method when no opponent is available"""
        if not available_players:
            return None
        
        # Simple ADP-based pick
        adp_sorted = sorted(available_players, key=lambda p: p.adp if p.adp < 999 else 999)
        return adp_sorted[0] if adp_sorted else available_players[0]
    
    def _make_strategic_pick(
        self,
        available_players: List[Player],
        current_roster: Roster,
        round_number: int,
        strategy_override: Optional[str] = None
    ) -> Player:
        """Make strategic pick using ADP range + VORP optimization"""
        
        # Calculate your current pick number
        if round_number % 2 == 1:  # Odd rounds
            pick_in_round = self.config.draft_position
        else:  # Even rounds (snake)
            pick_in_round = self.config.num_teams - self.config.draft_position + 1
        
        current_pick = (round_number - 1) * self.config.num_teams + pick_in_round
        
        # Define ADP range around your pick (±10 picks flexibility)
        adp_range = 10
        min_adp = max(1, current_pick - adp_range)
        max_adp = current_pick + adp_range
        
        # Filter to players within reasonable ADP range
        adp_candidates = []
        for player in available_players:
            player_adp = player.adp if player.adp < 999 else current_pick + 20  # Push unknown ADPs later
            if min_adp <= player_adp <= max_adp:
                adp_candidates.append(player)
        
        # If no one in range, expand the range
        if not adp_candidates:
            max_adp = current_pick + 25  # Expand range
            for player in available_players:
                player_adp = player.adp if player.adp < 999 else current_pick + 20
                if min_adp <= player_adp <= max_adp:
                    adp_candidates.append(player)
        
        # Final fallback to top 15 available
        if not adp_candidates:
            adp_candidates = available_players[:15]
        
        # Now use VORP and advanced metrics to pick the BEST within this ADP range
        from .recommendation_engine import PickRecommendationEngine
        rec_engine = PickRecommendationEngine(self.config, adp_candidates)
        recommendations = rec_engine.generate_round_recommendations(
            round_number, current_roster, []
        )
        
        if recommendations:
            return recommendations[0].player
        else:
            # Fallback to best projection in ADP range
            return max(adp_candidates, key=lambda p: p.projection)
    
    def _filter_realistic_strategic_candidates(self, available_players: List[Player], round_number: int) -> List[Player]:
        """Filter to realistic strategic picks (similar to AI logic but more aggressive)"""
        
        realistic = []
        
        for player in available_players:
            position = player.position.upper()
            
            # Realistic strategic drafting (slightly more aggressive than AI)
            if position == 'QB':
                if round_number == 1:
                    # Almost never take QB in Round 1 (only absolute elite)
                    if player.projection >= 350:  # Only Jayden Daniels level
                        realistic.append(player)
                elif round_number == 2:
                    # Sometimes take elite QBs in Round 2
                    if player.projection >= 315:
                        realistic.append(player)
                elif round_number <= 5:
                    # Most QB picks happen rounds 3-5
                    if player.projection >= 280:
                        realistic.append(player)
                else:
                    # Any QB later
                    realistic.append(player)
            
            elif position in ['RB', 'WR']:
                # RB/WR are flexible but still need to be good
                if round_number <= 3:
                    if player.projection >= 240:  # Elite only early
                        realistic.append(player)
                elif round_number <= 6:
                    if player.projection >= 170:  # Solid players mid-rounds
                        realistic.append(player)
                else:
                    realistic.append(player)  # Any RB/WR later
            
            elif position == 'TE':
                if round_number <= 3:
                    if player.projection >= 190:  # Only elite TEs early
                        realistic.append(player)
                elif round_number <= 6:
                    if player.projection >= 140:  # Solid TEs mid-rounds
                        realistic.append(player)
                else:
                    realistic.append(player)  # Any TE later
            
            elif position in ['K', 'DEF', 'DST']:
                if round_number >= 12:  # Never draft K/DEF early
                    realistic.append(player)
            
            else:
                realistic.append(player)  # Default: allow
        
        # Ensure we have enough candidates
        if len(realistic) < 10:
            # Add more high-projection players as fallback
            high_proj = sorted(available_players, key=lambda p: p.projection, reverse=True)
            for player in high_proj:
                if player not in realistic:
                    realistic.append(player)
                if len(realistic) >= 20:
                    break
        
        return realistic
    
    def _calculate_roster_strength(self, roster: Roster) -> float:
        """Calculate STARTING LINEUP strength score (not bench depth)"""
        if not roster.players:
            return 0.0
        
        # Get optimal starting lineup from roster
        starting_lineup = self._get_optimal_starting_lineup(roster.players)
        
        # Score starters only - bench depth doesn't matter for weekly scoring
        starter_projection = sum(player.projection for player in starting_lineup)
        
        # Bonus for complete starting lineup
        position_counts = defaultdict(int)
        for player in starting_lineup:
            position_counts[player.position] += 1
        
        # Penalty for missing essential starting positions
        balance_penalty = 0
        if position_counts.get('QB', 0) == 0:
            balance_penalty += 60  # No starting QB is catastrophic
        if position_counts.get('RB', 0) < 2:
            balance_penalty += 40 * (2 - position_counts.get('RB', 0))  # Need 2 starting RBs
        if position_counts.get('WR', 0) < 2:
            balance_penalty += 40 * (2 - position_counts.get('WR', 0))  # Need 2 starting WRs
        if position_counts.get('TE', 0) == 0:
            balance_penalty += 30  # No starting TE
        if position_counts.get('K', 0) == 0:
            balance_penalty += 15  # No kicker
        if position_counts.get('DEF', 0) == 0:
            balance_penalty += 15  # No defense
        
        return starter_projection - balance_penalty
    
    def _get_optimal_starting_lineup(self, roster_players: List[Player]) -> List[Player]:
        """Select optimal starting lineup from roster (QB:1, RB:2, WR:2, TE:1, FLEX:1, K:1, DEF:1)"""
        if not roster_players:
            return []
        
        # Sort players by position and projection
        by_position = {}
        for player in roster_players:
            pos = player.position.upper()
            if pos not in by_position:
                by_position[pos] = []
            by_position[pos].append(player)
        
        # Sort each position by projection (highest first)
        for pos in by_position:
            by_position[pos].sort(key=lambda p: p.projection, reverse=True)
        
        starting_lineup = []
        
        # Fill required starting positions
        # QB: 1
        if 'QB' in by_position and by_position['QB']:
            starting_lineup.append(by_position['QB'][0])
        
        # RB: 2 
        if 'RB' in by_position:
            starting_lineup.extend(by_position['RB'][:2])
        
        # WR: 2
        if 'WR' in by_position:
            starting_lineup.extend(by_position['WR'][:2])
        
        # TE: 1
        if 'TE' in by_position and by_position['TE']:
            starting_lineup.append(by_position['TE'][0])
        
        # FLEX: 1 (best remaining RB/WR/TE)
        flex_candidates = []
        if 'RB' in by_position and len(by_position['RB']) > 2:
            flex_candidates.extend(by_position['RB'][2:])  # RB3+
        if 'WR' in by_position and len(by_position['WR']) > 2:
            flex_candidates.extend(by_position['WR'][2:])  # WR3+
        if 'TE' in by_position and len(by_position['TE']) > 1:
            flex_candidates.extend(by_position['TE'][1:])  # TE2+
        
        if flex_candidates:
            best_flex = max(flex_candidates, key=lambda p: p.projection)
            starting_lineup.append(best_flex)
        
        # K: 1
        if 'K' in by_position and by_position['K']:
            starting_lineup.append(by_position['K'][0])
        
        # DEF: 1
        if 'DEF' in by_position and by_position['DEF']:
            starting_lineup.append(by_position['DEF'][0])
        
        return starting_lineup
    
    def generate_availability_report(self, results: MonteCarloResults, top_n: int = 50) -> str:
        """Generate human-readable availability report"""
        
        output = []
        output.append("🎲 MONTE CARLO SIMULATION RESULTS")
        output.append("=" * 60)
        output.append(f"Simulations Run: {results.total_simulations:,}")
        output.append(f"Average Roster Strength: {results.average_roster_strength:.1f}")
        output.append(f"Unique Roster Outcomes: {results.statistical_summary['total_unique_outcomes']:,}")
        output.append("")
        
        # Player availability by round
        for round_num in range(1, 7):  # First 6 rounds
            output.append(f"📍 ROUND {round_num} PLAYER AVAILABILITY:")
            if round_num == 1:
                output.append("(Players available when your Round 1 pick comes up)")
            elif round_num == 2:
                output.append("(Players available when your Round 2 pick comes up - varies by Round 1 outcome)")
            else:
                output.append(f"(Players available when your Round {round_num} pick comes up)")
            output.append("-" * 50)
            
            # Get players available in this round
            round_availability = []
            for player_name, round_rates in results.player_availability_rates.items():
                if round_num in round_rates:
                    rate = round_rates[round_num]
                    round_availability.append((player_name, rate))
            
            # Sort by availability rate (descending) 
            round_availability.sort(key=lambda x: x[1], reverse=True)
            
            # Filter to show meaningful availability (20%+ chance) and prioritize by ADP
            meaningful_availability = [(name, rate) for name, rate in round_availability if rate >= 0.20]
            
            # Get player objects to sort by ADP within meaningful availability
            players_with_rates = []
            for name, rate in meaningful_availability:
                player_obj = next((p for p in self.players if p.name == name), None)
                if player_obj:
                    players_with_rates.append((player_obj, rate))
            
            # Sort by ADP (best players first) within meaningful availability
            players_with_rates.sort(key=lambda x: x[0].adp if x[0].adp < 999 else 999)
            
            for i, (player_obj, rate) in enumerate(players_with_rates[:15], 1):
                player_name = player_obj.name
                percentage = rate * 100
                if percentage >= 80:
                    status = "🟢 Very Likely"
                elif percentage >= 50:
                    status = "🟡 Possible"
                elif percentage >= 20:
                    status = "🟠 Unlikely"
                else:
                    status = "🔴 Very Rare"
                
                adp_info = f"ADP:{player_obj.adp:5.1f}" if player_obj.adp < 999 else "ADP:  N/A"
                output.append(f"{i:2d}. {player_name:<20} {percentage:5.1f}% {status} ({adp_info})")
            
            output.append("")
        
        # Most common picks by round
        output.append("🎯 MOST COMMON PICKS BY ROUND:")
        output.append("-" * 50)
        
        for round_num, recommendations in results.pick_recommendations.items():
            if round_num <= 6:
                output.append(f"Round {round_num}:")
                for i, (player, selection_rate) in enumerate(recommendations[:5], 1):
                    percentage = selection_rate * 100
                    output.append(f"  {i}. {player.name:<20} {percentage:5.1f}%")
                output.append("")
        
        return "\n".join(output)
