"""
AI Strategy Summarizer
Integrates with Claude to provide intelligent strategy summaries and insights
"""

import json
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from .strategy_tree_search import DraftStrategy, PositionTarget
from .models import LeagueConfig


@dataclass
class StrategySummaryRequest:
    """Request for AI strategy summary"""
    strategy: DraftStrategy
    focus_areas: List[str]  # e.g., ["early_rounds", "risk_analysis", "positional_balance"]
    context: Dict[str, Any]  # Additional context like league trends, user preferences


@dataclass
class StrategySummary:
    """AI-generated strategy summary"""
    executive_summary: str
    round_by_round_analysis: Dict[int, str]
    key_strengths: List[str]
    potential_risks: List[str]
    strategic_alternatives: List[str]
    draft_day_tips: List[str]
    confidence_assessment: str


class StrategyAIAnalyzer:
    """Generate AI-powered strategy summaries and analysis"""
    
    def __init__(self):
        self.analysis_cache = {}
    
    def generate_comprehensive_summary(self, strategy: DraftStrategy, context: Dict[str, Any] = None) -> StrategySummary:
        """Generate a comprehensive AI summary of the strategy"""
        
        # Prepare strategy data for analysis
        strategy_data = self._prepare_strategy_data(strategy)
        
        # Generate different types of analysis
        executive_summary = self._generate_executive_summary(strategy, context or {})
        round_analysis = self._generate_round_analysis(strategy)
        strengths = self._identify_strengths(strategy)
        risks = self._identify_risks(strategy)
        alternatives = self._suggest_alternatives(strategy)
        tips = self._generate_draft_tips(strategy)
        confidence = self._assess_confidence(strategy)
        
        return StrategySummary(
            executive_summary=executive_summary,
            round_by_round_analysis=round_analysis,
            key_strengths=strengths,
            potential_risks=risks,
            strategic_alternatives=alternatives,
            draft_day_tips=tips,
            confidence_assessment=confidence
        )
    
    def _prepare_strategy_data(self, strategy: DraftStrategy) -> Dict[str, Any]:
        """Prepare strategy data for AI analysis"""
        return {
            "league_setup": {
                "teams": strategy.league_config.num_teams,
                "scoring": strategy.league_config.scoring_type,
                "draft_position": strategy.league_config.draft_position,
                "roster_spots": {
                    "QB": strategy.league_config.qb_slots,
                    "RB": strategy.league_config.rb_slots,
                    "WR": strategy.league_config.wr_slots,
                    "TE": strategy.league_config.te_slots,
                    "FLEX": strategy.league_config.flex_slots
                }
            },
            "strategy_overview": {
                "expected_value": strategy.expected_value,
                "confidence": strategy.confidence_score,
                "summary": strategy.strategy_summary,
                "insights": strategy.key_insights
            },
            "position_sequence": [
                {
                    "round": target.round_number,
                    "position": target.position,
                    "player": target.target_player.name if target.target_player else None,
                    "reasoning": target.reasoning,
                    "confidence": target.confidence,
                    "alternatives": target.alternatives,
                    "scarcity": target.scarcity_urgency
                }
                for target in strategy.position_targets
            ],
            "risk_profile": strategy.risk_tolerance_profile,
            "contingencies": strategy.contingency_plans
        }
    
    def _generate_executive_summary(self, strategy: DraftStrategy, context: Dict[str, Any]) -> str:
        """Generate executive summary of the strategy"""
        
        # Analyze strategy type
        early_positions = [t.position for t in strategy.position_targets[:3]]
        strategy_type = self._classify_strategy_type(early_positions)
        
        # Draft position context
        draft_pos = strategy.league_config.draft_position
        position_context = self._get_position_context(draft_pos, strategy.league_config.num_teams)
        
        # Risk assessment
        avg_risk = sum(strategy.risk_tolerance_profile.values()) / len(strategy.risk_tolerance_profile)
        risk_level = "aggressive" if avg_risk > 0.6 else "conservative" if avg_risk < 0.4 else "balanced"
        
        summary = f"""**{strategy_type} Strategy Analysis**

Your optimal draft strategy from the #{draft_pos} position employs a {strategy_type.lower()} approach with {risk_level} risk tolerance. {position_context}

**Core Strategy:** {strategy.strategy_summary.split('.')[0]}

**Expected Outcome:** This strategy projects to deliver {strategy.expected_value:.1f} points of total value with {strategy.confidence_score:.1%} confidence. The approach balances immediate impact with long-term roster construction.

**Key Philosophy:** {strategy.key_insights[0] if strategy.key_insights else 'Balanced positional approach targeting best available value'}"""
        
        return summary
    
    def _generate_round_analysis(self, strategy: DraftStrategy) -> Dict[int, str]:
        """Generate round-by-round strategic analysis"""
        analysis = {}
        
        for target in strategy.position_targets:
            round_num = target.round_number
            
            # Context for this round
            urgency_level = "critical" if target.scarcity_urgency > 0.7 else "moderate" if target.scarcity_urgency > 0.4 else "flexible"
            confidence_level = "high" if target.confidence > 0.7 else "medium" if target.confidence > 0.5 else "uncertain"
            
            # Generate analysis
            round_analysis = f"""**Round {round_num}: Target {target.position}**

*Reasoning:* {target.reasoning}

*Strategic Context:* This pick has {urgency_level} positional urgency and {confidence_level} confidence. """
            
            if target.alternatives:
                round_analysis += f"If your target {target.position} isn't available, pivot to {' or '.join(target.alternatives)}. "
            
            # Add specific advice based on scarcity
            if target.scarcity_urgency > 0.7:
                round_analysis += "⚠️ High scarcity - avoid waiting another round on this position."
            elif target.scarcity_urgency < 0.3:
                round_analysis += "✅ Position has depth - you can afford to wait if needed."
            
            analysis[round_num] = round_analysis
        
        return analysis
    
    def _identify_strengths(self, strategy: DraftStrategy) -> List[str]:
        """Identify key strengths of the strategy"""
        strengths = []
        
        # Analyze position balance
        position_counts = {}
        for target in strategy.position_targets[:8]:  # First 8 rounds
            pos = target.position
            position_counts[pos] = position_counts.get(pos, 0) + 1
        
        # Strength patterns
        if position_counts.get('RB', 0) >= 3:
            strengths.append("🏃 **RB Depth**: Excellent running back depth provides injury insurance and trade flexibility")
        
        if position_counts.get('WR', 0) >= 3:
            strengths.append("🎯 **WR Volume**: Strong wide receiver group maximizes target share and scoring consistency")
        
        if any(t.confidence > 0.8 for t in strategy.position_targets[:3]):
            strengths.append("💪 **Early Round Confidence**: High-confidence early picks provide solid foundation")
        
        # Risk management
        avg_risk = sum(strategy.risk_tolerance_profile.values()) / len(strategy.risk_tolerance_profile)
        if avg_risk < 0.4:
            strengths.append("🛡️ **Risk Management**: Conservative approach minimizes bust potential")
        elif avg_risk > 0.6:
            strengths.append("📈 **Upside Focus**: Aggressive stance maximizes ceiling outcomes")
        
        # Position scarcity awareness
        high_urgency_picks = sum(1 for t in strategy.position_targets if t.scarcity_urgency > 0.6)
        if high_urgency_picks >= 2:
            strengths.append("⏰ **Scarcity Awareness**: Strategy addresses positional scarcity at optimal timing")
        
        return strengths
    
    def _identify_risks(self, strategy: DraftStrategy) -> List[str]:
        """Identify potential risks and weaknesses"""
        risks = []
        
        # Analyze position concentration
        position_counts = {}
        for target in strategy.position_targets[:6]:  # First 6 rounds
            pos = target.position
            position_counts[pos] = position_counts.get(pos, 0) + 1
        
        # Risk patterns
        if position_counts.get('QB', 0) == 0:
            late_qb_round = next((t.round_number for t in strategy.position_targets if t.position == 'QB'), None)
            if late_qb_round and late_qb_round > 8:
                risks.append("⚠️ **Late QB Risk**: Waiting too long on quarterback could leave limited options")
        
        if position_counts.get('RB', 0) <= 1:
            risks.append("⚠️ **RB Scarcity**: Light on running backs - injury could be devastating")
        
        if position_counts.get('TE', 0) == 0:
            risks.append("⚠️ **TE Uncertainty**: No early tight end investment may force late-round dart throws")
        
        # Confidence risks
        low_confidence_picks = [t for t in strategy.position_targets[:5] if t.confidence < 0.6]
        if len(low_confidence_picks) >= 2:
            risks.append("📉 **Confidence Concerns**: Multiple early picks have lower confidence ratings")
        
        # Alternative position risks
        limited_alternatives = [t for t in strategy.position_targets if len(t.alternatives) <= 1]
        if len(limited_alternatives) >= 3:
            risks.append("🔒 **Flexibility Risk**: Limited alternative positions if targets are unavailable")
        
        return risks
    
    def _suggest_alternatives(self, strategy: DraftStrategy) -> List[str]:
        """Suggest strategic alternatives"""
        alternatives = []
        
        early_positions = [t.position for t in strategy.position_targets[:3]]
        
        # Alternative strategy suggestions
        if early_positions.count('RB') >= 2:
            alternatives.append("**Zero RB Pivot**: If elite RBs are gone, shift to WR-heavy approach for consistency")
        elif early_positions.count('WR') >= 2:
            alternatives.append("**RB Adjustment**: If unexpected RB value appears, consider pivoting for positional scarcity")
        
        if 'QB' not in early_positions[:4]:
            alternatives.append("**Early QB Escape**: If elite QBs fall, consider jumping on value even if not planned")
        
        if 'TE' not in early_positions[:6]:
            alternatives.append("**TE Premium Opportunity**: If elite tight ends available, consider position advantage")
        
        # Draft flow adjustments
        alternatives.append("**Flow-Based Adaptation**: Be ready to adjust if the draft takes unexpected turns")
        alternatives.append("**Value-Based Audible**: Don't force positions if significantly better value appears")
        
        return alternatives
    
    def _generate_draft_tips(self, strategy: DraftStrategy) -> List[str]:
        """Generate practical draft day tips"""
        tips = []
        
        # Early round tips
        early_targets = strategy.position_targets[:3]
        if any(t.scarcity_urgency > 0.7 for t in early_targets):
            tips.append("🎯 **Early Execution**: Don't overthink early picks - your high-urgency targets need to be secured")
        
        # Mid-round strategy
        mid_round_risk = sum(strategy.risk_tolerance_profile.get(i, 0.5) for i in range(4, 8)) / 4
        if mid_round_risk > 0.6:
            tips.append("📈 **Mid-Round Aggression**: Rounds 4-7 favor upside plays - target boom potential")
        elif mid_round_risk < 0.4:
            tips.append("🛡️ **Mid-Round Safety**: Focus on high-floor players in the middle rounds")
        
        # Contingency planning
        flexible_rounds = [t.round_number for t in strategy.position_targets if len(t.alternatives) >= 2]
        if flexible_rounds:
            tips.append(f"🔄 **Flexibility Windows**: Rounds {', '.join(map(str, flexible_rounds))} offer position flexibility")
        
        # League-specific advice
        if strategy.league_config.num_teams >= 14:
            tips.append("👥 **Deep League Depth**: In larger leagues, prioritize weekly starters over bench depth")
        elif strategy.league_config.num_teams <= 10:
            tips.append("⭐ **Shallow League Stars**: Smaller leagues reward elite talent - be aggressive on top players")
        
        # Draft position advice
        draft_pos = strategy.league_config.draft_position
        if draft_pos <= 3:
            tips.append("🏆 **Elite Pick Advantage**: Use your top pick to secure a foundational superstar")
        elif draft_pos >= 10:
            tips.append("⚡ **Turn Advantage**: Late pick gives you quick consecutive picks - plan 2-pick combinations")
        
        return tips
    
    def _assess_confidence(self, strategy: DraftStrategy) -> str:
        """Assess overall confidence in the strategy"""
        overall_confidence = strategy.confidence_score
        
        if overall_confidence > 0.8:
            assessment = "**High Confidence**: This strategy has strong analytical backing and clear execution path."
        elif overall_confidence > 0.6:
            assessment = "**Moderate Confidence**: Solid strategy with some areas requiring draft-day adaptation."
        else:
            assessment = "**Cautious Confidence**: Strategy provides good foundation but requires significant flexibility."
        
        # Add specific confidence factors
        high_conf_picks = sum(1 for t in strategy.position_targets if t.confidence > 0.7)
        low_conf_picks = sum(1 for t in strategy.position_targets if t.confidence < 0.5)
        
        assessment += f"\n\n**Confidence Breakdown**: {high_conf_picks} high-confidence picks, {low_conf_picks} uncertain picks."
        
        if low_conf_picks > high_conf_picks:
            assessment += " Focus extra attention on the uncertain rounds for optimal execution."
        
        return assessment
    
    def _classify_strategy_type(self, early_positions: List[str]) -> str:
        """Classify the overall strategy type"""
        if early_positions.count('RB') >= 2:
            return "Robust RB"
        elif early_positions.count('WR') >= 2:
            return "Zero RB"
        elif 'QB' in early_positions[:2]:
            return "Early QB"
        elif 'TE' in early_positions[:3]:
            return "TE Premium"
        else:
            return "Balanced Value"
    
    def _get_position_context(self, draft_position: int, num_teams: int) -> str:
        """Get context about the draft position"""
        if draft_position <= 3:
            return "Your early pick guarantees access to elite talent but creates longer waits between selections."
        elif draft_position >= num_teams - 2:
            return "Your late position provides quick consecutive picks but may miss top-tier players."
        else:
            return "Your middle position offers good balance between elite access and pick frequency."


def generate_strategy_summary_prompt(strategy: DraftStrategy) -> str:
    """Generate a prompt for Claude to analyze the strategy"""
    
    analyzer = StrategyAIAnalyzer()
    strategy_data = analyzer._prepare_strategy_data(strategy)
    
    prompt = f"""Please analyze this fantasy football draft strategy and provide insights:

**League Setup:**
- {strategy_data['league_setup']['teams']} teams, {strategy_data['league_setup']['scoring'].upper()} scoring
- Draft position #{strategy_data['league_setup']['draft_position']}
- Roster: {strategy_data['league_setup']['roster_spots']}

**Strategy Summary:**
{strategy_data['strategy_overview']['summary']}

**Round-by-Round Plan:**
"""
    
    for target_data in strategy_data['position_sequence']:
        prompt += f"Round {target_data['round']}: {target_data['position']}"
        if target_data['player']:
            prompt += f" (targeting {target_data['player']})"
        prompt += f" - {target_data['reasoning'][:100]}...\n"
    
    prompt += f"""
**Key Insights:**
{chr(10).join(strategy_data['strategy_overview']['insights'])}

Please provide:
1. An executive summary of this strategy's strengths and approach
2. Analysis of potential risks or weaknesses
3. 3-5 practical draft day tips for executing this strategy
4. Alternative approaches if the draft doesn't go as planned

Keep the analysis concise but insightful, focusing on actionable advice for draft day execution."""
    
    return prompt
