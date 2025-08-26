"""
Machine Learning projection tools for fantasy football predictions
"""

from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
from pathlib import Path

from ..core.base_tool import BaseTool, ToolResult, ToolStatus, ToolParameter
from ..core.workspace import WorkspaceManager


class PlayerProjectionTool(BaseTool):
    """Generate ML-based player projections"""
    
    def __init__(self):
        super().__init__()
        self.name = "player_projection"
        self.category = "ml_predictions"
        self.description = "Generate machine learning player projections"
        self.parameters = [
            ToolParameter(
                name="player_name",
                type="string",
                description="Player to project",
                required=True
            ),
            ToolParameter(
                name="weeks_ahead",
                type="int",
                description="Number of weeks to project",
                required=False,
                default=1
            ),
            ToolParameter(
                name="model_type",
                type="string",
                description="ML model type to use",
                required=False,
                default="ensemble",
                choices=["linear", "random_forest", "gradient_boost", "ensemble"]
            ),
            ToolParameter(
                name="include_factors",
                type="list",
                description="Factors to include in projection",
                required=False,
                default=["recent_form", "opponent", "weather", "injuries"]
            )
        ]
        self.workspace = WorkspaceManager()
        self.models = {}
    
    async def execute(self, player_name: str, weeks_ahead: int = 1,
                     model_type: str = "ensemble", 
                     include_factors: List[str] = None, **kwargs) -> ToolResult:
        """Generate player projection"""
        try:
            if not include_factors:
                include_factors = ["recent_form", "opponent", "weather", "injuries"]
            
            # Load or train model
            model = await self._get_or_train_model(player_name, model_type)
            
            # Prepare features
            features = await self._prepare_features(player_name, include_factors)
            
            # Generate projections
            projections = []
            confidence_intervals = []
            
            for week in range(1, weeks_ahead + 1):
                # Adjust features for future week
                week_features = self._adjust_features_for_week(features, week)
                
                if model_type == "ensemble":
                    # Use multiple models and average
                    predictions = []
                    for model_name in ["linear", "random_forest", "gradient_boost"]:
                        sub_model = await self._get_or_train_model(player_name, model_name)
                        pred = sub_model.predict(week_features.reshape(1, -1))[0]
                        predictions.append(pred)
                    
                    projection = np.mean(predictions)
                    std_dev = np.std(predictions)
                else:
                    projection = model.predict(week_features.reshape(1, -1))[0]
                    std_dev = self._estimate_std_dev(model, week_features)
                
                projections.append({
                    "week": week,
                    "projected_points": round(projection, 2),
                    "floor": round(projection - std_dev, 2),
                    "ceiling": round(projection + std_dev, 2),
                    "confidence": self._calculate_confidence(std_dev, projection)
                })
            
            # Generate analysis
            analysis = self._generate_projection_analysis(
                player_name, projections, include_factors
            )
            
            return ToolResult(
                status=ToolStatus.SUCCESS,
                data={
                    "player": player_name,
                    "projections": projections,
                    "analysis": analysis
                },
                metadata={
                    "model_type": model_type,
                    "factors_included": include_factors,
                    "weeks_projected": weeks_ahead
                }
            )
            
        except Exception as e:
            return ToolResult(
                status=ToolStatus.ERROR,
                error=f"Failed to generate projection: {str(e)}"
            )
    
    async def _get_or_train_model(self, player_name: str, model_type: str):
        """Get existing model or train new one"""
        model_key = f"{player_name}_{model_type}"
        
        if model_key in self.models:
            return self.models[model_key]
        
        # Train new model
        if model_type == "linear":
            model = LinearRegression()
        elif model_type == "random_forest":
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        elif model_type == "gradient_boost":
            model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        else:
            # Default to random forest for ensemble
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        
        # Generate synthetic training data (in production, use real historical data)
        X_train, y_train = self._generate_training_data(player_name)
        
        # Train model
        model.fit(X_train, y_train)
        
        # Cache model
        self.models[model_key] = model
        
        return model
    
    async def _prepare_features(self, player_name: str, factors: List[str]) -> np.ndarray:
        """Prepare feature vector for prediction"""
        features = []
        
        # Recent form (last 4 games average)
        if "recent_form" in factors:
            features.extend([15.5, 0.75, 0.85])  # Simulated: avg points, trend, consistency
        
        # Opponent strength
        if "opponent" in factors:
            features.extend([0.6, -2.5])  # Simulated: defense rating, points allowed diff
        
        # Weather impact
        if "weather" in factors:
            features.extend([0.0, 0.0])  # Simulated: wind impact, precipitation impact
        
        # Injury status
        if "injuries" in factors:
            features.append(1.0)  # Simulated: health score (0-1)
        
        # Additional features
        features.extend([
            0.65,  # Snap share
            0.22,  # Target share
            8.5,   # aDOT
            5.2    # YAC
        ])
        
        return np.array(features)
    
    def _adjust_features_for_week(self, features: np.ndarray, week: int) -> np.ndarray:
        """Adjust features for future week projection"""
        adjusted = features.copy()
        
        # Apply time decay to recent form
        decay_factor = 0.95 ** week
        adjusted[0] *= decay_factor  # Reduce recent form weight
        
        # Add uncertainty for future weeks
        noise = np.random.normal(0, 0.05 * week, size=adjusted.shape)
        adjusted += noise
        
        return adjusted
    
    def _generate_training_data(self, player_name: str) -> Tuple[np.ndarray, np.ndarray]:
        """Generate synthetic training data"""
        # In production, this would load real historical data
        n_samples = 100
        n_features = 10
        
        # Generate synthetic features
        X = np.random.randn(n_samples, n_features)
        
        # Generate synthetic targets with some pattern
        base_points = 15.0
        y = base_points + np.sum(X * np.random.randn(n_features), axis=1) + np.random.randn(n_samples) * 3
        
        return X, y
    
    def _estimate_std_dev(self, model, features: np.ndarray) -> float:
        """Estimate standard deviation for confidence interval"""
        # Simple estimation - in production use proper uncertainty quantification
        if hasattr(model, 'estimators_'):
            # For ensemble models, use prediction variance
            predictions = [est.predict(features.reshape(1, -1))[0] 
                         for est in model.estimators_[:10]]
            return np.std(predictions)
        else:
            # For simple models, use fixed percentage
            return 3.5  # Typical fantasy point std dev
    
    def _calculate_confidence(self, std_dev: float, projection: float) -> str:
        """Calculate confidence level"""
        cv = std_dev / projection if projection > 0 else 1
        
        if cv < 0.15:
            return "high"
        elif cv < 0.25:
            return "medium"
        else:
            return "low"
    
    def _generate_projection_analysis(self, player_name: str, 
                                     projections: List[Dict],
                                     factors: List[str]) -> Dict:
        """Generate analysis of projections"""
        avg_projection = np.mean([p["projected_points"] for p in projections])
        
        analysis = {
            "summary": f"{player_name} projected for {avg_projection:.1f} points average",
            "trend": "stable",
            "key_factors": [],
            "risks": [],
            "opportunities": []
        }
        
        # Analyze trend
        if len(projections) > 1:
            points = [p["projected_points"] for p in projections]
            if points[-1] > points[0]:
                analysis["trend"] = "improving"
            elif points[-1] < points[0]:
                analysis["trend"] = "declining"
        
        # Key factors
        if "opponent" in factors:
            analysis["key_factors"].append("Opponent matchup factored into projection")
        if "weather" in factors:
            analysis["key_factors"].append("Weather conditions considered")
        
        # Risks and opportunities
        if avg_projection < 10:
            analysis["risks"].append("Low projection - consider alternatives")
        if avg_projection > 20:
            analysis["opportunities"].append("High projection - strong start candidate")
        
        return analysis


class LineupOptimizerTool(BaseTool):
    """Optimize fantasy lineup using ML"""
    
    def __init__(self):
        super().__init__()
        self.name = "lineup_optimizer"
        self.category = "ml_predictions"
        self.description = "Optimize fantasy lineup using machine learning"
        self.parameters = [
            ToolParameter(
                name="available_players",
                type="list",
                description="List of available players",
                required=True
            ),
            ToolParameter(
                name="constraints",
                type="dict",
                description="Position constraints",
                required=False,
                default={"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 1}
            ),
            ToolParameter(
                name="optimization_method",
                type="string",
                description="Optimization method",
                required=False,
                default="maximize_points",
                choices=["maximize_points", "minimize_risk", "balanced"]
            ),
            ToolParameter(
                name="salary_cap",
                type="float",
                description="Salary cap for DFS",
                required=False
            )
        ]
    
    async def execute(self, available_players: List[str],
                     constraints: Dict = None,
                     optimization_method: str = "maximize_points",
                     salary_cap: Optional[float] = None, **kwargs) -> ToolResult:
        """Optimize lineup"""
        try:
            if not constraints:
                constraints = {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 1}
            
            # Get projections for all players
            player_projections = await self._get_all_projections(available_players)
            
            # Run optimization
            if optimization_method == "maximize_points":
                optimal_lineup = self._optimize_for_points(player_projections, constraints)
            elif optimization_method == "minimize_risk":
                optimal_lineup = self._optimize_for_safety(player_projections, constraints)
            else:  # balanced
                optimal_lineup = self._optimize_balanced(player_projections, constraints)
            
            # Apply salary cap if provided
            if salary_cap:
                optimal_lineup = self._apply_salary_constraint(optimal_lineup, salary_cap)
            
            # Calculate metrics
            total_projection = sum(p["projection"] for p in optimal_lineup.values())
            risk_score = self._calculate_risk_score(optimal_lineup)
            
            return ToolResult(
                status=ToolStatus.SUCCESS,
                data={
                    "optimal_lineup": optimal_lineup,
                    "total_projection": round(total_projection, 2),
                    "risk_score": risk_score,
                    "optimization_method": optimization_method
                },
                metadata={
                    "players_considered": len(available_players),
                    "constraints": constraints,
                    "has_salary_cap": salary_cap is not None
                }
            )
            
        except Exception as e:
            return ToolResult(
                status=ToolStatus.ERROR,
                error=f"Failed to optimize lineup: {str(e)}"
            )
    
    async def _get_all_projections(self, players: List[str]) -> Dict:
        """Get projections for all players"""
        projections = {}
        
        for player in players:
            # Simulated projections - in production, use real projections
            projections[player] = {
                "name": player,
                "position": self._get_player_position(player),
                "projection": np.random.uniform(5, 25),
                "floor": np.random.uniform(3, 15),
                "ceiling": np.random.uniform(15, 35),
                "salary": np.random.uniform(4000, 9000) if player else 5000,
                "risk": np.random.uniform(0.1, 0.9)
            }
        
        return projections
    
    def _get_player_position(self, player: str) -> str:
        """Get player position (simulated)"""
        # In production, look up from database
        positions = ["QB", "RB", "WR", "TE"]
        return np.random.choice(positions)
    
    def _optimize_for_points(self, projections: Dict, constraints: Dict) -> Dict:
        """Optimize for maximum projected points"""
        lineup = {}
        used_players = set()
        
        # Sort players by projection
        sorted_players = sorted(
            projections.items(),
            key=lambda x: x[1]["projection"],
            reverse=True
        )
        
        # Fill positions
        for position, count in constraints.items():
            if position == "FLEX":
                continue
            
            position_filled = 0
            for player_name, player_data in sorted_players:
                if (player_data["position"] == position and 
                    player_name not in used_players and
                    position_filled < count):
                    
                    slot = f"{position}{position_filled + 1}" if count > 1 else position
                    lineup[slot] = player_data
                    used_players.add(player_name)
                    position_filled += 1
        
        # Fill FLEX
        if "FLEX" in constraints:
            for player_name, player_data in sorted_players:
                if (player_data["position"] in ["RB", "WR", "TE"] and
                    player_name not in used_players):
                    
                    lineup["FLEX"] = player_data
                    break
        
        return lineup
    
    def _optimize_for_safety(self, projections: Dict, constraints: Dict) -> Dict:
        """Optimize for minimum risk"""
        lineup = {}
        used_players = set()
        
        # Sort players by risk (ascending) then projection (descending)
        sorted_players = sorted(
            projections.items(),
            key=lambda x: (x[1]["risk"], -x[1]["projection"])
        )
        
        # Similar filling logic as points optimization
        # but prioritizing low-risk players
        for position, count in constraints.items():
            if position == "FLEX":
                continue
            
            position_filled = 0
            for player_name, player_data in sorted_players:
                if (player_data["position"] == position and 
                    player_name not in used_players and
                    position_filled < count):
                    
                    slot = f"{position}{position_filled + 1}" if count > 1 else position
                    lineup[slot] = player_data
                    used_players.add(player_name)
                    position_filled += 1
        
        return lineup
    
    def _optimize_balanced(self, projections: Dict, constraints: Dict) -> Dict:
        """Balanced optimization between points and risk"""
        # Calculate composite score
        for player_data in projections.values():
            # Higher projection and lower risk = higher score
            player_data["composite_score"] = (
                player_data["projection"] * (1 - player_data["risk"] * 0.3)
            )
        
        # Sort by composite score
        sorted_players = sorted(
            projections.items(),
            key=lambda x: x[1]["composite_score"],
            reverse=True
        )
        
        # Fill lineup using composite score
        lineup = {}
        used_players = set()
        
        for position, count in constraints.items():
            if position == "FLEX":
                continue
            
            position_filled = 0
            for player_name, player_data in sorted_players:
                if (player_data["position"] == position and 
                    player_name not in used_players and
                    position_filled < count):
                    
                    slot = f"{position}{position_filled + 1}" if count > 1 else position
                    lineup[slot] = player_data
                    used_players.add(player_name)
                    position_filled += 1
        
        return lineup
    
    def _apply_salary_constraint(self, lineup: Dict, salary_cap: float) -> Dict:
        """Apply salary cap constraint to lineup"""
        total_salary = sum(p["salary"] for p in lineup.values())
        
        if total_salary <= salary_cap:
            return lineup
        
        # Need to optimize within salary cap
        # This is a simplified version - in production use proper optimization
        while total_salary > salary_cap:
            # Find highest salary player
            highest_salary_slot = max(lineup.keys(), key=lambda k: lineup[k]["salary"])
            
            # Replace with cheaper option (simplified)
            old_player = lineup[highest_salary_slot]
            old_player["salary"] *= 0.8  # Reduce salary for demo
            
            total_salary = sum(p["salary"] for p in lineup.values())
        
        return lineup
    
    def _calculate_risk_score(self, lineup: Dict) -> str:
        """Calculate overall risk score for lineup"""
        avg_risk = np.mean([p["risk"] for p in lineup.values()])
        
        if avg_risk < 0.3:
            return "low"
        elif avg_risk < 0.6:
            return "medium"
        else:
            return "high"