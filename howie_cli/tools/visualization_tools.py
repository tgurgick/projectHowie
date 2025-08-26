"""
Visualization tools for creating charts and graphs
"""

from typing import List, Dict, Optional, Any, Union
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO
import base64
from pathlib import Path

from ..core.base_tool import BaseTool, ToolResult, ToolStatus, ToolParameter
from ..core.workspace import WorkspaceManager


class CreateChartTool(BaseTool):
    """Create various types of charts and visualizations"""
    
    def __init__(self):
        super().__init__()
        self.name = "create_chart"
        self.category = "visualization"
        self.description = "Create charts and visualizations for fantasy data"
        self.parameters = [
            ToolParameter(
                name="data",
                type="dataframe",
                description="Data to visualize",
                required=True
            ),
            ToolParameter(
                name="chart_type",
                type="string",
                description="Type of chart (bar, line, scatter, heatmap, etc.)",
                required=True,
                choices=["bar", "line", "scatter", "heatmap", "box", "violin"]
            ),
            ToolParameter(
                name="x",
                type="string",
                description="X-axis column name",
                required=False
            ),
            ToolParameter(
                name="y",
                type="string",
                description="Y-axis column name",
                required=False
            ),
            ToolParameter(
                name="title",
                type="string",
                description="Chart title",
                required=False
            ),
            ToolParameter(
                name="save_path",
                type="string",
                description="Path to save the chart",
                required=False
            )
        ]
        self.workspace = WorkspaceManager()
        
        # Set style
        sns.set_theme(style="whitegrid")
        plt.rcParams['figure.figsize'] = (12, 6)
    
    async def execute(self, data: pd.DataFrame, chart_type: str,
                     x: Optional[str] = None, y: Optional[str] = None,
                     title: Optional[str] = None, save_path: Optional[str] = None,
                     **kwargs) -> ToolResult:
        """Create visualization"""
        try:
            fig, ax = plt.subplots(figsize=(12, 6))
            
            if chart_type == "bar":
                if x and y:
                    data.plot(kind='bar', x=x, y=y, ax=ax)
                else:
                    data.plot(kind='bar', ax=ax)
            
            elif chart_type == "line":
                if x and y:
                    data.plot(kind='line', x=x, y=y, ax=ax, marker='o')
                else:
                    data.plot(kind='line', ax=ax, marker='o')
            
            elif chart_type == "scatter":
                if x and y:
                    ax.scatter(data[x], data[y])
                    ax.set_xlabel(x)
                    ax.set_ylabel(y)
                else:
                    return ToolResult(
                        status=ToolStatus.ERROR,
                        error="Scatter plot requires x and y columns"
                    )
            
            elif chart_type == "heatmap":
                sns.heatmap(data.corr() if data.select_dtypes(include='number').shape[1] > 1 else data,
                           annot=True, cmap='coolwarm', ax=ax)
            
            elif chart_type == "box":
                data.plot(kind='box', ax=ax)
            
            elif chart_type == "violin":
                if y:
                    sns.violinplot(data=data, y=y, ax=ax)
                else:
                    sns.violinplot(data=data, ax=ax)
            
            else:
                return ToolResult(
                    status=ToolStatus.ERROR,
                    error=f"Unsupported chart type: {chart_type}"
                )
            
            # Set title
            if title:
                ax.set_title(title, fontsize=16, fontweight='bold')
            
            # Adjust layout
            plt.tight_layout()
            
            # Save if path provided
            if save_path:
                fig.savefig(save_path, dpi=100, bbox_inches='tight')
                saved_path = save_path
            else:
                # Save to workspace
                chart_name = f"{chart_type}_chart_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.png"
                saved_path = self.workspace.session_path / "charts" / chart_name
                saved_path.parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(saved_path, dpi=100, bbox_inches='tight')
            
            plt.close(fig)
            
            return ToolResult(
                status=ToolStatus.SUCCESS,
                data={"chart_path": str(saved_path)},
                metadata={
                    "chart_type": chart_type,
                    "title": title,
                    "x_column": x,
                    "y_column": y,
                    "data_shape": data.shape
                }
            )
            
        except Exception as e:
            return ToolResult(
                status=ToolStatus.ERROR,
                error=f"Failed to create chart: {str(e)}"
            )


class PlayerComparisonChartTool(BaseTool):
    """Create player comparison visualizations"""
    
    def __init__(self):
        super().__init__()
        self.name = "player_comparison_chart"
        self.category = "visualization"
        self.description = "Create visual comparisons between players"
        self.parameters = [
            ToolParameter(
                name="players_data",
                type="dict",
                description="Dictionary of player names to stats",
                required=True
            ),
            ToolParameter(
                name="metrics",
                type="list",
                description="List of metrics to compare",
                required=True
            ),
            ToolParameter(
                name="chart_style",
                type="string",
                description="Style of comparison (radar, bar, grouped)",
                required=False,
                default="bar",
                choices=["radar", "bar", "grouped"]
            )
        ]
        self.workspace = WorkspaceManager()
    
    async def execute(self, players_data: Dict, metrics: List[str],
                     chart_style: str = "bar", **kwargs) -> ToolResult:
        """Create player comparison chart"""
        try:
            import numpy as np
            
            fig, ax = plt.subplots(figsize=(12, 8))
            
            if chart_style == "bar":
                # Create grouped bar chart
                df = pd.DataFrame(players_data).T
                df = df[metrics]
                
                x = np.arange(len(metrics))
                width = 0.8 / len(players_data)
                
                for i, (player, data) in enumerate(players_data.items()):
                    values = [data.get(m, 0) for m in metrics]
                    offset = width * i - (width * len(players_data) / 2)
                    ax.bar(x + offset, values, width, label=player)
                
                ax.set_xlabel('Metrics')
                ax.set_ylabel('Value')
                ax.set_title('Player Comparison')
                ax.set_xticks(x)
                ax.set_xticklabels(metrics, rotation=45, ha='right')
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            elif chart_style == "radar":
                # Create radar chart
                from math import pi
                
                # Number of metrics
                num_vars = len(metrics)
                
                # Compute angle for each axis
                angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
                angles += angles[:1]
                
                # Initialize the plot
                ax = plt.subplot(111, polar=True)
                
                # Draw one axis per variable and add labels
                plt.xticks(angles[:-1], metrics)
                
                # Plot data for each player
                for player, data in players_data.items():
                    values = [data.get(m, 0) for m in metrics]
                    values += values[:1]  # Complete the circle
                    ax.plot(angles, values, 'o-', linewidth=2, label=player)
                    ax.fill(angles, values, alpha=0.25)
                
                ax.set_title('Player Comparison - Radar Chart')
                ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
            
            else:  # grouped
                # Create grouped visualization
                df = pd.DataFrame(players_data).T
                df = df[metrics]
                df.plot(kind='bar', ax=ax)
                ax.set_title('Player Statistical Comparison')
                ax.set_xlabel('Players')
                ax.set_ylabel('Value')
                ax.legend(title='Metrics')
                plt.xticks(rotation=45)
            
            plt.tight_layout()
            
            # Save chart
            chart_name = f"player_comparison_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.png"
            saved_path = self.workspace.session_path / "charts" / chart_name
            saved_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(saved_path, dpi=100, bbox_inches='tight')
            plt.close(fig)
            
            return ToolResult(
                status=ToolStatus.SUCCESS,
                data={"chart_path": str(saved_path)},
                metadata={
                    "players": list(players_data.keys()),
                    "metrics": metrics,
                    "chart_style": chart_style
                }
            )
            
        except Exception as e:
            return ToolResult(
                status=ToolStatus.ERROR,
                error=f"Failed to create comparison chart: {str(e)}"
            )


class SeasonTrendChartTool(BaseTool):
    """Create season trend visualizations"""
    
    def __init__(self):
        super().__init__()
        self.name = "season_trend_chart"
        self.category = "visualization"
        self.description = "Visualize player or team trends over the season"
        self.parameters = [
            ToolParameter(
                name="data",
                type="dataframe",
                description="Time series data with weeks/dates",
                required=True
            ),
            ToolParameter(
                name="player_column",
                type="string",
                description="Column containing player names",
                required=False
            ),
            ToolParameter(
                name="metric",
                type="string",
                description="Metric to plot over time",
                required=True
            ),
            ToolParameter(
                name="week_column",
                type="string",
                description="Column containing week/date info",
                required=False,
                default="week"
            )
        ]
        self.workspace = WorkspaceManager()
    
    async def execute(self, data: pd.DataFrame, metric: str,
                     player_column: Optional[str] = None,
                     week_column: str = "week", **kwargs) -> ToolResult:
        """Create trend chart"""
        try:
            fig, ax = plt.subplots(figsize=(14, 7))
            
            if player_column and player_column in data.columns:
                # Multiple players
                for player in data[player_column].unique():
                    player_data = data[data[player_column] == player]
                    ax.plot(player_data[week_column], player_data[metric],
                           marker='o', label=player, linewidth=2)
                
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                title = f"{metric} Trends by Player"
            else:
                # Single trend
                ax.plot(data[week_column], data[metric],
                       marker='o', linewidth=2, markersize=8)
                
                # Add trend line
                z = np.polyfit(data[week_column], data[metric], 1)
                p = np.poly1d(z)
                ax.plot(data[week_column], p(data[week_column]),
                       "--", alpha=0.5, label='Trend')
                
                title = f"{metric} Season Trend"
            
            ax.set_xlabel('Week')
            ax.set_ylabel(metric)
            ax.set_title(title, fontsize=16, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            plt.tight_layout()
            
            # Save chart
            chart_name = f"season_trend_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.png"
            saved_path = self.workspace.session_path / "charts" / chart_name
            saved_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(saved_path, dpi=100, bbox_inches='tight')
            plt.close(fig)
            
            return ToolResult(
                status=ToolStatus.SUCCESS,
                data={"chart_path": str(saved_path)},
                metadata={
                    "metric": metric,
                    "weeks": len(data[week_column].unique()),
                    "players": len(data[player_column].unique()) if player_column else 1
                }
            )
            
        except Exception as e:
            return ToolResult(
                status=ToolStatus.ERROR,
                error=f"Failed to create trend chart: {str(e)}"
            )


class ASCIIChartTool(BaseTool):
    """Create ASCII charts for terminal display"""
    
    def __init__(self):
        super().__init__()
        self.name = "ascii_chart"
        self.category = "visualization"
        self.description = "Create ASCII charts for terminal display"
        self.parameters = [
            ToolParameter(
                name="data",
                type="dict",
                description="Data to visualize",
                required=True
            ),
            ToolParameter(
                name="chart_type",
                type="string",
                description="Type of ASCII chart (bar, line)",
                required=False,
                default="bar",
                choices=["bar", "line"]
            ),
            ToolParameter(
                name="width",
                type="int",
                description="Chart width in characters",
                required=False,
                default=50
            )
        ]
    
    async def execute(self, data: Dict, chart_type: str = "bar",
                     width: int = 50, **kwargs) -> ToolResult:
        """Create ASCII chart"""
        try:
            output = StringIO()
            
            if chart_type == "bar":
                # Find max value for scaling
                max_val = max(data.values()) if data.values() else 1
                
                output.write("\n")
                for label, value in data.items():
                    # Calculate bar length
                    bar_length = int((value / max_val) * width)
                    bar = "█" * bar_length
                    
                    # Format output
                    output.write(f"{label:15s} |{bar} {value:.1f}\n")
                
            elif chart_type == "line":
                # Simple line chart
                values = list(data.values())
                max_val = max(values) if values else 1
                min_val = min(values) if values else 0
                height = 10
                
                # Create grid
                grid = [[' ' for _ in range(len(values))] for _ in range(height)]
                
                # Plot points
                for i, val in enumerate(values):
                    y = int((val - min_val) / (max_val - min_val) * (height - 1))
                    grid[height - 1 - y][i] = '●'
                
                # Draw grid
                for row in grid:
                    output.write(''.join(row) + '\n')
                
                # Add labels
                labels = list(data.keys())
                if len(labels) <= width:
                    output.write(' '.join(labels[:5]) + '...\n' if len(labels) > 5 else ' '.join(labels) + '\n')
            
            chart_str = output.getvalue()
            
            return ToolResult(
                status=ToolStatus.SUCCESS,
                data={"chart": chart_str},
                metadata={
                    "chart_type": chart_type,
                    "data_points": len(data),
                    "width": width
                }
            )
            
        except Exception as e:
            return ToolResult(
                status=ToolStatus.ERROR,
                error=f"Failed to create ASCII chart: {str(e)}"
            )