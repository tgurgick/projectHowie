"""
Distribution Display Tools
Adds variance/uncertainty context to player and position displays
"""

import sqlite3
from typing import Optional, Dict, Any
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from pathlib import Path

console = Console()


def get_database_path() -> str:
    """Get the database path"""
    script_dir = Path(__file__).parent.parent.parent
    db_path = script_dir / "data" / "fantasy_ppr.db"
    
    if db_path.exists():
        return str(db_path)
        
    # Fallback
    fallback_path = "data/fantasy_ppr.db"
    if Path(fallback_path).exists():
        return fallback_path
        
    raise FileNotFoundError("Fantasy database not found")


def get_player_distribution_info(player_name: str) -> Optional[Dict[str, Any]]:
    """Get distribution information for a specific player"""
    try:
        db_path = get_database_path()
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get player distribution profile
        cursor.execute("""
            SELECT player_name, position, mean_projection, coefficient_of_variation,
                   variance_bucket, injury_prob_healthy, injury_prob_minor, injury_prob_major,
                   distribution_type, confidence_score
            FROM player_distributions 
            WHERE LOWER(player_name) LIKE LOWER(?)
            LIMIT 1
        """, (f'%{player_name}%',))
        
        result = cursor.fetchone()
        conn.close()
        
        if not result:
            return None
            
        return {
            'player_name': result[0],
            'position': result[1],
            'mean_projection': result[2],
            'coefficient_of_variation': result[3],
            'variance_bucket': result[4],
            'injury_prob_healthy': result[5],
            'injury_prob_minor': result[6],
            'injury_prob_major': result[7],
            'distribution_type': result[8],
            'confidence_score': result[9]
        }
        
    except Exception as e:
        console.print(f"[dim]Note: Distribution data unavailable ({e})[/dim]")
        return None


def get_position_distribution_summary(position: str) -> Optional[Dict[str, Any]]:
    """Get distribution summary for a position"""
    try:
        db_path = get_database_path()
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get variance buckets for this position
        cursor.execute("""
            SELECT age_group, coefficient_of_variation, injury_prob_healthy, 
                   injury_prob_major, description
            FROM variance_buckets 
            WHERE position = ?
            ORDER BY coefficient_of_variation
        """, (position.upper(),))
        
        buckets = cursor.fetchall()
        
        # Get player count and variance range for this position
        cursor.execute("""
            SELECT COUNT(*), MIN(coefficient_of_variation), MAX(coefficient_of_variation),
                   AVG(coefficient_of_variation), MIN(mean_projection), MAX(mean_projection)
            FROM player_distributions 
            WHERE position = ?
        """, (position.lower(),))
        
        stats = cursor.fetchone()
        conn.close()
        
        if not buckets or not stats:
            return None
            
        return {
            'position': position,
            'buckets': [
                {
                    'age_group': bucket[0],
                    'cv': bucket[1],
                    'healthy_rate': bucket[2],
                    'major_injury_rate': bucket[3],
                    'description': bucket[4]
                }
                for bucket in buckets
            ],
            'player_count': stats[0],
            'cv_range': (stats[1], stats[2]),
            'avg_cv': stats[3],
            'projection_range': (stats[4], stats[5])
        }
        
    except Exception as e:
        console.print(f"[dim]Note: Position distribution data unavailable ({e})[/dim]")
        return None


def display_player_distribution_context(player_name: str):
    """Display distribution context for a specific player"""
    dist_info = get_player_distribution_info(player_name)
    
    if not dist_info:
        return
    
    console.print("\n[bold bright_green]📊 Variance & Uncertainty Analysis[/bold bright_green]")
    
    # Calculate key metrics
    cv = dist_info['coefficient_of_variation']
    mean_proj = dist_info['mean_projection']
    std_dev = cv * mean_proj
    
    # Variance category
    if cv <= 0.20:
        variance_label = "[green]LOW[/green] variance"
        variance_desc = "Consistent, predictable performance"
    elif cv <= 0.30:
        variance_label = "[yellow]MODERATE[/yellow] variance"
        variance_desc = "Some uncertainty, typical range"
    else:
        variance_label = "[red]HIGH[/red] variance"
        variance_desc = "Significant uncertainty, boom/bust potential"
    
    # Create variance panel
    variance_content = f"""[white]Projection: {mean_proj:.1f} ± {std_dev:.1f} points[/white]
[white]Coefficient of Variation: {cv:.1%}[/white]
[white]Variance Level: {variance_label}[/white]
[dim]{variance_desc}[/dim]

[white]Player Type: {dist_info['variance_bucket'].replace('_', ' ').title()}[/white]
[white]Injury Risk: {dist_info['injury_prob_major']:.0%} major, {dist_info['injury_prob_minor']:.0%} minor[/white]
[white]Healthy Seasons: {dist_info['injury_prob_healthy']:.0%}[/white]"""
    
    console.print(Panel(variance_content, title="🎲 Variance Profile", border_style="blue"))
    
    # Expected range
    p25 = mean_proj - 0.675 * std_dev  # Approximate 25th percentile
    p75 = mean_proj + 0.675 * std_dev  # Approximate 75th percentile
    
    console.print(f"[dim]Expected Range (50% of outcomes): {max(0, p25):.0f} - {p75:.0f} points[/dim]")
    console.print(f"[dim]Distribution: {dist_info['distribution_type'].replace('_', ' ').title()}[/dim]")


def display_position_distribution_summary(position: str):
    """Display distribution summary for a position"""
    dist_info = get_position_distribution_summary(position)
    
    if not dist_info:
        return
    
    console.print(f"\n[bold bright_green]📊 {position.upper()} Variance Patterns[/bold bright_green]")
    
    # Create variance table
    table = Table(title=f"{position.upper()} Player Types & Variance", show_header=True, header_style="bold bright_green")
    table.add_column("Player Type", style="bright_green", width=20)
    table.add_column("Variance", style="bright_green", width=12)
    table.add_column("Injury Risk", style="bright_green", width=15)
    table.add_column("Description", style="dim", width=30)
    
    for bucket in dist_info['buckets']:
        # Format variance level
        cv = bucket['cv']
        if cv <= 0.20:
            variance_str = f"[green]{cv:.1%}[/green] (Low)"
        elif cv <= 0.30:
            variance_str = f"[yellow]{cv:.1%}[/yellow] (Mod)"
        else:
            variance_str = f"[red]{cv:.1%}[/red] (High)"
        
        # Format injury risk
        major_risk = bucket['major_injury_rate']
        if major_risk <= 0.05:
            injury_str = f"[green]{major_risk:.0%}[/green] major"
        elif major_risk <= 0.08:
            injury_str = f"[yellow]{major_risk:.0%}[/yellow] major"
        else:
            injury_str = f"[red]{major_risk:.0%}[/red] major"
        
        age_group = bucket['age_group'].replace('_', ' ').title()
        table.add_row(age_group, variance_str, injury_str, bucket['description'][:30])
    
    console.print(table)
    
    # Summary stats
    cv_min, cv_max = dist_info['cv_range']
    proj_min, proj_max = dist_info['projection_range']
    
    console.print(f"\n[dim]📊 {dist_info['player_count']} {position.upper()}s analyzed[/dim]")
    console.print(f"[dim]   Variance range: {cv_min:.1%} - {cv_max:.1%}[/dim]")
    console.print(f"[dim]   Projection range: {proj_min:.0f} - {proj_max:.0f} points[/dim]")


def add_distribution_context_to_stats_table(table: Table, position: str, include_header: bool = True):
    """Add a distribution context note to existing stats tables"""
    if include_header:
        console.print(f"\n[dim]💡 Variance Note: {position.upper()} players have different uncertainty levels based on experience and position type.[/dim]")
        console.print(f"[dim]   Use '/player/PlayerName' for individual variance analysis or see position summary below.[/dim]")


# Convenience function for integration
def generate_variance_bar(player_name: str, fantasy_points: float, bar_width: int = 12) -> str:
    """Generate a visual variance bar for a player showing uncertainty spread"""
    try:
        dist_info = get_player_distribution_info(player_name)
        
        if not dist_info:
            return "─" * bar_width  # Default neutral bar
        
        cv = dist_info['coefficient_of_variation']
        mean_proj = fantasy_points or dist_info['mean_projection']
        std_dev = cv * mean_proj
        
        # Calculate range (±1 std dev covers ~68% of outcomes)
        low_range = max(0, mean_proj - std_dev)
        high_range = mean_proj + std_dev
        
        # Variance category and color
        if cv <= 0.20:
            color = "green"
            bar_char = "━"  # Solid bar for low variance
        elif cv <= 0.30:
            color = "yellow"
            bar_char = "─"  # Medium bar for moderate variance
        else:
            color = "red"
            bar_char = "┅"  # Dotted bar for high variance
        
        # Create visual bar with uncertainty indicators
        left_pad = bar_width // 4
        center_width = bar_width // 2
        right_pad = bar_width - left_pad - center_width
        
        # Show range visually: low ──██──── high
        bar = "┢" + "┅" * left_pad + bar_char * center_width + "┅" * right_pad + "┪"
        
        return f"[{color}]{bar}[/{color}]"
        
    except Exception:
        return "─" * bar_width  # Safe fallback


def get_variance_category_emoji(player_name: str) -> str:
    """Get emoji indicating variance category"""
    try:
        dist_info = get_player_distribution_info(player_name)
        if not dist_info:
            return "❓"
        
        cv = dist_info['coefficient_of_variation']
        if cv <= 0.20:
            return "🟢"  # Low variance
        elif cv <= 0.30:
            return "🟡"  # Moderate variance
        else:
            return "🔴"  # High variance
    except Exception:
        return "❓"


def show_distribution_context(player_name: str = None, position: str = None):
    """Show distribution context for player or position"""
    if player_name:
        display_player_distribution_context(player_name)
    elif position:
        display_position_distribution_summary(position)
