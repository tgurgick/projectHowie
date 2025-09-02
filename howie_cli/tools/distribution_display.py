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
def generate_variance_bar(player_name: str, fantasy_points: float, bar_width: int = 18, position_context: list = None) -> str:
    """Generate a visual variance bar showing actual outcome ranges relative to peers"""
    try:
        dist_info = get_player_distribution_info(player_name)
        
        if not dist_info:
            return "─" * bar_width  # Default neutral bar
        
        cv = dist_info['coefficient_of_variation']
        mean_proj = fantasy_points or dist_info['mean_projection']
        std_dev = cv * mean_proj
        
        # Calculate P25 and P75 ranges (~50% of outcomes) with realistic caps
        raw_low = mean_proj - 0.675 * std_dev
        raw_high = mean_proj + 0.675 * std_dev
        
        # Apply position-specific realistic floors and ceilings
        position = dist_info.get('position', '').lower()
        
        if position == 'qb':
            # QBs: Floor ~50 (backup level), Ceiling ~500 (historic peak)
            floor_cap = max(50, mean_proj * 0.25)
            ceiling_cap = min(500, mean_proj * 1.75)
        elif position == 'rb':
            # RBs: Floor ~20 (injury/committee), Ceiling ~450 (historic peak)
            floor_cap = max(20, mean_proj * 0.15)
            ceiling_cap = min(450, mean_proj * 1.85)
        elif position == 'wr':
            # WRs: Floor ~30 (low usage), Ceiling ~400 (historic peak)
            floor_cap = max(30, mean_proj * 0.20)
            ceiling_cap = min(400, mean_proj * 1.80)
        elif position == 'te':
            # TEs: Floor ~15 (minimal usage), Ceiling ~300 (elite season)
            floor_cap = max(15, mean_proj * 0.15)
            ceiling_cap = min(300, mean_proj * 1.75)
        elif position in ['k', 'kicker']:
            # Kickers: Floor ~80, Ceiling ~180 (more constrained)
            floor_cap = max(80, mean_proj * 0.60)
            ceiling_cap = min(180, mean_proj * 1.40)
        elif position in ['dst', 'def']:
            # Defense: Floor ~40, Ceiling ~200 (very constrained)
            floor_cap = max(40, mean_proj * 0.40)
            ceiling_cap = min(200, mean_proj * 1.60)
        else:
            # Default caps
            floor_cap = max(0, mean_proj * 0.20)
            ceiling_cap = mean_proj * 1.80
        
        # Apply caps to outcomes
        low_outcome = max(floor_cap, raw_low)
        high_outcome = min(ceiling_cap, raw_high)
        
        # If position context provided, scale relative to position
        if position_context:
            # Find min/max projections in the position
            all_projections = [p for p in position_context if p > 0]
            if all_projections:
                pos_min = min(all_projections) * 0.8  # Add some padding
                pos_max = max(all_projections) * 1.1
                
                # Scale values to bar width
                def scale_to_bar(value):
                    if pos_max <= pos_min:
                        return bar_width // 2
                    return int((value - pos_min) / (pos_max - pos_min) * bar_width)
                
                low_pos = max(0, scale_to_bar(low_outcome))
                mean_pos = max(0, scale_to_bar(mean_proj))
                high_pos = min(bar_width, scale_to_bar(high_outcome))
                
                # Ensure proper ordering
                low_pos = min(low_pos, mean_pos - 1) if mean_pos > 0 else 0
                high_pos = max(high_pos, mean_pos + 1) if mean_pos < bar_width else bar_width
                
            else:
                # Fallback to centered
                low_pos = bar_width // 3
                mean_pos = bar_width // 2  
                high_pos = 2 * bar_width // 3
        else:
            # Fallback to centered positioning
            low_pos = bar_width // 3
            mean_pos = bar_width // 2
            high_pos = 2 * bar_width // 3
        
        # Variance category and styling (adjusted for individual player variance)
        if cv <= 0.15:
            color = "green"
            range_char = "━"     # Solid for elite consistency (Mahomes, CMC, Kelce)
            mean_char = "█"      # Solid center
        elif cv <= 0.35:
            color = "yellow" 
            range_char = "▬"     # Medium for moderate variance (Josh Allen, veteran players)
            mean_char = "█"
        else:
            color = "red"
            range_char = "┅"     # Dotted for high variance (boom/bust, backups, rookies)
            mean_char = "█"
        
        # Build the visual bar
        bar_chars = [" "] * bar_width
        
        # Fill the range with appropriate character
        for i in range(low_pos, high_pos + 1):
            if i < bar_width:
                bar_chars[i] = range_char
        
        # Mark the mean projection
        if mean_pos < bar_width:
            bar_chars[mean_pos] = mean_char
        
        # Add range indicators
        if low_pos < bar_width:
            bar_chars[low_pos] = "┢"
        if high_pos < bar_width:
            bar_chars[high_pos] = "┪"
        
        bar_visual = "".join(bar_chars)
        
        # Add upside/downside numbers
        upside_text = f"{high_outcome:.0f}"
        downside_text = f"{low_outcome:.0f}"
        
        # Format: "Low: 250 ┢━━█━━┪ High: 350"
        return f"[dim]{downside_text}[/dim] [{color}]{bar_visual}[/{color}] [dim]{upside_text}[/dim]"
        
    except Exception as e:
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
