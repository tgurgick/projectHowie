# 🏈 Fantasy Football Database

A comprehensive fantasy football database system with advanced analytics, multiple scoring formats, and historical data from 2018-2024.

## 📊 Features

### ✅ Core Data
- **Players**: 24,522 unique players with complete profiles
- **Games**: 1,942 games across 7 seasons (2018-2024)
- **Fantasy Points**: Calculated for PPR, Half-PPR, and Standard scoring
- **Player Stats**: 38,235 records per scoring type

### 📈 Advanced Analytics
- **Snap Share**: Offensive snap percentage (10,274 records across 2018-2024)
- **Target Share**: Percentage of team targets (31,109 records)
- **Air Yards**: Passing yards before catch (31,109 records)
- **aDOT**: Average depth of target (31,109 records)
- **YAC**: Yards after catch (31,109 records)
- **EPA/CPOE**: Expected points added and completion percentage over expected (16,968 records)
- **Route Running**: PFF route grades, YPRR, route participation (3,558 records across 2018-2024)

### 📊 Market Data
- **ECR Rankings**: Expert consensus rankings (3,691 records)
- **ADP Data**: FantasyPros ADP data (2,074 records across all formats)
  - PPR: 884 players with ADP rankings (2021-2024)
  - Half-PPR: 492 players with ADP rankings (2023-2024)
  - Standard: 698 players with ADP rankings (2022-2024)

### 🔄 Multiple Scoring Formats
- **PPR**: Point per reception scoring
- **Half-PPR**: 0.5 points per reception
- **Standard**: Traditional scoring (no PPR)

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip or conda
- **PFF+ Subscription** (for route running data - BYOD)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/fantasy-football-database.git
   cd fantasy-football-database
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Build the databases**
   ```bash
   # Build all scoring types (recommended)
   python3 scripts/build_all_scoring.py
   
   # Or build individual databases
   python3 scripts/build_fantasy_db_simple.py --start 2018 --end 2024 --scoring ppr
   python3 scripts/build_fantasy_db_simple.py --start 2018 --end 2024 --scoring half_ppr
   python3 scripts/build_fantasy_db_simple.py --start 2018 --end 2024 --scoring standard
   ```

4. **Add advanced stats and market data**
   ```bash
   python3 scripts/build_all_stats.py
   ```

5. **Add PFF route running data (BYOD)**
   ```bash
   # Download CSV files from PFF+ and place in data/pff_csv/
   python3 scripts/build_pff_route_data_batch.py
   ```

5. **Verify the installation**
   ```bash
   python3 tests/verify_all_databases.py
   ```

## 📁 Database Files

After building, you'll have three main database files in the `data/` directory:

- **`data/fantasy_ppr.db`** (9.1 MB) - PPR scoring database
- **`data/fantasy_halfppr.db`** (9.1 MB) - Half-PPR scoring database  
- **`data/fantasy_standard.db`** (9.1 MB) - Standard scoring database

Each database contains:
- `players` - Player information and metadata
- `games` - Game schedules and results
- `player_game_stats` - Fantasy points and basic stats
- `player_advanced_stats` - Advanced analytics (snap share, air yards, etc.)
- `fantasy_market` - ECR rankings and ADP data

## 🔧 Usage Examples

### Basic Query Examples

```python
import pandas as pd
from sqlalchemy import create_engine, text

# Connect to PPR database
engine = create_engine("sqlite:///data/fantasy_ppr.db", future=True)

# Top 10 players by fantasy points (2024)
query = """
SELECT p.name, p.position, p.team, 
       SUM(pgs.fantasy_points) as total_points,
       COUNT(*) as games_played
FROM players p
JOIN player_game_stats pgs ON p.player_id = pgs.player_id
JOIN games g ON pgs.game_id = g.game_id
WHERE g.season = 2024
GROUP BY p.player_id
HAVING games_played >= 10
ORDER BY total_points DESC
LIMIT 10
"""

df = pd.read_sql(text(query), engine)
print(df)
```

### Advanced Stats Query

```python
# Players with highest air yards (2024)
query = """
SELECT p.name, p.position, p.team,
       pas.air_yards, pas.target_share, pas.snap_share
FROM players p
JOIN player_advanced_stats pas ON p.player_id = pas.player_id
JOIN games g ON pas.game_id = g.game_id
WHERE g.season = 2024 
  AND pas.air_yards IS NOT NULL
ORDER BY pas.air_yards DESC
LIMIT 10
"""

df = pd.read_sql(text(query), engine)
print(df)
```

### Market Data Query

```python
# Top players by ECR rank
query = """
SELECT p.name, p.position, p.team, fm.ecr_rank
FROM players p
JOIN fantasy_market fm ON p.player_id = fm.player_id
WHERE fm.season = 2024
ORDER BY fm.ecr_rank ASC
LIMIT 10
"""

df = pd.read_sql(text(query), engine)
print(df)
```

## 📊 Available Scripts

### Core Building Scripts
- **`scripts/build_fantasy_db_simple.py`** - Build core database (players, games, stats)
- **`scripts/build_all_scoring.py`** - Build all three scoring databases
- **`scripts/build_advanced_stats.py`** - Add advanced analytics
- **`scripts/build_fantasy_market.py`** - Add market data (ECR)
- **`scripts/build_fantasypros_adp.py`** - Add FantasyPros ADP data (single year)
- **`scripts/build_historical_adp.py`** - Add historical ADP data (2021-2024)
- **`scripts/build_pff_route_data_batch.py`** - Import PFF route running data (2018-2024, BYOD)
- **`scripts/build_all_stats.py`** - Master script for all advanced features

### Verification & Testing
- **`tests/verify_all_databases.py`** - Comprehensive database verification
- **`tests/test_all_databases.py`** - Detailed testing suite
- **`tests/database_summary.py`** - Quick database overview
- **`tests/show_advanced_stats.py`** - Display available advanced stats
- **`tests/show_route_data.py`** - Display PFF route running data

### Analysis Scripts
- **`scripts/compare_scoring.py`** - Compare fantasy points across scoring types
- **`scripts/ext_compute_replacement_vor.py`** - Calculate Value Over Replacement

## 📈 Data Sources

### Primary Sources
- **NFL Data**: [nfl_data_py](https://github.com/cooperdff/nfl_data_py) - Official NFL statistics
- **DynastyProcess**: Player IDs and ECR rankings
- **PFR**: Pro Football Reference snap counts

### Data Access Requirements
- **Free Data**: NFL stats, ECR rankings, basic advanced stats
- **PFF+ Subscription**: Route running analytics (BYOD - Bring Your Own Data)
- **FantasyPros**: ADP data (web scraping, no subscription required)

### Current Integrations
- **FantasyPros**: ✅ ADP data and expert rankings (implemented)
- **DynastyProcess**: ✅ ECR rankings and player IDs (implemented)
- **nfl_data_py**: ✅ Official NFL statistics (implemented)
- **PFF**: ✅ Route running analytics and grades (BYOD - Bring Your Own Data via PFF+)

### Planned Integrations
- **Sleeper**: Real-time ADP and league data
- **NFL Next Gen Stats**: Player tracking data

## 🔄 Incremental Updates

The system supports incremental loading to avoid duplicates:

```bash
# Add new season data
python3 scripts/build_fantasy_db_simple.py --start 2025 --end 2025 --scoring ppr

# Update advanced stats for new season
python3 scripts/build_advanced_stats.py --start 2025 --end 2025 --db-url sqlite:///data/fantasy_ppr.db
```

## 📊 Database Schema

### Players Table
```sql
CREATE TABLE players (
    player_id TEXT PRIMARY KEY,
    name TEXT,
    position TEXT,
    team TEXT,
    gsis_id TEXT,
    -- additional metadata
);
```

### Games Table
```sql
CREATE TABLE games (
    game_id TEXT PRIMARY KEY,
    season INTEGER,
    week INTEGER,
    home_team TEXT,
    away_team TEXT,
    date DATE,
    -- game metadata
);
```

### Player Game Stats Table
```sql
CREATE TABLE player_game_stats (
    game_id TEXT,
    player_id TEXT,
    fantasy_points REAL,
    pass_yards REAL,
    rush_yards REAL,
    rec_yards REAL,
    -- additional stats
    PRIMARY KEY (game_id, player_id)
);
```

### Advanced Stats Table
```sql
CREATE TABLE player_advanced_stats (
    game_id TEXT,
    player_id TEXT,
    snap_share REAL,
    target_share REAL,
    air_yards REAL,
    aDOT REAL,
    yac REAL,
    epa_per_play REAL,
    cpoe REAL,
    -- additional advanced metrics
    PRIMARY KEY (game_id, player_id)
);
```

## 🛠️ Development

### Project Structure
```
fantasy-etl/
├── 📁 scripts/                  # Core building and analysis scripts
│   ├── build_fantasy_db.py          # Main ETL script
│   ├── build_fantasy_db_simple.py   # Simplified core builder
│   ├── build_advanced_stats.py      # Advanced analytics
│   ├── build_fantasy_market.py      # Market data
│   ├── build_all_scoring.py         # Multi-scoring orchestrator
│   ├── build_all_stats.py           # Master stats builder
│   ├── compare_scoring.py           # Scoring comparison
│   └── ext_compute_replacement_vor.py # VOR calculations
├── 🧪 tests/                    # Testing and verification scripts
│   ├── verify_all_databases.py      # Comprehensive verification
│   ├── test_all_databases.py        # Testing framework
│   ├── database_summary.py          # Quick overview
│   └── show_advanced_stats.py       # Stats display
├── 📄 docs/                     # Documentation
│   ├── README.md                    # This file
│   ├── ROADMAP.md                   # Development roadmap
│   ├── CONTRIBUTING.md              # Contribution guidelines
│   └── MULTI_SCORING_GUIDE.md       # Scoring guide
├── 🗄️ data/                     # Database files
│   ├── fantasy_ppr.db              # PPR scoring database
│   ├── fantasy_halfppr.db          # Half-PPR scoring database
│   └── fantasy_standard.db         # Standard scoring database
├── 📦 Project Files
│   ├── requirements.txt             # Dependencies
│   ├── .gitignore                  # Git ignore rules
│   └── LICENSE                     # MIT License
```

### Adding New Features

1. **Create new script** following naming convention
2. **Add error handling** and logging
3. **Implement incremental loading** to avoid duplicates
4. **Add verification** to test suite
5. **Update documentation** and examples

### Testing

```bash
# Run comprehensive tests
python3 tests/verify_all_databases.py

# Run specific test suite
python3 tests/test_all_databases.py

# Check database integrity
python3 tests/database_summary.py
```

## 📋 Requirements

### Core Dependencies
```
nfl_data_py>=0.3.0
pandas>=1.3.0
sqlalchemy>=2.0.0
numpy>=1.21.0
requests>=2.25.0
```

### Optional Dependencies
```
beautifulsoup4>=4.9.0  # For web scraping
plotly>=5.0.0         # For visualizations
scikit-learn>=1.0.0   # For predictive models
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) and [Development Roadmap](ROADMAP.md) for details.

### How to Contribute
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests and documentation
5. Submit a pull request

### Development Setup
```bash
# Clone and setup
git clone https://github.com/yourusername/fantasy-football-database.git
cd fantasy-football-database
pip install -r requirements.txt

# Run tests
python3 verify_all_databases.py
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **nfl_data_py**: Official NFL data access
- **DynastyProcess**: Player IDs and ECR data
- **Pro Football Reference**: Additional statistics
- **Fantasy Football Community**: Feedback and suggestions

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/fantasy-football-database/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/fantasy-football-database/discussions)
- **Documentation**: [Wiki](https://github.com/yourusername/fantasy-football-database/wiki)

## 📈 Roadmap

See [ROADMAP.md](ROADMAP.md) for detailed development plans including:
- Enhanced advanced stats (EPA, CPOE, route running)
- FantasyPros ADP integration
- Predictive models and analytics
- API development
- Real-time data integration

---

**🏈 Ready to dominate your fantasy football league with data-driven insights!**
