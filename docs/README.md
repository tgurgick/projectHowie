# Fantasy Football Database

A comprehensive fantasy football database system with advanced analytics, route running data, and scheme-specific coverage analysis.

## 🚀 Features

- **Complete Player Database**: 2018-2024 player rosters and game statistics
- **Multi-Scoring Support**: PPR, Half-PPR, and Standard scoring databases
- **Advanced Analytics**: EPA, CPOE, snap share, and other advanced metrics
- **Fantasy Market Data**: ADP and ECR from FantasyPros (2021-2024)
- **Route Running Analysis**: PFF route participation and efficiency metrics
- **Scheme Splits Analysis**: Man vs Zone coverage performance (2018-2024)
- **Incremental Loading**: Efficient data updates without duplicates
- **Comprehensive Testing**: Automated verification and data quality checks
- **🤖 Enhanced Chat System**: AI-powered multi-agent chat interface with GPT-4o

## 📊 Data Coverage

### Core Statistics (2018-2024)
- **Players**: 24,522+ player records
- **Games**: 1,995+ games across all seasons
- **Player Game Stats**: 58,000+ weekly performance records
- **Advanced Stats**: EPA, CPOE, snap share for all positions
- **Market Data**: 3,957+ ADP/ECR records (2021-2024)

### Route Running Data (2018-2024)
- **Route Participation**: 3,558+ player-season records
- **Route Efficiency**: Yards per route run, route grades
- **Route Types**: Various route patterns and success rates
- **BYOD Requirement**: Requires PFF+ subscription for CSV data

### Scheme Splits Analysis (2018-2024)
- **Man Coverage**: 3,558+ player-season records with man coverage metrics
- **Zone Coverage**: Comprehensive zone coverage performance data
- **Coverage Specialists**: Players who excel vs specific coverage types
- **Route Grades**: PFF grades for route running vs man/zone
- **Contested Catch Rates**: Performance on contested targets by coverage type

## 🛠️ Scripts

### Core Database Building
- `scripts/build_fantasy_db.py` - Main ETL script for core statistics
- `scripts/build_all_scoring.py` - Build all three scoring databases
- `scripts/build_advanced_stats.py` - Advanced analytics (EPA, CPOE, snap share)
- `scripts/build_fantasy_market.py` - Market data (ECR, ADP)

### Route Running & Scheme Analysis
- `scripts/build_pff_route_data_batch.py` - Import PFF route running data
- `scripts/build_pff_scheme_data.py` - Import PFF scheme splits (Man vs Zone)
- `scripts/build_fantasypros_adp.py` - Scrape FantasyPros ADP data
- `scripts/build_historical_adp.py` - Batch load historical ADP data

### Testing & Verification
- `tests/test_all_databases.py` - Comprehensive database verification
- `tests/database_summary.py` - Quick database overview
- `tests/verify_all_databases.py` - Master verification script
- `tests/show_advanced_stats.py` - Display advanced stats samples
- `tests/show_route_data.py` - Display route running data
- `tests/show_scheme_data.py` - Display scheme splits analysis

## 📁 Project Structure

```
projectHowie/
├── fantasy-etl/          # Data processing and ETL pipeline
│   ├── scripts/         # Database building scripts
│   ├── tests/          # Testing and verification
│   ├── data/           # Database files and CSV data
│   │   ├── pff_csv/    # PFF CSV files (BYOD)
│   │   ├── fantasy_ppr.db
│   │   ├── fantasy_halfppr.db
│   │   └── fantasy_standard.db
│   └── requirements.txt # ETL dependencies
├── chat_system/         # AI-powered chat interface
│   ├── enhanced_agents.py  # Main chat system with GPT-4o
│   ├── test_demo.py        # Demo without API key
│   ├── requirements.txt    # Chat system dependencies
│   └── README.md          # Chat system documentation
└── docs/                # Project documentation
    ├── README.md        # Main documentation
    ├── CHAT_SYSTEM_DESIGN.md
    ├── AGENT_POWER_SYSTEM.md
    ├── ROADMAP.md
    └── ...
```

## 🚀 Quick Start

### 1. Data Processing (ETL)
```bash
git clone <repository>
cd fantasy-etl
pip install -r requirements.txt

# Build PPR database (2018-2024)
python3 scripts/build_fantasy_db.py --db-url sqlite:///data/fantasy_ppr.db

# Build all scoring types
python3 scripts/build_all_scoring.py

# Add advanced stats (EPA, CPOE, snap share)
python3 scripts/build_advanced_stats.py --db-url sqlite:///data/fantasy_ppr.db
```

### 2. AI Chat System
```bash
cd ../chat_system
pip install -r requirements.txt

# Set up API key
echo "OPENAI_API_KEY=your-api-key-here" > .env

# View demo (no API key required)
python test_demo.py

# Start interactive chat
python enhanced_agents.py chat

# Ask a single question
python enhanced_agents.py ask "Tell me about Justin Jefferson"
```

# Add market data (ADP, ECR)
python3 scripts/build_fantasy_market.py --db-url sqlite:///data/fantasy_ppr.db
```

### 4. Add Route Running Data (BYOD)
```bash
# Place PFF CSV files in data/pff_csv/
# Then import route running data
python3 scripts/build_pff_route_data_batch.py --db-url sqlite:///data/fantasy_ppr.db

# Import scheme splits data
python3 scripts/build_pff_scheme_data.py --db-url sqlite:///data/fantasy_ppr.db
```

### 5. Verify Data
```bash
# Run comprehensive verification
python3 tests/verify_all_databases.py

# Show sample data
python3 tests/show_advanced_stats.py --db-url sqlite:///data/fantasy_ppr.db
python3 tests/show_route_data.py --db-url sqlite:///data/fantasy_ppr.db
python3 tests/show_scheme_data.py --db-url sqlite:///data/fantasy_ppr.db
```

## 📈 Data Sources

- **NFL Statistics**: `nfl_data_py` package
- **Player IDs**: DynastyProcess API
- **ADP/ECR**: FantasyPros (web scraping)
- **Route Running**: PFF (BYOD - requires PFF+ subscription)
- **Scheme Splits**: PFF (BYOD - requires PFF+ subscription)

## 🔍 Key Insights

### Scheme Splits Analysis
The database now includes comprehensive Man vs Zone coverage analysis:

- **Man Coverage Specialists**: Players who excel against man coverage (e.g., CeeDee Lamb: 4.44 YPRR vs man)
- **Zone Coverage Specialists**: Players who thrive against zone coverage (e.g., Tyreek Hill: 3.84 YPRR vs zone)
- **Coverage Efficiency Differences**: YPRR and route grade differences between man/zone
- **Contested Catch Performance**: Success rates on contested targets by coverage type

### Route Running Metrics
- **Route Participation**: Percentage of plays running routes
- **Yards Per Route Run**: Efficiency metric for route running
- **Route Grades**: PFF grades for route running quality
- **Route Types**: Various route patterns and success rates

## 📋 Data Requirements

### PFF Data (BYOD - Bring Your Own Data)
- **PFF+ Subscription**: Required for CSV data access
- **Route Running Data**: `receiving_YYYY_reg.csv` files (2018-2024)
- **Scheme Splits Data**: `receiving_scheme_YYYY.csv` files (2018-2024)
- **File Placement**: Place CSV files in `data/pff_csv/` directory

## 🤝 Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on contributing to this project.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🗺️ Roadmap

See [ROADMAP.md](docs/ROADMAP.md) for detailed development plans and completed features.

## 📚 Additional Documentation

- [Multi-Scoring Guide](docs/MULTI_SCORING_GUIDE.md) - Multi-scoring database options
- [PFF Route Stats Guide](docs/PFF_ROUTE_STATS_GUIDE.md) - PFF data requirements and usage
- [Contributing Guidelines](CONTRIBUTING.md) - How to contribute to the project
