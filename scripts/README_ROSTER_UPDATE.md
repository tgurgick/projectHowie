# NFL Roster Update System

This system allows you to keep your fantasy football database updated with the most current NFL roster information.

## Quick Start

### Using the CLI Command
```bash
# Update rosters using the built-in command
howie update-rosters
```

### Using the Script Directly
```bash
# Run the API-based updater
python scripts/update_rosters_api.py

# Run the web scraper (more comprehensive)
python scripts/update_rosters.py
```

## Features

### 1. Multiple Data Sources
- **ESPN API**: Fast, reliable roster data
- **Pro Football Reference**: Comprehensive player information
- **NFL.com**: Official team rosters
- **Manual CSV Import**: For custom roster data

### 2. Automatic Database Updates
- Updates all three databases (PPR, Half-PPR, Standard)
- Maintains player IDs and relationships
- Adds timestamps for tracking updates

### 3. Data Validation
- Validates team abbreviations
- Normalizes position codes
- Generates unique player IDs

## Data Sources

### ESPN API
The ESPN API provides current roster information for all NFL teams. This is the primary source for the roster updater.

**URL Format**: `https://site.api.espn.com/apis/site/v2/sports/football/nfl/teams/{team_abbrev}/roster`

### Pro Football Reference
Pro Football Reference provides comprehensive player statistics and roster information.

**URL Format**: `https://www.pro-football-reference.com/teams/{team_abbrev}/2025.htm`

### Manual CSV Import
You can create a CSV file with roster information and import it manually.

**CSV Format**:
```csv
name,position,team,height,weight,birthdate,college,experience
A.J. Brown,WR,PHI,6-1,226,1997-06-30,Ole Miss,5
DeVonta Smith,WR,PHI,6-0,170,1998-11-14,Alabama,3
```

## Database Schema

The roster updater creates/updates the `players` table with the following structure:

```sql
CREATE TABLE players (
    player_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    position TEXT,
    team TEXT,
    height TEXT,
    weight TEXT,
    birthdate TEXT,
    college TEXT,
    experience TEXT,
    updated_at TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
```

## Team Abbreviations

The system uses the following team abbreviations:

| Team | Abbreviation |
|------|--------------|
| Arizona Cardinals | ARI |
| Atlanta Falcons | ATL |
| Baltimore Ravens | BAL |
| Buffalo Bills | BUF |
| Carolina Panthers | CAR |
| Chicago Bears | CHI |
| Cincinnati Bengals | CIN |
| Cleveland Browns | CLE |
| Dallas Cowboys | DAL |
| Denver Broncos | DEN |
| Detroit Lions | DET |
| Green Bay Packers | GBP |
| Houston Texans | HOU |
| Indianapolis Colts | IND |
| Jacksonville Jaguars | JAC |
| Kansas City Chiefs | KCC |
| Las Vegas Raiders | LVR |
| Los Angeles Chargers | LAC |
| Los Angeles Rams | LAR |
| Miami Dolphins | MIA |
| Minnesota Vikings | MIN |
| New England Patriots | NEP |
| New Orleans Saints | NOS |
| New York Giants | NYG |
| New York Jets | NYJ |
| Philadelphia Eagles | PHI |
| Pittsburgh Steelers | PIT |
| San Francisco 49ers | SFO |
| Seattle Seahawks | SEA |
| Tampa Bay Buccaneers | TBB |
| Tennessee Titans | TEN |
| Washington Commanders | WAS |

## Position Codes

The system normalizes position codes to:

- **QB**: Quarterback
- **RB**: Running Back
- **WR**: Wide Receiver
- **TE**: Tight End
- **K**: Kicker
- **DEF**: Defense

## Usage Examples

### Update All Rosters
```bash
howie update-rosters
```

### Manual CSV Import
```python
from scripts.update_rosters_api import ManualRosterUpdater

updater = ManualRosterUpdater()
updater.load_from_csv("my_rosters.csv")
rosters = updater.get_rosters()
```

### Custom Team Addition
```python
from scripts.update_rosters_api import ManualRosterUpdater

updater = ManualRosterUpdater()
updater.add_team_roster("PHI", [
    {
        "name": "A.J. Brown",
        "position": "WR",
        "team": "PHI",
        "height": "6-1",
        "weight": "226",
        "birthdate": "1997-06-30",
        "college": "Ole Miss",
        "experience": "5"
    }
])
```

## Troubleshooting

### Common Issues

1. **API Rate Limiting**: The scripts include rate limiting to avoid being blocked
2. **Missing Teams**: Check that team abbreviations match the official list
3. **Database Permissions**: Ensure write permissions to the database files
4. **Network Issues**: Check internet connectivity for API calls

### Error Messages

- **"Database not found"**: Check that database files exist in the `data/` directory
- **"No roster data available"**: API calls failed, try manual CSV import
- **"Error parsing player row"**: Check CSV format matches expected structure

## Scheduling Updates

You can schedule regular roster updates using cron jobs:

```bash
# Update rosters daily at 6 AM
0 6 * * * cd /path/to/projectHowie && howie update-rosters

# Update rosters weekly on Sundays at 8 AM
0 8 * * 0 cd /path/to/projectHowie && howie update-rosters
```

## Contributing

To add new data sources or improve the roster updater:

1. Add new scraping methods to `NFLRosterScraper` class
2. Update the `main()` function to use new sources
3. Test with a small subset of teams first
4. Add error handling and logging

## Dependencies

The roster updater requires:

- `aiohttp`: For async HTTP requests
- `beautifulsoup4`: For HTML parsing
- `pandas`: For CSV processing
- `sqlite3`: For database operations

Install with:
```bash
pip install aiohttp beautifulsoup4 pandas
```
