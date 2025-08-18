# PFF Route Running Statistics Guide

## ⚠️ **BYOD (Bring Your Own Data) Requirement**

**PFF route running data requires a PFF+ subscription** to access and download the CSV files. This integration is designed as a BYOD (Bring Your Own Data) solution.

### **PFF+ Subscription Required**
- **PFF+ Elite**: Access to advanced receiving statistics and route running data
- **CSV Export**: Download route data for seasons 2018-2024
- **Data Format**: Regular season receiving statistics with route metrics

### **What You Need**
1. **Active PFF+ subscription** (Elite tier recommended)
2. **Access to receiving statistics** in PFF dashboard
3. **Ability to export CSV files** from PFF platform

## 📁 **Data Directory**
Your PFF CSV files should be placed in: `data/pff_csv/`

## 🎯 **Key Route Running Statistics to Look For**

### **Primary Route Metrics**
1. **Routes Run** - Total number of routes run by a player
2. **Route Participation** - Percentage of team's routes run by the player
3. **Route Efficiency** - Success rate of routes (targets/receptions per route)

### **Advanced Route Metrics**
4. **Route Depth** - Average depth of routes run
5. **Route Separation** - Average separation from defender when targeted
6. **Route Cushion** - Average cushion from defender at snap
7. **Route Win Rate** - Percentage of routes where player creates separation

### **Target & Reception Data**
8. **Targets** - Number of times targeted on routes
9. **Receptions** - Number of catches made
10. **Target Share** - Percentage of team targets
11. **Catch Rate** - Receptions divided by targets

### **Route Types & Success**
12. **Route Types** - Breakdown by route type (slant, post, corner, etc.)
13. **Route Success Rate** - Success rate by route type
14. **Route Yards** - Yards gained on each route type

## 📊 **PFF-Specific Metrics**

### **Route Running Grades**
- **Route Running Grade** - PFF's overall route running grade
- **Route Running Score** - Numerical route running score
- **Route Running Rank** - Player's rank among position group

### **Separation Metrics**
- **Average Separation** - Average yards of separation when targeted
- **Separation vs. Coverage** - Separation against different coverage types
- **Separation vs. Route Type** - Separation by route type

### **Route Efficiency**
- **Route Win Rate** - Percentage of routes where player "wins"
- **Route Success Rate** - Percentage of routes that result in positive outcomes
- **Route Efficiency Score** - PFF's efficiency rating

## 🔍 **What to Download from PFF+**

### **Receiving Route Data**
With your PFF+ subscription, look for these export options:
- **"Receiving Routes"** or **"Route Running"** data
- **"Advanced Receiving Stats"** 
- **"Route Efficiency"** data
- **"Separation Metrics"**
- **"Regular Season Receiving Statistics"** (recommended)

### **Data Granularity**
- **Weekly data** (preferred) - Game-by-game route stats
- **Seasonal data** - Aggregated season totals
- **Player-level data** - Individual player route statistics

### **Time Periods**
- **2024 Season** (current)
- **2023 Season** (for historical analysis)
- **2022 Season** (if available)
- **Earlier seasons** (2018-2021 if available)

## 📋 **Expected CSV Columns**

When you download from PFF, look for columns like:
```
Player Name, Team, Position, Season, Week, Game ID,
Routes Run, Route Participation %, Route Efficiency,
Average Route Depth, Average Separation, Average Cushion,
Targets, Receptions, Yards, Route Win Rate,
Route Running Grade, Route Running Score
```

## 🚀 **Next Steps**

1. **Log into PFF+** with your subscription and navigate to receiving/route data
2. **Look for CSV export options** in the route running section
3. **Download data** for 2024 (and earlier seasons if available)
4. **Save files** to `data/pff_csv/` with descriptive names like:
   - `receiving_2024_reg.csv`
   - `receiving_2023_reg.csv`
   - `receiving_2022_reg.csv`

5. **Run analysis** to understand the structure:
   ```bash
   python3 tests/analyze_csv_structure.py data/pff_csv/your_file.csv
   ```

## 💡 **Alternative Data Sources**

If you don't have PFF+ access, consider these alternatives:
- **Next Gen Stats** (from NFL) - Separation and cushion data (free)
- **ESPN Advanced Stats** - Route participation metrics (free)
- **Pro Football Reference** - Basic route running stats (free)
- **nfl_data_py** - Limited route metrics (free, already integrated)

## 🔧 **Customization**

Once you have the CSV files, the import script will:
1. **Analyze the structure** automatically
2. **Map columns** to our expected format
3. **Handle different encodings** and formats
4. **Create player ID mappings**
5. **Import into your fantasy database**

**Ready to download your PFF+ data?** Place the CSV files in `data/pff_csv/` and we'll analyze and import them!

## 📝 **Note on Data Access**

This integration is designed for users who already have PFF+ subscriptions and want to enhance their fantasy football analysis with route running data. The system provides tools to import and analyze PFF data but does not provide access to the data itself.
