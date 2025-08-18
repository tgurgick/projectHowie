# PFF Route Running Statistics Guide

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

## 🔍 **What to Download from PFF**

### **Receiving Route Data**
Look for these export options in PFF:
- **"Receiving Routes"** or **"Route Running"** data
- **"Advanced Receiving Stats"** 
- **"Route Efficiency"** data
- **"Separation Metrics"**

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

1. **Log into PFF** and navigate to receiving/route data
2. **Look for CSV export options** in the route running section
3. **Download data** for 2024 (and earlier seasons if available)
4. **Save files** to `data/pff_csv/` with descriptive names like:
   - `pff_route_data_2024.csv`
   - `pff_route_data_2023.csv`
   - `pff_advanced_receiving_2024.csv`

5. **Run analysis** to understand the structure:
   ```bash
   python3 tests/analyze_csv_structure.py data/pff_csv/your_file.csv
   ```

## 💡 **Alternative Data Sources**

If PFF doesn't have the exact route data you need, also look for:
- **Next Gen Stats** (from NFL) - Separation and cushion data
- **ESPN Advanced Stats** - Route participation metrics
- **Pro Football Reference** - Basic route running stats

## 🔧 **Customization**

Once you have the CSV files, the import script will:
1. **Analyze the structure** automatically
2. **Map columns** to our expected format
3. **Handle different encodings** and formats
4. **Create player ID mappings**
5. **Import into your fantasy database**

**Ready to download your PFF data?** Place the CSV files in `data/pff_csv/` and we'll analyze and import them!
