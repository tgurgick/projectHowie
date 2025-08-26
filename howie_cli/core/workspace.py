"""
Workspace management for file operations and data handling
"""

import os
import shutil
import tempfile
from pathlib import Path
from typing import Optional, List, Dict, Any, Union
import pandas as pd
import json
import csv
from datetime import datetime
import hashlib


class WorkspaceManager:
    """Manages workspace for file operations and temporary data"""
    
    def __init__(self, base_path: Optional[Path] = None):
        if base_path:
            self.base_path = Path(base_path)
        else:
            self.base_path = Path.home() / ".howie" / "workspace"
        
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.session_path = self._create_session_workspace()
        self.file_registry: Dict[str, Dict] = {}
        
    def _create_session_workspace(self) -> Path:
        """Create a unique session workspace"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_id = hashlib.md5(timestamp.encode()).hexdigest()[:8]
        session_path = self.base_path / f"session_{timestamp}_{session_id}"
        session_path.mkdir(parents=True, exist_ok=True)
        return session_path
    
    def read_file(self, file_path: Union[str, Path], file_type: Optional[str] = None) -> Any:
        """Read a file and return its contents"""
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Auto-detect file type if not provided
        if not file_type:
            file_type = path.suffix.lower()[1:]  # Remove the dot
        
        # Register file
        self._register_file(path, "read")
        
        if file_type in ['csv', 'tsv']:
            delimiter = '\t' if file_type == 'tsv' else ','
            return pd.read_csv(path, delimiter=delimiter)
        
        elif file_type in ['xlsx', 'xls']:
            return pd.read_excel(path)
        
        elif file_type == 'json':
            with open(path, 'r') as f:
                return json.load(f)
        
        elif file_type in ['txt', 'md']:
            with open(path, 'r') as f:
                return f.read()
        
        elif file_type == 'parquet':
            return pd.read_parquet(path)
        
        else:
            # Default to text
            with open(path, 'r') as f:
                return f.read()
    
    def write_file(self, data: Any, file_name: str, file_type: Optional[str] = None,
                   subfolder: Optional[str] = None) -> Path:
        """Write data to a file in the workspace"""
        # Determine subfolder
        if subfolder:
            target_dir = self.session_path / subfolder
            target_dir.mkdir(parents=True, exist_ok=True)
        else:
            target_dir = self.session_path
        
        # Construct file path
        file_path = target_dir / file_name
        
        # Auto-detect file type if not provided
        if not file_type:
            file_type = file_path.suffix.lower()[1:]
        
        # Write based on type
        if isinstance(data, pd.DataFrame):
            if file_type == 'csv':
                data.to_csv(file_path, index=False)
            elif file_type in ['xlsx', 'xls']:
                data.to_excel(file_path, index=False)
            elif file_type == 'json':
                data.to_json(file_path, orient='records', indent=2)
            elif file_type == 'parquet':
                data.to_parquet(file_path)
            elif file_type == 'html':
                data.to_html(file_path)
            else:
                # Default to CSV
                data.to_csv(file_path, index=False)
        
        elif isinstance(data, dict):
            if file_type == 'json':
                with open(file_path, 'w') as f:
                    json.dump(data, f, indent=2, default=str)
            else:
                # Convert to text representation
                with open(file_path, 'w') as f:
                    f.write(str(data))
        
        elif isinstance(data, (list, tuple)):
            if file_type == 'json':
                with open(file_path, 'w') as f:
                    json.dump(data, f, indent=2, default=str)
            elif file_type == 'csv':
                with open(file_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    for row in data:
                        if isinstance(row, (list, tuple)):
                            writer.writerow(row)
                        else:
                            writer.writerow([row])
            else:
                with open(file_path, 'w') as f:
                    for item in data:
                        f.write(str(item) + '\n')
        
        else:
            # Write as text
            with open(file_path, 'w') as f:
                f.write(str(data))
        
        # Register file
        self._register_file(file_path, "write")
        
        return file_path
    
    def _register_file(self, file_path: Path, operation: str):
        """Register file operation"""
        file_key = str(file_path)
        if file_key not in self.file_registry:
            self.file_registry[file_key] = {
                "path": str(file_path),
                "first_accessed": datetime.now(),
                "operations": []
            }
        
        self.file_registry[file_key]["operations"].append({
            "type": operation,
            "timestamp": datetime.now()
        })
        self.file_registry[file_key]["last_accessed"] = datetime.now()
    
    def import_roster(self, file_path: Union[str, Path], platform: str = "generic") -> pd.DataFrame:
        """Import fantasy roster from various platforms"""
        data = self.read_file(file_path)
        
        if isinstance(data, pd.DataFrame):
            df = data
        else:
            # Try to convert to DataFrame
            df = pd.DataFrame(data)
        
        # Platform-specific parsing
        if platform.lower() == "espn":
            return self._parse_espn_roster(df)
        elif platform.lower() == "yahoo":
            return self._parse_yahoo_roster(df)
        elif platform.lower() == "sleeper":
            return self._parse_sleeper_roster(df)
        else:
            return self._parse_generic_roster(df)
    
    def _parse_generic_roster(self, df: pd.DataFrame) -> pd.DataFrame:
        """Parse generic roster format"""
        # Look for common column names
        player_cols = ['player', 'name', 'player_name', 'player name']
        pos_cols = ['position', 'pos', 'positions']
        team_cols = ['team', 'team_abbr', 'nfl_team']
        
        # Standardize column names
        col_mapping = {}
        for col in df.columns:
            col_lower = col.lower()
            if col_lower in player_cols:
                col_mapping[col] = 'player_name'
            elif col_lower in pos_cols:
                col_mapping[col] = 'position'
            elif col_lower in team_cols:
                col_mapping[col] = 'team'
        
        if col_mapping:
            df = df.rename(columns=col_mapping)
        
        return df
    
    def _parse_espn_roster(self, df: pd.DataFrame) -> pd.DataFrame:
        """Parse ESPN roster format"""
        # ESPN-specific parsing logic
        return self._parse_generic_roster(df)
    
    def _parse_yahoo_roster(self, df: pd.DataFrame) -> pd.DataFrame:
        """Parse Yahoo roster format"""
        # Yahoo-specific parsing logic
        return self._parse_generic_roster(df)
    
    def _parse_sleeper_roster(self, df: pd.DataFrame) -> pd.DataFrame:
        """Parse Sleeper roster format"""
        # Sleeper-specific parsing logic
        return self._parse_generic_roster(df)
    
    def create_analysis_report(self, analysis: Dict, report_name: str = "analysis_report") -> Path:
        """Create a formatted analysis report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = self.session_path / "reports" / f"{report_name}_{timestamp}.md"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create markdown report
        report_content = f"# Fantasy Football Analysis Report\n\n"
        report_content += f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        for section, content in analysis.items():
            report_content += f"## {section}\n\n"
            
            if isinstance(content, pd.DataFrame):
                report_content += content.to_markdown() + "\n\n"
            elif isinstance(content, dict):
                for key, value in content.items():
                    report_content += f"**{key}:** {value}\n"
                report_content += "\n"
            elif isinstance(content, list):
                for item in content:
                    report_content += f"- {item}\n"
                report_content += "\n"
            else:
                report_content += f"{content}\n\n"
        
        with open(report_path, 'w') as f:
            f.write(report_content)
        
        self._register_file(report_path, "write")
        return report_path
    
    def list_files(self, pattern: Optional[str] = None) -> List[Dict]:
        """List all files in the workspace"""
        files = []
        
        for path in self.session_path.rglob(pattern or "*"):
            if path.is_file():
                files.append({
                    "name": path.name,
                    "path": str(path),
                    "size": path.stat().st_size,
                    "modified": datetime.fromtimestamp(path.stat().st_mtime),
                    "relative_path": str(path.relative_to(self.session_path))
                })
        
        return files
    
    def export_workspace(self, export_path: Optional[Path] = None) -> Path:
        """Export the entire workspace as an archive"""
        if not export_path:
            export_path = self.base_path / "exports"
            export_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_name = export_path / f"workspace_{timestamp}"
        
        # Create archive
        shutil.make_archive(str(archive_name), 'zip', self.session_path)
        
        return Path(f"{archive_name}.zip")
    
    def cleanup(self, keep_reports: bool = True):
        """Clean up temporary files"""
        if keep_reports and (self.session_path / "reports").exists():
            # Keep reports, clean everything else
            for item in self.session_path.iterdir():
                if item.name != "reports":
                    if item.is_dir():
                        shutil.rmtree(item)
                    else:
                        item.unlink()
        else:
            # Clean everything
            shutil.rmtree(self.session_path)
    
    def get_workspace_info(self) -> Dict:
        """Get information about the current workspace"""
        total_size = sum(f.stat().st_size for f in self.session_path.rglob("*") if f.is_file())
        file_count = len(list(self.session_path.rglob("*")))
        
        return {
            "path": str(self.session_path),
            "total_files": file_count,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "created": datetime.fromtimestamp(self.session_path.stat().st_ctime),
            "files_accessed": len(self.file_registry)
        }