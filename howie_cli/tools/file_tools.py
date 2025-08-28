"""
File operation tools for Howie CLI
"""

from typing import Any, Dict, Optional, Union
from pathlib import Path
import pandas as pd
import json

from ..core.base_tool import BaseTool, ToolResult, ToolStatus, ToolParameter
from ..core.workspace import WorkspaceManager


class ReadFileTool(BaseTool):
    """Read and parse files in various formats (CSV, JSON, Excel, etc.)"""
    
    def __init__(self):
        super().__init__()
        self.name = "read_file"
        self.category = "file_operations"
        self.description = "Read and parse files in various formats"
        self.parameters = [
            ToolParameter(
                name="file_path",
                type="string",
                description="Path to the file to read (alias: filename)",
                required=True
            ),
            ToolParameter(
                name="file_type",
                type="string",
                description="File type (csv, json, excel, txt, etc.)",
                required=False,
                default="auto"
            ),
            ToolParameter(
                name="encoding",
                type="string",
                description="File encoding",
                required=False,
                default="utf-8"
            )
        ]
        self.workspace = WorkspaceManager()
    
    async def execute(self, file_path: str, file_type: Optional[str] = None, encoding: str = "utf-8", **kwargs) -> ToolResult:
        """Execute file read operation"""
        try:
            data = self.workspace.read_file(file_path, file_type)
            
            # Prepare metadata
            metadata = {
                "file_path": str(file_path),
                "file_type": file_type or Path(file_path).suffix[1:]
            }
            
            if isinstance(data, pd.DataFrame):
                metadata["rows"] = len(data)
                metadata["columns"] = list(data.columns)
                metadata["shape"] = data.shape
            elif isinstance(data, dict):
                metadata["keys"] = list(data.keys())
            elif isinstance(data, list):
                metadata["length"] = len(data)
            
            return ToolResult(
                status=ToolStatus.SUCCESS,
                data=data,
                metadata=metadata
            )
            
        except Exception as e:
            return ToolResult(
                status=ToolStatus.ERROR,
                error=f"Failed to read file: {str(e)}"
            )


class WriteFileTool(BaseTool):
    """Write data to files in various formats"""
    
    def __init__(self):
        super().__init__()
        self.name = "write_file"
        self.category = "file_operations"
        self.description = "Write data to files in various formats"
        self.parameters = [
            ToolParameter(
                name="data",
                type="any",
                description="Data to write to file (alias: content)",
                required=True
            ),
            ToolParameter(
                name="file_name",
                type="string",
                description="Name of the file to create (alias: filename)",
                required=True
            ),
            ToolParameter(
                name="file_type",
                type="string",
                description="File type (csv, json, excel, txt, etc.)",
                required=False,
                default="txt"
            ),
            ToolParameter(
                name="subfolder",
                type="string",
                description="Subfolder within workspace",
                required=False,
                default=""
            ),
            ToolParameter(
                name="overwrite",
                type="boolean",
                description="Whether to overwrite existing files",
                required=False,
                default=True
            )
        ]
        self.workspace = WorkspaceManager()
    
    async def execute(self, data: Any, file_name: str, 
                     file_type: Optional[str] = None,
                     subfolder: Optional[str] = None,
                     overwrite: bool = True, **kwargs) -> ToolResult:
        """Execute file write operation"""
        try:
            file_path = self.workspace.write_file(
                data=data,
                file_name=file_name,
                file_type=file_type,
                subfolder=subfolder
            )
            
            return ToolResult(
                status=ToolStatus.SUCCESS,
                data={"file_path": str(file_path)},
                metadata={
                    "file_name": file_name,
                    "file_type": file_type or Path(file_name).suffix[1:],
                    "subfolder": subfolder,
                    "absolute_path": str(file_path)
                }
            )
            
        except Exception as e:
            return ToolResult(
                status=ToolStatus.ERROR,
                error=f"Failed to write file: {str(e)}"
            )


class ImportRosterTool(BaseTool):
    """Import fantasy roster from CSV/Excel files"""
    
    def __init__(self):
        super().__init__()
        self.name = "import_roster"
        self.category = "file_operations"
        self.description = "Import fantasy roster from various platforms"
        self.parameters = [
            ToolParameter(
                name="file_path",
                type="string",
                description="Path to the roster file (alias: roster_data)",
                required=False
            ),
            ToolParameter(
                name="roster_data",
                type="string",
                description="CSV roster data as string",
                required=False
            ),
            ToolParameter(
                name="platform",
                type="string",
                description="Platform format (espn, yahoo, sleeper, generic)",
                required=False,
                default="generic",
                choices=["espn", "yahoo", "sleeper", "generic"]
            ),
            ToolParameter(
                name="format",
                type="string",
                description="Data format (csv, json, excel)",
                required=False,
                default="csv"
            ),
            ToolParameter(
                name="team_name",
                type="string",
                description="Team name for the roster",
                required=False,
                default="My Team"
            )
        ]
        self.workspace = WorkspaceManager()
    
    async def execute(self, file_path: Optional[str] = None, roster_data: Optional[str] = None, 
                     platform: str = "generic", format: str = "csv", team_name: str = "My Team", **kwargs) -> ToolResult:
        """Import and parse roster file"""
        try:
            # Handle either file_path or roster_data
            if roster_data:
                # Parse CSV data from string
                import io
                import pandas as pd
                roster_df = pd.read_csv(io.StringIO(roster_data))
            elif file_path:
                # Import from file
                roster_df = self.workspace.import_roster(file_path, platform)
            else:
                return ToolResult(
                    status=ToolStatus.ERROR,
                    error="Either file_path or roster_data must be provided"
                )
            
            # Extract roster information
            roster_info = {
                "total_players": len(roster_df),
                "columns": list(roster_df.columns),
                "platform": platform
            }
            
            if "position" in roster_df.columns:
                position_counts = roster_df["position"].value_counts().to_dict()
                roster_info["position_breakdown"] = position_counts
            
            if "player_name" in roster_df.columns:
                roster_info["players"] = roster_df["player_name"].tolist()
            
            return ToolResult(
                status=ToolStatus.SUCCESS,
                data=roster_df,
                metadata=roster_info
            )
            
        except Exception as e:
            return ToolResult(
                status=ToolStatus.ERROR,
                error=f"Failed to import roster: {str(e)}"
            )


class CreateReportTool(BaseTool):
    """Create formatted analysis reports"""
    
    def __init__(self):
        super().__init__()
        self.name = "create_report"
        self.category = "file_operations"
        self.description = "Create formatted analysis reports"
        self.parameters = [
            ToolParameter(
                name="analysis",
                type="dict",
                description="Analysis data to include in report",
                required=True
            ),
            ToolParameter(
                name="report_name",
                type="string",
                description="Name for the report",
                required=False,
                default="analysis_report"
            )
        ]
        self.workspace = WorkspaceManager()
    
    async def execute(self, analysis: Dict, report_name: str = "analysis_report", **kwargs) -> ToolResult:
        """Create analysis report"""
        try:
            report_path = self.workspace.create_analysis_report(analysis, report_name)
            
            return ToolResult(
                status=ToolStatus.SUCCESS,
                data={"report_path": str(report_path)},
                metadata={
                    "report_name": report_name,
                    "sections": list(analysis.keys()),
                    "path": str(report_path)
                }
            )
            
        except Exception as e:
            return ToolResult(
                status=ToolStatus.ERROR,
                error=f"Failed to create report: {str(e)}"
            )


class ListFilesTool(BaseTool):
    """List files in the workspace"""
    
    def __init__(self):
        super().__init__()
        self.name = "list_files"
        self.category = "file_operations"
        self.description = "List files in the current workspace"
        self.parameters = [
            ToolParameter(
                name="pattern",
                type="string",
                description="File pattern to match (e.g., '*.csv')",
                required=False
            )
        ]
        self.workspace = WorkspaceManager()
    
    async def execute(self, pattern: Optional[str] = None, **kwargs) -> ToolResult:
        """List workspace files"""
        try:
            files = self.workspace.list_files(pattern)
            
            return ToolResult(
                status=ToolStatus.SUCCESS,
                data=files,
                metadata={
                    "total_files": len(files),
                    "pattern": pattern or "*",
                    "workspace": str(self.workspace.session_path)
                }
            )
            
        except Exception as e:
            return ToolResult(
                status=ToolStatus.ERROR,
                error=f"Failed to list files: {str(e)}"
            )