#!/usr/bin/env python3
"""
Data migration script for Howie CLI
Helps transfer databases and configurations between computers
"""

import os
import sys
import shutil
from pathlib import Path
import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
import tarfile
import json
from datetime import datetime

console = Console()


@click.group()
def cli():
    """Howie Data Migration Tool"""
    pass


@cli.command()
@click.option('--source', type=click.Path(exists=True), required=True, 
              help='Source directory containing databases')
@click.option('--output', type=click.Path(), help='Output archive path')
@click.option('--include-config', is_flag=True, help='Include configuration files')
def export(source, output, include_config):
    """Export databases and configuration to an archive"""
    
    source_path = Path(source)
    
    if not output:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output = f"howie_data_{timestamp}.tar.gz"
    
    output_path = Path(output)
    
    console.print(Panel.fit(
        f"[bold cyan]Exporting Howie Data[/bold cyan]\n"
        f"Source: {source_path}\n"
        f"Output: {output_path}",
        border_style="cyan"
    ))
    
    # Find database files
    db_files = []
    for pattern in ['*.db', '*.sqlite', '*.sqlite3']:
        db_files.extend(source_path.rglob(pattern))
    
    if not db_files:
        console.print("[red]No database files found![/red]")
        return
    
    console.print(f"\n[green]Found {len(db_files)} database files:[/green]")
    for db_file in db_files:
        size_mb = db_file.stat().st_size / (1024 * 1024)
        console.print(f"  {db_file.name} ({size_mb:.1f} MB)")
    
    # Create archive
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Creating archive...", total=None)
        
        with tarfile.open(output_path, "w:gz") as tar:
            # Add database files
            for db_file in db_files:
                arcname = f"data/{db_file.name}"
                tar.add(db_file, arcname=arcname)
                progress.console.print(f"  Added: {db_file.name}")
            
            # Add configuration files if requested
            if include_config:
                config_files = [
                    source_path / "models_config.json",
                    source_path / "config.json",
                    Path.home() / ".howie" / "models.json"
                ]
                
                for config_file in config_files:
                    if config_file.exists():
                        arcname = f"config/{config_file.name}"
                        tar.add(config_file, arcname=arcname)
                        progress.console.print(f"  Added config: {config_file.name}")
        
        progress.update(task, completed=True)
    
    console.print(f"\n[green]✅ Export complete: {output_path}[/green]")
    console.print(f"Archive size: {output_path.stat().st_size / (1024 * 1024):.1f} MB")


@cli.command()
@click.option('--archive', type=click.Path(exists=True), required=True,
              help='Archive file to import')
@click.option('--target', type=click.Path(), help='Target directory (default: current)')
@click.option('--force', is_flag=True, help='Overwrite existing files')
def import_data(archive, target, force):
    """Import databases and configuration from an archive"""
    
    archive_path = Path(archive)
    target_path = Path(target) if target else Path.cwd()
    
    console.print(Panel.fit(
        f"[bold cyan]Importing Howie Data[/bold cyan]\n"
        f"Archive: {archive_path}\n"
        f"Target: {target_path}",
        border_style="cyan"
    ))
    
    # Create target directories
    data_dir = target_path / "data"
    data_dir.mkdir(exist_ok=True)
    
    config_dir = target_path / "config"
    config_dir.mkdir(exist_ok=True)
    
    # Extract archive
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Extracting archive...", total=None)
        
        with tarfile.open(archive_path, "r:gz") as tar:
            members = tar.getmembers()
            
            for member in members:
                if member.isfile():
                    # Determine destination
                    if member.name.startswith("data/"):
                        dest_path = target_path / member.name
                    elif member.name.startswith("config/"):
                        if "models.json" in member.name:
                            # Special handling for models config
                            dest_path = Path.home() / ".howie" / "models.json"
                            dest_path.parent.mkdir(parents=True, exist_ok=True)
                        else:
                            dest_path = target_path / member.name
                    else:
                        dest_path = target_path / member.name
                    
                    # Check if file exists
                    if dest_path.exists() and not force:
                        console.print(f"[yellow]Skipping existing file: {dest_path.name}[/yellow]")
                        continue
                    
                    # Extract file
                    tar.extract(member, path=target_path)
                    progress.console.print(f"  Extracted: {member.name}")
        
        progress.update(task, completed=True)
    
    console.print(f"\n[green]✅ Import complete![/green]")
    
    # Verify databases
    verify_databases(target_path)


@cli.command()
@click.option('--path', type=click.Path(), help='Path to check (default: current directory)')
def verify(path):
    """Verify database files and show information"""
    
    check_path = Path(path) if path else Path.cwd()
    verify_databases(check_path)


def verify_databases(path: Path):
    """Verify and show database information"""
    
    console.print("\n[bold]Database Verification:[/bold]")
    
    # Find database files
    db_files = []
    for pattern in ['*.db', '*.sqlite', '*.sqlite3']:
        db_files.extend(path.rglob(pattern))
    
    if not db_files:
        console.print("[red]No database files found![/red]")
        return
    
    # Create verification table
    table = Table(title="Database Files", show_header=True, header_style="bold magenta")
    table.add_column("File", style="cyan")
    table.add_column("Size", style="green")
    table.add_column("Modified", style="yellow")
    table.add_column("Status", style="white")
    
    import sqlite3
    
    for db_file in sorted(db_files):
        size_mb = db_file.stat().st_size / (1024 * 1024)
        modified = datetime.fromtimestamp(db_file.stat().st_mtime)
        
        # Test database connection
        try:
            conn = sqlite3.connect(db_file)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table'")
            table_count = cursor.fetchone()[0]
            conn.close()
            status = f"✅ {table_count} tables"
        except Exception as e:
            status = f"❌ {str(e)[:30]}..."
        
        table.add_row(
            db_file.name,
            f"{size_mb:.1f} MB",
            modified.strftime("%Y-%m-%d"),
            status
        )
    
    console.print(table)


@cli.command()
@click.option('--old-location', type=click.Path(exists=True), required=True,
              help='Path to old Howie installation')
@click.option('--new-location', type=click.Path(), help='Path to new installation')
def migrate(old_location, new_location):
    """Migrate from old Howie installation to new location"""
    
    old_path = Path(old_location)
    new_path = Path(new_location) if new_location else Path.cwd()
    
    console.print(Panel.fit(
        f"[bold cyan]Migrating Howie Installation[/bold cyan]\n"
        f"From: {old_path}\n"
        f"To: {new_path}",
        border_style="cyan"
    ))
    
    # Find what to migrate
    items_to_migrate = []
    
    # Database files
    for pattern in ['*.db', '*.sqlite', '*.sqlite3']:
        for db_file in old_path.rglob(pattern):
            items_to_migrate.append(("database", db_file, new_path / "data" / db_file.name))
    
    # Configuration files
    config_files = [
        old_path / "models_config.json",
        old_path / "config.json",
        old_path / ".howie" / "models.json"
    ]
    
    for config_file in config_files:
        if config_file.exists():
            if config_file.name == "models.json":
                dest = Path.home() / ".howie" / "models.json"
                dest.parent.mkdir(parents=True, exist_ok=True)
            else:
                dest = new_path / config_file.name
            items_to_migrate.append(("config", config_file, dest))
    
    if not items_to_migrate:
        console.print("[yellow]No files found to migrate![/yellow]")
        return
    
    # Show what will be migrated
    console.print(f"\n[bold]Will migrate {len(items_to_migrate)} items:[/bold]")
    for item_type, source, dest in items_to_migrate:
        console.print(f"  {item_type}: {source.name} → {dest}")
    
    if not click.confirm("\nProceed with migration?"):
        console.print("Migration cancelled.")
        return
    
    # Perform migration
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Migrating files...", total=len(items_to_migrate))
        
        for item_type, source, dest in items_to_migrate:
            try:
                # Create destination directory
                dest.parent.mkdir(parents=True, exist_ok=True)
                
                # Copy file
                shutil.copy2(source, dest)
                progress.console.print(f"  ✅ Migrated: {source.name}")
                
            except Exception as e:
                progress.console.print(f"  ❌ Failed: {source.name} - {e}")
            
            progress.advance(task)
    
    console.print(f"\n[green]✅ Migration complete![/green]")
    verify_databases(new_path)


@cli.command()
def setup_sync():
    """Set up data synchronization between computers"""
    
    console.print(Panel.fit(
        "[bold cyan]Howie Data Sync Setup[/bold cyan]\n"
        "Choose a synchronization method:",
        border_style="cyan"
    ))
    
    methods = [
        ("Cloud Storage", "Use Dropbox/Google Drive/OneDrive"),
        ("Git LFS", "Use Git Large File Storage for databases"),
        ("Manual Export", "Regular export/import cycles"),
        ("Network Share", "Shared network folder")
    ]
    
    console.print("\n[bold]Available methods:[/bold]")
    for i, (name, desc) in enumerate(methods, 1):
        console.print(f"{i}. [cyan]{name}[/cyan] - {desc}")
    
    choice = click.prompt("\nSelect method", type=int, default=1)
    
    if choice == 1:
        setup_cloud_sync()
    elif choice == 2:
        setup_git_lfs()
    elif choice == 3:
        setup_manual_sync()
    elif choice == 4:
        setup_network_sync()


def setup_cloud_sync():
    """Set up cloud storage synchronization"""
    console.print("\n[bold cyan]Cloud Storage Sync Setup[/bold cyan]")
    console.print("""
1. Create a folder in your cloud storage: 'HowieData'
2. Export your data: howie migrate export --source ./data --output ~/CloudStorage/HowieData/howie_data.tar.gz
3. On other computers, import: howie migrate import --archive ~/CloudStorage/HowieData/howie_data.tar.gz
4. Set up regular exports with cron/scheduled tasks
""")


def setup_git_lfs():
    """Set up Git LFS for large database files"""
    console.print("\n[bold cyan]Git LFS Setup[/bold cyan]")
    console.print("""
1. Install Git LFS: git lfs install
2. Track database files: git lfs track "*.db" "*.sqlite"
3. Add .gitattributes: git add .gitattributes
4. Commit and push: git add data/*.db && git commit -m "Add databases" && git push
5. On other computers: git lfs pull
""")


def setup_manual_sync():
    """Set up manual synchronization"""
    console.print("\n[bold cyan]Manual Sync Setup[/bold cyan]")
    console.print("""
1. Regular export: python migrate_data.py export --source ./data
2. Transfer archive to other computer
3. Import: python migrate_data.py import --archive howie_data_*.tar.gz
4. Set up regular schedule (weekly/monthly)
""")


def setup_network_sync():
    """Set up network folder synchronization"""
    console.print("\n[bold cyan]Network Share Setup[/bold cyan]")
    console.print("""
1. Set up shared network folder
2. Create symbolic links: ln -s /network/share/howie_data ./data
3. All computers use the same shared data
4. Ensure proper backup of network share
""")


if __name__ == "__main__":
    cli()