from __future__ import annotations

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

from textual.app import App, ComposeResult
from textual.widgets import Input, ListView, ListItem, Static
from rich.text import Text
from textual.reactive import reactive
from textual import events
from textual.widgets._list_view import ListView as _LV  # type: ignore

# Load environment variables first
load_dotenv()

# Import after environment setup
from howie_cli.core.commands import REGISTRY
from howie_cli.core.paths import get_db_path, get_db_url, get_data_dir

def _ensure_database_available():
    """Ensure the PPR database is available in user directory"""
    try:
        user_db_path = get_db_path("ppr")
        
        # If user database doesn't exist, try to copy from package installation
        if not user_db_path.exists():
            # Try to find the database in the package installation
            try:
                import howie_cli
                package_dir = Path(howie_cli.__file__).parent.parent
                source_db = package_dir / "data" / "fantasy_ppr.db"
                
                if source_db.exists():
                    import shutil
                    shutil.copy2(source_db, user_db_path)
                    print(f"Copied database to: {user_db_path}")
                else:
                    print(f"Warning: Database not found at {source_db}")
            except Exception as e:
                print(f"Warning: Could not copy database: {e}")
        
        return str(user_db_path)
    except Exception as e:
        print(f"Error setting up database: {e}")
        return None

# Set up database environment using portable paths (same as CLI)
if not os.getenv('DB_URL'):
    # Ensure database is available and use the portable database path system
    _ensure_database_available()
    os.environ['DB_URL'] = get_db_url("ppr")


# Optional: FIGlet banner support
try:
    import pyfiglet  # type: ignore
except Exception:
    pyfiglet = None

WORD = "HOWIE"
FIGLET_FONT = "rectangles"

# Solid filled block font - all letters same width for alignment
BLOCK_FONT = {
    "H": [
        "██  ██",
        "██  ██",
        "██████",
        "██████",
        "██  ██",
        "██  ██",
        "██  ██",
    ],
    "O": [
        " ████ ",
        "██  ██",
        "██  ██",
        "██  ██",
        "██  ██",
        "██  ██",
        " ████ ",
    ],
    "W": [
        "██   ██",
        "██   ██",
        "██   ██",
        "██ █ ██",
        "██ █ ██",
        "███ ███",
        "██   ██",
    ],
    "I": [
        "██████",
        "  ██  ",
        "  ██  ",
        "  ██  ",
        "  ██  ",
        "  ██  ",
        "██████",
    ],
    "E": [
        "██████",
        "██    ",
        "██    ",
        "█████ ",
        "██    ",
        "██    ",
        "██████",
    ],
    " ": ["  ", "  ", "  ", "  ", "  ", "  ", "  "],
}


def _render_block_word(word: str, hspace: int = 2) -> str:
    rows = [""] * 7
    for ch in word.upper():
        glyph = BLOCK_FONT.get(ch, BLOCK_FONT[" "])
        for i in range(7):
            rows[i] += glyph[i] + (" " * hspace)
    return "\n".join(rows)


class Banner(Static):
    def __init__(self, text: str = WORD, font: str | None = FIGLET_FONT) -> None:
        super().__init__()
        self._text = text
        self._font = font

    def _figlet(self, text: str) -> str | None:
        if not pyfiglet:
            return None
        try:
            return pyfiglet.figlet_format(text, font=self._font or "rectangles")
        except Exception:
            try:
                return pyfiglet.figlet_format(text, font="standard")
            except Exception:
                return None

    def render(self) -> Text:
        # Always use the filled block font for consistent solid lettering
        banner = _render_block_word(self._text)
        out = Text()
        # Kelly green foreground, no shadow
        out.append(banner, style="bold #4CBB17")
        return out


class HowieTUI(App):
    CSS = """
    Screen { layout: vertical; }
    #stream { height: auto; border: none; padding: 0 0; background: transparent; }
    #input { height: auto; border: none; background: transparent; }
    #input:focus { border: none; background: transparent; }
    /* Reserve space to avoid layout jumping for suggestions */
    #slash { width: 64; height: 14; border: none; background: transparent; }
    #palette { width: 72; height: 16; border: none; background: transparent; }
    #palette_input { width: 100%; border: none; background: transparent; }
    #palette_input:focus { border: none; background: transparent; }
    #footer { height: auto; background: transparent; }
    /* Spacers to position input area ~75% down (top heavier) */
    #spacer_top { height: 3fr; }
    #spacer_bottom { height: 1fr; }
    """

    query: reactive[str] = reactive("")
    stream_buf: reactive[str] = reactive("")
    tokens_used: reactive[int] = reactive(0)
    context_left: reactive[int] = reactive(100)
    agent = None
    session_log_file = None
    
    def _setup_database_environment(self):
        """Set up database environment like main CLI"""
        try:
            project_root = Path(__file__).parent.parent.parent
            
            # Ensure we're using the PPR database with all the data
            ppr_db_path = project_root / "data" / "fantasy_ppr.db"
            
            # Set up environment for ALL database access - main CLI and draft CLI
            os.environ["DB_URL"] = f"sqlite:///{ppr_db_path}"
            
            # Verify database exists and log status (but don't use _append_stream during init)
            if ppr_db_path.exists():
                size_mb = ppr_db_path.stat().st_size // 1024 // 1024
                self._log_to_session(f"📁 Database: {ppr_db_path.name} ({size_mb}MB)")
                
                # Test database connectivity
                import sqlite3
                with sqlite3.connect(str(ppr_db_path)) as conn:
                    cursor = conn.cursor()
                    cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table'")
                    table_count = cursor.fetchone()[0]
                    self._log_to_session(f"📊 Tables available: {table_count}")
            else:
                self._log_to_session(f"⚠️  Database not found: {ppr_db_path}")
                
        except Exception as e:
            self._log_to_session(f"Database setup error: {e}")
    
    def _setup_session_logging(self):
        """Set up automatic session logging to files."""
        try:
            from datetime import datetime
            
            # Use portable data directory for logs (same as CLI)
            data_dir = get_data_dir()
            logs_dir = data_dir / "logs"
            logs_dir.mkdir(exist_ok=True)
            
            # Create session log file with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_filename = f"tui_session_{timestamp}.log"
            self.session_log_file = logs_dir / log_filename
            
            # Write session header
            with open(self.session_log_file, 'w') as f:
                f.write(f"Howie TUI Session Log\n")
                f.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"=" * 50 + "\n\n")
            
            # DON'T call _append_stream during initialization - no widgets yet!
            # Just log directly to file
            self._log_to_session(f"📝 Session logging: {log_filename}")
            
        except Exception as e:
            # DON'T call _append_stream during initialization
            # Just log the error to file if possible
            if hasattr(self, 'session_log_file') and self.session_log_file:
                self._log_to_session(f"⚠️  Session logging failed: {e}")
    
    def _show_welcome_message(self):
        """Show welcome message after TUI is fully initialized."""
        self._append_stream("")
        self._append_stream("🏈 Welcome to Howie TUI!")
        
        if self.agent is None:
            self._append_stream("📋 Draft commands available (AI features limited):")
        else:
            self._append_stream("🤖 Full AI + Draft system ready! Try these:")
            
        self._append_stream("• /draft/help - Show all draft commands")
        self._append_stream("• /draft/test - Test database connection") 
        self._append_stream("• /draft/quick - Quick draft analysis")
        
        if self.agent is None:
            self._append_stream("💡 Set OPENAI_API_KEY for AI chat features")
        else:
            self._append_stream("• Ask questions - Natural language queries")
            
        self._append_stream("")
    
    def on_unmount(self) -> None:
        """Log when TUI exits."""
        self._log_session_event("TUI_EXIT", "Session ended")
    
    def _log_session_event(self, event_type: str, message: str) -> None:
        """Log special session events."""
        if not self.session_log_file:
            return
            
        try:
            from datetime import datetime
            with open(self.session_log_file, 'a') as f:
                timestamp = datetime.now().strftime("%H:%M:%S")
                f.write(f"[{timestamp}] [{event_type}] {message}\n")
                f.flush()
        except:
            pass

    def compose(self) -> ComposeResult:
        yield Banner()
        yield Static(id="stream")
        yield Static(id="spacer_top")
        yield Static(id="sep")
        # Place input before slash so the menu appears BELOW the input
        yield Input(placeholder="Type /help or ask a question…", id="input")
        yield ListView(id="slash")
        yield Static(id="spacer_bottom")
        yield Static(id="footer")

    def on_mount(self) -> None:
        # Set up session logging
        self._setup_session_logging()
        self._log_session_event("TUI_START", "Initializing TUI session")
        
        # Set up database paths like main CLI
        self._setup_database_environment()
        
        # Keep suggestions panel present to avoid layout jumping (no hint row)
        sl = self.query_one("#slash", ListView)
        sl.visible = True
        sl.clear()
        # Minimal intro and focus input
        stream = self.query_one("#stream", Static)
        # Add a blank line so executed text appears one line below the HOWIE banner
        self.stream_buf = "\n"
        stream.update(self.stream_buf)
        # subtle separation line between stream and input
        self.query_one("#sep", Static).update("[dim]────────────────────────────────────────────────────────────────────────────────[/dim]")
        self.query_one("#input", Input).focus()
        self._update_footer()
        
        # Try to initialize Howie agent for command execution
        try:
            # Check for API key first
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                self._append_stream(f"[dim]⚠️  No OPENAI_API_KEY found. Draft commands will work, AI chat limited.[/dim]")
                self.agent = None
            else:
                from howie_cli.core.enhanced_agent import EnhancedHowieAgent
                self.agent = EnhancedHowieAgent()
        except Exception as e:
            self._append_stream(f"[red]Agent initialization failed: {e}[/red]")
            self.agent = None
        
        # Show welcome message once after everything is set up
        self._show_welcome_message()


    async def on_key(self, event: events.Key) -> None:
        # Cmd+C / Ctrl+C copies output to clipboard
        if event.key == "c" and (event.ctrl or event.meta):
            self._copy_output_to_clipboard()
            event.stop()
        # Tab: move focus to slash menu
        if event.key == "tab":
            if self.query_one("#slash", ListView).visible:
                self.query_one("#slash", ListView).focus()
                event.stop()
                return

    def _toggle_palette(self) -> None:
        pi = self.query_one("#palette_input", Input)
        pl = self.query_one("#palette", ListView)
        show = not pi.visible
        pi.visible = show
        pl.visible = show
        if show:
            pi.value = ""
            pl.clear()
            self._populate_palette("")
            pi.focus()
        else:
            self.query_one("#input", Input).focus()

    def _hide_palette(self) -> None:
        self.query_one("#palette_input", Input).visible = False
        self.query_one("#palette", ListView).visible = False
        self.query_one("#input", Input).focus()

    def _populate_palette(self, query: str) -> None:
        q = query.lower()
        items = [c for c in REGISTRY if not q or q in c.id.lower() or q in c.title.lower() or (c.keywords and any(q in k for k in c.keywords))]
        lv = self.query_one("#palette", ListView)
        lv.clear()
        for c in items[:20]:
            label = f"{c.id} — {c.title}"
            li = ListItem(Static(label))
            setattr(li, "data", label)
            lv.append(li)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        # If palette input submitted, treat as filter/execute top item
        if event.input.id == "palette_input":
            # Execute first visible item
            lv = self.query_one("#palette", ListView)
            if lv.children:
                first = lv.children[0]
                label = self._label_from_listitem(first)
                self._execute_command_label(label)
            self._hide_palette()
            return
        # If main input submitted, execute first suggestion if present; otherwise dispatch raw input
        if event.input.id == "input":
            lv = self.query_one("#slash", ListView)
            if lv.children:
                first = lv.children[0]
                label = self._label_from_listitem(first)
                self._execute_command_label(label)
            else:
                # No suggestions; send raw text (supports normal queries and @model override)
                raw = (self.query or "").strip()
                if raw:
                    # Log user input
                    self._log_to_session(f"USER INPUT: {raw}")
                    
                    # Clear input and suggestions
                    try:
                        self.query_one("#input", Input).value = ""
                        lv.clear()
                    except Exception:
                        pass
                    self._append_stream(f"› Executed: {raw}")
                    try:
                        import asyncio
                        asyncio.get_running_loop().create_task(self._run_command(raw))
                    except Exception as e:
                        error_msg = f"Error scheduling {raw}: {e}"
                        self._append_stream(f"[red]{error_msg}[/red]")
                        self._log_session_event("SCHEDULING_ERROR", f"{raw} -> {e}")
            return

    def on_input_changed_palette(self, value: str) -> None:
        self._populate_palette(value)

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "palette_input":
            self._populate_palette(event.value)
            return
        # default input handling (slash)
        value = event.value
        self.query = value
        slash_view = self.query_one("#slash", ListView)
        slash_view.clear()
        if "/" in value:
            q = value.split("/")[-1].lower()
            matches = [c for c in REGISTRY if q in c.id.lower() or q in c.title.lower() or (c.keywords and any(q in k for k in c.keywords))]
            for c in matches[:14]:
                label = f"{c.id} — {c.title}"
                li = ListItem(Static(label))
                setattr(li, "data", label)
                slash_view.append(li)
        else:
            # leave empty (area is reserved via CSS height)
            pass

    def _append_stream(self, text: str) -> None:
        stream = self.query_one("#stream", Static)
        self.stream_buf = f"{self.stream_buf}\n{text}" if self.stream_buf else text
        stream.update(self.stream_buf)
        self._update_footer()
        
        # Simple auto-scroll: use Textual's built-in scroll action
        try:
            self.call_after_refresh(self._scroll_to_bottom)
        except Exception:
            pass  # If scroll fails, don't break functionality
        
        # Log to session file
        self._log_to_session(text)
    
    def _scroll_to_bottom(self) -> None:
        """Scroll to the bottom of the screen."""
        try:
            self.action_scroll_end()
        except Exception:
            pass

    def _execute_command_label(self, label: str) -> None:
        # label format: "id — title"; ignore placeholders/hints
        if " — " not in label or "Type '/'" in label:
            return
        cmd_id = label.split(" — ")[0]
        # Use the current input text as the full command if it starts with '/'
        full_input = self.query or ""
        cmd_text = full_input.strip() if full_input.strip().startswith('/') else f"/{cmd_id}"
        
        # Log command execution
        self._log_to_session(f"COMMAND: {cmd_text}")
        self._append_stream(f"› Executed: {cmd_text}")
        # Clear input box after sending
        try:
            self.query_one("#input", Input).value = ""
        except Exception:
            pass
        # Clear suggestions list
        try:
            self.query_one("#slash", ListView).clear()
        except Exception:
            pass
        # Dispatch to Howie command handlers asynchronously
        try:
            import asyncio
            asyncio.get_running_loop().create_task(self._run_command(cmd_text))
        except Exception as e:
            self._append_stream(f"[red]Error scheduling {cmd_text}: {e}[/red]")

    # Handle list selection for both palette and slash
    def on_list_view_selected(self, event: _LV.Selected) -> None:  # type: ignore
        list_id = event.list_view.id if hasattr(event, "list_view") else ""
        try:
            item = event.item  # type: ignore[attr-defined]
            label = self._label_from_listitem(item)
            self._execute_command_label(label)
            if list_id == "palette":
                self._hide_palette()
            elif list_id == "slash":
                # keep slash visible to avoid layout jumps
                pass
            # Return focus to input
            self.query_one("#input", Input).focus()
        except Exception:
            pass

    def _label_from_listitem(self, li: ListItem) -> str:
        data = getattr(li, "data", None)
        if isinstance(data, str):
            return data
        try:
            st = li.query_one(Static)
            # Static content can be any renderable; get plain text if possible
            r = getattr(st, "renderable", None)
            if r is None:
                return st.render()
            return getattr(r, "plain", str(r))
        except Exception:
            return str(li)

    async def _run_command(self, cmd_text: str) -> None:
        """Run a command by replicating the CLI execution flow"""
        try:
            # CRITICAL: Ensure database environment is set for this command execution
            if not os.getenv('DB_URL'):
                os.environ['DB_URL'] = get_db_url("ppr")
            
            # Import everything we need
            import howie_enhanced as HE
            from io import StringIO
            from rich.console import Console as RichConsole
            
        except Exception as e:
            self._append_stream(f"[red]Import error: {e}[/red]")
            return

        # Capture output like the CLI does
        buf = StringIO()
        rich_console = RichConsole(file=buf, force_terminal=False, no_color=False, width=120)
        
        # Save original console
        orig_console = getattr(HE, "console", None)
        
        try:
            # Set our console for output capture
            setattr(HE, "console", rich_console)
            
            # Also capture any draft CLI console output
            import howie_cli.draft.draft_cli as draft_cli_module
            orig_draft_console = getattr(draft_cli_module, "console", None)
            
            # Create a shared console for all output
            shared_console = RichConsole(file=buf, force_terminal=False, no_color=False, width=100)
            setattr(HE, "console", shared_console)
            
            # Set draft console to same buffer - this is key!
            setattr(draft_cli_module, "console", shared_console)
            
            # Execute command using CLI logic (mimicking enhanced_chat_loop)
            low = cmd_text.lower().strip()
            
            # Handle normal queries (no slash) 
            if not low.startswith('/'):
                if self.agent is None:
                    self._append_stream("[yellow]AI agent not available. Draft commands still work with /draft/[/yellow]")
                    self._append_stream("[dim]To enable AI chat, ensure OPENAI_API_KEY is set in .env[/dim]")
                    return
                    
                # Handle @model queries
                if low.startswith('@'):
                    parts = cmd_text.split(' ', 1)
                    if len(parts) == 2:
                        model_name = parts[0][1:]
                        query = parts[1]
                        try:
                            resp = await self.agent.process_with_model(query, model_name)
                            self._append_stream(f"[bold bright_green]Howie ({model_name}):[/bold bright_green]")
                            from rich.markdown import Markdown
                            # Render markdown to text for TUI
                            markdown_obj = Markdown(resp)
                            with StringIO() as md_buf:
                                md_console = RichConsole(file=md_buf, force_terminal=False, no_color=False, width=100)
                                md_console.print(markdown_obj)
                                self._append_stream(md_buf.getvalue())
                            return
                        except Exception as e:
                            self._append_stream(f"[red]@{model_name} error: {e}[/red]")
                            return
                
                # Handle normal AI queries
                try:
                    self._append_stream("[dim]Processing query...[/dim]")
                    
                    # Show model selection like CLI
                    recommended_model = self.agent.model_manager.recommend_model(cmd_text)
                    self._append_stream(f"[dim]Selected model: {recommended_model}[/dim]")
                    
                    # Process the message
                    response = await self.agent.process_message(cmd_text)
                    
                    # Display response like CLI
                    self._append_stream(f"[bold bright_green]Howie:[/bold bright_green]")
                    from rich.markdown import Markdown
                    # Render markdown to text for TUI
                    markdown_obj = Markdown(response)
                    with StringIO() as md_buf:
                        md_console = RichConsole(file=md_buf, force_terminal=False, no_color=False, width=100)
                        md_console.print(markdown_obj)
                        self._append_stream(md_buf.getvalue())
                    
                    return
                except Exception as e:
                    self._append_stream(f"[red]Query error: {e}[/red]")
                    return

            # Handle slash commands exactly like CLI
            if low.startswith('/help') or low in ('/?', '?', '/'):
                HE.show_enhanced_help()
            elif low.startswith('/model'):
                if self.agent is not None:
                    HE.handle_model_command(self.agent, cmd_text[7:])
                else:
                    self._append_stream("[dim]Agent not initialized.[/dim]")
            elif low.startswith('/agent'):
                if self.agent is not None:
                    HE.handle_agent_command(self.agent, cmd_text[7:])
                else:
                    self._append_stream("[dim]Agent not initialized.[/dim]")
            elif low.startswith('/cost'):
                if self.agent is not None:
                    HE.handle_cost_command(self.agent, cmd_text[6:])
                else:
                    self._append_stream("[dim]Agent not initialized.[/dim]")
            elif low.startswith('/logs'):
                if self.agent is not None:
                    HE.handle_logs_command(self.agent, cmd_text[6:])
                else:
                    self._append_stream("[dim]Agent not initialized.[/dim]")
            elif low.startswith('/adp'):
                HE.handle_adp_command(cmd_text[1:])
            elif low.startswith('/tiers'):
                HE.handle_tiers_command(cmd_text[1:])
            elif low.startswith('/intel'):
                HE.handle_intel_command(cmd_text[1:])
            elif low.startswith('/player'):
                await HE.handle_player_search_command(cmd_text[1:])
            elif low.startswith('/draft'):
                HE.handle_draft_command(cmd_text[7:])
            elif low.startswith('/update'):
                if self.agent is not None:
                    await HE.handle_update_command(self.agent, cmd_text[8:])
                else:
                    self._append_stream("[dim]Agent not initialized.[/dim]")
            elif (low.startswith('/wr/') or low.startswith('/qb/') or 
                  low.startswith('/rb/') or low.startswith('/te/') or 
                  low.startswith('/k/') or low.startswith('/def/') or 
                  low.startswith('/dst/')):
                HE.handle_rapid_stats_command(cmd_text[1:])
            elif low in ('/quit', '/end', '/e'):
                self.exit()
                return
            else:
                self._append_stream("[dim]Unknown command. Type '/help' for available commands.[/dim]")
                return

            # Capture and display output
            output = buf.getvalue().strip()
            if output:
                # Process Rich markup for TUI display
                self._append_stream(output)
            else:
                # For commands that don't produce console output, show confirmation
                if low.startswith('/draft'):
                    self._append_stream(f"[green]✅ Draft command '{cmd_text}' executed[/green]")
                else:
                    self._append_stream(f"[dim]✅ Command '{cmd_text}' executed[/dim]")
                
        except Exception as e:
            self._append_stream(f"[red]Execution error: {e}[/red]")
            import traceback
            self._append_stream(f"[dim]{traceback.format_exc()}[/dim]")
        finally:
            # Restore original consoles
            if orig_console is not None:
                setattr(HE, "console", orig_console)
            try:
                # Restore draft console
                if orig_draft_console is not None:
                    import howie_cli.draft.draft_cli as draft_cli_module
                    setattr(draft_cli_module, "console", orig_draft_console)
            except:
                pass
            
            # Log command completion immediately
            self._log_session_event("COMMAND_COMPLETE", f"Finished executing: {cmd_text}")

    def _dispatch_command(self, cmd_text: str) -> str:
        """Route simple commands to howie_enhanced handlers and capture output.
        Returns plain text output to append to the stream.
        """
        return "[dim]Use async _run_command instead[/dim]"

    def _update_footer(self) -> None:
        """Render a Codex-style footer with key hints and metrics (dimmed)."""
        footer = self.query_one("#footer", Static)
        kelly = "#4CBB17"
        
        # Add session log info
        log_info = ""
        if self.session_log_file:
            log_name = self.session_log_file.name
            log_info = f"📝 {log_name}   "
        
        cmds = f"[{kelly}]⏎ send   Tab slash-menu   Cmd+C copy-output[/]"
        metrics = f"{log_info}{self.tokens_used} tokens used   {self.context_left}% context left"
        footer.update(f"[dim]{cmds}   {metrics}[/dim]")
    
    def _copy_output_to_clipboard(self) -> None:
        """Copy current output to clipboard for sharing."""
        try:
            # Get the current stream content
            import subprocess
            
            # Remove rich markup for clean clipboard copy
            clean_text = self._strip_rich_markup(self.stream_buf)
            
            # Add some debug info
            debug_info = f"[DEBUG] Platform: {sys.platform}, Text length: {len(clean_text)}\n"
            self._append_stream(f"[dim]{debug_info}[/dim]")
            
            # Copy to clipboard using system commands
            if sys.platform == "darwin":  # macOS
                self._append_stream("[dim]🍎 Using macOS pbcopy...[/dim]")
                process = subprocess.Popen(['pbcopy'], stdin=subprocess.PIPE)
                stdout, stderr = process.communicate(clean_text.encode('utf-8'))
                if process.returncode != 0:
                    raise Exception(f"pbcopy failed with code {process.returncode}")
                    
            elif sys.platform == "linux":  # Linux
                self._append_stream("[dim]🐧 Using Linux xclip...[/dim]")
                process = subprocess.Popen(['xclip', '-selection', 'clipboard'], stdin=subprocess.PIPE)
                stdout, stderr = process.communicate(clean_text.encode('utf-8'))
                if process.returncode != 0:
                    raise Exception(f"xclip failed with code {process.returncode}")
                    
            elif sys.platform == "win32":  # Windows
                self._append_stream("[dim]🪟 Using Windows clip...[/dim]")
                process = subprocess.Popen(['clip'], stdin=subprocess.PIPE, shell=True)
                stdout, stderr = process.communicate(clean_text.encode('utf-8'))
                if process.returncode != 0:
                    raise Exception(f"clip failed with code {process.returncode}")
            else:
                raise Exception(f"Unsupported platform: {sys.platform}")
            
            # Show confirmation
            self._append_stream("[green]✅ Output copied to clipboard! Try Cmd+V to paste[/green]")
            
        except Exception as e:
            self._append_stream(f"[red]❌ Copy failed: {e}[/red]")
            self._append_stream(f"[yellow]💡 Try selecting text manually and using system copy[/yellow]")
    
    def _strip_rich_markup(self, text: str) -> str:
        """Remove Rich markup tags for clean text output."""
        import re
        # Remove Rich markup like [red], [/red], [dim], etc.
        clean = re.sub(r'\[/?[^\]]*\]', '', text)
        # Clean up extra whitespace
        clean = re.sub(r'\n\s*\n\s*\n', '\n\n', clean)
        return clean.strip()
    
    def _log_to_session(self, text: str) -> None:
        """Log text to the session log file with immediate write."""
        if not self.session_log_file:
            return
            
        try:
            from datetime import datetime
            
            # Strip Rich markup for clean log
            clean_text = self._strip_rich_markup(text)
            
            # Write to log file with timestamp - IMMEDIATE write
            with open(self.session_log_file, 'a', buffering=1) as f:  # Line buffering
                timestamp = datetime.now().strftime("%H:%M:%S")
                f.write(f"[{timestamp}] {clean_text}\n")
                f.flush()  # Force immediate write to disk
                os.fsync(f.fileno())  # Force OS to write to disk
                
        except Exception as e:
            # Log the error but don't disrupt UI
            try:
                with open(self.session_log_file, 'a') as f:
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    f.write(f"[{timestamp}] [LOGGING ERROR] {e}\n")
                    f.flush()
                    os.fsync(f.fileno())
            except:
                pass

    def _intro_banner(self) -> str:
        return ""


def run() -> None:
    """Run the TUI with simple, working approach."""
    HowieTUI().run()
