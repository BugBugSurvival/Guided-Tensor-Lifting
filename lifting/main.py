
"""
Script to process multiple benchmarks by calling another Python script (`run.py`) with 
specific arguments. The script:

1. Collects all '.c' files from a specified directory (`benchmarks_dir`).
2. Derives a list of benchmark names from those files.
3. Iterates over each benchmark name, constructing a command to invoke `run.py`.
4. Uses `subprocess.run` to execute `run.py` with a specified grammar style, enumerator, timeout,
   and optional pre-check flag.
5. Uses the Rich library to provide a progress bar and real-time output for each benchmark.
"""


from run import Runner
import subprocess
import glob
import os
from rich.progress import Progress, BarColumn, TimeElapsedColumn, TextColumn
from rich.console import Console, Group
from rich.live import Live
from rich.text import Text

console = Console()

# Directory containing benchmark '.c' files
benchmarks_dir = 'data/benchmarks'  

# Path to the script that will process each benchmark
script_path = './run.py'

# Grammar style for the synthesizer
grammar_style = 'wcfg'

# Timeout in seconds for each individual benchmark run
timeout = 60 * 60  

# Enumerator type for the synthesizer
enumerator = 'bottom_up'
enumerator = 'top_down'

# Flag to determine if we should run a pre-check on LLM solutions
pre_check = 'no'

# Collect all '.c' benchmark files from the specified directory
files = glob.glob(f'{benchmarks_dir}/*.c')

# Derive the benchmark names (without the '.c' extension)
benchmarks = [os.path.basename(f).replace('.c', '') for f in sorted(files)]

# Print out summary information for the run
console.print(f"[bold blue]Number of benchmarks:[/] [bold yellow]{len(files)}[/]")
console.print(f"[bold blue]Grammar Style:[/] [bold yellow]{grammar_style}[/]")
console.print(f"[bold blue]Timeout:[/] [bold yellow]{int(timeout/60)} minutes[/]")
console.print(f"[bold blue]Enumerator:[/] [bold yellow]{enumerator.replace('_', ' ')}[/]")
console.print(f"[bold blue]Check LLM Solutions:[/] [bold yellow]{pre_check}[/]")

# Create a Rich Progress object for a visual progress bar
progress = Progress(
    BarColumn(bar_width=None),  
    TextColumn("[progress.percentage]{task.percentage:>3.0f}%", justify="right"),
    TimeElapsedColumn(),
    console=console,
    expand=True,
)

# Iterate over each benchmark and process it
task_id = progress.add_task("Processing Benchmarks...", total=len(benchmarks))

# Use the Live context manager to continuously update output
with Live(console=console, refresh_per_second=10) as live:
    for benchmark in benchmarks:
        processing_text = Text(f"Processing: {benchmark}", style="green")
        live.update(Group(processing_text, progress))
        command = [
            'python3', script_path,
            '--grammar_style', grammar_style,
            '--benchmark', benchmark,
            '--timeout', str(timeout),
            '--enumerator', enumerator,
            '--pre_check', str(pre_check)
        ]
        try:
            subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as e:
            console.log(f"[bold red]Error with {benchmark}[/]: {e.stderr.decode()}")
        progress.advance(task_id)
console.print("[bold green]All benchmarks completed.[/]")
