import asyncio
import random
import pandas as pd
from rich.progress import (
    Progress,
    TextColumn,
    BarColumn,
    TimeRemainingColumn,
    TimeElapsedColumn,
    TaskProgressColumn,
)

async def run_batched(
    wrapper_class,
    df: pd.DataFrame,
    batch_size: int = 100,
    pause_seconds: float = 60,
    **kwargs
) -> pd.DataFrame:
    """
    Run any gender inference wrapper (GenderAPI, NamSor, Genderize, etc.)
    in batched mode with adaptive cooldown handling and rich visual progress bars.
    """

    total = len(df)
    results = []

    # Create progress bar layout
    progress = Progress(
        TextColumn("[bold blue]{task.description}[/]"),
        BarColumn(bar_width=40),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    )

    # Rich allows you to "live" manage multiple bars
    with progress:
        # Main dataset progress
        overall_task = progress.add_task("Total progress", total=total)
        # Placeholder for cooldown per batch
        cooldown_task = None

        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            batch_df = df.iloc[start:end]

            progress.log(f"[green]Processing rows {start}-{end}...[/green]")
            wrap_instance = wrapper_class(batch_df)

            # Run async API calls (simulate with asyncio.sleep here)
            batch_result = await wrap_instance.get_prediction_async(**kwargs)
            results.append(batch_result)

            # Update main task
            progress.update(overall_task, advance=len(batch_df))

            # Cooldown between batches (if not last)
            if end < total:
                sleep_time = pause_seconds + random.uniform(1, 5)
                progress.log(f"[yellow]Cooling down for {sleep_time:.1f}s to respect API limits...[/yellow]")

                # Create a temporary "cooldown" progress bar
                cooldown_task = progress.add_task("Batch cooldown", total=sleep_time)
                start_time = asyncio.get_event_loop().time()

                while not progress.finished:
                    now = asyncio.get_event_loop().time()
                    elapsed = now - start_time
                    progress.update(cooldown_task, completed=elapsed)
                    if elapsed >= sleep_time:
                        break
                    await asyncio.sleep(0.2)

                progress.remove_task(cooldown_task)

        progress.console.print("\n [bold green]All batches completed successfully![/bold green]")

    # Combine results into a single DataFrame
    combined_df = pd.concat(results, ignore_index=True)
    return combined_df
