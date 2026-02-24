import os
import glob
import argparse
import subprocess
from collections import deque

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.live import Live
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
    from rich.layout import Layout
    from rich.text import Text
    HAS_RICH = True
except ImportError:
    HAS_RICH = False
    print("rich library not found. Install with: pip install rich")
    print("Falling back to basic output...\n")

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', type=str, required=True, help='Directory containing character subdirectories with FBX files')
parser.add_argument('--output_dir', type=str, required=True, help='Output directory for processed data')
parser.add_argument('--max_log_lines', type=int, default=20, help='Maximum lines to show in subprocess output panel')
parser.add_argument('--n_cam', type=int, default=1, help='Number of cameras to create')
parser.add_argument('--follow_bone', type=str, default='mixamorig:Hips', help='Bone name for the camera to follow')
parser.add_argument('--cam_height', type=float, default=3.0, help='Camera height from the ground')
parser.add_argument('--cam_radius', type=float, default=4.5, help='Radius of camera circle around the character')
parser.add_argument('--img_width', type=int, default=1280, help='Image width')
parser.add_argument('--img_height', type=int, default=720, help='Image height')
parser.add_argument('--run_blender', action='store_true', default=False, help='Enable blender execution')
parser.add_argument('--run_projection', action='store_true', default=False, help='Enable 2D projection after rendering')
parser.add_argument('--use_gpu', action='store_true', default=False, help='Use GPU for rendering in Blender')
parser.add_argument('--only_body_joints', action='store_true', default=False, help='Only render body joints without the fingers')
parser.add_argument('--skip_plot_map', action='store_true', default=False, help='Skip plotting the 2D joint heatmap and skeleton overlay on the rendered images')
args = parser.parse_args()

def run_with_rich_ui(character_dirs):
    """Run processing with rich UI showing main progress and subprocess output."""
    console = Console()
    
    # Count total motion files across all characters
    total_motions = sum(len(glob.glob(os.path.join(d, "*.fbx"))) for d in character_dirs)
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        
        main_task = progress.add_task(
            "[cyan]Processing motion files...", 
            total=total_motions
        )
        
        motion_count = 0
        for char_idx, char_dir in enumerate(character_dirs, 1):
            char_name = os.path.basename(char_dir.rstrip('/'))
            fbx_files = glob.glob(os.path.join(char_dir, "*.fbx"))
            
            if not fbx_files:
                console.print(f"[yellow]⚠ No FBX files found in {char_dir}, skipping.")
                continue

            for motion_idx, motion_file in enumerate(fbx_files, 1):
                motion_count += 1
                motion_name = os.path.basename(motion_file)
                console.print(f"\n[bold cyan]→[/bold cyan] Processing [{motion_count}/{total_motions}]: {char_name}/{motion_name}")

                char_output_dir = os.path.join(args.output_dir, char_name)
                os.makedirs(char_output_dir, exist_ok=True)
                
                render_settings = f"--n_cam {args.n_cam} --follow_bone {args.follow_bone} --cam_height {args.cam_height} --cam_radius {args.cam_radius} --img_width {args.img_width} --img_height {args.img_height}"
                blender_cmd = f"python run.py --fbx \"{motion_file}\" --out_dir \"{char_output_dir}\" {render_settings}"
                blender_cmd += " --use_gpu" if args.use_gpu else ""
                blender_cmd += " --run_blender" if args.run_blender else ""
                blender_cmd += " --run_projection" if args.run_projection else ""
                blender_cmd += " --only_body_joints" if args.only_body_joints else ""
                blender_cmd += " --skip_plot_map" if args.skip_plot_map else ""
                print(f"Command: {blender_cmd}\n")
                # Update main progress
                progress.update(
                    main_task, 
                    description=f"[cyan]Processing [{motion_count}/{total_motions}]: {char_name}/{motion_name}"
                )
                
                # Create panel for subprocess output
                subprocess_lines = deque(maxlen=args.max_log_lines)
                
                with Live(console=console, refresh_per_second=4) as live:
                    # Run subprocess and capture output
                    process = subprocess.Popen(
                        blender_cmd,
                        shell=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        bufsize=1
                    )
                    
                    for line in process.stdout:
                        line = line.rstrip()
                        if line:
                            subprocess_lines.append(line)
                            
                            # Create panel with recent output
                            output_text = "\n".join(subprocess_lines)
                            panel = Panel(
                                output_text,
                                title=f"[bold green]{char_name}/{motion_name}[/bold green]",
                                subtitle=f"[dim]Showing last {min(len(subprocess_lines), args.max_log_lines)} lines[/dim]",
                                border_style="blue"
                            )
                            live.update(panel)
                    
                    process.wait()
                    
                    if process.returncode == 0:
                        console.print(f"[green]✓[/green] Completed: {char_name}/{motion_name}")
                    else:
                        console.print(f"[red]✗[/red] Failed: {char_name}/{motion_name} (exit code: {process.returncode})")
                
                progress.advance(main_task)
    
    console.print("\n[bold green]All motion files processed![/bold green]")


def run_basic(character_dirs):
    """Fallback to basic output without rich UI."""
    total_motions = sum(len(glob.glob(os.path.join(d, "*.fbx"))) for d in character_dirs)
    
    motion_count = 0
    for char_idx, char_dir in enumerate(character_dirs, 1):
        char_name = os.path.basename(char_dir.rstrip('/'))
        
        fbx_files = glob.glob(os.path.join(char_dir, "*.fbx"))
        
        if not fbx_files:
            print(f"⚠ No FBX files found in {char_dir}, skipping.")
            continue
            
        for motion_idx, motion_file in enumerate(fbx_files, 1):
            motion_count += 1
            motion_name = os.path.splitext(os.path.basename(motion_file))[0]
            
            print(f"\n{'='*60}")
            print(f"Processing [{motion_count}/{total_motions}]: {char_name}/{motion_name}")
            print('='*60)

            char_output_dir = os.path.join(args.output_dir, char_name)
            os.makedirs(char_output_dir, exist_ok=True)
            
            render_settings = f"--n_cam {args.n_cam} --follow_bone {args.follow_bone} --cam_height {args.cam_height} --cam_radius {args.cam_radius} --img_width {args.img_width} --img_height {args.img_height}"
            blender_cmd = f"python run.py --fbx {motion_file} --out_dir {char_output_dir} {render_settings}"
            blender_cmd += " --use_gpu" if args.use_gpu else ""
            blender_cmd += " --run_blender" if args.run_blender else ""
            blender_cmd += " --run_projection" if args.run_projection else ""
            blender_cmd += " --only_body_joints" if args.only_body_joints else ""
            blender_cmd += " --skip_plot_map" if args.skip_plot_map else ""
            print(f"Command: {blender_cmd}\n")
            
            result = os.system(blender_cmd)
            
            if result == 0:
                print(f"\n✓ Completed: {char_name}/{motion_name}")
            else:
                print(f"\n✗ Failed: {char_name}/{motion_name} (exit code: {result})")
    
    print(f"\n{'='*60}")
    print(f"All {motion_count} motion files processed!")
    print('='*60)


if __name__ == "__main__":
    character_dirs = glob.glob(f'{args.input_dir}/*/')
    
    if not character_dirs:
        print(f"No character directories found in {args.input_dir}")
        exit(1)
    
    total_motions = sum(len(glob.glob(os.path.join(d, "*.fbx"))) for d in character_dirs)
    print(f"Found {len(character_dirs)} character(s) with {total_motions} motion file(s) to process\n")
    
    if HAS_RICH:
        run_with_rich_ui(character_dirs)
    else:
        run_basic(character_dirs)