import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import heapq
from PIL import Image, ImageOps
import io

# Global state variables
if 'maze' not in st.session_state:
    st.session_state.maze = None
if 'start_point' not in st.session_state:
    st.session_state.start_point = (0, 0)
if 'end_point' not in st.session_state:
    st.session_state.end_point = None

def dijkstra(maze, start, end):
    if maze is None or start is None or end is None:
        return []
    
    rows, cols = maze.shape
    distances = np.full((rows, cols), np.inf)
    previous = np.full((rows, cols, 2), -1, dtype=int)
    distances[start] = 0

    pq = [(0, start)]
    moves = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # right, down, left, up

    while pq:
        current_dist, current = heapq.heappop(pq)
        current_r, current_c = current
        
        if current == end:
            break

        for dr, dc in moves:
            r, c = current_r + dr, current_c + dc
            if 0 <= r < rows and 0 <= c < cols and maze[r, c] == 0:  # 0 is path
                new_dist = current_dist + 1
                if new_dist < distances[r, c]:
                    distances[r, c] = new_dist
                    previous[r, c] = [current_r, current_c]
                    heapq.heappush(pq, (new_dist, (r, c)))

    # Reconstruct path
    path = []
    if distances[end] == np.inf:  # No path found
        return []
    
    current = end
    while current != start:
        path.append(current)
        prev_r, prev_c = previous[current]
        if prev_r == -1 and prev_c == -1:  # No path
            return []
        current = (prev_r, prev_c)
    
    path.append(start)
    path.reverse()
    return path

def create_empty_maze(rows, cols):
    """Create an empty maze filled with paths (0)"""
    return np.zeros((rows, cols), dtype=int)

def plot_maze_realistic(maze, path=None, start=None, end=None):
    """Plot maze with realistic walls"""
    if maze is None:
        return None
    
    rows, cols = maze.shape
    
    # Create a higher resolution grid for better visualization
    scale = 10  # Scale factor for higher resolution
    grid = np.zeros((rows * scale, cols * scale))
    
    # Fill in walls with thicker lines
    wall_thickness = scale // 3
    
    # Function to draw a wall segment
    def draw_wall(grid, r1, c1, r2, c2, thickness=wall_thickness):
        r1, c1, r2, c2 = r1*scale, c1*scale, r2*scale, c2*scale
        if r1 == r2:  # Horizontal wall
            for i in range(max(0, r1 - thickness//2), min(grid.shape[0], r1 + thickness//2 + 1)):
                for j in range(min(c1, c2), max(c1, c2) + 1):
                    if 0 <= i < grid.shape[0] and 0 <= j < grid.shape[1]:
                        grid[i, j] = 1
        else:  # Vertical wall
            for i in range(min(r1, r2), max(r1, r2) + 1):
                for j in range(max(0, c1 - thickness//2), min(grid.shape[1], c1 + thickness//2 + 1)):
                    if 0 <= i < grid.shape[0] and 0 <= j < grid.shape[1]:
                        grid[i, j] = 1
    
    # Draw outer boundary
    draw_wall(grid, 0, 0, 0, cols, thickness=wall_thickness*2)  # Top
    draw_wall(grid, rows, 0, rows, cols, thickness=wall_thickness*2)  # Bottom
    draw_wall(grid, 0, 0, rows, 0, thickness=wall_thickness*2)  # Left
    draw_wall(grid, 0, cols, rows, cols, thickness=wall_thickness*2)  # Right
    
    # Draw inner walls
    for r in range(rows):
        for c in range(cols):
            if maze[r, c] == 1:  # Wall
                # Check adjacent cells and draw walls accordingly
                if r > 0 and maze[r-1, c] == 1:  # Wall above
                    draw_wall(grid, r, c, r-1, c)
                if c > 0 and maze[r, c-1] == 1:  # Wall to the left
                    draw_wall(grid, r, c, r, c-1)
                if r < rows-1 and maze[r+1, c] == 1:  # Wall below
                    draw_wall(grid, r, c, r+1, c)
                if c < cols-1 and maze[r, c+1] == 1:  # Wall to the right
                    draw_wall(grid, r, c, r, c+1)
                
                # Always draw a block for isolated walls
                center_r, center_c = r*scale + scale//2, c*scale + scale//2
                block_size = scale // 2
                for i in range(center_r - block_size, center_r + block_size):
                    for j in range(center_c - block_size, center_c + block_size):
                        if 0 <= i < grid.shape[0] and 0 <= j < grid.shape[1]:
                            grid[i, j] = 1
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 12))
    
    # Plot the maze with a black (walls) and white (paths) colormap
    ax.imshow(grid, cmap='binary', interpolation='nearest')
    
    # Draw the path if provided
    if path and len(path) > 1:
        path_scaled = [(p[0]*scale + scale//2, p[1]*scale + scale//2) for p in path]
        path_y, path_x = zip(*path_scaled)
        ax.plot(path_x, path_y, 'r-', linewidth=3, alpha=0.7)
    
    # Mark start and end points
    if start:
        start_y, start_x = start[0]*scale + scale//2, start[1]*scale + scale//2
        ax.scatter(start_x, start_y, c='green', s=200, marker='o', edgecolor='black', linewidth=1.5)
    if end:
        end_y, end_x = end[0]*scale + scale//2, end[1]*scale + scale//2
        ax.scatter(end_x, end_y, c='blue', s=200, marker='X', edgecolor='black', linewidth=1.5)
    
    # Remove axes
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)
    plt.tight_layout()
    
    return fig

def convert_image_to_maze(uploaded_image):
    """Convert a black and white maze image to a binary numpy array"""
    img = Image.open(uploaded_image).convert('L')  # Convert to grayscale
    
    # Resize image if it's too large
    max_size = 50  # Maximum size in either dimension
    if img.width > max_size or img.height > max_size:
        ratio = min(max_size / img.width, max_size / img.height)
        new_width = int(img.width * ratio)
        new_height = int(img.height * ratio)
        img = img.resize((new_width, new_height), Image.LANCZOS)
    
    # Optional: enhance contrast
    img = ImageOps.autocontrast(img, cutoff=0.5)
    
    # Convert to numpy array and threshold to binary
    maze_array = np.array(img)
    # Threshold to binary (0 = path, 1 = wall)
    maze_array = (maze_array < 128).astype(int)
    
    return maze_array

# Safely parse CSV with error handling
def parse_maze_csv(uploaded_file):
    try:
        # First try to read it directly
        maze_df = pd.read_csv(uploaded_file)
        
        # Check if 'cell' column exists
        if 'cell' not in maze_df.columns:
            st.error("CSV format error: Missing 'cell' column")
            return None
        
        # Check for E, W, N, S columns
        required_cols = ['E', 'W', 'N', 'S']
        missing_cols = [col for col in required_cols if col not in maze_df.columns]
        if missing_cols:
            st.error(f"CSV format error: Missing columns: {', '.join(missing_cols)}")
            return None
        
        # Try to parse the cell column
        try:
            # Display first few rows for debugging
            st.write("First few rows of the CSV:")
            st.write(maze_df.head())
            
            # Clean cell column (handle various formats)
            # First, ensure cell is a string type
            maze_df['cell'] = maze_df['cell'].astype(str)
            
            # Remove quotes, parentheses, and extra spaces
            maze_df['cell'] = maze_df['cell'].str.replace('"', '').str.replace('(', '').str.replace(')', '').str.strip()
            
            # Split by comma and convert to tuple of integers
            maze_df['cell'] = maze_df['cell'].str.split(',').apply(
                lambda x: (int(x[0].strip()), int(x[1].strip())) if len(x) == 2 else None
            )
            
            # Check for any None values (parsing failures)
            if maze_df['cell'].isna().any():
                st.error("Error parsing cell coordinates in CSV")
                return None
            
            # Determine maze dimensions
            max_row = max(maze_df['cell'].apply(lambda x: x[0]))
            max_col = max(maze_df['cell'].apply(lambda x: x[1]))
            
            # Create empty maze
            maze = np.zeros((max_row, max_col), dtype=int)
            
            # Fill maze with walls based on E,W,N,S columns
            for _, row in maze_df.iterrows():
                r, c = row['cell']
                # If any direction has a wall (1), mark the cell as a wall
                if any([row['E'], row['W'], row['N'], row['S']]):
                    maze[r-1, c-1] = 1  # Adjust for 0-indexing
            
            return maze
            
        except Exception as e:
            st.error(f"Error parsing cell data: {e}")
            return None
            
    except Exception as e:
        st.error(f"Error reading CSV file: {e}")
        return None

# Streamlit app
st.title("Maze Solver with Dijkstra's Algorithm")

# Tabs for different maze input methods
tab1, tab2, tab3 = st.tabs(["Upload Image", "Create Maze", "Upload CSV"])

# Tab 1: Upload image of a maze
with tab1:
    st.header("Upload a Maze Image")
    st.write("Upload a black and white image of a maze. Black lines will be walls, white spaces will be paths.")
    
    uploaded_image = st.file_uploader("Upload maze image", type=["png", "jpg", "jpeg", "bmp"])
    
    if uploaded_image:
        maze = convert_image_to_maze(uploaded_image)
        st.session_state.maze = maze
        st.write(f"Maze dimensions: {maze.shape[0]} rows × {maze.shape[1]} columns")
        
        # Display the processed maze
        fig = plot_maze_realistic(maze)
        if fig:
            st.pyplot(fig)
            
        # Set default start and end points
        if st.session_state.start_point == (0, 0) or st.session_state.start_point[0] >= maze.shape[0] or st.session_state.start_point[1] >= maze.shape[1]:
            st.session_state.start_point = (0, 0)
        
        if st.session_state.end_point is None or st.session_state.end_point[0] >= maze.shape[0] or st.session_state.end_point[1] >= maze.shape[1]:
            st.session_state.end_point = (maze.shape[0]-1, maze.shape[1]-1)

# Tab 2: Create maze manually
with tab2:
    st.header("Create a Maze")
    
    # Maze dimensions
    col1, col2 = st.columns(2)
    with col1:
        rows = st.number_input("Rows", min_value=5, max_value=50, value=15)
    with col2:
        cols = st.number_input("Columns", min_value=5, max_value=50, value=20)
    
    if st.button("Create Empty Maze"):
        maze = create_empty_maze(rows, cols)
        st.session_state.maze = maze
        st.session_state.start_point = (0, 0)
        st.session_state.end_point = (rows-1, cols-1)
        st.success("Empty maze created!")
    
    # If maze exists, show editing tools
    if st.session_state.maze is not None and tab2._active:
        maze = st.session_state.maze
        
        st.subheader("Edit Maze")
        st.write("Click to toggle between wall and path at specific coordinates:")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            edit_row = st.number_input("Row", min_value=0, max_value=maze.shape[0]-1, value=0)
        with col2:
            edit_col = st.number_input("Column", min_value=0, max_value=maze.shape[1]-1, value=0)
        with col3:
            cell_type = st.selectbox("Type", ["Path (0)", "Wall (1)"])
        
        if st.button("Set Cell"):
            value = 1 if cell_type == "Wall (1)" else 0
            maze[edit_row, edit_col] = value
            st.session_state.maze = maze
            st.success(f"Cell ({edit_row}, {edit_col}) set to {cell_type}")
            
        # Show current maze
        fig = plot_maze_realistic(maze, start=st.session_state.start_point, end=st.session_state.end_point)
        if fig:
            st.pyplot(fig)

# Tab 3: Upload CSV
with tab3:
    st.header("Upload Maze CSV")
    
    # Sample CSV format explanation
    with st.expander("CSV Format Information"):
        st.write("""
        Your CSV file should have the following columns:
        - `cell`: Format should be like "(row, column)" or "row, column"
        - `E`, `W`, `N`, `S`: Binary indicators (0 or 1) for walls in East, West, North, South directions
        
        Example:
        ```
        cell,E,W,N,S
        "(1, 1)",1,0,0,0
        "(1, 2)",0,1,0,1
        ```
        """)
    
    uploaded_file = st.file_uploader("Upload maze CSV file", type=["csv"], key="csv_uploader")
    
    if uploaded_file:
        # Parse CSV with improved error handling
        maze = parse_maze_csv(uploaded_file)
        
        if maze is not None:
            st.session_state.maze = maze
            st.success(f"Maze loaded from CSV. Dimensions: {maze.shape[0]} rows × {maze.shape[1]} columns")
            
            # Set default start and end points
            st.session_state.start_point = (0, 0)
            st.session_state.end_point = (maze.shape[0]-1, maze.shape[1]-1)
            
            # Display the maze
            fig = plot_maze_realistic(maze)
            if fig:
                st.pyplot(fig)

# Common controls for all tabs
if st.session_state.maze is not None:
    st.header("Maze Navigation")
    
    # Set start and end points
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Start Point")
        start_row = st.number_input("Start Row", min_value=0, max_value=st.session_state.maze.shape[0]-1, 
                                   value=st.session_state.start_point[0], key="start_row")
        start_col = st.number_input("Start Column", min_value=0, max_value=st.session_state.maze.shape[1]-1, 
                                   value=st.session_state.start_point[1], key="start_col")
    
    with col2:
        st.subheader("End Point")
        end_row = st.number_input("End Row", min_value=0, max_value=st.session_state.maze.shape[0]-1, 
                                 value=st.session_state.end_point[0] if st.session_state.end_point else 0, key="end_row")
        end_col = st.number_input("End Column", min_value=0, max_value=st.session_state.maze.shape[1]-1, 
                                 value=st.session_state.end_point[1] if st.session_state.end_point else 0, key="end_col")
    
    if st.button("Update Start/End Points"):
        # Ensure start and end points are on paths, not walls
        maze = st.session_state.maze
        if maze[start_row, start_col] == 1:
            st.warning("Start point is on a wall. It will be cleared to create a path.")
            maze[start_row, start_col] = 0
            
        if maze[end_row, end_col] == 1:
            st.warning("End point is on a wall. It will be cleared to create a path.")
            maze[end_row, end_col] = 0
            
        st.session_state.start_point = (start_row, start_col)
        st.session_state.end_point = (end_row, end_col)
        st.session_state.maze = maze
        st.success("Start and End points updated!")
        
    # Solve maze
    st.header("Solve Maze")
    if st.button("Find Path using Dijkstra"):
        if st.session_state.maze is None:
            st.error("Please create or upload a maze first.")
        elif st.session_state.end_point is None:
            st.error("Please set an end point first.")
        else:
            path = dijkstra(st.session_state.maze, st.session_state.start_point, st.session_state.end_point)
            if path:
                st.success(f"Path found! Length = {len(path)}")
                fig = plot_maze_realistic(st.session_state.maze, path=path, 
                               start=st.session_state.start_point, end=st.session_state.end_point)
                if fig:
                    st.pyplot(fig)
            else:
                st.error("No path found between start and end points.")
                fig = plot_maze_realistic(st.session_state.maze, 
                               start=st.session_state.start_point, end=st.session_state.end_point)
                if fig:
                    st.pyplot(fig)
    
    # Download option
    st.header("Download Maze")
    
    if st.button("Generate CSV"):
        maze = st.session_state.maze
        rows, cols = maze.shape
        
        # Create data for CSV
        data = []
        for r in range(rows):
            for c in range(cols):
                if maze[r, c] == 1:  # Wall
                    data.append({
                        'cell': f"({r+1}, {c+1})",
                        'E': 1, 'W': 1, 'N': 1, 'S': 1  # Mark all directions as walls
                    })
        
        # Convert to DataFrame and then to CSV
        if data:
            df = pd.DataFrame(data)
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Maze as CSV", data=csv, file_name="maze.csv", mime="text/csv")
        else:
            st.warning("Current maze has no walls to export.")
            
    # Add option to download as image
    if st.button("Generate Image"):
        fig = plot_maze_realistic(st.session_state.maze, path=None,
                       start=st.session_state.start_point, end=st.session_state.end_point)
        if fig:
            # Save figure to BytesIO buffer
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=300, bbox_inches='tight')
            buf.seek(0)
            
            # Download button for the image
            st.download_button("Download Maze as Image", data=buf, file_name="maze.png", mime="image/png")