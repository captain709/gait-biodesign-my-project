root="/mnt/ExpDrive/SparkLabLongRun/PeamProject/results"
 
# Check if root directory exists
if [ ! -d "$root" ]; then
    echo "Error: Root directory '$root' does not exist."
    exit 1
fi
 
# Get current date and time in format: YYYY-MM-DD_HH-MM-SS
timestamp=$(date +"%Y-%m-%d_%H-%M-%S")
 
# Define the main folder name with timestamp
main_folder="$root/$timestamp"
 
# Create the main folder structure
echo "Creating folder structure..."
mkdir -p "$main_folder/img/subject"
 
# Verify creation
if [ -d "$main_folder/img/subject" ]; then
    echo "✓ Successfully created folder structure:"
    echo "  Main folder: $main_folder"
    echo "  Subdirectories:"
    echo "    - $main_folder/img"
    echo "    - $main_folder/img/subject"
    echo ""
    echo "Timestamp: $timestamp"
else
    echo "Error: Failed to create folder structure."
    exit 1
fi
 
# Display the created structure
echo ""
echo "Folder tree:"
tree "$main_folder" 2>/dev/null || find "$main_folder" -type d | sed 's|[^/]*/| |g'