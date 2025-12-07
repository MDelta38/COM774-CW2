import urllib.request
import re

def messageDecoder():
    url = input("Enter Google Docs URL: ").strip()
    
    try:
        # Get the doc
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        html = urllib.request.urlopen(req).read().decode('utf-8')
        
        # Remove markup junk to get the co-ordinate raw data
        text = re.sub(r'<[^>]+>', ' ', html)  # Remove tags
        text = re.sub(r'\s+', ' ', text)      
        
        # Find pattern x, (unicode), y
        matches = re.findall(r'(\d+)\s*([█░])\s*(\d+)', text) 
        
        if not matches:
            print("No data")
            return
        
        # Convert to points, remove duplicates
        points = list(set((int(x), char, int(y)) for x, char, y in matches))
        
        # Get grid size
        MaxX = max(x for x, _, _ in points)
        MaxY = max(y for _, _, y in points)
        
        print(f"Grid: {MaxX + 1}x{MaxY + 1}")
        
        # Build and display grid
        grid = [[' '] * (MaxX + 1) for _ in range(MaxY + 1)]
        for x, char, y in points:
            grid[y][x] = char
        
        print("\nSecret Message:")
        for row in grid:
            print(''.join(row))
            
    except Exception as e:
        print(f"Error: {e}")



if __name__ == "__main__":
    messageDecoder()