import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import requests
import time

def create_sample_data(filename="2024-2025.csv", games_per_player=20):
    """
    Create a sample NBA dataset with real active NBA players
    
    Args:
        filename: Output CSV file name
        games_per_player: Number of games per player
        
    Returns:
        DataFrame with the generated data
    """
    # NBA teams
    teams = ["LAL", "GSW", "BKN", "BOS", "MIL", "MIA", "CHI", "PHX", "DAL", 
             "PHI", "LAC", "NYK", "ATL", "DEN", "TOR", "UTA", "POR", "MEM",
             "NOP", "SAC", "SAS", "IND", "CHA", "WAS", "DET", "OKC", "ORL",
             "HOU", "CLE", "MIN"]
    
    # Fetch real NBA players
    print("Fetching active NBA players...")
    player_names = fetch_active_nba_players()
    if not player_names:
        print("Warning: Could not fetch active players. Using default list.")
        player_names = [
            "LeBron James", "Stephen Curry", "Kevin Durant", "Giannis Antetokounmpo",
            "Nikola Jokić", "Joel Embiid", "Luka Dončić", "Jayson Tatum", "Damian Lillard",
            "Ja Morant", "Zion Williamson", "Anthony Edwards", "Trae Young", "Bam Adebayo",
            "Jaylen Brown", "Domantas Sabonis", "Devin Booker", "Shai Gilgeous-Alexander"
        ]
    
    # Generate game dates
    end_date = datetime.now()
    start_date = end_date - timedelta(days=180)  # About 6 months of games
    date_range = pd.date_range(start=start_date, end=end_date, periods=50)
    game_dates = [d.strftime("%Y-%m-%d") for d in date_range]
    
    # Create data
    data = []
    
    print(f"Generating data for {len(player_names)} players...")
    for player in player_names:
        # Assign player to a team
        team = np.random.choice(teams)
        
        # Player's baseline stats with some randomness
        # Stars get higher stats
        is_star = player in ["LeBron James", "Stephen Curry", "Kevin Durant", "Giannis Antetokounmpo",
                           "Nikola Jokić", "Joel Embiid", "Luka Dončić", "Jayson Tatum", "Damian Lillard"]
        
        if is_star:
            base_pts = np.random.normal(25, 5)   # Stars score more
            base_mp = np.random.normal(34, 3)    # Stars play more minutes
        else:
            base_pts = np.random.normal(15, 8)   # Mean 15 points with std dev 8
            base_mp = np.random.normal(28, 6)    # Mean 28 minutes with std dev 6
        
        # Ensure base stats are positive
        base_pts = max(5, base_pts)
        base_mp = max(10, base_mp)
        
        # Generate games for this player
        for _ in range(games_per_player):
            # Choose a random game date
            game_date = np.random.choice(game_dates)
            
            # Choose opponent (not the player's team)
            other_teams = [t for t in teams if t != team]
            opponent = np.random.choice(other_teams)
            
            # Randomize stats around baseline with some game-to-game variation
            points = max(0, np.random.normal(base_pts, abs(base_pts)/4))  # More variance for higher scorers
            minutes = max(0, np.random.normal(base_mp, 5))
            
            # Generate other stats
            fg_made = max(0, int(points * 0.4))  # About 40% of points from field goals
            fg_att = max(fg_made, int(fg_made * (1 + np.random.normal(0.3, 0.1))))  # ~40-50% FG%
            
            fg3_made = max(0, int(points * 0.2 * np.random.normal(1, 0.3)))  # About 20% from 3s
            fg3_att = max(fg3_made, int(fg3_made * (1 + np.random.normal(0.5, 0.2))))  # ~33% 3P%
            
            ft_made = max(0, int((points - (fg_made * 2) - (fg3_made * 3))))  # Remaining points from FTs
            ft_att = max(ft_made, int(ft_made * (1 + np.random.normal(0.2, 0.1))))  # ~75-85% FT%
            
            # Calculate percentages (avoid division by zero)
            fg_pct = round(fg_made / max(1, fg_att), 3)
            fg3_pct = round(fg3_made / max(1, fg3_att), 3)
            ft_pct = round(ft_made / max(1, ft_att), 3)
            
            # Other box score stats
            rebounds = max(0, int(np.random.normal(base_pts/4, 3)))
            assists = max(0, int(np.random.normal(base_pts/5, 2)))
            steals = max(0, int(np.random.normal(1, 0.7)))
            blocks = max(0, int(np.random.normal(0.5, 0.5)))
            turnovers = max(0, int(np.random.normal(2, 1)))
            fouls = max(0, int(np.random.normal(2.5, 1.2)))
            
            # Split rebounds
            offensive_rebounds = int(rebounds * np.random.normal(0.3, 0.1))
            defensive_rebounds = rebounds - offensive_rebounds
            
            # Format opponent for the game log
            opp_format = f"@ {opponent}" if np.random.random() > 0.5 else f"vs {opponent}"
            
            # Add the game to the dataset
            game = {
                'Player': player,
                'Tm': team,
                'Data': game_date,
                'Opp': opp_format,
                'MP': str(int(minutes)) + ":" + str(int((minutes % 1) * 60)).zfill(2),
                'FG': fg_made,
                'FGA': fg_att,
                'FG%': fg_pct,
                '3P': fg3_made,
                '3PA': fg3_att,
                '3P%': fg3_pct,
                'FT': ft_made,
                'FTA': ft_att,
                'FT%': ft_pct,
                'ORB': offensive_rebounds,
                'DRB': defensive_rebounds,
                'REB': rebounds,
                'AST': assists,
                'STL': steals,
                'BLK': blocks,
                'TOV': turnovers,
                'PF': fouls,
                'PTS': int(points)
            }
            
            data.append(game)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Sort by player and date
    df = df.sort_values(['Player', 'Data'])
    
    # Save to CSV
    df.to_csv(filename, index=False)
    
    print(f"Sample NBA data created and saved to {filename}")
    print(f"{len(df)} game records for {len(player_names)} players")
    
    return df

def fetch_active_nba_players():
    """Fetch active NBA players from the NBA Stats API"""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Referer': 'https://www.nba.com/',
        'x-nba-stats-origin': 'stats',
        'x-nba-stats-token': 'true',
        'Accept-Language': 'en-US,en;q=0.9'
    }
    
    try:
        url = "https://stats.nba.com/stats/playerindex"
        params = {
            'LeagueID': '00',  # NBA
            'Season': '2023-24',  # Use last complete season
            'IsOnlyCurrentSeason': '1'
        }
        
        response = requests.get(url, headers=headers, params=params)
        
        if response.status_code == 200:
            data = response.json()
            headers = data['resultSets'][0]['headers']
            rows = data['resultSets'][0]['rowSet']
            
            # Create DataFrame
            players_df = pd.DataFrame(rows, columns=headers)
            
            # Create full names
            player_names = [f"{player['PLAYER_FIRST_NAME']} {player['PLAYER_LAST_NAME']}" 
                           for _, player in players_df.iterrows()]
            
            # Limit to a reasonable number of players (e.g., 100)
            if len(player_names) > 100:
                return player_names[:100]
            
            return player_names
        else:
            print(f"Error fetching players: {response.status_code}")
            return []
            
    except Exception as e:
        print(f"Error fetching players: {e}")
        return []

if __name__ == "__main__":
    create_sample_data()