import requests
import pandas as pd
import json
import time
from datetime import datetime

class NBAStatsAPI:
    """
    Class to interact with the NBA Stats API to get the latest player data
    """
    def __init__(self):
        self.base_url = "https://stats.nba.com/stats"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Referer': 'https://www.nba.com/',
            'x-nba-stats-origin': 'stats',
            'x-nba-stats-token': 'true',
            'Accept-Language': 'en-US,en;q=0.9'
        }
        self.player_cache = {}  # Cache player IDs and info
        
    def get_player_id(self, player_name):
        """
        Get the NBA.com player ID for a given player name
        
        Args:
            player_name: The name of the player (e.g., "LeBron James")
            
        Returns:
            Player ID (int) or None if not found
        """
        # Check cache first
        if player_name in self.player_cache:
            return self.player_cache[player_name]['id']
            
        # Fetch all players
        url = f"{self.base_url}/playerindex"
        params = {
            'LeagueID': '00',  # NBA League ID
            'Season': self._get_current_season(),
            'IsOnlyCurrentSeason': '1'
        }
        
        try:
            response = requests.get(url, headers=self.headers, params=params)
            
            if response.status_code == 200:
                data = response.json()
                headers = data['resultSets'][0]['headers']
                rows = data['resultSets'][0]['rowSet']
                
                # Convert to DataFrame
                players_df = pd.DataFrame(rows, columns=headers)
                
                # Search for player name
                player_name_lower = player_name.lower()
                for _, player in players_df.iterrows():
                    full_name = f"{player['PLAYER_FIRST_NAME']} {player['PLAYER_LAST_NAME']}".lower()
                    if player_name_lower in full_name or full_name in player_name_lower:
                        player_id = player['PERSON_ID']
                        
                        # Cache this player
                        self.player_cache[player_name] = {
                            'id': player_id,
                            'team_id': player['TEAM_ID'],
                            'team_name': player['TEAM_NAME'],
                            'team_abbreviation': player['TEAM_ABBREVIATION']
                        }
                        
                        return player_id
                
                print(f"Player '{player_name}' not found in active NBA players")
                
                # Try with a more fuzzy search
                best_match = None
                highest_similarity = 0
                
                for _, player in players_df.iterrows():
                    full_name = f"{player['PLAYER_FIRST_NAME']} {player['PLAYER_LAST_NAME']}".lower()
                    name_parts = player_name_lower.split()
                    
                    # Check if any part of the search name is in the player's name
                    for part in name_parts:
                        if part in full_name and len(part) > 2:  # Avoid matching short parts like "de" or "la"
                            similarity = len(part) / len(full_name)
                            if similarity > highest_similarity:
                                highest_similarity = similarity
                                best_match = player
                
                if best_match is not None and highest_similarity > 0.2:
                    player_id = best_match['PERSON_ID']
                    matched_name = f"{best_match['PLAYER_FIRST_NAME']} {best_match['PLAYER_LAST_NAME']}"
                    print(f"Using closest match: '{matched_name}' for '{player_name}'")
                    
                    # Cache this player
                    self.player_cache[player_name] = {
                        'id': player_id,
                        'team_id': best_match['TEAM_ID'],
                        'team_name': best_match['TEAM_NAME'],
                        'team_abbreviation': best_match['TEAM_ABBREVIATION']
                    }
                    
                    return player_id
                
                return None
            else:
                print(f"Error fetching player list: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"Error fetching player ID: {str(e)}")
            return None
            
    def get_player_game_logs(self, player_name, season_type="Regular Season", season=None):
        """
        Get game logs for a specific player
        
        Args:
            player_name: Player name
            season_type: "Regular Season", "Playoffs", or "All Star"
            season: Season in format YYYY-YY (e.g., "2023-24"), None for current season
            
        Returns:
            DataFrame with player game logs or None if not found
        """
        player_id = self.get_player_id(player_name)
        
        if not player_id:
            print(f"Could not find NBA player ID for '{player_name}'")
            return None
            
        if not season:
            season = self._get_current_season()
            
        # Endpoint for game logs
        url = f"{self.base_url}/playergamelog"
        
        params = {
            'PlayerID': player_id,
            'Season': season,
            'SeasonType': season_type
        }
        
        try:
            response = requests.get(url, headers=self.headers, params=params)
            
            if response.status_code == 200:
                data = response.json()
                headers = data['resultSets'][0]['headers']
                rows = data['resultSets'][0]['rowSet']
                
                # Check if we have data
                if not rows:
                    print(f"No game logs found for {player_name} in {season} {season_type}")
                    return None
                
                # Convert to DataFrame
                df = pd.DataFrame(rows, columns=headers)
                
                # Add some convenience columns
                df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
                
                return df
            else:
                print(f"Error fetching game logs: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"Error fetching player game logs: {str(e)}")
            return None
            
    def convert_to_prediction_format(self, game_logs, player_name):
        """
        Convert NBA Stats API game logs to the format used by our prediction system
        
        Args:
            game_logs: DataFrame from get_player_game_logs
            player_name: Player name
            
        Returns:
            DataFrame in the format used by the prediction system
        """
        if game_logs is None or game_logs.empty:
            return None
            
        # Create new DataFrame
        prediction_data = pd.DataFrame()
        
        # Get player team info
        team_abbrev = self.player_cache.get(player_name, {}).get('team_abbreviation', '')
        
        # Map columns
        prediction_data['Player'] = player_name
        prediction_data['Tm'] = team_abbrev
        prediction_data['Data'] = game_logs['GAME_DATE'].dt.strftime('%Y-%m-%d')
        
        # Handle matchups (format: "vs LAL" or "@ LAL")
        prediction_data['Opp'] = game_logs.apply(
            lambda row: f"@ {row['MATCHUP'].split()[-1]}" if '@' in row['MATCHUP'] else f"vs {row['MATCHUP'].split()[-1]}",
            axis=1
        )
        
        # Convert minutes (format: "MM:SS")
        prediction_data['MP'] = game_logs['MIN']
        
        # Add other stats
        prediction_data['FG'] = game_logs['FGM']
        prediction_data['FGA'] = game_logs['FGA']
        prediction_data['FG%'] = game_logs.apply(lambda row: row['FGM'] / row['FGA'] if row['FGA'] > 0 else 0, axis=1)
        
        prediction_data['3P'] = game_logs['FG3M']
        prediction_data['3PA'] = game_logs['FG3A']
        prediction_data['3P%'] = game_logs.apply(lambda row: row['FG3M'] / row['FG3A'] if row['FG3A'] > 0 else 0, axis=1)
        
        prediction_data['FT'] = game_logs['FTM']
        prediction_data['FTA'] = game_logs['FTA']
        prediction_data['FT%'] = game_logs.apply(lambda row: row['FTM'] / row['FTA'] if row['FTA'] > 0 else 0, axis=1)
        
        # Rebounds
        prediction_data['ORB'] = game_logs['OREB']
        prediction_data['DRB'] = game_logs['DREB']
        prediction_data['REB'] = game_logs['REB']
        
        # Other box score stats
        prediction_data['AST'] = game_logs['AST']
        prediction_data['STL'] = game_logs['STL']
        prediction_data['BLK'] = game_logs['BLK']
        prediction_data['TOV'] = game_logs['TOV']
        prediction_data['PF'] = game_logs['PF']
        prediction_data['PTS'] = game_logs['PTS']
        
        return prediction_data
    
    def _get_current_season(self):
        """Get current NBA season in format YYYY-YY"""
        now = datetime.now()
        year = now.year
        month = now.month
        
        # NBA season typically starts in October
        # If it's between January and September, we're in the previous season
        if 1 <= month <= 9:
            return f"{year-1}-{str(year)[2:]}"
        else:
            return f"{year}-{str(year+1)[2:]}"