import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import json
from datetime import datetime, timedelta
import time
import traceback
import os

class NBADataEnhancer:
    """
    Enhanced NBA data fetcher using real-time data from external sources
    """
    def __init__(self, existing_df=None):
        """Initialize with existing dataframe if available"""
        self.existing_df = existing_df
        self.injury_data = {}
        self.team_schedule = {}
        self.player_stats_cache = {}
        self.player_database = set()
        
        # Build player database for better matching
        if existing_df is not None and 'Player' in existing_df.columns:
            self.player_database = set(existing_df['Player'].unique())
            
        # Initialize cache
        self._initialize_cache()
    
    def _initialize_cache(self):
        """Initialize cache with common data to reduce API calls"""
        # Get team abbreviations and mappings
        self.team_mapping = self.get_team_abbreviations()
        
        # Pre-fetch injury data
        self.injury_data = self.get_current_injuries()
    
    def get_team_abbreviations(self):
        """Get NBA team abbreviations and map them to IDs"""
        # Standard NBA team abbreviations
        team_mapping = {
            "ATL": 1, "BOS": 2, "BKN": 3, "CHA": 4, "CHI": 5,
            "CLE": 6, "DAL": 7, "DEN": 8, "DET": 9, "GSW": 10,
            "HOU": 11, "IND": 12, "LAC": 13, "LAL": 14, "MEM": 15,
            "MIA": 16, "MIL": 17, "MIN": 18, "NOP": 19, "NYK": 20,
            "OKC": 21, "ORL": 22, "PHI": 23, "PHX": 24, "POR": 25,
            "SAC": 26, "SAS": 27, "TOR": 28, "UTA": 29, "WAS": 30
        }
        
        # Try to get more up-to-date information from balldontlie API
        try:
            response = requests.get("https://api.balldontlie.io/v1/teams", timeout=5)
            if response.status_code == 200:
                teams_data = response.json()['data']
                api_mapping = {team['abbreviation']: team['id'] for team in teams_data}
                team_mapping.update(api_mapping)
                print(f"Successfully fetched team data from API: {len(teams_data)} teams")
        except Exception as e:
            print(f"Using pre-defined team mapping due to API error: {e}")
        
        return team_mapping
    
    def get_current_injuries(self):
        """Get current NBA injury data from ESPN or other sources"""
        injury_data = {}
        
        try:
            # Try ESPN API first
            response = requests.get(
                "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries",
                timeout=5
            )
            
            if response.status_code == 200:
                data = response.json()
                
                for team in data.get('injuries', []):
                    team_abbrev = team.get('team', {}).get('abbreviation')
                    if not team_abbrev:
                        continue
                    
                    team_injuries = []
                    for player in team.get('injuries', []):
                        player_info = {
                            'name': player.get('athlete', {}).get('displayName'),
                            'position': player.get('athlete', {}).get('position', {}).get('abbreviation'),
                            'status': player.get('status'),
                            'description': player.get('longDescription', ''),
                            'date': player.get('date')
                        }
                        team_injuries.append(player_info)
                    
                    injury_data[team_abbrev] = team_injuries
                
                print(f"Successfully fetched injury data: {sum(len(inj) for inj in injury_data.values())} injuries")
                return injury_data
            
            # If ESPN API fails, try scraping an alternative source
            else:
                print(f"ESPN API returned status {response.status_code}, trying alternative source")
                
                # Example of scraping basketball-reference or another site
                # This would need to be customized based on the site structure
                alt_url = "https://www.basketball-reference.com/friv/injuries.html"
                response = requests.get(alt_url, timeout=5)
                
                if response.status_code == 200:
                    soup = BeautifulSoup(response.text, 'html.parser')
                    injury_table = soup.find('table', {'id': 'injuries'})
                    
                    if injury_table:
                        rows = injury_table.find_all('tr')[1:]  # Skip header
                        
                        current_team = None
                        for row in rows:
                            # Check if this is a team header row
                            team_header = row.find('th', {'class': 'over_header'})
                            if team_header:
                                current_team = team_header.text.strip()
                                # Convert team name to abbreviation
                                for abbr, name in self.team_name_mapping.items():
                                    if name in current_team:
                                        current_team = abbr
                                        break
                                
                                if current_team not in injury_data:
                                    injury_data[current_team] = []
                                continue
                            
                            # Regular player injury row
                            cols = row.find_all('td')
                            if cols and current_team:
                                player_name = cols[0].text.strip()
                                injury_type = cols[1].text.strip()
                                status = cols[2].text.strip()
                                
                                injury_data[current_team].append({
                                    'name': player_name,
                                    'description': injury_type,
                                    'status': status,
                                    'date': datetime.now().strftime('%Y-%m-%d')
                                })
                    
                    print(f"Scraped injury data from alternative source: {sum(len(inj) for inj in injury_data.values())} injuries")
                    return injury_data
        
        except Exception as e:
            print(f"Error fetching injury data: {e}")
        
        return injury_data
    
    def get_player_recent_stats(self, player_name, n_games=5):
        """Get player's recent game statistics from NBA.com or other sources"""
        # Check cache first
        if player_name in self.player_stats_cache:
            return self.player_stats_cache[player_name]
        
        player_stats = []
        
        try:
            # Try balldontlie API first
            search_url = f"https://api.balldontlie.io/v1/players?search={player_name}"
            response = requests.get(search_url)
            
            if response.status_code == 200:
                players = response.json()['data']
                
                if not players:
                    print(f"No players found matching '{player_name}'")
                else:
                    # Find best match
                    player_id = players[0]['id']
                    player_team = players[0]['team']['abbreviation']
                    
                    # Fetch stats for this player
                    stats_url = f"https://api.balldontlie.io/v1/stats?player_ids[]={player_id}&per_page={n_games}"
                    stats_response = requests.get(stats_url)
                    
                    if stats_response.status_code == 200:
                        stats_data = stats_response.json()['data']
                        
                        # Convert to our format
                        for stat in stats_data:
                            game_stats = {
                                'date': stat['game']['date'][:10],  # YYYY-MM-DD format
                                'team': stat['team']['abbreviation'],
                                'opponent': stat['game']['home_team_id'] == stat['team']['id'] 
                                          and stat['game']['visitor_team']['abbreviation']
                                          or stat['game']['home_team']['abbreviation'],
                                'is_home': stat['game']['home_team_id'] == stat['team']['id'],
                                'min': stat.get('min', '0'),
                                'pts': stat.get('pts', 0),
                                'fg_pct': stat.get('fg_pct', 0),
                                'fg3_pct': stat.get('fg3_pct', 0),
                                'ft_pct': stat.get('ft_pct', 0),
                                'reb': stat.get('reb', 0),
                                'ast': stat.get('ast', 0),
                                'stl': stat.get('stl', 0),
                                'blk': stat.get('blk', 0),
                                'turnover': stat.get('turnover', 0),
                            }
                            player_stats.append(game_stats)
            
            # If balldontlie API fails or returns no data, try an alternative source
            if not player_stats:
                print(f"No stats found from primary API, trying alternative source")
                
                # Alternative: Scrape from basketball-reference
                # Convert player name to basketball-reference format (lowercase, first 5 letters of last name + first 2 letters of first name)
                player_parts = player_name.split(' ')
                if len(player_parts) >= 2:
                    last_name = player_parts[-1].lower()
                    first_name = player_parts[0].lower()
                    player_url_id = f"{last_name[:5]}{first_name[:2]}01"  # Adding "01" for first player with this pattern
                    
                    url = f"https://www.basketball-reference.com/players/{last_name[0]}/{player_url_id}/gamelog/2025"
                    response = requests.get(url)
                    
                    if response.status_code == 200:
                        soup = BeautifulSoup(response.text, 'html.parser')
                        game_log_table = soup.find('table', {'id': 'pgl_basic'})
                        
                        if game_log_table:
                            rows = game_log_table.find('tbody').find_all('tr')
                            
                            for row in rows:
                                if 'thead' in row.get('class', []):
                                    continue
                                    
                                if not row.find('td'):  # Skip rows without data cells
                                    continue
                                    
                                # Extract game stats
                                date_cell = row.find('td', {'data-stat': 'date_game'})
                                if not date_cell:
                                    continue
                                    
                                date = date_cell.text.strip()
                                if not date:
                                    continue
                                    
                                game_stats = {
                                    'date': datetime.strptime(date, '%Y-%m-%d').strftime('%Y-%m-%d'),
                                    'team': row.find('td', {'data-stat': 'team_id'}).text.strip(),
                                    'opponent': row.find('td', {'data-stat': 'opp_id'}).text.strip(),
                                    'is_home': row.find('td', {'data-stat': 'game_location'}).text.strip() != '@',
                                    'min': row.find('td', {'data-stat': 'mp'}).text.strip(),
                                    'pts': int(row.find('td', {'data-stat': 'pts'}).text.strip() or 0),
                                    'fg_pct': float(row.find('td', {'data-stat': 'fg_pct'}).text.strip() or 0),
                                    'fg3_pct': float(row.find('td', {'data-stat': 'fg3_pct'}).text.strip() or 0),
                                    'ft_pct': float(row.find('td', {'data-stat': 'ft_pct'}).text.strip() or 0),
                                    'reb': int(row.find('td', {'data-stat': 'trb'}).text.strip() or 0),
                                    'ast': int(row.find('td', {'data-stat': 'ast'}).text.strip() or 0),
                                    'stl': int(row.find('td', {'data-stat': 'stl'}).text.strip() or 0),
                                    'blk': int(row.find('td', {'data-stat': 'blk'}).text.strip() or 0),
                                    'turnover': int(row.find('td', {'data-stat': 'tov'}).text.strip() or 0),
                                }
                                player_stats.append(game_stats)
                            
                            print(f"Fetched {len(player_stats)} games from alternative source")
        except Exception as e:
            print(f"Error fetching player stats: {e}")
            traceback.print_exc()
        
        # Cache results
        self.player_stats_cache[player_name] = player_stats
        return player_stats
    
    def get_team_schedule(self, team_abbrev, days=14):
        """Get team's upcoming and recent games"""
        # Check cache first
        if team_abbrev in self.team_schedule:
            return self.team_schedule[team_abbrev]
        
        team_id = self.team_mapping.get(team_abbrev)
        if not team_id:
            print(f"Team abbreviation '{team_abbrev}' not found")
            return []
        
        try:
            # Get today's date and date range for schedule
            today = datetime.now()
            start_date = (today - timedelta(days=days//2)).strftime("%Y-%m-%d")
            end_date = (today + timedelta(days=days//2)).strftime("%Y-%m-%d")
            
            # Try balldontlie API
            response = requests.get(
                f"https://api.balldontlie.io/v1/games",
                params={
                    "team_ids[]": team_id,
                    "start_date": start_date,
                    "end_date": end_date,
                    "per_page": 100
                }
            )
            
            if response.status_code == 200:
                games_data = response.json()['data']
                
                # Process games data
                games = []
                for game in games_data:
                    game_info = {
                        'date': game['date'][:10],  # YYYY-MM-DD format
                        'home_team': game['home_team']['abbreviation'],
                        'away_team': game['visitor_team']['abbreviation'],
                        'is_home': game['home_team']['abbreviation'] == team_abbrev,
                        'status': game['status'],
                        'time': game.get('time', '')
                    }
                    games.append(game_info)
                
                # Sort by date
                games.sort(key=lambda x: x['date'])
                self.team_schedule[team_abbrev] = games
                return games
                
            else:
                print(f"Error fetching schedule: {response.status_code}")
                
                # Try alternative source - ESPN
                espn_team_id = self.get_espn_team_id(team_abbrev)
                if espn_team_id:
                    url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/teams/{espn_team_id}/schedule"
                    response = requests.get(url)
                    
                    if response.status_code == 200:
                        events = response.json().get('events', [])
                        
                        games = []
                        for event in events:
                            date = event.get('date')
                            if date:
                                date = date[:10]  # YYYY-MM-DD format
                                
                                # Filter to requested date range
                                if start_date <= date <= end_date:
                                    competitions = event.get('competitions', [])
                                    if competitions:
                                        competition = competitions[0]
                                        competitors = competition.get('competitors', [])
                                        
                                        if len(competitors) == 2:
                                            home_team = None
                                            away_team = None
                                            is_home = False
                                            
                                            for competitor in competitors:
                                                if competitor.get('homeAway') == 'home':
                                                    home_team = competitor.get('team', {}).get('abbreviation')
                                                else:
                                                    away_team = competitor.get('team', {}).get('abbreviation')
                                                
                                                if competitor.get('homeAway') == 'home' and competitor.get('team', {}).get('abbreviation') == team_abbrev:
                                                    is_home = True
                                            
                                            if home_team and away_team:
                                                games.append({
                                                    'date': date,
                                                    'home_team': home_team,
                                                    'away_team': away_team,
                                                    'is_home': is_home,
                                                    'status': competition.get('status', {}).get('type', {}).get('name', ''),
                                                    'time': competition.get('status', {}).get('type', {}).get('shortDetail', '')
                                                })
                        
                        # Sort by date
                        games.sort(key=lambda x: x['date'])
                        self.team_schedule[team_abbrev] = games
                        return games
        
        except Exception as e:
            print(f"Error fetching schedule: {e}")
        
        return []
    
    def get_espn_team_id(self, team_abbrev):
        """Get ESPN team ID from abbreviation"""
        # Mapping of NBA abbreviations to ESPN team IDs
        espn_mapping = {
            "ATL": 1, "BOS": 2, "BKN": 17, "CHA": 30, "CHI": 4,
            "CLE": 5, "DAL": 6, "DEN": 7, "DET": 8, "GSW": 9,
            "HOU": 10, "IND": 11, "LAC": 12, "LAL": 13, "MEM": 29,
            "MIA": 14, "MIL": 15, "MIN": 16, "NOP": 3, "NYK": 18,
            "OKC": 25, "ORL": 19, "PHI": 20, "PHX": 21, "POR": 22,
            "SAC": 23, "SAS": 24, "TOR": 28, "UTA": 26, "WAS": 27
        }
        
        return espn_mapping.get(team_abbrev)
    
    def is_player_injured(self, player_name):
        """Check if a player is currently injured"""
        if not self.injury_data:
            self.get_current_injuries()
        
        # Search for player across all teams
        for team, injuries in self.injury_data.items():
            for player in injuries:
                if player['name'].lower() == player_name.lower():
                    return player
        
        return None
    
    def get_team_injuries(self, team_abbrev):
        """Get all injuries for a team"""
        if not self.injury_data:
            self.get_current_injuries()
        
        return self.injury_data.get(team_abbrev, [])
    
    def get_team_back_to_back_status(self, team_abbrev):
        """Check if team is on a back-to-back game"""
        if team_abbrev not in self.team_schedule:
            self.get_team_schedule(team_abbrev)
        
        schedule = self.team_schedule.get(team_abbrev, [])
        if not schedule:
            return False
        
        # Find the most recent and next game
        today = datetime.now().strftime("%Y-%m-%d")
        
        # Find most recent game
        past_games = [g for g in schedule if g['date'] <= today]
        if not past_games:
            return False
        
        most_recent = max(past_games, key=lambda x: x['date'])
        most_recent_date = datetime.strptime(most_recent['date'], "%Y-%m-%d")
        
        # Find next game
        upcoming_games = [g for g in schedule if g['date'] >= today]
        if not upcoming_games:
            return False
        
        next_game = min(upcoming_games, key=lambda x: x['date'])
        next_game_date = datetime.strptime(next_game['date'], "%Y-%m-%d")
        
        # Check if the most recent game was yesterday
        yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        
        # Check if next game is today and most recent was yesterday
        return (most_recent['date'] == yesterday and next_game['date'] == today)
    
    def enhance_prediction_data(self, player_name, opponent_team):
        """
        Gather all API data that could enhance the prediction model
        Returns a dictionary of features
        """
        enhanced_features = {}
        
        # Get player's team
        player_team = self._get_player_team(player_name)
        enhanced_features['team'] = player_team
        
        # Get player injury status
        injury_status = self.is_player_injured(player_name)
        enhanced_features['is_injured'] = 1 if injury_status else 0
        enhanced_features['injury_status'] = injury_status['status'] if injury_status else 'Active'
        
        # Get team's injury status (percentage of team injured)
        if player_team:
            team_injuries = self.get_team_injuries(player_team)
            enhanced_features['team_injury_count'] = len(team_injuries)
            
            # Get opponent injuries
            opp_injuries = self.get_team_injuries(opponent_team)
            enhanced_features['opp_injury_count'] = len(opp_injuries)
        
        # Check back-to-back status
        if player_team:
            enhanced_features['is_back_to_back'] = 1 if self.get_team_back_to_back_status(player_team) else 0
            enhanced_features['opp_is_back_to_back'] = 1 if self.get_team_back_to_back_status(opponent_team) else 0
        
        # Get recent player performance
        recent_stats = self.get_player_recent_stats(player_name)
        if recent_stats:
            # Calculate recency-weighted averages
            weights = [1.0, 0.9, 0.8, 0.7, 0.6]  # More weight to recent games
            stats_to_average = ['pts', 'min', 'fg_pct', 'fg3_pct', 'ft_pct', 'reb', 'ast', 'stl', 'blk']
            
            for stat in stats_to_average:
                values = [game.get(stat, 0) for game in recent_stats[:min(5, len(recent_stats))]]
                if values:
                    # Apply weights up to available games
                    w = weights[:len(values)]
                    weighted_avg = sum(v * w[i] for i, v in enumerate(values)) / sum(w[:len(values)])
                    enhanced_features[f'recent_weighted_{stat}'] = weighted_avg
        
        return enhanced_features
    
    def _get_player_team(self, player_name):
        """Determine player's team based on recent stats"""
        recent_stats = self.get_player_recent_stats(player_name)
        if recent_stats and len(recent_stats) > 0:
            return recent_stats[0]['team']
        
        # If we can't determine from recent stats, try to find in our existing data
        if self.existing_df is not None and 'Player' in self.existing_df.columns and 'Tm' in self.existing_df.columns:
            player_data = self.existing_df[self.existing_df['Player'] == player_name]
            if len(player_data) > 0:
                return player_data['Tm'].iloc[0]
        
        return None
    
    def check_if_playing_now(self, player_name, team):
        """Check if player is currently in a live game"""
        try:
            # Get current live games
            response = requests.get("https://api.balldontlie.io/v1/games", 
                                  params={"per_page": 15, "start_date": datetime.now().strftime("%Y-%m-%d")})
                                  
            if response.status_code == 200:
                games = response.json()['data']
                
                # Filter only live games
                live_games = [g for g in games if g.get('status') == 'in progress']
                
                # Check if player's team is in a live game
                for game in live_games:
                    home_team = game['home_team']['abbreviation']
                    away_team = game['visitor_team']['abbreviation']
                    
                    if home_team == team or away_team == team:
                        return {
                            'game_id': game['id'],
                            'home_team': home_team,
                            'away_team': away_team,
                            'is_home': home_team == team
                        }
                        
            return None
        except Exception as e:
            print(f"Error checking live games: {str(e)}")
            return None
    
    def get_live_game_stats(self, player_name, game_info):
        """Get live game stats for a player - uses real API when possible"""
        try:
            # Try to get real-time stats from NBA API
            game_id = game_info.get('game_id')
            
            if game_id:
                # Try balldontlie API first
                stats_url = f"https://api.balldontlie.io/v1/stats?game_ids[]={game_id}&player_names[]={player_name}"
                response = requests.get(stats_url)
                
                if response.status_code == 200:
                    stats_data = response.json()['data']
                    
                    if stats_data:
                        player_stats = stats_data[0]
                        
                        return {
                            "points": player_stats.get('pts', 0),
                            "minutes_played": self._convert_min_string(player_stats.get('min', '0')),
                            "quarter": game_info.get('period', 1),
                            "elapsed_game_time": self._estimate_elapsed_time(player_stats.get('min', '0'), game_info.get('period', 1)),
                            "fouls": player_stats.get('pf', 0),
                            "fg_pct": player_stats.get('fg_pct', 0),
                            "score_difference": self._calculate_score_diff(game_id, game_info.get('home_team'), game_info.get('away_team'), game_info.get('is_home'))
                        }
            
            # If we can't get real-time data, use a more accurate simulation
            print("Using simulated live game stats (API data not available)")
            
            # Use time-based simulation for more realistic values
            current_hour = datetime.now().hour
            current_minute = datetime.now().minute
            
            # Determine which quarter based on time
            # For simulation, assume each quarter is 15 minutes of real time
            minutes_elapsed = (current_hour % 4) * 60 + current_minute
            game_progress = min(1.0, max(0.1, minutes_elapsed / 240))  # 4 hours to complete a game
            
            # Get player's season average points
            avg_pts = 0
            if self.existing_df is not None and 'Player' in self.existing_df.columns and 'PTS' in self.existing_df.columns:
                player_data = self.existing_df[self.existing_df['Player'] == player_name]
                if len(player_data) > 0:
                    avg_pts = player_data['PTS'].mean()
            
            if avg_pts == 0:
                # Get from recent stats
                recent_stats = self.get_player_recent_stats(player_name)
                if recent_stats:
                    avg_pts = sum(game.get('pts', 0) for game in recent_stats) / len(recent_stats)
                else:
                    avg_pts = 15  # Default if no data
            
            # Simulate current game stats
            stats = {
                "points": int(avg_pts * game_progress * (0.8 + 0.4 * np.random.random())),  # Realistic points based on avg
                "minutes_played": min(48, int(game_progress * 36)),  # Minutes played
                "quarter": min(4, int(game_progress * 4.5) + 1),  # Current quarter
                "elapsed_game_time": game_progress * 48,  # Minutes elapsed in the game
                "fouls": min(5, int(np.random.poisson(2 * game_progress))),  # Random fouls
                "fg_pct": max(0, min(1, np.random.normal(0.45, 0.1))),  # Random FG%
                "score_difference": int(np.random.normal(0, 10))  # Random score difference
            }
            
            print(f"Simulated game progress: Q{stats['quarter']}, {stats['minutes_played']} mins played")
            return stats
            
        except Exception as e:
            print(f"Error getting live stats: {str(e)}")
            traceback.print_exc()
            return None
    
    def _convert_min_string(self, min_str):
        """Convert minutes string to decimal format"""
        try:
            if isinstance(min_str, str) and ':' in min_str:
                mins, secs = map(int, min_str.split(':'))
                return mins + secs / 60
            elif isinstance(min_str, str) and min_str.isdigit():
                return float(min_str)
            elif isinstance(min_str, (int, float)):
                return float(min_str)
            else:
                return 0
        except:
            return 0
    
    def _estimate_elapsed_time(self, min_str, quarter):
        """Estimate elapsed game time based on minutes played and quarter"""
        minutes_played = self._convert_min_string(min_str)
        
        # Each quarter is 12 minutes
        return (quarter - 1) * 12 + min(12, minutes_played)
    
    def _calculate_score_diff(self, game_id, home_team, away_team, is_home):
        """Calculate score difference for the player's team"""
        try:
            response = requests.get(f"https://api.balldontlie.io/v1/games/{game_id}")
            
            if response.status_code == 200:
                game_data = response.json()
                
                home_score = game_data.get('home_team_score', 0)
                away_score = game_data.get('visitor_team_score', 0)
                
                # Positive means player's team is winning
                return (home_score - away_score) if is_home else (away_score - home_score)
            
            return 0
        except:
            return 0
    
    def get_live_game_factors(self, player_name, team):
        """Get live game factors for a player"""
        live_factors = {}
        
        try:
            print("Checking for live game factors...")
            
            # Check for live game status
            live_game = self.check_if_playing_now(player_name, team)
            
            if live_game:
                print(f"📊 Live game detected: {live_game['home_team']} vs {live_game['away_team']}")
                
                # Get current game stats if available
                live_stats = self.get_live_game_stats(player_name, live_game)
                
                if live_stats:
                    live_factors["current_pts"] = live_stats.get("points", 0)
                    live_factors["current_mins"] = live_stats.get("minutes_played", 0)
                    live_factors["current_quarter"] = live_stats.get("quarter", 1)
                    live_factors["game_status"] = "LIVE"
                    live_factors["foul_trouble"] = live_stats.get("fouls", 0) >= 4
                    live_factors["shooting_hot"] = live_stats.get("fg_pct", 0) > 0.55
                    
                    # Project rest of game based on current pace
                    if live_factors["current_mins"] > 0:
                        projected_mins = min(36, live_factors["current_mins"] * (48/live_stats.get("elapsed_game_time", 12)))
                        scoring_pace = live_factors["current_pts"] / live_factors["current_mins"]
                        live_factors["projected_final_pts"] = scoring_pace * projected_mins
                    
                    # Game context factors
                    live_factors["close_game"] = abs(live_stats.get("score_difference", 0)) < 8
                    live_factors["blowout"] = abs(live_stats.get("score_difference", 0)) > 20
                    live_factors["team_winning"] = live_stats.get("score_difference", 0) > 0
                    
                    return live_factors
            else:
                # Check if game is scheduled for today
                today_game = self.check_if_playing_today(player_name, team)
                if today_game:
                    print(f"🏀 Game scheduled today: {today_game['home_team']} vs {today_game['away_team']}")
                    live_factors["game_status"] = "SCHEDULED"
                    live_factors["game_time"] = today_game["time"]
                    live_factors["is_home"] = today_game["is_home"]
                    return live_factors
                else:
                    print("No live or scheduled game found today")
                    
        except Exception as e:
            print(f"Error getting live factors: {str(e)}")
            traceback.print_exc()
            
        return live_factors
    
    def check_if_playing_today(self, player_name, team):
        """Check if player has a game scheduled for today"""
        try:
            today = datetime.now().strftime("%Y-%m-%d")
            
            # Get today's games
            response = requests.get("https://api.balldontlie.io/v1/games", 
                                  params={"per_page": 30, "dates[]": today})
                                  
            if response.status_code == 200:
                games = response.json()['data']
                
                for game in games:
                    home_team = game['home_team']['abbreviation']
                    away_team = game['visitor_team']['abbreviation']
                    
                    if home_team == team or away_team == team:
                        return {
                            'game_id': game['id'],
                            'home_team': home_team,
                            'away_team': away_team,
                            'is_home': home_team == team,
                            'time': game.get('time', 'TBD')
                        }
            
            return None
        except Exception as e:
            print(f"Error checking today's games: {str(e)}")
            return None
    
    def _adjust_prediction_for_injuries(self, original_prediction, enhanced_features):
        """
        Adjust prediction based on injury data and other API features
        """
        adjusted_prediction = original_prediction
        
        # Player is injured - significant reduction
        if enhanced_features.get('is_injured', 0) == 1:
            status = enhanced_features.get('injury_status', '').lower()
            if 'out' in status:
                return 0.0  # Player won't play
            elif 'doubtful' in status:
                return original_prediction * 0.3  # Severe reduction if doubtful
            elif 'questionable' in status:
                return original_prediction * 0.7  # Moderate reduction if questionable
            elif 'probable' in status:
                adjusted_prediction *= 0.9  # Slight reduction if probable
        
        # Back-to-back games typically reduce performance
        if enhanced_features.get('is_back_to_back', 0) == 1:
            adjusted_prediction *= 0.95  # 5% reduction for back-to-back games
        
        # If opponent has significant injuries, might increase scoring
        opp_injuries = enhanced_features.get('opp_injury_count', 0)
        if opp_injuries >= 3:
            adjusted_prediction *= 1.07  # 7% boost if opponent has 3+ injuries
        elif opp_injuries >= 1:
            adjusted_prediction *= 1.03  # 3% boost if opponent has any injuries
        
        # If team has multiple injuries, might increase usage for available players
        team_injuries = enhanced_features.get('team_injury_count', 0)
        if team_injuries >= 2:
            adjusted_prediction *= 1.05  # 5% boost if team is missing multiple players
        
        # Recent performance trending
        if 'recent_weighted_pts' in enhanced_features:
            recent_avg = enhanced_features['recent_weighted_pts']
            # If recent average is significantly different from prediction, adjust slightly
            if recent_avg > original_prediction * 1.15:  # Recent average 15% higher
                adjusted_prediction = (adjusted_prediction + recent_avg) / 2
            elif recent_avg < original_prediction * 0.85:  # Recent average 15% lower
                adjusted_prediction = (adjusted_prediction * 2 + recent_avg) / 3
        
        return adjusted_prediction
    
    def integrate_with_prediction_model(self, model_function, player_name, opponent_team, line=None):
        """
        Integrate API data with the existing prediction model
        
        Args:
            model_function: Original prediction function that takes player_name and opponent_team
            player_name: Player name to predict
            opponent_team: Opponent team abbreviation
            line: Optional betting line
            
        Returns:
            Enhanced prediction result
        """
        # First get the original prediction
        original_prediction = model_function(player_name, opponent_team)
        
        # If prediction returned an error message, return it
        if isinstance(original_prediction, str):
            return original_prediction
        
        # Get enhanced features from API
        enhanced_features = self.enhance_prediction_data(player_name, opponent_team)
        
        # Add API data to the prediction result
        result = original_prediction.copy()
        result["api_enhanced_data"] = enhanced_features
        
        # Adjust prediction based on injury information
        adjusted_prediction = self._adjust_prediction_for_injuries(
            original_prediction["predicted_points"],
            enhanced_features
        )
        
        # Update the prediction with adjusted value
        result["original_prediction"] = original_prediction["predicted_points"]
        result["predicted_points"] = adjusted_prediction
        
        # Generate a suggested line if one isn't provided in original_prediction
        if "suggested_line" not in result or result["suggested_line"] is None:
            # Generate a line based on the adjusted prediction with a small random factor
            variance = 0.02  # 2% variance
            random_factor = 1 + variance * (np.random.random() - 0.5)
            result["suggested_line"] = round(adjusted_prediction * random_factor, 0.5)
        
        # If line is provided by user, use that, otherwise use our generated line
        betting_line = line if line is not None else result["suggested_line"]
        
        # Calculate over/under with the line (either provided or auto-generated)
        from nba_prediction import evaluate_over_under
        result = evaluate_over_under(result, betting_line)
        
        return result