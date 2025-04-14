import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor
import joblib
from difflib import get_close_matches
import requests
import json
from datetime import datetime, timedelta
import time
import traceback
import argparse
import sys
import re
import os

# === Load and clean data ===
def load_and_prepare_data(csv_path="2024-2025.csv"):
    """Load and prepare the NBA data from CSV"""
    # Check if file exists
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file {csv_path} not found")
        
    df = pd.read_csv(csv_path)
    
    # Convert minutes to decimal format
    def convert_minutes(mp):
        try:
            if isinstance(mp, str) and ':' in mp:
                mins, secs = map(int, mp.split(':'))
                return mins + secs / 60
            else:
                return float(mp)
        except:
            return None
    
    def get_live_game_stats(self, player_name, game_info):
        """Get live game stats for a player"""
        try:
            # This would typically require a real-time stats API
            # For demonstration, we'll simulate live stats
            
            # In a real implementation, you would use:
            # 1. ESPN API (although it requires authorization)
            # 2. NBA Stats API
            # 3. Commercial sports data APIs like SportRadar
            
            # Simulate live stats with random data for demonstration
            current_hour = datetime.now().hour
            game_progress = min(1.0, max(0.1, (current_hour % 12) / 12))  # Simulate game progress
            
            stats = {
                "points": int(np.random.normal(8, 3) * game_progress),  # Random points based on game progress
                "minutes_played": min(36, int(game_progress * 36)),  # Minutes played
                "quarter": min(4, int(game_progress * 4.5) + 1),  # Current quarter
                "elapsed_game_time": game_progress * 48,  # Minutes elapsed in the game
                "fouls": min(5, int(np.random.poisson(2))),  # Random fouls
                "fg_pct": max(0, min(1, np.random.normal(0.45, 0.1))),  # Random FG%
                "score_difference": int(np.random.normal(0, 10))  # Random score difference
            }
            
            # Add a note about the simulated stats
            print("Note: Using simulated live game stats for demonstration")
            print(f"Current game progress: Q{stats['quarter']}, {stats['minutes_played']} mins played")
            
            return stats
        except Exception as e:
            print(f"Error getting live stats: {str(e)}")
            return None
    
    def run_interactive(self):
        """Run the prediction system in interactive mode"""
        if not self.predictor or not self.enhancer:
            if not self.setup():
                print("Failed to set up prediction system. Exiting.")
                return
        
        print("\n" + "="*60)
        print("🏀  ENHANCED NBA PLAYER POINTS PREDICTOR  🏀")
        print("="*60)
        print("This tool predicts NBA player points based on historical data")
        print("and real-time factors like injuries, matchups, and game context.")
        print("="*60 + "\n")
        
        while True:
            try:
                # Get player name
                player_input = input("\nEnter player name (or 'quit' to exit): ")
                if player_input.lower() in ('quit', 'exit', 'q'):
                    break
                
                # Check if we should update data from API
                update_data = False
                if player_input.startswith("update "):
                    update_data = True
                    player_input = player_input[7:].strip()
                
                # Normalize player name
                player_name = self.normalize_player_name(player_input)
                
                # Get opponent
                opponent_input = input("Enter opponent team (e.g. 'LAL', 'Warriors'): ")
                opponent_team = self.normalize_team_name(opponent_input)
                
                # Get betting line (optional)
                line_input = input("Enter betting line (optional, press enter to skip): ")
                line = float(line_input) if line_input and line_input.replace('.', '', 1).isdigit() else None
                
                print("\n⚙️ Processing prediction...")
                
                # Update data if requested
                if update_data and hasattr(self.enhancer, 'fetch_latest_player_data'):
                    self.enhancer.update_player_data(player_name, opponent_team)
                
                # Get enhanced prediction
                result = self.enhancer.integrate_with_prediction_model(
                    self.predictor,
                    player_name,
                    opponent_team,
                    line
                )
                
                # Handle string result (error message)
                if isinstance(result, str):
                    print(f"ℹ️ {result}")
                    continue
                
                # Get live game factors
                player_team = self._get_player_team(result)
                if player_team:
                    live_factors = self.get_live_game_factors(player_name, player_team)
                    
                    # Incorporate live factors into prediction
                    if live_factors and "game_status" in live_factors:
                        result["live_game_factors"] = live_factors
                        
                        # Adjust prediction based on live factors
                        if live_factors["game_status"] == "LIVE":
                            # Player already has some points
                            current_pts = live_factors.get("current_pts", 0)
                            
                            # Adjust prediction based on current performance
                            if "projected_final_pts" in live_factors:
                                result["predicted_points"] = (result["predicted_points"] + live_factors["projected_final_pts"]) / 2
                            
                            # Factor in current points
                            result["current_points"] = current_pts
                            
                            # Adjust for foul trouble
                            if live_factors.get("foul_trouble", False):
                                result["predicted_points"] *= 0.85
                                result["foul_trouble"] = True
                            
                            # Adjust for blowout
                            if live_factors.get("blowout", False):
                                result["predicted_points"] *= 0.9
                                result["blowout_factor"] = True
                
                # Display results
                self._display_prediction_result(result, player_name, opponent_team, line)
                
                # Option to continue
                again = input("\nMake another prediction? (y/n): ")
                if again.lower() not in ('y', 'yes'):
                    break
                    
            except KeyboardInterrupt:
                print("\nExiting prediction system.")
                break
            except Exception as e:
            
             df['MP'] = df['MP'].apply(convert_minutes)
    
    # Convert percentage columns to numeric
    for col in ['FG%', '3P%', 'FT%']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
    # Drop rows with missing critical values
    df = df.dropna(subset=['PTS', 'MP'])
    
    # Sort by player and date
    df = df.sort_values(['Player', 'Data'])
    
    # Add advanced statistics
    box_score_stats = ['MP', 'FGA', '3PA', 'FTA', 'AST', 'TOV', 'FG%', '3P%', 'FT%',
                      'PTS', 'REB', 'ORB', 'DRB', 'STL', 'BLK', 'PF']
                      
    # Calculate advanced metrics
    df['USG%'] = (df['FGA'] + 0.44*df['FTA'] + df['TOV']) / df['MP']  # Usage rate
    df['TS%'] = df['PTS'] / (2 * (df['FGA'] + 0.44*df['FTA']))  # True shooting percentage
    df['PACE'] = df.groupby(['Data', 'Tm'])['MP'].transform('sum')  # Team pace estimation
    
    # Calculate efficiency metrics
    df['PTS_per_MIN'] = df['PTS'] / df['MP']
    df['PTS_per_FGA'] = df['PTS'] / df['FGA'].replace(0, np.nan)
    
    # Create rolling averages with different window sizes
    windows = [3, 5, 10]  # Last 3, 5, and 10 games
    for window in windows:
        for stat in box_score_stats:
            if stat in df.columns:
                df[f'{stat}_avg{window}'] = df.groupby('Player')[stat].transform(
                    lambda x: x.shift(1).rolling(window).mean())
                    
    # Add matchup-specific statistics
    df['vs_opp_PTS'] = df.groupby(['Player', 'Opp'])['PTS'].transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).mean())
        
    # Add contextual factors
    df['rest_days'] = df.groupby('Player')['Data'].transform(
        lambda x: (pd.to_datetime(x) - pd.to_datetime(x).shift(1)).dt.days)
        
    # Calculate home/away game factor
    df['is_home'] = df['Tm'] != df['Opp'].str.split(' @ | vs ').str[0]
    
    return df

# Function to evaluate over/under betting lines
def evaluate_over_under(prediction_result, line):
    """
    Evaluate over/under betting line based on prediction
    
    Args:
        prediction_result (dict): Prediction result from model
        line (float): Betting line
    
    Returns:
        dict: Updated prediction result with over/under evaluation
    """
    predicted_points = prediction_result["predicted_points"]
    
    # Calculate difference from line
    diff = predicted_points - line
    
    # Determine confidence level based on difference magnitude
    confidence = 0
    if abs(diff) > 5:  # High confidence if predicted > 5 points from line
        confidence = 3
    elif abs(diff) > 3:  # Medium confidence
        confidence = 2
    elif abs(diff) > 1:  # Low confidence
        confidence = 1
    
    # Add fields to result
    prediction_result["line"] = line
    prediction_result["diff_from_line"] = diff
    prediction_result["recommendation"] = "OVER" if diff > 0 else "UNDER"
    prediction_result["confidence"] = confidence
    
    return prediction_result

# Model training and prediction
def train_prediction_model(df):
    """
    Train the XGBoost regression model on the provided dataframe
    
    Args:
        df (DataFrame): Prepared NBA data
    
    Returns:
        tuple: Trained model and feature list
    """
    # Create a copy of the original dataframe
    df_original = df.copy()
    
    # Encode categorical variables
    if 'Tm' in df.columns and 'Opp' in df.columns:
        df_encoded = pd.get_dummies(df, columns=['Tm', 'Opp'], prefix=['team', 'opp'])
    else:
        df_encoded = df
    
    # Define features
    rolling_features = [col for col in df_encoded.columns if any(f'_avg{w}' in col for w in [3, 5, 10])]
    team_features = [col for col in df_encoded.columns if col.startswith('team_') or col.startswith('opp_')]
    context_features = ['rest_days', 'is_home'] if all(f in df_encoded.columns for f in ['rest_days', 'is_home']) else []
    advanced_features = ['USG%', 'TS%', 'PACE', 'PTS_per_MIN', 'PTS_per_FGA'] 
    advanced_features = [f for f in advanced_features if f in df_encoded.columns]
    matchup_features = ['vs_opp_PTS'] if 'vs_opp_PTS' in df_encoded.columns else []
    
    # Combine all features
    features = rolling_features + team_features + context_features + advanced_features + matchup_features
    
    # Remove duplicates and check existence
    features = list(set(features))
    features = [f for f in features if f in df_encoded.columns]
    
    # Split data
    X = df_encoded[features]
    y = df_encoded['PTS']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    # Train XGBoost model
    model = XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=6)
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Model performance - MAE: {mae:.2f}, R²: {r2:.4f}")
    
    return model, features, df_original

# Function to fetch the latest box scores for a player
def fetch_latest_box_scores(player_name, df_full, num_games=5):
    """
    Fetch the latest box scores for a player
   
    Args:
        player_name (str): Player name
        df_full (DataFrame): Full dataset with player history
        num_games (int): Number of most recent games to fetch
       
    Returns:
        DataFrame: Recent box scores for the player
    """
    player_games = df_full[df_full['Player'] == player_name].sort_values('Data', ascending=False)
   
    if len(player_games) == 0:
        return None
   
    recent_games = player_games.head(num_games)
   
    # Select relevant box score columns
    box_score_columns = ['Data', 'Opp', 'MP', 'FG', 'FGA', 'FG%', '3P', '3PA', '3P%',
                         'FT', 'FTA', 'FT%', 'ORB', 'DRB', 'REB', 'AST', 'STL',
                         'BLK', 'TOV', 'PF', 'PTS']
   
    available_columns = [col for col in box_score_columns if col in recent_games.columns]
   
    return recent_games[available_columns]

# Function to predict player points
def predict_player_points(player_name, opponent_team, model, features, df_full):
    """
    Predict points for a player against a specific opponent
   
    Args:
        player_name (str): Player name
        opponent_team (str): Opponent team abbreviation (e.g., MIA)
        model: Trained regression model
        features: List of features used by the model
        df_full (DataFrame): Full dataset with player history
       
    Returns:
        dict or str: Prediction result or error message
    """
    # Check if player exists in the dataset
    if player_name not in df_full['Player'].values:
        # Try to find close matches
        all_players = df_full['Player'].unique()
        close_matches = get_close_matches(player_name, all_players, n=3, cutoff=0.6)
       
        if close_matches:
            return f"Player '{player_name}' not found. Did you mean: {', '.join(close_matches)}?"
        else:
            return f"Player '{player_name}' not found in the dataset."
   
    # Check if opponent exists
    opp_cols = [col for col in df_full.columns if col.startswith('opp_')]
    opp_col = f'opp_{opponent_team}'
    
    # Handle opponent team validation
    all_opps = []
    if opp_cols:
        all_opps = [col[4:] for col in opp_cols]  # Remove 'opp_' prefix
    else:
        # Try to extract opponent teams from the 'Opp' column if available
        if 'Opp' in df_full.columns:
            all_opps = df_full['Opp'].unique()
    
    if opponent_team not in all_opps:
        close_matches = get_close_matches(opponent_team, all_opps, n=3, cutoff=0.6)
        
        if close_matches:
            return f"Opponent '{opponent_team}' not found. Did you mean: {', '.join(close_matches)}?"
        else:
            return f"Opponent '{opponent_team}' not found in the dataset."
   
    player_games = df_full[df_full['Player'] == player_name].sort_values('Data', ascending=False)
   
    if len(player_games) < 3:
        return f"Not enough data for {player_name}. Only {len(player_games)} games available."
   
    # Get recent box scores for analysis and display
    recent_box_scores = fetch_latest_box_scores(player_name, df_full, 5)
   
    # Calculate input features from available data
    input_dict = {}
   
    # Add rolling averages for all window sizes
    windows = [3, 5, 10]
    box_score_stats = ['MP', 'FGA', '3PA', 'FTA', 'AST', 'TOV', 'FG%', '3P%', 'FT%',
                      'PTS', 'REB', 'ORB', 'DRB', 'STL', 'BLK', 'PF']
                      
    for window in windows:
        for stat in box_score_stats:
            feature = f'{stat}_avg{window}'
            if feature in features:
                avg_value = player_games.head(window)[stat].mean() if stat in player_games.columns else 0
                input_dict[feature] = avg_value
   
    # Add advanced stats if they exist
    advanced_features = ['USG%', 'TS%', 'PACE', 'PTS_per_MIN', 'PTS_per_FGA']
    for feat in advanced_features:
        if feat in features:
            input_dict[feat] = player_games[feat].iloc[0] if feat in player_games.columns else 0
   
    # Add matchup specific features
    for feat in features:
        if feat.startswith('vs_opp_'):
            # Try to find games against this specific opponent
            if 'Opp' in player_games.columns:
                opp_games = player_games[player_games['Opp'].str.contains(opponent_team)]
                if len(opp_games) > 0 and feat.replace('vs_opp_', '') in opp_games.columns:
                    input_dict[feat] = opp_games[feat.replace('vs_opp_', '')].mean()
                else:
                    # Use average for the stat if no games against this opponent
                    input_dict[feat] = player_games[feat.replace('vs_opp_', '')].mean() if feat.replace('vs_opp_', '') in player_games.columns else 0
   
    # Add contextual features
    for feat in ['rest_days', 'is_home']:
        if feat in features:
            if feat == 'rest_days':
                input_dict[feat] = 2  # Default to 2 days of rest if unknown
            elif feat == 'is_home':
                # Determine if the game is home or away based on opponent
                input_dict[feat] = 1  # Default to home game if unknown
   
    # Determine player's team from the team_ columns
    player_team = None
    team_cols = [col for col in player_games.columns if col.startswith('team_')]
    for col in team_cols:
        # Find the first team column that has a value of 1
        if player_games.iloc[0][col] == 1:
            player_team = col[5:]  # Remove 'team_' prefix
            break
   
    # If we couldn't determine team from columns, try from recent stats
    if player_team is None and 'Tm' in player_games.columns:
        player_team = player_games.iloc[0]['Tm']
   
    if player_team is None:
        return f"Could not determine team for player {player_name}"
   
    # Set all team and opponent columns to 0, then set player's team and opponent to 1
    for col in features:
        if col.startswith('team_'):
            input_dict[col] = 1 if col == f'team_{player_team}' else 0
        elif col.startswith('opp_'):
            input_dict[col] = 1 if col == f'opp_{opponent_team}' else 0
   
    # Make prediction
    try:
        # Ensure all model features are included
        X_input = pd.DataFrame([input_dict])
        missing_cols = set(model.feature_names_in_) - set(X_input.columns)
        for col in missing_cols:
            X_input[col] = 0
        
        # Keep only the features used by the model
        X_input = X_input[model.feature_names_in_]
        
        predicted_pts = model.predict(X_input)[0]
    except Exception as e:
        return f"Error during prediction: {str(e)}"
   
    # Get player stats for context
    season_avg = player_games['PTS'].mean()
    recent_avg = player_games.head(3)['PTS'].mean()
   
    # Calculate suggested line with more sophisticated approach
    recent_variance = player_games.head(5)['PTS'].std()
    market_adjustment = 0.02  # 2% adjustment to simulate market inefficiency
    suggested_line = round(predicted_pts * (1 + market_adjustment * np.random.randn()), 1)
   
    # Create result dictionary
    result = {
        "player": player_name,
        "opponent": opponent_team,
        "predicted_points": predicted_pts,
        "season_avg": season_avg,
        "recent_avg": recent_avg,
        "recent_variance": recent_variance,
        "suggested_line": suggested_line,
        "recent_games": recent_box_scores.to_dict('records') if recent_box_scores is not None else []
    }
   
    return result
   
    def get_team_back_to_back_status(self, team_abbrev):
        """Check if team is on a back-to-back game"""
        if team_abbrev not in self.team_schedule:
            self.get_team_schedule(team_abbrev)
       
        schedule = self.team_schedule.get(team_abbrev, [])
        if not schedule:
            return False
       
        # Find the next game
        today = datetime.now().strftime("%Y-%m-%d")
        upcoming_games = [g for g in schedule if g['date'] >= today]
        if not upcoming_games:
            return False
       
        next_game = min(upcoming_games, key=lambda x: x['date'])
        next_game_date = datetime.strptime(next_game['date'], "%Y-%m-%d")
       
        # Check if there was a game the day before
        prev_day = (next_game_date - timedelta(days=1)).strftime("%Y-%m-%d")
        return any(g['date'] == prev_day for g in schedule)
   
    def enhance_prediction_data(self, player_name, opponent_team):
        """
        Gather all API data that could enhance the prediction model
        Returns a dictionary of features
        """
        enhanced_features = {}
       
        # Get player injury status
        injury_status = self.is_player_injured(player_name)
        enhanced_features['is_injured'] = 1 if injury_status else 0
        enhanced_features['injury_status'] = injury_status['status'] if injury_status else 'Active'
       
        # Get team's injury status (percentage of team injured)
        player_team = self._get_player_team(player_name)
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
            stats_to_average = ['pts', 'min', 'fg_pct', 'fg3_pct', 'ft_pct', 'reb', 'ast']
           
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
        if recent_stats:
            return recent_stats[0]['team']
        return None
   
    def _adjust_prediction_for_injuries(self, original_prediction, enhanced_features):
        """
        Adjust prediction based on injury data and other API features
        Uses heuristics that can be refined over time
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
            variance = 0.02 # 2% variance
            random_factor = 1 + variance * (np.random.random() - 0.5)
            result["suggested_line"] = round(adjusted_prediction * random_factor, 0.5)
        
        # If line is provided by user, use that, otherwise use our generated line
        betting_line = line if line is not None else result["suggested_line"]
        
        # Calculate over/under with the line (either provided or auto-generated)
        result = evaluate_over_under(result, betting_line)
        
        return result
        
    def fetch_latest_player_data(self, player_name):
        """
        Fetch the latest game data for a player from the NBA Stats API
        and add it to our dataset
        
        Args:
            player_name: Player name to fetch data for
            
        Returns:
            DataFrame with the player's latest game data or None if failed
        """
        if not self.has_nba_api:
            print("NBA Stats API not available")
            return None
            
        print(f"Fetching latest NBA data for {player_name}...")
        
        try:
            # Get game logs from the NBA Stats API
            game_logs = self.nba_api.get_player_game_logs(player_name)
            
            if game_logs is None or game_logs.empty:
                print(f"No game data found for {player_name}")
                return None
                
            # Convert to our prediction format
            prediction_data = self.nba_api.convert_to_prediction_format(game_logs, player_name)
            
            if prediction_data is None or prediction_data.empty:
                print("Failed to convert game data to prediction format")
                return None
                
            # Add to player database
            if player_name not in self.player_database:
                self.player_database.add(player_name)
                
            # Update existing dataframe if we have one
            if self.existing_df is not None:
                # Check if this player exists in our data
                existing_player_data = self.existing_df[self.existing_df['Player'] == player_name]
                
                if len(existing_player_data) > 0:
                    # Remove existing player data
                    self.existing_df = self.existing_df[self.existing_df['Player'] != player_name]
                    
                # Append new data
                self.existing_df = pd.concat([self.existing_df, prediction_data])
                
                # Sort dataframe
                self.existing_df = self.existing_df.sort_values(['Player', 'Data'])
                
                print(f"Added {len(prediction_data)} games for {player_name} to dataset")
            else:
                # Create new dataframe if we don't have one
                self.existing_df = prediction_data
                print(f"Created new dataset with {len(prediction_data)} games for {player_name}")
                
            return prediction_data
            
        except Exception as e:
            print(f"Error fetching latest player data: {str(e)}")
            return None
    
    def update_player_data(self, player_name, opponent_team=None):
        """
        Update data for a specific player and opponent matchup
        
        Args:
            player_name: Player to update
            opponent_team: Optional opponent team to get specific matchup data
            
        Returns:
            True if successful, False otherwise
        """
        # First update player data
        player_data = self.fetch_latest_player_data(player_name)
        
        if player_data is None:
            return False
            
        # If opponent specified, get specific matchup data
        if opponent_team and self.has_nba_api:
            player_team = self._get_player_team(player_name)
            
            if player_team:
                # Get matchup stats
                matchup_data = self.nba_api.get_matchup_stats(player_team, opponent_team)
                
                if matchup_data is not None:
                    print(f"Found {len(matchup_data) // 2} matchups between {player_team} and {opponent_team}")
                    
                    # We could process this further to extract player-specific 
                    # matchup stats, but that would require additional API calls
                    
        return True

# User Interface for NBA Prediction
class NBAPredictionUI:
    """
    User interface for the NBA prediction system
    """
    def __init__(self, predictor_model=None):
        """Initialize the prediction UI"""
        self.predictor = predictor_model
        self.enhancer = None
        
    def setup(self):
        """Set up the prediction system"""
        try:
            # Check if model already exists
            try:
                model = joblib.load('nba_points_model.joblib')
                features = joblib.load('model_features.joblib')
                
                # Check if CSV file exists
                if not os.path.exists("2024-2025.csv"):
                    print("❌ Data file 2024-2025.csv not found")
                    return False
                    
                df = pd.read_csv("2024-2025.csv")
                print("✓ Loaded existing model and data")
            except FileNotFoundError:
                print("⚙️ Training new model (this may take a moment)...")
                if not os.path.exists("2024-2025.csv"):
                    print("❌ Data file 2024-2025.csv not found")
                    return False
                    
                df = load_and_prepare_data("2024-2025.csv")
                model, features, df = train_prediction_model(df)
                # Save model for future use
                joblib.dump(model, 'nba_points_model.joblib')
                joblib.dump(features, 'model_features.joblib')
                print("✓ Model trained and saved")
            
            # Define predictor function
            def predict_wrapper(player_name, opponent_team):
                return predict_player_points(player_name, opponent_team, model, features, df)
            
            self.predictor = predict_wrapper
            
            # Initialize NBA data enhancer
            self.enhancer = NBADataEnhancer(df)
            
            return True
        except Exception as e:
            print(f"❌ Error setting up prediction system: {str(e)}")
            traceback.print_exc()
            return False
    
    def normalize_player_name(self, player_name):
        """Normalize player name format for better matching"""
        # Convert to title case
        normalized = player_name.title()
        
        # Handle common nicknames
        nickname_map = {
            "Lebron": "LeBron James",
            "Luka": "Luka Dončić",
            "Giannis": "Giannis Antetokounmpo",
            "Steph": "Stephen Curry",
            "KD": "Kevin Durant",
            "CP3": "Chris Paul",
            "Kawhi": "Kawhi Leonard",
            "Dame": "Damian Lillard",
            "Jimmy Buckets": "Jimmy Butler",
            "Kyrie": "Kyrie Irving",
            "The Joker": "Nikola Jokić",
            "Joker": "Nikola Jokić",
            "Greek Freak": "Giannis Antetokounmpo",
            "Zion": "Zion Williamson",
            "AD": "Anthony Davis",
            "PG": "Paul George",
            "PG13": "Paul George"
        }
        
        # Check for nickname matches
        if normalized in nickname_map:
            return nickname_map[normalized]
            
        # Add last name only handling by checking against full dataset
        if self.enhancer and ' ' not in normalized:
            # Try to find matches in player database
            all_players = self.enhancer.player_database
            possible_matches = [p for p in all_players if p.endswith(f" {normalized}")]
            if len(possible_matches) == 1:
                return possible_matches[0]
        
        return normalized
        
    def normalize_team_name(self, team_name):
        """Normalize team name to standard abbreviation"""
        team_name = team_name.upper()
        
        # Common team name mappings
        team_map = {
            "LAKERS": "LAL",
            "WARRIORS": "GSW",
            "NETS": "BKN",
            "CELTICS": "BOS",
            "BUCKS": "MIL",
            "HEAT": "MIA",
            "BULLS": "CHI",
            "SUNS": "PHX",
            "MAVS": "DAL",
            "MAVERICKS": "DAL",
            "SIXERS": "PHI",
            "76ERS": "PHI",
            "CLIPPERS": "LAC",
            "KNICKS": "NYK",
            "HAWKS": "ATL",
            "NUGGETS": "DEN",
            "RAPTORS": "TOR",
            "JAZZ": "UTA",
            "BLAZERS": "POR",
            "TRAILBLAZERS": "POR",
            "TRAIL BLAZERS": "POR",
            "GRIZZLIES": "MEM",
            "PELICANS": "NOP",
            "KINGS": "SAC",
            "SPURS": "SAS",
            "PACERS": "IND",
            "HORNETS": "CHA",
            "WIZARDS": "WAS",
            "PISTONS": "DET",
            "THUNDER": "OKC",
            "MAGIC": "ORL",
            "ROCKETS": "HOU",
            "CAVS": "CLE",
            "CAVALIERS": "CLE",
            "TIMBERWOLVES": "MIN",
            "WOLVES": "MIN"
        }
        
        if team_name in team_map:
            return team_map[team_name]
            
        # If it's already a standard abbreviation, return as is
        common_abbrevs = {"LAL", "GSW", "BKN", "BOS", "MIL", "MIA", "CHI", "PHX", "DAL",
                        "PHI", "LAC", "NYK", "ATL", "DEN", "TOR", "UTA", "POR", "MEM",
                        "NOP", "SAC", "SAS", "IND", "CHA", "WAS", "DET", "OKC", "ORL",
                        "HOU", "CLE", "MIN"}
        if team_name in common_abbrevs:
            return team_name
        
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
            
        return live_factors
    
    def check_if_playing_now(self, player_name, team):
        """Check if player is currently in a live game"""
        try:
            # Get current live games
            response = requests.get("https://api.balldontlie.io/v1/games", 
                                  params={"per_page": 15, "start_date": datetime.now().strftime("%Y-%m-%d")})
                                  
            if response.status_code == 200:
                games = response.json()['data']
                
                # Filter only live games
                live_games = [g for g in games if g['status'] == 'in progress']
                
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
            
        return team_name

# === NBA API Enhancer Class ===
class NBADataEnhancer:
    """
    Class to enhance NBA prediction model with real-time data from public APIs
    """
    def __init__(self, existing_df=None):
        """Initialize with existing dataframe if available"""
        self.existing_df = existing_df
        self.injury_data = {}
        self.team_schedule = {}
        self.player_stats_cache = {}
        self.team_mapping = {}  # Cache for team ID mapping
        
        # Build player database for better matching
        self.player_database = set()
        if existing_df is not None and 'Player' in existing_df.columns:
            self.player_database = set(existing_df['Player'].unique())
            
        # Cache common info to reduce API calls
        self._initialize_cache()
        
        # Try to import NBAStatsAPI if available
        try:
            from nba_stats_api import NBAStatsAPI
            self.nba_api = NBAStatsAPI()
            self.has_nba_api = True
        except ImportError:
            self.has_nba_api = False
       
    def _initialize_cache(self):
        """Initialize cache with common data to reduce API calls"""
        # Cache team info
        self.get_team_abbreviations()
        
        # Pre-fetch injury data
        self.get_current_injuries()
            
    def get_team_abbreviations(self):
        """Get NBA team abbreviations and map them to API team IDs"""
        # Use cached mapping if available
        if self.team_mapping:
            return self.team_mapping
           
        try:
            # First try balldontlie API
            response = requests.get("https://api.balldontlie.io/v1/teams")
            if response.status_code == 200:
                teams_data = response.json()['data']
                self.team_mapping = {team['abbreviation']: team['id'] for team in teams_data}
                return self.team_mapping
            else:
                print(f"Error fetching teams: {response.status_code}")
                # Fallback to hardcoded common team abbreviations
                self.team_mapping = {
                    "ATL": 1, "BOS": 2, "BKN": 3, "CHA": 4, "CHI": 5,
                    "CLE": 6, "DAL": 7, "DEN": 8, "DET": 9, "GSW": 10,
                    "HOU": 11, "IND": 12, "LAC": 13, "LAL": 14, "MEM": 15,
                    "MIA": 16, "MIL": 17, "MIN": 18, "NOP": 19, "NYK": 20,
                    "OKC": 21, "ORL": 22, "PHI": 23, "PHX": 24, "POR": 25,
                    "SAC": 26, "SAS": 27, "TOR": 28, "UTA": 29, "WAS": 30
                }
                return self.team_mapping
        except Exception as e:
            print(f"Error fetching teams: {e}")
            # Fallback to hardcoded team IDs in case of API failure
            self.team_mapping = {
                "ATL": 1, "BOS": 2, "BKN": 3, "CHA": 4, "CHI": 5,
                "CLE": 6, "DAL": 7, "DEN": 8, "DET": 9, "GSW": 10,
                "HOU": 11, "IND": 12, "LAC": 13, "LAL": 14, "MEM": 15,
                "MIA": 16, "MIL": 17, "MIN": 18, "NOP": 19, "NYK": 20,
                "OKC": 21, "ORL": 22, "PHI": 23, "PHX": 24, "POR": 25,
                "SAC": 26, "SAS": 27, "TOR": 28, "UTA": 29, "WAS": 30
            }
            return self.team_mapping
            
    def get_current_injuries(self):
        """Get current injury data from ESPN API"""
        try:
            # ESPN API for injury reports
            response = requests.get("https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries")
           
            if response.status_code == 200:
                injury_data = response.json()
               
                # Process and structure injury data
                processed_injuries = {}
               
                for team in injury_data.get('injuries', []):
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
                   
                    processed_injuries[team_abbrev] = team_injuries
               
                self.injury_data = processed_injuries
                return processed_injuries
            else:
                print(f"Error fetching injury data: {response.status_code}")
                return {}
        except Exception as e:
            print(f"Error fetching injury data: {e}")
            return {}
   
    def get_team_schedule(self, team_abbrev, days=14):
        """Get team's upcoming and recent games"""
        team_mapping = self.get_team_abbreviations()
        team_id = team_mapping.get(team_abbrev)
       
        if not team_id:
            print(f"Team abbreviation '{team_abbrev}' not found")
            return []
       
        try:
            # Get today's date and date range for schedule
            today = datetime.now()
            start_date = (today - timedelta(days=days//2)).strftime("%Y-%m-%d")
            end_date = (today + timedelta(days=days//2)).strftime("%Y-%m-%d")
           
            # Get games in the date range
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
                        'date': game['date'],
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
                return []
        except Exception as e:
            print(f"Error fetching schedule: {e}")
            return []
   
    def get_player_recent_stats(self, player_name):
        """Get player's recent game statistics"""
        # Check if we have NBA Stats API available
        if self.has_nba_api:
            try:
                # Try to use NBA Stats API first
                game_logs = self.nba_api.get_player_game_logs(player_name)
                
                if game_logs is not None and not game_logs.empty:
                    # Convert to our format
                    recent_stats = []
                    for _, row in game_logs.iterrows():
                        game_stats = {
                            'date': row['GAME_DATE'].strftime('%Y-%m-%d'),
                            'team': row['TEAM_ABBREVIATION'] if 'TEAM_ABBREVIATION' in row else self.nba_api.player_cache.get(player_name, {}).get('team_abbreviation', ''),
                            'opponent': row['MATCHUP'].split()[-1],
                            'is_home': '@' not in row['MATCHUP'],
                            'min': row['MIN'],
                            'pts': row['PTS'],
                            'fg_pct': row['FG_PCT'],
                            'fg3_pct': row['FG3_PCT'],
                            'ft_pct': row['FT_PCT'],
                            'reb': row['REB'],
                            'ast': row['AST'],
                            'stl': row['STL'],
                            'blk': row['BLK'],
                            'turnover': row['TOV']
                        }
                        recent_stats.append(game_stats)
                    
                    # Cache the result
                    self.player_stats_cache[player_name] = recent_stats
                    return recent_stats
            except Exception as e:
                print(f"Error using NBA Stats API: {e}")
                # Fall back to BallDontLie API
        
        # Check cache first
        if player_name in self.player_stats_cache:
            return self.player_stats_cache[player_name]
       
        try:
            # Search for player
            response = requests.get(
                "https://api.balldontlie.io/v1/players",
                params={"search": player_name}
            )
           
            if response.status_code != 200:
                print(f"Error searching for player: {response.status_code}")
                return []
           
            player_data = response.json()['data']
            if not player_data:
                print(f"Player '{player_name}' not found")
                return []
           
            # Find best match for player name
            player_id = None
            for player in player_data:
                full_name = f"{player['first_name']} {player['last_name']}"
                if full_name.lower() == player_name.lower():
                    player_id = player['id']
                    break
           
            if not player_id:
                # Take first match if exact match not found
                player_id = player_data[0]['id']
           
            # Get player's recent stats
            today = datetime.now()
            start_date = (today - timedelta(days=30)).strftime("%Y-%m-%d")
           
            stats_response = requests.get(
                "https://api.balldontlie.io/v1/stats",
                params={
                    "player_ids[]": player_id,
                    "start_date": start_date,
                    "per_page": 100
                }
            )
           
            if stats_response.status_code != 200:
                print(f"Error fetching player stats: {stats_response.status_code}")
                return []
           
            stats_data = stats_response.json()['data']
           
            # Process stats data
            player_stats = []
            for stat in stats_data:
                # Handle potential missing data in API response
                try:
                    game_stats = {
                        'date': stat['game']['date'],
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
                except (KeyError, TypeError) as e:
                    # Skip this stat entry if there are missing keys
                    continue
           
            # Cache the result
            self.player_stats_cache[player_name] = player_stats
            return player_stats
        except Exception as e:
            print(f"Error fetching player stats: {e}")
            return []
   
    def is_player_injured(self, player_name, team_abbrev=None):
        """Check if a player is currently injured"""
        if not self.injury_data:
            self.get_current_injuries()
       
        if team_abbrev and team_abbrev in self.injury_data:
            # If we know the team, check that team's injury list
            for player in self.injury_data[team_abbrev]:
                if player['name'].lower() == player_name.lower():
                    return player
        else:
            # Otherwise, check all teams
            for team, players in self.injury_data.items():
                for player in players:
                    if player['name'].lower() == player_name.lower():
                        return player
       
        return None
   
    def get_team_injuries(self, team_abbrev):
        """Get all injuries for a team"""
        if not self.injury_data:
            self.get_current_injuries()
       
        return self.injury_data.get(team_abbrev, [])