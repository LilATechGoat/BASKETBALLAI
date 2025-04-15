import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor
import joblib
from difflib import get_close_matches
import requests
import json
from datetime import datetime, timedelta
import time
import traceback
import os
import re

# === Load and prepare data ===
def load_and_prepare_data(csv_path="2024-2025.csv", boxscores_path=None):
    """
    Load and prepare NBA data from CSV and optional boxscores
    
    Args:
        csv_path: Path to the main dataset CSV
        boxscores_path: Optional path to boxscores CSV
        
    Returns:
        DataFrame with prepared NBA data
    """
    # Check if file exists
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file {csv_path} not found")
        
    # Load main dataset
    df = pd.read_csv(csv_path)
    
    # Load boxscores if provided
    if boxscores_path and os.path.exists(boxscores_path):
        boxscores_df = pd.read_csv(boxscores_path)
        print(f"Loaded boxscores data with {len(boxscores_df)} rows")
        
        # Add any additional data from boxscores if needed
        # This would merge relevant stats from boxscores into main dataset
        # Example: (implementation depends on boxscores structure)
    
    # Convert minutes to decimal format if in MM:SS format
    def convert_minutes(mp):
        try:
            if isinstance(mp, str) and ':' in mp:
                mins, secs = map(int, mp.split(':'))
                return mins + secs / 60
            else:
                return float(mp)
        except:
            return None
    
    # Apply minutes conversion    
    df['MP'] = df['MP'].apply(convert_minutes)
    
    # Convert percentage columns to numeric
    for col in ['FG%', '3P%', 'FT%']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Drop rows with missing critical values
    df = df.dropna(subset=['PTS', 'MP'])
    
    # Sort by player and date
    if 'Data' in df.columns:
        df = df.sort_values(['Player', 'Data'])
    
    # Clean team and opponent formatting
    if 'Opp' in df.columns:
        # Extract opponent from formats like "@ LAL" or "vs LAL"
        df['Opponent'] = df['Opp'].str.extract(r'(?:vs|@)\s+([A-Z]{3})')
        df['IsHome'] = df['Opp'].str.contains('vs')
    
    # Add advanced statistics
    if all(col in df.columns for col in ['FGA', 'FTA', 'TOV', 'MP']):
        # Usage rate calculation
        df['USG%'] = (df['FGA'] + 0.44*df['FTA'] + df['TOV']) / df['MP']
    
    if all(col in df.columns for col in ['PTS', 'FGA', 'FTA']):
        # True shooting percentage
        df['TS%'] = df['PTS'] / (2 * (df['FGA'] + 0.44*df['FTA']))
    
    # Calculate points per minute and per shot
    if all(col in df.columns for col in ['PTS', 'MP', 'FGA']):
        df['PTS_per_MIN'] = df['PTS'] / df['MP']
        df['PTS_per_FGA'] = df['PTS'] / df['FGA'].replace(0, np.nan)
    
    # Create rolling averages with different window sizes
    if 'Data' in df.columns:
        windows = [3, 5, 10]  # Last 3, 5, and 10 games
        stats_to_roll = ['MP', 'PTS', 'FGA', '3PA', 'FTA', 'AST', 'TOV', 'FG%', '3P%', 'FT%', 'REB', 'STL', 'BLK']
        
        for window in windows:
            for stat in stats_to_roll:
                if stat in df.columns:
                    col_name = f'{stat}_avg{window}'
                    df[col_name] = df.groupby('Player')[stat].transform(
                        lambda x: x.shift(1).rolling(window, min_periods=1).mean())
    
    # Add matchup-specific statistics if applicable
    if all(col in df.columns for col in ['Player', 'Opponent', 'PTS']):
        df['vs_opp_PTS'] = df.groupby(['Player', 'Opponent'])['PTS'].transform(
            lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    
    # Add rest days information if date column exists
    if 'Data' in df.columns:
        df['Data'] = pd.to_datetime(df['Data'])
        df['rest_days'] = df.groupby('Player')['Data'].transform(
            lambda x: (x - x.shift(1)).dt.days)
    
    # Fill NaN values with reasonable defaults
    for col in df.columns:
        if df[col].dtype in [np.float64, np.int64]:
            if col.endswith(('_avg3', '_avg5', '_avg10')) or col.startswith('vs_opp_'):
                # Fill rolling averages with column means
                df[col] = df[col].fillna(df[col].mean())
    
    print(f"Prepared dataset with {len(df)} rows and {len(df.columns)} columns")
    return df

# === Model training ===
def train_prediction_model(df, tune_hyperparams=False):
    """
    Train the prediction model on the provided dataframe
    
    Args:
        df: DataFrame with prepared NBA data
        tune_hyperparams: Whether to perform hyperparameter tuning
        
    Returns:
        tuple: Trained model, feature list, and processed dataframe
    """
    # Create a copy to avoid modifying the original
    df_model = df.copy()
    
    # Encode categorical variables
    categorical_cols = []
    for col in ['Tm', 'Opponent']:
        if col in df_model.columns:
            categorical_cols.append(col)
            
    if categorical_cols:
        df_encoded = pd.get_dummies(df_model, columns=categorical_cols, prefix=['team', 'opp'])
    else:
        df_encoded = df_model
    
    # Define feature groups
    rolling_features = [col for col in df_encoded.columns if any(f'_avg{w}' in col for w in [3, 5, 10])]
    team_features = [col for col in df_encoded.columns if col.startswith('team_') or col.startswith('opp_')]
    advanced_features = ['USG%', 'TS%', 'PTS_per_MIN', 'PTS_per_FGA']
    advanced_features = [f for f in advanced_features if f in df_encoded.columns]
    
    # Context features (like rest, home/away)
    context_features = []
    for cf in ['rest_days', 'IsHome']:
        if cf in df_encoded.columns:
            context_features.append(cf)
    
    # Matchup features
    matchup_features = [col for col in df_encoded.columns if col.startswith('vs_opp_')]
    
    # Combine all features and remove duplicates
    features = rolling_features + team_features + context_features + advanced_features + matchup_features
    features = list(set(features))
    features = [f for f in features if f in df_encoded.columns]
    
    print(f"Using {len(features)} features for model training")
    
    # Split data
    X = df_encoded[features].copy()
    y = df_encoded['PTS'].copy()
    
    # Handle missing values in features
    X = X.fillna(X.mean())
    
    # Time-based split to avoid data leakage
    train_size = 0.8
    train_idx = int(len(X) * train_size)
    X_train, X_test = X.iloc[:train_idx], X.iloc[train_idx:]
    y_train, y_test = y.iloc[:train_idx], y.iloc[train_idx:]
    
    # Model with default parameters
    model = XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        objective='reg:squarederror',
        random_state=42
    )
    
    # Hyperparameter tuning if requested
    if tune_hyperparams:
        print("Performing hyperparameter tuning...")
        param_grid = {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [4, 6, 8],
            'min_child_weight': [1, 3, 5],
            'subsample': [0.8, 0.9, 1.0]
        }
        
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            scoring='neg_mean_absolute_error',
            cv=5,
            n_jobs=-1
        )
        
        grid_search.fit(X_train, y_train)
        print(f"Best parameters: {grid_search.best_params_}")
        model = grid_search.best_estimator_
    else:
        # Train with default parameters
        model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Model performance - MAE: {mae:.2f}, R²: {r2:.4f}")
    
    # Feature importance
    importance = model.feature_importances_
    importance_df = pd.DataFrame({
        'Feature': features,
        'Importance': importance
    }).sort_values('Importance', ascending=False)
    
    print("Top 10 important features:")
    print(importance_df.head(10))
    
    return model, features, df_encoded

# === Player stats functions ===
def fetch_player_recent_stats(player_name, n_games=5):
    """
    Fetch recent real-world stats for a player using the NBA API
    
    Args:
        player_name: Player name to search for
        n_games: Number of recent games to retrieve
        
    Returns:
        DataFrame with player's recent game stats
    """
    # Define headers for NBA.com API requests
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Referer': 'https://www.nba.com/',
        'x-nba-stats-origin': 'stats',
        'x-nba-stats-token': 'true'
    }
    
    try:
        # First, get player ID
        first_name, last_name = player_name.split(' ', 1)
        
        # Try NBA.com API - typically needs player ID first
        # For demonstration, let's use a publicly accessible NBA API
        # There are several NBA APIs like balldontlie.io, rapidapi, etc.
        
        # Fetch from balldontlie.io API
        search_url = f"https://www.balldontlie.io/api/v1/players?search={player_name}"
        
        response = requests.get(search_url)
        if response.status_code == 200:
            players = response.json()['data']
            
            if not players:
                print(f"No players found matching '{player_name}'")
                return None
            
            # Find best match
            player_id = players[0]['id']
            
            # Fetch stats for this player
            stats_url = f"https://www.balldontlie.io/api/v1/stats?player_ids[]={player_id}&per_page={n_games}"
            stats_response = requests.get(stats_url)
            
            if stats_response.status_code == 200:
                stats_data = stats_response.json()['data']
                
                # Convert to DataFrame
                if stats_data:
                    stats_df = pd.DataFrame(stats_data)
                    
                    # Restructure if needed
                    # This depends on the API response structure
                    
                    return stats_df
                else:
                    print(f"No recent stats found for {player_name}")
            else:
                print(f"Failed to fetch stats: {stats_response.status_code}")
        else:
            print(f"Failed to search player: {response.status_code}")
            
        # Fallback to NBA.com API or other sources could be implemented here
        
    except Exception as e:
        print(f"Error fetching player stats: {str(e)}")
        traceback.print_exc()
    
    return None

# === Prediction function ===
def predict_player_points(player_name, opponent_team, model, features, df_full):
    """
    Predict points for a player against a specific opponent
    
    Args:
        player_name: Player name
        opponent_team: Opponent team abbreviation
        model: Trained prediction model
        features: List of features used by the model
        df_full: Full dataset with player history
        
    Returns:
        dict: Prediction results
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
    
    # Get player's recent games
    player_games = df_full[df_full['Player'] == player_name].sort_values('Data', ascending=False)
    
    if len(player_games) < 3:
        return f"Not enough data for {player_name}. Only {len(player_games)} games available."
    
    # Get recent boxscores for display
    recent_games = player_games.head(5)
    
    # Calculate input features for prediction
    input_dict = {}
    
    # Add rolling averages
    for feature in features:
        if feature.endswith(('_avg3', '_avg5', '_avg10')):
            # Get the most recent value for this feature
            if feature in player_games.columns:
                input_dict[feature] = player_games[feature].iloc[0]
            else:
                # If feature doesn't exist, use appropriate default
                base_stat = feature.split('_avg')[0]
                if base_stat in player_games.columns:
                    window_size = int(feature.split('_avg')[1])
                    input_dict[feature] = player_games[base_stat].head(window_size).mean()
                else:
                    input_dict[feature] = 0
    
    # Add advanced stats
    for feat in ['USG%', 'TS%', 'PTS_per_MIN', 'PTS_per_FGA']:
        if feat in features and feat in player_games.columns:
            input_dict[feat] = player_games[feat].iloc[0]
        elif feat in features:
            input_dict[feat] = 0
    
    # Add team and opponent encoding
    # Get player's team
    player_team = player_games['Tm'].iloc[0] if 'Tm' in player_games.columns else None
    
    # Set all team features to 0
    for feat in features:
        if feat.startswith('team_'):
            team_code = feat[5:]  # Extract team code
            input_dict[feat] = 1 if player_team == team_code else 0
        elif feat.startswith('opp_'):
            opp_code = feat[4:]  # Extract opponent code
            input_dict[feat] = 1 if opponent_team == opp_code else 0
    
    # Add context features
    if 'rest_days' in features:
        input_dict['rest_days'] = 2  # Default to 2 days of rest
    
    if 'IsHome' in features:
        # Determine if game is likely home or away based on recent patterns
        # This is just an estimate
        input_dict['IsHome'] = 0.5  # Default to 50/50
    
    # Add matchup-specific features
    for feat in features:
        if feat.startswith('vs_opp_'):
            stat_name = feat[7:]  # Extract stat name
            
            # Look for games against this opponent
            opp_games = player_games[player_games['Opponent'] == opponent_team]
            
            if len(opp_games) > 0 and stat_name in opp_games.columns:
                input_dict[feat] = opp_games[stat_name].mean()
            else:
                # Use overall average if no games against this opponent
                if stat_name in player_games.columns:
                    input_dict[feat] = player_games[stat_name].mean()
                else:
                    input_dict[feat] = 0
    
    # Create DataFrame for prediction
    input_df = pd.DataFrame([input_dict])
    
    # Ensure all model features are present
    for feat in features:
        if feat not in input_df.columns:
            input_df[feat] = 0
    
    # Make prediction
    try:
        predicted_pts = model.predict(input_df[features])[0]
    except Exception as e:
        return f"Error during prediction: {str(e)}"
    
    # Get player stats for context
    season_avg = player_games['PTS'].mean()
    recent_avg = player_games.head(3)['PTS'].mean()
    recent_variance = player_games.head(5)['PTS'].std()
    
    # Create result dictionary
    result = {
        "player": player_name,
        "opponent": opponent_team,
        "predicted_points": predicted_pts,
        "season_avg": season_avg,
        "recent_avg": recent_avg,
        "recent_variance": recent_variance,
        "recent_games": recent_games[['Data', 'Opp', 'PTS']].to_dict('records') if 'Data' in recent_games.columns else []
    }
    
    return result

# === Evaluate betting line ===
def evaluate_over_under(prediction_result, line):
    """
    Evaluate over/under betting line based on prediction
    
    Args:
        prediction_result: Dictionary with prediction results
        line: Betting line value
    
    Returns:
        dict: Updated prediction results with betting evaluation
    """
    result = prediction_result.copy()
    
    predicted_points = result["predicted_points"]
    recent_variance = result.get("recent_variance", 5.0)  # Default if not available
    
    # Calculate difference from line
    diff = predicted_points - line
    
    # Calculate confidence based on difference and variance
    # Higher variance = lower confidence
    confidence = min(3, max(0, abs(diff) / (recent_variance/2)))
    
    # Add results to dictionary
    result["line"] = line
    result["diff_from_line"] = diff
    result["recommendation"] = "OVER" if diff > 0 else "UNDER"
    result["confidence"] = int(confidence)  # 0-3 scale
    
    return result

# === Main functions to use directly ===
def initialize_model(csv_path="2024-2025.csv", model_path="nba_points_model.joblib", features_path="model_features.joblib", force_retrain=False):
    """
    Initialize the prediction model
    
    Args:
        csv_path: Path to the NBA data CSV
        model_path: Path to save/load the model
        features_path: Path to save/load the feature list
        force_retrain: Whether to force retraining the model
    
    Returns:
        tuple: Model, features, and dataframe
    """
    # Check if model exists and we're not forcing a retrain
    if not force_retrain:
        try:
            print(f"Looking for existing model at {model_path}...")
            model = joblib.load(model_path)
            features = joblib.load(features_path)
            
            print("Loading dataset...")
            df = load_and_prepare_data(csv_path)
            
            print("Model and data loaded successfully")
            return model, features, df
            
        except (FileNotFoundError, Exception) as e:
            print(f"Need to train new model: {str(e)}")
    else:
        print("Forcing model retraining...")
    
    # Load and prepare data
    print("Loading and preparing data...")
    df = load_and_prepare_data(csv_path)
    
    # Train model
    print("Training model...")
    model, features, df = train_prediction_model(df)
    
    # Save model
    print("Saving model...")
    joblib.dump(model, model_path)
    joblib.dump(features, features_path)
    
    return model, features, df

# Function to normalize player/team names
def normalize_name(name, name_type="player"):
    """
    Normalize player or team name
    
    Args:
        name: Name to normalize
        name_type: "player" or "team"
    
    Returns:
        Normalized name
    """
    if name_type == "player":
        # Convert to title case
        normalized = name.title()
        
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
            "The Joker": "Nikola Jokić",
            "Joker": "Nikola Jokić",
            "Greek Freak": "Giannis Antetokounmpo",
            "Zion": "Zion Williamson",
            "AD": "Anthony Davis"
        }
        
        if normalized in nickname_map:
            return nickname_map[normalized]
        
        return normalized
    
    elif name_type == "team":
        name = name.upper()
        
        # Team name mappings
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
        
        if name in team_map:
            return team_map[name]
        
        # If it's already a standard abbreviation, return as is
        common_abbrevs = {"LAL", "GSW", "BKN", "BOS", "MIL", "MIA", "CHI", "PHX", "DAL",
                        "PHI", "LAC", "NYK", "ATL", "DEN", "TOR", "UTA", "POR", "MEM",
                        "NOP", "SAC", "SAS", "IND", "CHA", "WAS", "DET", "OKC", "ORL",
                        "HOU", "CLE", "MIN"}
        if name in common_abbrevs:
            return name
        
        return name

# === Interactive prediction ===
def predict_interactive(model, features, df):
    """
    Interactive prediction loop
    
    Args:
        model: Trained model
        features: List of features
        df: Prepared dataset
    """
    print("\n" + "="*60)
    print("🏀  ENHANCED NBA PLAYER POINTS PREDICTOR  🏀")
    print("="*60)
    print("This tool predicts NBA player points based on real historical data.")
    print("="*60 + "\n")
    
    while True:
        try:
            # Get player name
            player_input = input("\nEnter player name (or 'quit' to exit): ")
            if player_input.lower() in ('quit', 'exit', 'q'):
                break
            
            # Normalize player name
            player_name = normalize_name(player_input, "player")
            
            # Get opponent
            opponent_input = input("Enter opponent team (e.g. 'LAL', 'Warriors'): ")
            opponent_team = normalize_name(opponent_input, "team")
            
            # Get betting line (optional)
            line_input = input("Enter betting line (optional, press enter to skip): ")
            line = float(line_input) if line_input and line_input.replace('.', '', 1).isdigit() else None
            
            print("\n⚙️ Processing prediction...")
            
            # Get prediction
            result = predict_player_points(player_name, opponent_team, model, features, df)
            
            # Handle string result (error message)
            if isinstance(result, str):
                print(f"ℹ️ {result}")
                continue
            
            # Evaluate betting line if provided
            if line is not None:
                result = evaluate_over_under(result, line)
            
            # Display results
            print("\n" + "="*60)
            print(f"🏀 PREDICTION FOR {result['player'].upper()} vs {result['opponent']}")
            print("="*60)
            
            print(f"\nPredicted points: {result['predicted_points']:.1f}")
            print(f"Season average: {result['season_avg']:.1f}")
            print(f"Recent average (3 games): {result['recent_avg']:.1f}")
            
            if line is not None:
                print("\n💰 BETTING RECOMMENDATION:")
                print(f"Line: {line} points")
                print(f"Recommendation: {result['recommendation']}")
                
                # Confidence level
                confidence_levels = ["No Confidence", "Low", "Medium", "High"]
                confidence_index = min(max(result.get('confidence', 0), 0), 3)
                print(f"Confidence: {confidence_levels[confidence_index]}")
            
            # Recent games
            if result['recent_games']:
                print("\n📜 RECENT PERFORMANCES:")
                for i, game in enumerate(result['recent_games'][:3], 1):
                    pts = game.get('PTS', 'N/A')
                    opp = game.get('Opp', 'Unknown')
                    date = game.get('Data', 'Unknown Date')
                    if isinstance(date, pd.Timestamp):
                        date = date.strftime('%Y-%m-%d')
                    print(f"  Game {i}: {pts} pts vs {opp} ({date})")
            
            # Option to continue
            again = input("\nMake another prediction? (y/n): ")
            if again.lower() not in ('y', 'yes'):
                break
                
        except KeyboardInterrupt:
            print("\nExiting prediction system.")
            break
        except Exception as e:
            print(f"\n❌ Error during prediction: {str(e)}")
            traceback.print_exc()

# === Main function ===
def main():
    """Main function to run the NBA prediction model"""
    try:
        # Initialize the model
        model, features, df = initialize_model()
        
        # Run interactive prediction
        predict_interactive(model, features, df)
        
    except KeyboardInterrupt:
        print("\nExiting prediction system.")
    except Exception as e:
        print(f"Error: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    main()