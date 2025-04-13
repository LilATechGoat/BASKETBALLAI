import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor
import joblib
from difflib import get_close_matches


# === Load and clean data ===
df = pd.read_csv("2024-2025.csv")


def convert_minutes(mp):
    try:
        if isinstance(mp, str) and ':' in mp:
            mins, secs = map(int, mp.split(':'))
            return mins + secs / 60
        else:
            return float(mp)
    except:
        return None


df['MP'] = df['MP'].apply(convert_minutes)


for col in ['FG%', '3P%', 'FT%']:
    df[col] = pd.to_numeric(df[col], errors='coerce')


df = df.dropna(subset=['PTS', 'MP'])
df = df.sort_values(['Player', 'Data'])


# Create rolling averages for key stats
rolling_stats = ['MP', 'FGA', '3PA', 'FTA', 'AST', 'TOV', 'FG%', '3P%', 'FT%']
for stat in rolling_stats:
    df[f'{stat}_avg3'] = df.groupby('Player')[stat].transform(lambda x: x.shift(1).rolling(3).mean())


df = df.dropna()


# Important: Save original team data before encoding
# Extract team info from encoded column names
teams = []
for col in df.columns:
    if col.startswith('team_'):
        teams.append(col[5:])  # Extract team abbreviation


# Create a copy of the original dataframe for later reference
df_original = df.copy()


# Encode categorical variables
# Check if Tm and Opp columns exist
if 'Tm' not in df.columns:
    print("'Tm' column not found. Using team_ columns for team information.")
    # No need to encode again, it's already encoded
    df_encoded = df
else:
    # Encode if Tm and Opp columns exist
    df_encoded = pd.get_dummies(df, columns=['Tm', 'Opp'], prefix=['team', 'opp'])


# Define features for model
rolling_features = [col for col in df_encoded.columns if '_avg3' in col]
team_features = [col for col in df_encoded.columns if col.startswith('team_') or col.startswith('opp_')]
features = rolling_features + team_features


X = df_encoded[features]
y = df_encoded['PTS']


# Split data for training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)


# Train XGBoost model
xgb_regressor = XGBRegressor(n_estimators=300, learning_rate=0.1, max_depth=5,
                            subsample=0.8, colsample_bytree=0.8, random_state=42)
xgb_regressor.fit(X_train, y_train)


# Evaluate model
y_pred = xgb_regressor.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Model MAE: {mae:.2f}")
print(f"Model R²: {r2:.2f}")


# Save the model
joblib.dump(xgb_regressor, 'nba_points_predictor.joblib')


def predict_player_points(player_name, opponent_team, model, df_full):
    """
    Predict points for a player against a specific opponent
   
    Args:
        player_name (str): Player name
        opponent_team (str): Opponent team abbreviation (e.g., MIA)
        model: Trained regression model
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
   
    # Check if opponent exists in the dataset
    opp_cols = [col for col in df_full.columns if col.startswith('opp_')]
    opp_col = f'opp_{opponent_team}'
    if opp_col not in opp_cols:
        # Try to find close matches for opponent
        all_opps = [col[4:] for col in opp_cols]  # Remove 'opp_' prefix
        close_matches = get_close_matches(opponent_team, all_opps, n=3, cutoff=0.6)
       
        if close_matches:
            return f"Opponent '{opponent_team}' not found. Did you mean: {', '.join(close_matches)}?"
        else:
            return f"Opponent '{opponent_team}' not found in the dataset."
   
    player_games = df_full[df_full['Player'] == player_name].sort_values('Data', ascending=False)
   
    if len(player_games) < 3:
        return f"Not enough data for {player_name}. Only {len(player_games)} games available."
   
    # Calculate rolling averages from last 3 games
    avg_stats = player_games.head(3)[rolling_stats].mean()
   
    # Prepare input dictionary for prediction
    input_dict = {f'{stat}_avg3': avg_stats[stat] for stat in rolling_stats}
   
    # Add team and opponent one-hot encoded features
    # Determine player's team from the team_ columns
    player_team = None
    team_cols = [col for col in player_games.columns if col.startswith('team_')]
    for col in team_cols:
        # Find the first team column that has a value of 1
        if player_games.iloc[0][col] == 1:
            player_team = col[5:]  # Remove 'team_' prefix
            break
   
    if player_team is None:
        return f"Could not determine team for player {player_name}"
   
    # Set all team columns to 0, then set player's team to 1
    for col in df_full.columns:
        if col.startswith('team_'):
            input_dict[col] = 1 if col == f'team_{player_team}' else 0
        elif col.startswith('opp_'):
            input_dict[col] = 1 if col == opp_col else 0
   
    # Ensure all features are included
    all_features = model.feature_names_in_
    for col in all_features:
        if col not in input_dict:
            input_dict[col] = 0
   
    # Make prediction
    X_input = pd.DataFrame([input_dict])[all_features]
    predicted_pts = model.predict(X_input)[0]
   
    # Get player stats for context
    season_avg = player_games['PTS'].mean()
    recent_avg = player_games.head(3)['PTS'].mean()
   
    # Calculate suggested line based on prediction with a slight adjustment
    # Using a simple approach: predicted points with a small random factor
    suggested_line = round(predicted_pts * np.random.uniform(0.97, 1.03), 1)
    # Make sure it ends in .5 for a true over/under line
    if suggested_line % 1 == 0:
        suggested_line += 0.5
    else:
        suggested_line = round(suggested_line - (suggested_line % 1) + 0.5, 1)
    
    result = {
        "player": player_name,
        "team": player_team,
        "opponent": opponent_team,
        "predicted_points": round(predicted_pts, 2),
        "suggested_line": suggested_line,
        "recent_games_avg": round(recent_avg, 2),
        "season_avg": round(season_avg, 2)
    }
    
    # ENHANCED: Additional metrics for better confidence calculation
    # Calculate standard deviation of recent performances (consistency metric)
    recent_games = player_games.head(5)
    result["recent_games_std"] = recent_games['PTS'].std()
    
    # Calculate trend more precisely (positive means scoring is increasing)
    if len(recent_games) >= 5:
        recent_trend = recent_games.iloc[0:2]['PTS'].mean() - recent_games.iloc[3:5]['PTS'].mean()
        result["recent_trend"] = recent_trend
    
    # Add opponent history count
    opp_games = player_games[player_games[f'opp_{opponent_team}'] == 1]
    result["vs_opponent_games"] = len(opp_games)
    
    # Use R² as a proxy for model accuracy for this player
    result["model_accuracy"] = r2  # Use global r2 or calculate player-specific r2 if possible
    
    # Add line movement data if available (would need external source)
    # result["line_movement"] = get_line_movement(player_name, opponent_team)  # Mock function
    
    return result


def evaluate_over_under(prediction_result, line):
    """
    Significantly enhanced evaluation for over/under prediction with better confidence calculation
    
    Args:
        prediction_result (dict): Output from predict_player_points
        line (float): Over/under betting line
        
    Returns:
        dict: Prediction result with over/under evaluation
    """
    predicted_pts = prediction_result["predicted_points"]
    prediction = "OVER" if predicted_pts > line else "UNDER"
    
    # Calculate baseline difference (as percentage of the line)
    points_difference = abs(predicted_pts - line)
    percentage_difference = points_difference / line
    
    # Base confidence starts higher (25-30% minimum)
    base_confidence = 25 + (percentage_difference * 100)
    
    # Factors that can increase confidence:
    confidence_factors = []
    
    # 1. Recent performance consistency (if available)
    recent_std = prediction_result.get("recent_games_std", 5.0)  # Default to medium variability
    consistency_factor = max(0, 20 - (recent_std * 2))  # Lower std = higher consistency
    confidence_factors.append(consistency_factor)
    
    # 2. History vs. specific opponent
    opp_history_points = 0
    if prediction_result.get("vs_opponent_avg", "N/A") != "N/A":
        opp_history_games = prediction_result.get("vs_opponent_games", 0)
        if opp_history_games > 0:
            opp_history_points = min(15, opp_history_games * 3)  # Up to 15 points for opponent history
    confidence_factors.append(opp_history_points)
    
    # 3. Model accuracy factor - increases confidence if the model has been accurate
    accuracy_boost = prediction_result.get("model_accuracy", 0.7) * 20  # 0-20 points based on accuracy
    confidence_factors.append(accuracy_boost)
    
    # 4. Strong trend factor
    trend_factor = 0
    if "recent_trend" in prediction_result:
        trend = prediction_result["recent_trend"]
        # If trend strongly supports prediction, boost confidence
        if (prediction == "OVER" and trend > 3) or (prediction == "UNDER" and trend < -3):
            trend_factor = 15
        elif (prediction == "OVER" and trend > 0) or (prediction == "UNDER" and trend < 0):
            trend_factor = 7
    confidence_factors.append(trend_factor)
    
    # 5. Line movement alignment (if available)
    line_movement_boost = 0
    if "line_movement" in prediction_result and prediction_result["line_movement"] is not None:
        movement = prediction_result["line_movement"]
        # If line is moving in same direction as our prediction, boost confidence
        if (prediction == "OVER" and movement > 0) or (prediction == "UNDER" and movement < 0):
            line_movement_boost = 10
    confidence_factors.append(line_movement_boost)
    
    # Apply distance from line effect - closer to line = lower confidence
    line_proximity_factor = 1.0
    if points_difference < 3:
        # Reduce confidence when very close to the line, but not below minimum
        line_proximity_factor = 0.5 + (points_difference / 6)  # 0.5 to 1.0
    
    # Calculate final confidence with all factors
    total_boost = sum(confidence_factors)
    adjusted_confidence = (base_confidence + total_boost) * line_proximity_factor
    
    # Ensure reasonable bounds (30-95%)
    final_confidence = min(95, max(30, adjusted_confidence))
    
    # Add explainability
    confidence_explanation = {
        "base_confidence": round(base_confidence, 1),
        "consistency_boost": round(consistency_factor, 1),
        "opponent_history_boost": round(opp_history_points, 1),
        "model_accuracy_boost": round(accuracy_boost, 1),
        "trend_boost": round(trend_factor, 1),
        "line_movement_boost": round(line_movement_boost, 1),
        "line_proximity_factor": round(line_proximity_factor, 2),
        "final_confidence": round(final_confidence, 1)
    }
    
    result = prediction_result.copy()
    result["line"] = line
    result["prediction"] = prediction
    result["confidence"] = f"{final_confidence:.1f}%"
    result["confidence_explanation"] = confidence_explanation
    result["points_diff_from_line"] = points_difference
    
    return result


# Function to display detailed confidence breakdown
def display_confidence_details(result):
    """
    Display detailed breakdown of confidence calculation
    """
    if "confidence_explanation" in result:
        exp = result["confidence_explanation"]
        print("\n=== Confidence Calculation Breakdown ===")
        print(f"Base confidence from prediction-line difference: {exp['base_confidence']}%")
        print(f"Boost from player scoring consistency: +{exp['consistency_boost']}%")
        print(f"Boost from history vs this opponent: +{exp['opponent_history_boost']}%")
        print(f"Boost from model accuracy: +{exp['model_accuracy_boost']}%")
        print(f"Boost from recent scoring trend: +{exp['trend_boost']}%")
        print(f"Boost from betting line movement: +{exp['line_movement_boost']}%")
        print(f"Line proximity adjustment factor: x{exp['line_proximity_factor']}")
        print(f"Final confidence: {exp['final_confidence']}%")
        
        # Add additional context for the prediction
        points_diff = result.get("points_diff_from_line", 0)
        if points_diff < 1:
            print("\nCAUTION: Prediction is extremely close to the line.")
        elif points_diff < 3:
            print("\nNOTE: Prediction is somewhat close to the line.")
        else:
            print("\nStrong point differential from the line increases confidence.")


# === Interactive CLI ===
if __name__ == "__main__":
    print("========================================")
    print("Welcome to the NBA Over/Under Predictor!")
    print("========================================")
   
    try:
        # Allow loading a saved model if it exists
        try:
            model_choice = input("Use saved model? (y/n): ").lower()
            if model_choice == 'y':
                xgb_regressor = joblib.load('nba_points_predictor.joblib')
                print("Saved model loaded successfully.")
        except FileNotFoundError:
            print("No saved model found, using newly trained model.")
       
        while True:
            print("\nEnter player details (or 'quit' to exit, 'list players' to see available players):")
            player_input = input("Player name: ")
           
            if player_input.lower() == 'quit':
                break
           
            if player_input.lower() == 'list players':
                print("\nAvailable players in dataset:")
                for player in sorted(df['Player'].unique()):
                    print(f"- {player}")
                continue
               
            opponent_input = input("Opponent team abbreviation (e.g., MIA): ")
           
            if opponent_input.lower() == 'list teams':
                print("\nAvailable teams in dataset:")
                opp_cols = [col for col in df_encoded.columns if col.startswith('opp_')]
                for team in sorted([col[4:] for col in opp_cols]):
                    print(f"- {team}")
                continue
           
            # First get the point prediction
            prediction_result = predict_player_points(
                player_name=player_input,
                opponent_team=opponent_input,
                model=xgb_regressor,
                df_full=df_encoded
            )
           
            # If we got an error message, display it and continue
            if isinstance(prediction_result, str):
                print("\n===== Prediction Result =====")
                print(prediction_result)
                continue
           
            # Display the automatic point prediction first
            print("\n===== Automatic Point Prediction =====")
            print(f"Player: {prediction_result['player']} ({prediction_result['team']})")
            print(f"Opponent: {prediction_result['opponent']}")
            print(f"Predicted Points: {prediction_result['predicted_points']}")
            print(f"Suggested Line: {prediction_result['suggested_line']}")
            print(f"Recent 3-Game Average: {prediction_result['recent_games_avg']} points")
            print(f"Season Average: {prediction_result['season_avg']} points")
           
            # Ask if the user wants to evaluate an over/under line
            line_choice = input("\nWould you like to evaluate an over/under line? (y/n): ").lower()
           
            if line_choice == 'y':
                # Use the suggested line as default
                suggested = prediction_result['suggested_line']
                line_input_str = input(f"Enter Over/Under line (press Enter for suggested {suggested}): ")
               
                if line_input_str.strip() == '':
                    line_input = suggested
                else:
                    try:
                        line_input = float(line_input_str)
                    except ValueError:
                        print(f"Invalid line value. Using suggested line of {suggested}")
                        line_input = suggested
               
                # Evaluate the over/under
                result = evaluate_over_under(prediction_result, line_input)
               
                print("\n===== Over/Under Evaluation =====")
                print(f"Player: {result['player']} ({result['team']})")
                print(f"Opponent: {result['opponent']}")
                print(f"Line: {result['line']} points")
                print(f"Prediction: {result['prediction']}")
                print(f"Confidence: {result['confidence']}")
                print(f"Predicted Points: {result['predicted_points']}")
                
                # NEW: Display detailed confidence explanation
                display_confidence_details(result)
               
    except KeyboardInterrupt:
        print("\nExiting program.")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        import traceback
        traceback.print_exc()
   
    print("\nThank you for using the NBA Over/Under Predictor!")