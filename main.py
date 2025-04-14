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

# Import the necessary functions and classes from your NBA prediction module
from nba_prediction import (
    load_and_prepare_data,
    train_prediction_model,
    predict_player_points,
    NBADataEnhancer
)

# === User Interface for NBA Prediction ===
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
    
    def _get_player_team(self, result):
        """Extract player's team from prediction result"""
        if self.enhancer is None:
            return None
            
        api_data = result.get("api_enhanced_data", {})
        if "recent_weighted_pts" in api_data:
            return self.enhancer._get_player_team(result["player"])
        return None
        
    def _display_prediction_result(self, result, player_name, opponent_team, line):
        """Display the prediction results in a user-friendly format"""
        print("\n" + "="*60)
        print(f"🏀 PREDICTION FOR {player_name.upper()} vs {opponent_team}")
        print("="*60)
        
        # Handle live game scenario
        if "live_game_factors" in result:
            live_factors = result["live_game_factors"]
            
            if live_factors["game_status"] == "LIVE":
                print(f"📊 LIVE GAME STATUS: Q{live_factors.get('current_quarter', 1)}")
                print(f"Current points: {live_factors.get('current_pts', 0)}")
                print(f"Minutes played: {live_factors.get('current_mins', 0)}")
                
                if live_factors.get("foul_trouble", False):
                    print("⚠️ ALERT: Player in foul trouble")
                    
                if live_factors.get("blowout", False):
                    print("⚠️ ALERT: Game is a blowout")
                    
                print("\n🔮 FINAL PROJECTION:")
                
            elif live_factors["game_status"] == "SCHEDULED":
                print(f"📅 Game scheduled today at {live_factors.get('game_time', 'TBD')}")
                print(f"Location: {'Home' if live_factors.get('is_home', False) else 'Away'}")
        
        # Prediction details
        print(f"\nPredicted points: {result['predicted_points']:.1f}")
        print(f"Season average: {result['season_avg']:.1f}")
        print(f"Recent average (3 games): {result['recent_avg']:.1f}")
        
        # Confidence indicators
        print("\n📊 PREDICTION CONFIDENCE:")
        confidence = self._calculate_confidence(result)
        confidence_bar = "█" * min(10, int(confidence * 10))
        print(f"[{confidence_bar.ljust(10)}] {int(confidence * 100)}%")
        
        # Injury information
        api_data = result.get("api_enhanced_data", {})
        injury_status = api_data.get("injury_status", "Active")
        if injury_status != "Active":
            print(f"\n⚠️ INJURY ALERT: {injury_status}")
        
        # Back-to-back status
        if api_data.get("is_back_to_back", 0) == 1:
            print("⚠️ ALERT: Team is on back-to-back games")
        
        # Betting recommendation - always show this now
        is_auto_line = line is None or line == result.get("suggested_line")
        line_to_use = result.get("suggested_line") if line is None else line
        
        print("\n💰 BETTING RECOMMENDATION:")
        print(f"Line: {line_to_use} points {'(auto-generated)' if is_auto_line else ''}")
        print(f"Recommendation: {result['recommendation']}")
        
        # Display confidence level
        confidence_level = ["", "Low", "Medium", "High"][min(3, result.get('confidence', 0))]
        print(f"Confidence: {confidence_level}")
        
        # Add reasoning
        diff = abs(result['predicted_points'] - line_to_use)
        print(f"Margin: {diff:.1f} points {'above' if result['predicted_points'] > line_to_use else 'below'} the line")

        # Recent games
        if "recent_games" in result and result["recent_games"]:
            print("\n📜 RECENT PERFORMANCES:")
            for i, game in enumerate(result["recent_games"][:3], 1):
                print(f"  Game {i}: {game['PTS']} pts vs {game['Opp']} ({game['Data']})")

    def _calculate_confidence(self, result):
        """Calculate confidence level for prediction (0-1 scale)"""
        # Start with base confidence
        confidence = 0.7
        
        # Factors that increase confidence
        if "recent_games" in result and len(result["recent_games"]) >= 3:
            confidence += 0.05  # More recent games data
            
        if "api_enhanced_data" in result:
            api_data = result["api_enhanced_data"]
            if "recent_weighted_pts" in api_data:
                confidence += 0.05  # Have recent weighted stats
                
            # Injury affects confidence
            if api_data.get("injury_status", "Active") != "Active":
                confidence -= 0.2  # Lower confidence if injured
                
            # Back-to-back affects confidence
            if api_data.get("is_back_to_back", 0) == 1:
                confidence -= 0.05
        
        # Live game affects confidence
        if "live_game_factors" in result:
            live_data = result["live_game_factors"]
            if live_data.get("game_status") == "LIVE":
                confidence += 0.1  # Higher confidence with live data
                
                # But lower if in foul trouble or blowout
                if live_data.get("foul_trouble", False):
                    confidence -= 0.15
                if live_data.get("blowout", False):
                    confidence -= 0.1
        
        # Variance affects confidence
        if "recent_variance" in result:
            variance = result["recent_variance"]
            norm_variance = max(0, min(0.3, variance / 15))  # Normalize to 0-0.3 range
            confidence -= norm_variance  # Higher variance = lower confidence
        
        return max(0.2, min(0.95, confidence))  # Keep in range 0.2-0.95
    
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
                result = self.integrate_with_prediction_model(
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
                print(f"\n❌ Error during prediction: {str(e)}")
                traceback.print_exc()

    def integrate_with_prediction_model(self, player_name, opponent_team, line=None):
        """
        Wrapper method to handle predictions in case the enhancer lacks the method
        """
        try:
            # Try using the enhancer's method if it exists
            if hasattr(self.enhancer, 'integrate_with_prediction_model'):
                return self.enhancer.integrate_with_prediction_model(
                    self.predictor, player_name, opponent_team, line
                )
            else:
                # Fallback to direct prediction if method doesn't exist
                print("Using basic prediction (no enhancement)")
                result = self.predictor(player_name, opponent_team)
                
                # If the result is a string (error message), return it
                if isinstance(result, str):
                    return result
                    
                # For a numeric line, handle over/under evaluation
                if line is not None:
                    from nba_prediction import evaluate_over_under
                    result = evaluate_over_under(result, line)
                    
                return result
        except Exception as e:
            print(f"Error in prediction: {str(e)}")
            return f"Prediction failed: {str(e)}"


# Main Function
def main():
    """Main function to run the NBA prediction model"""
    parser = argparse.ArgumentParser(description='NBA Player Points Predictor')
    parser.add_argument('--player', type=str, help='Player name to predict')
    parser.add_argument('--opponent', type=str, help='Opponent team abbreviation')
    parser.add_argument('--line', type=float, help='Betting line (optional)')
    parser.add_argument('--interactive', action='store_true', help='Run in interactive mode')
    parser.add_argument('--csv', type=str, default='2024-2025.csv', help='CSV file path with player data')
    
    args = parser.parse_args()
    
    try:
        # Initialize the UI
        ui = NBAPredictionUI()
        
        if args.interactive or not (args.player and args.opponent):
            # Run in interactive mode
            ui.run_interactive()
        else:
            # Run in command line mode
            if not ui.setup():
                print("Failed to set up prediction system. Exiting.")
                return
                
            # Normalize input
            player_name = ui.normalize_player_name(args.player)
            opponent_team = ui.normalize_team_name(args.opponent)
            
            # Get prediction
            result = ui.enhancer.integrate_with_prediction_model(
                ui.predictor,
                player_name,
                opponent_team,
                args.line
            )
            
            # Display results
            if isinstance(result, str):
                print(result)
            else:
                ui._display_prediction_result(result, player_name, opponent_team, args.line)
        
    except KeyboardInterrupt:
        print("\nExiting prediction system.")
    except Exception as e:
        print(f"Error: {str(e)}")
        traceback.print_exc()


if __name__ == "__main__":
    main()