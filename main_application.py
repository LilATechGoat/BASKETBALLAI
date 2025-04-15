import os
import argparse
import pandas as pd
import numpy as np
import joblib
import traceback
from datetime import datetime
import time

# Import our custom modules
from nba_prediction import (
    load_and_prepare_data,
    train_prediction_model,
    predict_player_points,
    evaluate_over_under,
    normalize_name,
    initialize_model
)
from real_data_enhancement import NBADataEnhancer

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
            # Initialize model
            model, features, df = initialize_model()
            
            # Define predictor function
            def predict_wrapper(player_name, opponent_team):
                return predict_player_points(player_name, opponent_team, model, features, df)
            
            self.predictor = predict_wrapper
            
            # Initialize NBA data enhancer
            self.enhancer = NBADataEnhancer(df)
            
            print("✅ Prediction system initialized successfully")
            return True
        except Exception as e:
            print(f"❌ Error setting up prediction system: {str(e)}")
            traceback.print_exc()
            return False
    
    def _get_player_team(self, result):
        """Extract player's team from prediction result"""
        if self.enhancer is None:
            return None
            
        api_data = result.get("api_enhanced_data", {})
        if "recent_weighted_pts" in api_data:
            return self.enhancer._get_player_team(result["player"])
        return None
    
    def _calculate_confidence(self, result):
        """Calculate confidence level for prediction (0-3 scale)"""
        # Start with base confidence
        confidence = 1.5
        
        # Factors that increase confidence
        if "recent_games" in result and len(result["recent_games"]) >= 3:
            confidence += 0.3  # More recent games data
            
        if "api_enhanced_data" in result:
            api_data = result["api_enhanced_data"]
            if "recent_weighted_pts" in api_data:
                confidence += 0.3  # Have recent weighted stats
                
            # Injury affects confidence
            if api_data.get("injury_status", "Active") != "Active":
                confidence -= 0.8  # Lower confidence if injured
                
            # Back-to-back affects confidence
            if api_data.get("is_back_to_back", 0) == 1:
                confidence -= 0.3
        
        # Live game affects confidence
        if "live_game_factors" in result:
            live_data = result["live_game_factors"]
            if live_data.get("game_status") == "LIVE":
                confidence += 0.5  # Higher confidence with live data
                
                # But lower if in foul trouble or blowout
                if live_data.get("foul_trouble", False):
                    confidence -= 0.7
                if live_data.get("blowout", False):
                    confidence -= 0.5
        
        # Variance affects confidence
        if "recent_variance" in result:
            variance = result["recent_variance"]
            norm_variance = max(0, min(1.0, variance / 15))  # Normalize to 0-1 range
            confidence -= norm_variance  # Higher variance = lower confidence
        
        # Line difference affects confidence
        if "diff_from_line" in result:
            diff = abs(result["diff_from_line"])
            if diff < 1:
                confidence -= 0.5  # Very close to line = lower confidence
            elif diff > 5:
                confidence += 0.5  # Far from line = higher confidence
        
        # Round to nearest integer in 0-3 range
        return max(0, min(3, round(confidence)))
    
    def _display_prediction_result(self, result, player_name, opponent_team, line):
        """Display the prediction result in a readable format"""
        # Safely extract and handle different result scenarios
        if isinstance(result, str):
            print(f"❌ Prediction Error: {result}")
            return
        
        # Ensure all expected keys exist with default values
        result = {
            'player': result.get('player', player_name),
            'opponent': result.get('opponent', opponent_team),
            'predicted_points': result.get('predicted_points', 0),
            'recommendation': result.get('recommendation', 'No recommendation'),
            'season_avg': result.get('season_avg', 0),
            'recent_avg': result.get('recent_avg', 0),
            'line': result.get('line', line),
            'confidence': result.get('confidence', 0),
            'recent_games': result.get('recent_games', [])
        }
        
        # Check for error in prediction
        if 'error' in result:
            print(f"❌ Prediction Error: {result['error']}")
            return

        print("\n" + "="*60)
        print(f"🏀 PREDICTION FOR {result['player'].upper()} vs {result['opponent']}")
        print("="*60)
    
        # Prediction details
        print(f"\nPredicted points: {result['predicted_points']:.1f}")
        print(f"Season average: {result['season_avg']:.1f}")
        print(f"Recent average (3 games): {result['recent_avg']:.1f}")
    
        # Live game information if available
        if "live_game_factors" in result:
            live_factors = result["live_game_factors"]
            if live_factors.get("game_status") == "LIVE":
                print("\n🔴 LIVE GAME STATUS:")
                print(f"Current points: {live_factors.get('current_pts', 0)}")
                print(f"Current minutes: {live_factors.get('current_mins', 0)}")
                print(f"Current quarter: Q{live_factors.get('current_quarter', 1)}")
                
                if live_factors.get("foul_trouble", False):
                    print("⚠️ Player is in foul trouble")
                
                if live_factors.get("blowout", False):
                    print("⚠️ Game is a blowout")
            
            elif live_factors.get("game_status") == "SCHEDULED":
                print(f"\n📅 Game scheduled today at {live_factors.get('game_time', 'TBD')}")
    
        # Enhanced data information if available
        if "api_enhanced_data" in result:
            api_data = result["api_enhanced_data"]
            
            if api_data.get("is_injured", 0) == 1:
                print(f"\n⚠️ INJURY ALERT: {api_data.get('injury_status', 'Unknown status')}")
            
            if api_data.get("is_back_to_back", 0) == 1:
                print("⚠️ Player is on a back-to-back game")
            
            team_injuries = api_data.get("team_injury_count", 0)
            opp_injuries = api_data.get("opp_injury_count", 0)
            
            if team_injuries > 0 or opp_injuries > 0:
                print(f"\n📋 TEAM STATUS:")
                if team_injuries > 0:
                    print(f"  • {team_injuries} player(s) injured on {result['player']}'s team")
                if opp_injuries > 0:
                    print(f"  • {opp_injuries} player(s) injured on opponent's team")
    
        # Betting recommendation
        if result['line'] is not None:
            print("\n💰 BETTING RECOMMENDATION:")
            line_to_display = result['line'] if result['line'] is not None else "No line"
            print(f"Line: {line_to_display} points")
            print(f"Recommendation: {result['recommendation']}")
    
            # Confidence level
            confidence_levels = ["No Confidence", "Low", "Medium", "High"]
            confidence_index = min(max(result['confidence'], 0), 3)
            print(f"Confidence: {confidence_levels[confidence_index]}")
    
        # Recent games (if available)
        if result['recent_games']:
            print("\n📜 RECENT PERFORMANCES:")
            for i, game in enumerate(result['recent_games'][:3], 1):
                # Safely extract game details
                pts = game.get('PTS', 'N/A')
                opp = game.get('Opp', 'Unknown')
                date = game.get('Data', 'Unknown Date')
                if isinstance(date, pd.Timestamp):
                    date = date.strftime('%Y-%m-%d')
                print(f"  Game {i}: {pts} pts vs {opp} ({date})")
    
    def integrate_with_prediction_model(self, player_name, opponent_team, line=None):
        """
        Wrapper method to handle predictions with enhancer integration
        """
        try:
            # First get the base prediction
            result = self.predictor(player_name, opponent_team)
            
            # If the result is a string (error message), return it
            if isinstance(result, str):
                return result
            
            # If enhancer is available, use it to improve prediction
            if self.enhancer and hasattr(self.enhancer, 'integrate_with_prediction_model'):
                # Get enhanced prediction
                result = self.enhancer.integrate_with_prediction_model(
                    self.predictor, player_name, opponent_team, line
                )
            else:
                # For a numeric line, handle over/under evaluation directly
                if line is not None:
                    result = evaluate_over_under(result, line)
            
            # Calculate confidence
            result['confidence'] = self._calculate_confidence(result)
            
            return result
        except Exception as e:
            print(f"Error in prediction: {str(e)}")
            traceback.print_exc()
            return f"Prediction failed: {str(e)}"
    
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
                player_name = normalize_name(player_input, "player")
                
                # Get opponent
                opponent_input = input("Enter opponent team (e.g. 'LAL', 'Warriors'): ")
                opponent_team = normalize_name(opponent_input, "team")
                
                # Get betting line (optional)
                line_input = input("Enter betting line (optional, press enter to skip): ")
                line = float(line_input) if line_input and line_input.replace('.', '', 1).isdigit() else None
                
                print("\n⚙️ Processing prediction...")
                
                # Get integrated prediction
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
                    live_factors = self.enhancer.get_live_game_factors(player_name, player_team)
                    
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
                
                # Recalculate confidence after all adjustments
                result['confidence'] = self._calculate_confidence(result)
                
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

def main():
    """Main function to run the NBA prediction model"""
    parser = argparse.ArgumentParser(description='NBA Player Points Predictor')
    parser.add_argument('--player', type=str, help='Player name to predict')
    parser.add_argument('--opponent', type=str, help='Opponent team abbreviation')
    parser.add_argument('--line', type=float, help='Betting line (optional)')
    parser.add_argument('--interactive', action='store_true', help='Run in interactive mode')
    parser.add_argument('--csv', type=str, default='2024-2025.csv', help='CSV file path with player data')
    parser.add_argument('--boxscores', type=str, help='Path to boxscores CSV (optional)')
    parser.add_argument('--retrain', action='store_true', help='Force retraining of model')
    
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
            player_name = normalize_name(args.player, "player")
            opponent_team = normalize_name(args.opponent, "team")
            
            # Get prediction
            result = ui.integrate_with_prediction_model(
                player_name,
                opponent_team,
                args.line
            )
            
            # Get live factors if applicable
            player_team = ui._get_player_team(result)
            if player_team and isinstance(result, dict):  # Make sure result is not an error string
                live_factors = ui.enhancer.get_live_game_factors(player_name, player_team)
                if live_factors and "game_status" in live_factors:
                    result["live_game_factors"] = live_factors
            
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