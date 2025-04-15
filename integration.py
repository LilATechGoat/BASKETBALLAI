import argparse
import os
import sys
import pandas as pd
import numpy as np
import traceback
from datetime import datetime

# Import our modules
from nba_prediction import (
    load_and_prepare_data,
    train_prediction_model,
    predict_player_points,
    evaluate_over_under,
    normalize_name,
    initialize_model
)
from real_data_enhancement import NBADataEnhancer
from main_application import NBAPredictionUI
from visualization import NBAVisualizer
from data_converter import convert_standard_nba_data, convert_nba_boxscores, integrate_datasets

class NBAPredictorApp:
    """
    Integrated application for NBA Player Points Prediction
    """
    def __init__(self):
        """Initialize the NBA Predictor App"""
        self.ui = NBAPredictionUI()
        self.visualizer = NBAVisualizer()
        self.df = None
        self.model = None
        self.features = None
    
    def setup(self, csv_path="2024-2025.csv", force_retrain=False):
        """
        Set up the prediction system
        
        Args:
            csv_path: Path to the main dataset
            force_retrain: Whether to force retraining the model
        
        Returns:
            bool: Whether setup was successful
        """
        try:
            # First check if CSV exists
            if not os.path.exists(csv_path):
                print(f"❌ Data file {csv_path} not found")
                return False
            
            # Initialize model and data
            self.model, self.features, self.df = initialize_model(
                csv_path=csv_path,
                force_retrain=force_retrain
            )
            
            # Set up the UI with our model
            def predict_wrapper(player_name, opponent_team):
                return predict_player_points(player_name, opponent_team, self.model, self.features, self.df)
            
            self.ui.predictor = predict_wrapper
            
            # Initialize data enhancer
            self.ui.enhancer = NBADataEnhancer(self.df)
            
            print("✅ Predictor system initialized successfully")
            return True
        
        except Exception as e:
            print(f"❌ Error setting up prediction system: {str(e)}")
            traceback.print_exc()
            return False
    
    def run_interactive(self, visualize=True):
        """
        Run the prediction system in interactive mode
        
        Args:
            visualize: Whether to generate visualizations
        """
        if not self.ui.predictor or not self.ui.enhancer:
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
                
                # Check for special commands
                if player_input.lower() == "help":
                    self._show_help()
                    continue
                
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
                result = self.ui.integrate_with_prediction_model(
                    player_name,
                    opponent_team,
                    line
                )
                
                # Handle string result (error message)
                if isinstance(result, str):
                    print(f"ℹ️ {result}")
                    continue
                
                # Get live game factors
                player_team = self.ui._get_player_team(result)
                if player_team:
                    live_factors = self.ui.enhancer.get_live_game_factors(player_name, player_team)
                    
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
                result['confidence'] = self.ui._calculate_confidence(result)
                
                # Display results
                self.ui._display_prediction_result(result, player_name, opponent_team, line)
                
                # Generate visualizations if requested
                if visualize:
                    # Filter data for this player
                    player_data = self.df[self.df['Player'] == player_name]
                    
                    if len(player_data) > 0:
                        print("\n📊 Generating visualizations...")
                        
                        # Create trend visualization
                        trend_file = f"{player_name.replace(' ', '_')}_trend.png"
                        self.visualizer.player_performance_trend(
                            player_data,
                            player_name,
                            result['predicted_points'],
                            line,
                            trend_file
                        )
                        
                        # Create matchup analysis
                        matchup_file = f"{player_name.replace(' ', '_')}_vs_{opponent_team}.png"
                        self.visualizer.matchup_analysis(
                            player_data,
                            opponent_team,
                            player_name,
                            result['predicted_points'],
                            matchup_file
                        )
                        
                        # Create full prediction analysis
                        prediction_file = f"{player_name.replace(' ', '_')}_{opponent_team}_prediction.png"
                        self.visualizer.prediction_analysis(
                            result,
                            player_data,
                            prediction_file
                        )
                        
                        print(f"✅ Visualizations saved to {self.visualizer.save_dir} directory")
                        
                        # Ask if user wants a full PDF report
                        pdf_choice = input("\nGenerate a complete PDF report? (y/n): ")
                        if pdf_choice.lower() in ('y', 'yes'):
                            pdf_file = f"{player_name.replace(' ', '_')}_{opponent_team}_report.pdf"
                            self.visualizer.create_pdf_report(result, player_data, pdf_file)
                            print(f"✅ PDF report saved to {os.path.join(self.visualizer.save_dir, pdf_file)}")
                    else:
                        print("\n⚠️ Not enough data for visualizations")
                
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
    
    def _show_help(self):
        """Display help information about the app"""
        print("\n" + "="*60)
        print("NBA PLAYER POINTS PREDICTOR - HELP")
        print("="*60)
        print("\nCOMMANDS:")
        print("  update [player]  - Update data for specific player")
        print("  quit/exit        - Exit the application")
        print("  help             - Show this help message")
        print("\nUSAGE TIPS:")
        print("• Enter full player names for best results (e.g., 'LeBron James')")
        print("• Team names can be full names or abbreviations (e.g., 'Lakers' or 'LAL')")
        print("• Betting lines should be entered as decimal numbers (e.g., '25.5')")
        print("• Visualizations are saved in the './visualizations' directory")
        print("="*60)
    
    def predict_single(self, player_name, opponent_team, line=None, visualize=True):
        """
        Make a single prediction and return the result
        
        Args:
            player_name: Player name
            opponent_team: Opponent team
            line: Optional betting line
            visualize: Whether to generate visualizations
            
        Returns:
            Prediction result
        """
        if not self.ui.predictor or not self.ui.enhancer:
            if not self.setup():
                return "Failed to set up prediction system"
        
        # Normalize names
        player_name = normalize_name(player_name, "player")
        opponent_team = normalize_name(opponent_team, "team")
        
        # Get prediction
        result = self.ui.integrate_with_prediction_model(
            player_name,
            opponent_team,
            line
        )
        
        # Handle string result (error message)
        if isinstance(result, str):
            return result
        
        # Get live game factors
        player_team = self.ui._get_player_team(result)
        if player_team:
            live_factors = self.ui.enhancer.get_live_game_factors(player_name, player_team)
            if live_factors and "game_status" in live_factors:
                result["live_game_factors"] = live_factors
                
                # Adjust prediction if game is live
                if live_factors["game_status"] == "LIVE":
                    self._adjust_for_live_factors(result, live_factors)
        
        # Recalculate confidence after all adjustments
        result['confidence'] = self.ui._calculate_confidence(result)
        
        # Generate visualizations if requested
        if visualize:
            # Filter data for this player
            player_data = self.df[self.df['Player'] == player_name]
            
            if len(player_data) > 0:
                # Create visualizations
                prediction_file = f"{player_name.replace(' ', '_')}_{opponent_team}_prediction.png"
                self.visualizer.prediction_analysis(
                    result,
                    player_data,
                    prediction_file
                )
        
        return result
    
    def _adjust_for_live_factors(self, result, live_factors):
        """
        Adjust prediction based on live game factors
        
        Args:
            result: Prediction result to adjust
            live_factors: Live game factors
        """
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
    
    def batch_predict(self, input_file, output_file=None, visualize=False):
        """
        Run predictions on a batch of players from a CSV file
        
        Args:
            input_file: Path to input CSV with columns: Player,Opponent,Line
            output_file: Optional path to save results
            visualize: Whether to generate visualizations
            
        Returns:
            DataFrame with prediction results
        """
        if not self.ui.predictor or not self.ui.enhancer:
            if not self.setup():
                print("Failed to set up prediction system. Exiting.")
                return None
        
        try:
            # Load input file
            input_df = pd.read_csv(input_file)
            
            if 'Player' not in input_df.columns or 'Opponent' not in input_df.columns:
                print("❌ Input file must contain 'Player' and 'Opponent' columns")
                return None
            
            print(f"Processing {len(input_df)} predictions...")
            
            # Initialize results list
            results = []
            
            # Process each row
            for i, row in input_df.iterrows():
                player_name = row['Player']
                opponent_team = row['Opponent']
                line = float(row['Line']) if 'Line' in row and not pd.isna(row['Line']) else None
                
                print(f"({i+1}/{len(input_df)}) Predicting {player_name} vs {opponent_team}...")
                
                # Get prediction
                result = self.predict_single(player_name, opponent_team, line, visualize=visualize)
                
                # Handle string result (error message)
                if isinstance(result, str):
                    print(f"  ℹ️ {result}")
                    error_dict = {
                        'Player': player_name,
                        'Opponent': opponent_team,
                        'Line': line,
                        'Error': result
                    }
                    results.append(error_dict)
                else:
                    print(f"  ✅ Predicted points: {result['predicted_points']:.1f}")
                    
                    # Add result to list
                    results.append(result)
            
            # Convert results to DataFrame
            results_df = pd.DataFrame(results)
            
            # Save to output file if specified
            if output_file:
                results_df.to_csv(output_file, index=False)
                print(f"Results saved to {output_file}")
            
            return results_df
        
        except Exception as e:
            print(f"❌ Error processing batch predictions: {str(e)}")
            traceback.print_exc()
            return None

def main():
    """Main function to run the NBA prediction app"""
    parser = argparse.ArgumentParser(description='NBA Player Points Predictor')
    parser.add_argument('--player', type=str, help='Player name to predict')
    parser.add_argument('--opponent', type=str, help='Opponent team abbreviation')
    parser.add_argument('--line', type=float, help='Betting line (optional)')
    parser.add_argument('--interactive', action='store_true', help='Run in interactive mode')
    parser.add_argument('--csv', type=str, default='2024-2025.csv', help='CSV file path with player data')
    parser.add_argument('--boxscores', type=str, help='Path to boxscores CSV (optional)')
    parser.add_argument('--retrain', action='store_true', help='Force retraining of model')
    parser.add_argument('--batch', type=str, help='Path to CSV file with batch prediction requests')
    parser.add_argument('--output', type=str, help='Path to save batch prediction results')
    parser.add_argument('--convert', type=str, help='Convert data file to prediction format')
    parser.add_argument('--no-viz', action='store_true', help='Disable visualizations')
    
    args = parser.parse_args()
    
    # Create the app
    app = NBAPredictorApp()
    
    try:
        # Handle data conversion if requested
        if args.convert:
            output_path = args.output or args.convert.replace('.csv', '_converted.csv')
            
            if args.boxscores:
                print(f"Integrating {args.convert} with {args.boxscores}...")
                integrate_datasets(args.convert, args.boxscores, output_path)
            else:
                print(f"Converting {args.convert}...")
                convert_standard_nba_data(args.convert, output_path)
            
            print("✅ Conversion completed")
            return
        
        # Determine the data source
        csv_path = args.csv
        if args.boxscores:
            # Integrate CSV with boxscores
            print(f"Integrating {args.csv} with {args.boxscores}...")
            integrated_path = "integrated_data.csv"
            integrate_datasets(args.csv, args.boxscores, integrated_path)
            csv_path = integrated_path
        
        # Initialize the app
        app.setup(csv_path=csv_path, force_retrain=args.retrain)
        
        # Run in appropriate mode
        if args.batch:
            # Batch mode
            output_path = args.output or args.batch.replace('.csv', '_results.csv')
            app.batch_predict(args.batch, output_path, visualize=not args.no_viz)
        elif args.interactive or not (args.player and args.opponent):
            # Interactive mode
            app.run_interactive(visualize=not args.no_viz)
        else:
            # Single prediction mode
            result = app.predict_single(args.player, args.opponent, args.line, visualize=not args.no_viz)
            
            # Display results
            if isinstance(result, str):
                print(result)
            else:
                app.ui._display_prediction_result(result, args.player, args.opponent, args.line)
        
    except KeyboardInterrupt:
        print("\nExiting prediction system.")
    except Exception as e:
        print(f"Error: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    main()