import pandas as pd
import numpy as np
import os
import re
import argparse
from datetime import datetime

def convert_nba_boxscores(boxscores_path, output_path=None):
    """
    Convert the NBA boxscores dataset to our prediction format
    
    Args:
        boxscores_path: Path to the NBA-BoxScores CSV
        output_path: Optional path to save the converted data
    
    Returns:
        DataFrame in prediction format
    """
    print(f"Loading NBA boxscores from {boxscores_path}...")
    
    # Load the boxscores data
    boxscores_df = pd.read_csv(boxscores_path)
    
    # Print the column names to understand the structure
    print(f"Original columns: {boxscores_df.columns.tolist()}")
    
    # Create a new dataframe in our prediction format
    prediction_data = []
    
    # Extract team abbreviations and player info
    for _, row in boxscores_df.iterrows():
        try:
            # Extract relevant data
            team_abbrev = row.get('TEAM_ABBREVIATION')
            team_city = row.get('TEAM_CITY')
            player_id = row.get('PLAYER_ID')
            player_name = row.get('PLAYER_NAME')
            nickname = row.get('NICKNAME', '')
            position = row.get('START_POSITION')
            minutes = row.get('MIN')
            
            # Game info
            game_id = row.get('GAME_ID')
            game_date = None
            
            # Try to extract date from game ID or other fields
            if isinstance(game_id, str) and len(game_id) >= 8:
                try:
                    # NBA game IDs often start with date in format YYYYMMDD
                    date_part = game_id[:8]
                    game_date = datetime.strptime(date_part, '%Y%m%d').strftime('%Y-%m-%d')
                except:
                    # If date extraction fails, try to get from other fields
                    if 'GAME_DATE' in row:
                        game_date = row['GAME_DATE']
            
            # If we still don't have a date, use a placeholder
            if not game_date:
                game_date = '2024-10-22'  # Use default date
            
            # Extract opponent from available data
            opponent = None
            if 'OPPONENT_TEAM_ABBREVIATION' in boxscores_df.columns:
                opponent = row.get('OPPONENT_TEAM_ABBREVIATION')
            else:
                # Try to determine opponent from game context
                # This would require more data about the game
                opponent = 'OPP'  # Placeholder
            
            # Format opponent for display (home/away)
            is_home = False
            if 'HOME_AWAY' in boxscores_df.columns:
                is_home = row.get('HOME_AWAY') == 'HOME'
            elif 'HOME_GAME' in boxscores_df.columns:
                is_home = row.get('HOME_GAME') == 1
            
            opp_display = f"vs {opponent}" if is_home else f"@ {opponent}"
            
            # Extract stats
            fg = float(row.get('FGM', 0))
            fga = float(row.get('FGA', 0))
            fg_pct = float(row.get('FG_PCT', 0)) if 'FG_PCT' in row else (fg / fga if fga > 0 else 0)
            
            fg3 = float(row.get('FG3M', 0))
            fg3a = float(row.get('FG3A', 0))
            fg3_pct = float(row.get('FG3_PCT', 0)) if 'FG3_PCT' in row else (fg3 / fg3a if fg3a > 0 else 0)
            
            ft = float(row.get('FTM', 0))
            fta = float(row.get('FTA', 0))
            ft_pct = float(row.get('FT_PCT', 0)) if 'FT_PCT' in row else (ft / fta if fta > 0 else 0)
            
            # Rebounds
            orb = float(row.get('OREB', 0))
            drb = float(row.get('DREB', 0))
            reb = float(row.get('REB', 0)) if 'REB' in row else (orb + drb)
            
            # Other stats
            ast = float(row.get('AST', 0))
            stl = float(row.get('STL', 0))
            blk = float(row.get('BLK', 0))
            tov = float(row.get('TOV', 0))
            pf = float(row.get('PF', 0))
            pts = float(row.get('PTS', 0))
            
            # Create entry in prediction format
            entry = {
                'Player': player_name,
                'Tm': team_abbrev,
                'Data': game_date,
                'Opp': opp_display,
                'MP': minutes,
                'FG': fg,
                'FGA': fga,
                'FG%': fg_pct,
                '3P': fg3,
                '3PA': fg3a,
                '3P%': fg3_pct,
                'FT': ft,
                'FTA': fta,
                'FT%': ft_pct,
                'ORB': orb,
                'DRB': drb,
                'REB': reb,
                'AST': ast,
                'STL': stl,
                'BLK': blk,
                'TOV': tov,
                'PF': pf,
                'PTS': pts
            }
            
            prediction_data.append(entry)
            
        except Exception as e:
            print(f"Error processing row: {e}")
    
    # Convert to DataFrame
    prediction_df = pd.DataFrame(prediction_data)
    
    # Save to output file if specified
    if output_path:
        prediction_df.to_csv(output_path, index=False)
        print(f"Converted data saved to {output_path}")
    
    print(f"Successfully converted {len(prediction_df)} rows of data")
    return prediction_df

def convert_standard_nba_data(input_path, output_path=None):
    """
    Convert standard NBA data format (like the one in screenshots) to prediction format
    
    Args:
        input_path: Path to the NBA data CSV
        output_path: Optional path to save the converted data
    
    Returns:
        DataFrame in prediction format
    """
    print(f"Loading NBA data from {input_path}...")
    
    # Load the data
    df = pd.read_csv(input_path)
    
    # Check for known format based on column names
    if all(col in df.columns for col in ['Player', 'Tm', 'Opp', 'MP', 'PTS']):
        print("Data already in prediction format, no conversion needed")
        
        # Save to output file if specified
        if output_path and output_path != input_path:
            df.to_csv(output_path, index=False)
            print(f"Data saved to {output_path}")
        
        return df
    
    # Print the column names to understand the structure
    print(f"Original columns: {df.columns.tolist()}")
    
    # Try to detect the exact format of the CSV
    # Here are common formats we want to handle:
    # 1. The format from the first screenshot (2024-2025.csv)
    # 2. The format from the second screenshot (NBA-BoxScores)
    # 3. Other standard formats from basketball reference or NBA.com
    
    # Look for key identifiers in column names
    is_player_game_log = any('Player' in col for col in df.columns) and any('PTS' in col for col in df.columns)
    is_boxscore = any('PLAYER_NAME' in col for col in df.columns) or any('TEAM_ABBREVIATION' in col for col in df.columns)
    
    if is_boxscore:
        return convert_nba_boxscores(input_path, output_path)
    
    # Create a new dataframe in our prediction format
    prediction_data = []
    
    # Handle the format from screenshot 1 (similar to basketball reference)
    for _, row in df.iterrows():
        try:
            # Determine the key columns
            player_col = [col for col in df.columns if 'Player' in col][0] if any('Player' in col for col in df.columns) else None
            team_col = [col for col in df.columns if 'Tm' in col][0] if any('Tm' in col for col in df.columns) else None
            opp_col = [col for col in df.columns if 'Opp' in col][0] if any('Opp' in col for col in df.columns) else None
            date_col = [col for col in df.columns if 'Data' in col or 'Date' in col][0] if any(col in ['Data', 'Date'] for col in df.columns) else None
            
            # If we can't find the necessary columns, try a different approach
            if not all([player_col, team_col, opp_col]):
                if len(df.columns) >= 3:
                    # Assume the first columns are Player, Tm, and Opp
                    player_col = df.columns[0]
                    team_col = df.columns[1] 
                    opp_col = df.columns[2]
            
            # Extract player info
            player_name = row[player_col] if player_col else f"Player_{_}"
            team = row[team_col] if team_col else "TEAM"
            opp = row[opp_col] if opp_col else "OPP"
            game_date = row[date_col] if date_col else "2024-10-22"
            
            # Initialize entry with core info
            entry = {
                'Player': player_name,
                'Tm': team,
                'Opp': opp,
                'Data': game_date
            }
            
            # Add all available stats
            for col in df.columns:
                if col in [player_col, team_col, opp_col, date_col]:
                    continue
                
                # Map column names to our standard format
                std_col = self._standardize_column_name(col)
                entry[std_col] = row[col]
            
            prediction_data.append(entry)
            
        except Exception as e:
            print(f"Error processing row: {e}")
    
    # Convert to DataFrame
    prediction_df = pd.DataFrame(prediction_data)
    
    # Save to output file if specified
    if output_path:
        prediction_df.to_csv(output_path, index=False)
        print(f"Converted data saved to {output_path}")
    
    print(f"Successfully converted {len(prediction_df)} rows of data")
    return prediction_df

def _standardize_column_name(col_name):
    """Standardize column names to our prediction format"""
    # Common mappings
    mapping = {
        'MIN': 'MP',
        'MINUTES': 'MP',
        'FGM': 'FG',
        'FG_PCT': 'FG%',
        'FG3M': '3P',
        'FG3A': '3PA',
        'FG3_PCT': '3P%',
        'FTM': 'FT',
        'FT_PCT': 'FT%',
        'OREB': 'ORB',
        'DREB': 'DRB',
        'REBOUNDS': 'REB',
        'ASSISTS': 'AST',
        'STEALS': 'STL',
        'BLOCKS': 'BLK',
        'TURNOVERS': 'TOV',
        'FOULS': 'PF',
        'POINTS': 'PTS',
        'DATE': 'Data',
        'TEAM': 'Tm',
        'OPPONENT': 'Opp'
    }
    
    # Check for exact match
    if col_name in mapping:
        return mapping[col_name]
    
    # Check for case-insensitive match
    for k, v in mapping.items():
        if col_name.upper() == k.upper():
            return v
    
    # Return original if no match
    return col_name

def integrate_datasets(main_path, boxscores_path, output_path):
    """
    Integrate main dataset with boxscores for more comprehensive data
    
    Args:
        main_path: Path to main dataset
        boxscores_path: Path to boxscores dataset
        output_path: Path to save integrated dataset
    
    Returns:
        Integrated DataFrame
    """
    # Load and convert both datasets
    main_df = convert_standard_nba_data(main_path)
    boxscores_df = convert_nba_boxscores(boxscores_path)
    
    # Combine datasets
    combined_df = pd.concat([main_df, boxscores_df], ignore_index=True)
    
    # Remove duplicates based on Player, Date, and Team
    if 'Data' in combined_df.columns and 'Player' in combined_df.columns and 'Tm' in combined_df.columns:
        combined_df = combined_df.drop_duplicates(subset=['Player', 'Data', 'Tm'], keep='first')
    
    # Sort by player and date
    if 'Data' in combined_df.columns and 'Player' in combined_df.columns:
        combined_df = combined_df.sort_values(['Player', 'Data'])
    
    # Save integrated dataset
    if output_path:
        combined_df.to_csv(output_path, index=False)
        print(f"Integrated data saved to {output_path}")
    
    return combined_df

def main():
    """Main function to run the data converter"""
    parser = argparse.ArgumentParser(description='NBA Data Converter')
    parser.add_argument('--input', type=str, required=True, help='Path to input CSV file')
    parser.add_argument('--output', type=str, help='Path to output CSV file')
    parser.add_argument('--boxscores', type=str, help='Path to boxscores CSV (for integration)')
    parser.add_argument('--format', type=str, choices=['auto', 'boxscores', 'standard'], default='auto', 
                       help='Format of input data (auto-detect by default)')
    
    args = parser.parse_args()
    
    try:
        if args.boxscores:
            # Integrate datasets
            integrate_datasets(args.input, args.boxscores, args.output)
        else:
            # Convert single dataset
            if args.format == 'boxscores' or (args.format == 'auto' and 'boxscores' in args.input.lower()):
                convert_nba_boxscores(args.input, args.output)
            else:
                convert_standard_nba_data(args.input, args.output)
    
    except Exception as e:
        print(f"Error during conversion: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()