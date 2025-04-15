import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from datetime import datetime
import os

class NBAVisualizer:
    """
    Class for creating visualizations of NBA player data and predictions
    """
    def __init__(self, save_dir='./visualizations'):
        """Initialize the visualizer with a directory to save images"""
        self.save_dir = save_dir
        
        # Create directory if it doesn't exist
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # Set style
        self.set_plot_style()
    
    def set_plot_style(self):
        """Set the visual style for plots"""
        sns.set_style("darkgrid")
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 12
        
        # Use NBA colors
        self.nba_colors = {
            'main_blue': '#17408B',
            'main_red': '#C9082A',
            'main_orange': '#F58426',
            'main_black': '#000000',
            'main_white': '#FFFFFF',
            'lakers_purple': '#552583',
            'lakers_gold': '#FDB927',
            'celtics_green': '#007A33',
            'warriors_blue': '#1D428A',
            'warriors_gold': '#FFC72C'
        }
        
        # Team-specific colors
        self.team_colors = {
            'LAL': self.nba_colors['lakers_purple'],
            'BOS': self.nba_colors['celtics_green'],
            'GSW': self.nba_colors['warriors_blue'],
            # Add more teams as needed
        }
    
    def player_performance_trend(self, player_data, player_name, prediction=None, line=None, output_file=None):
        """
        Create a visualization of a player's recent performance trend
        
        Args:
            player_data: DataFrame with player's historical data
            player_name: Name of the player
            prediction: Optional predicted points value
            line: Optional betting line value
            output_file: Optional filename to save the visualization
        
        Returns:
            Path to saved visualization file or None
        """
        if 'Data' not in player_data.columns or 'PTS' not in player_data.columns:
            print("Required columns 'Data' and 'PTS' not found in player data")
            return None
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Convert date column if needed
        if player_data['Data'].dtype == object:
            player_data['Data'] = pd.to_datetime(player_data['Data'])
        
        # Sort by date
        player_data = player_data.sort_values('Data')
        
        # Calculate rolling averages
        player_data['PTS_rolling_3'] = player_data['PTS'].rolling(window=3, min_periods=1).mean()
        player_data['PTS_rolling_5'] = player_data['PTS'].rolling(window=5, min_periods=1).mean()
        
        # Plot actual points
        ax.scatter(
            player_data['Data'],
            player_data['PTS'],
            color=self.nba_colors['main_orange'],
            s=100,
            alpha=0.7,
            label='Actual Points'
        )
        
        # Connect points with line
        ax.plot(
            player_data['Data'],
            player_data['PTS'],
            color=self.nba_colors['main_orange'],
            alpha=0.3
        )
        
        # Plot rolling averages
        ax.plot(
            player_data['Data'],
            player_data['PTS_rolling_3'],
            color=self.nba_colors['main_blue'],
            linewidth=2,
            label='3-Game Avg'
        )
        
        ax.plot(
            player_data['Data'],
            player_data['PTS_rolling_5'],
            color=self.nba_colors['main_red'],
            linewidth=2,
            label='5-Game Avg'
        )
        
        # Add season average line
        season_avg = player_data['PTS'].mean()
        ax.axhline(
            y=season_avg,
            color=self.nba_colors['main_black'],
            linestyle='--',
            alpha=0.7,
            label=f'Season Avg: {season_avg:.1f}'
        )
        
        # Add prediction and betting line if provided
        if prediction is not None:
            # Add a prediction point at the next game date (or just after the last game)
            last_date = player_data['Data'].max()
            next_date = last_date + pd.Timedelta(days=2)  # Assume next game is 2 days later
            
            ax.scatter(
                next_date,
                prediction,
                color=self.nba_colors['main_blue'],
                s=200,
                marker='*',
                label=f'Prediction: {prediction:.1f}'
            )
            
            # Add line from last game to prediction
            ax.plot(
                [player_data['Data'].iloc[-1], next_date],
                [player_data['PTS'].iloc[-1], prediction],
                color=self.nba_colors['main_blue'],
                linestyle=':',
                alpha=0.5
            )
        
        if line is not None:
            # Add betting line
            ax.axhline(
                y=line,
                color=self.nba_colors['main_red'],
                linestyle='-.',
                alpha=0.7,
                label=f'Betting Line: {line}'
            )
        
        # Add matchup info to the plot (show opponents)
        if 'Opp' in player_data.columns:
            for i, row in player_data.iterrows():
                opponent = row['Opp']
                points = row['PTS']
                game_date = row['Data']
                
                # Only add labels for some games to avoid cluttering
                if i % 3 == 0:  # every 3rd game
                    ax.annotate(
                        opponent,
                        (game_date, points),
                        xytext=(0, 10),
                        textcoords='offset points',
                        ha='center',
                        fontsize=8,
                        alpha=0.7
                    )
        
        # Set title and labels
        ax.set_title(f"{player_name} - Points Trend", fontsize=16, fontweight='bold')
        ax.set_xlabel('Game Date', fontsize=12)
        ax.set_ylabel('Points', fontsize=12)
        
        # Format x-axis to show dates nicely
        plt.xticks(rotation=45)
        fig.tight_layout()
        
        # Add legend
        ax.legend(loc='upper left')
        
        # Save figure if output file specified
        if output_file:
            save_path = os.path.join(self.save_dir, output_file)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            return save_path
        else:
            plt.show()
            return None
    
    def matchup_analysis(self, player_data, opponent_team, player_name, prediction=None, output_file=None):
        """
        Create a visualization of a player's performance against specific opponents
        
        Args:
            player_data: DataFrame with player's historical data
            opponent_team: Opponent team abbreviation
            player_name: Name of the player
            prediction: Optional predicted points value
            output_file: Optional filename to save the visualization
            
        Returns:
            Path to saved visualization file or None
        """
        if 'Opp' not in player_data.columns or 'PTS' not in player_data.columns:
            print("Required columns 'Opp' and 'PTS' not found in player data")
            return None
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # 1. Performance against all opponents
        # Group by opponent and calculate average points
        opp_avg = player_data.groupby('Opp')['PTS'].agg(['mean', 'count', 'std']).reset_index()
        opp_avg = opp_avg.sort_values('mean', ascending=False)
        
        # Color for the specific opponent
        colors = [self.nba_colors['main_red'] if opp in opponent_team else self.nba_colors['main_blue'] for opp in opp_avg['Opp']]
        
        # Plot average points by opponent (for opponents with at least 2 games)
        opp_filtered = opp_avg[opp_avg['count'] >= 2]
        
        ax1.bar(
            opp_filtered['Opp'],
            opp_filtered['mean'],
            color=colors[:len(opp_filtered)],
            alpha=0.7
        )
        
        # Add error bars for standard deviation
        if 'std' in opp_filtered.columns:
            ax1.errorbar(
                x=opp_filtered['Opp'],
                y=opp_filtered['mean'],
                yerr=opp_filtered['std'],
                fmt='none',
                ecolor=self.nba_colors['main_black'],
                elinewidth=1,
                capsize=5
            )
        
        # Highlight current opponent
        if opponent_team in opp_filtered['Opp'].values:
            opp_idx = opp_filtered[opp_filtered['Opp'] == opponent_team].index[0]
            ax1.bar(
                opponent_team,
                opp_filtered.loc[opp_idx, 'mean'],
                color=self.nba_colors['main_red'],
                alpha=1.0,
                linewidth=2,
                edgecolor=self.nba_colors['main_black']
            )
            
            # Add text with opponent average
            opp_mean = opp_filtered.loc[opp_idx, 'mean']
            ax1.text(
                opp_idx,
                opp_mean + 1,
                f"{opp_mean:.1f}",
                ha='center',
                va='bottom',
                fontweight='bold'
            )
        
        # Set title and labels
        ax1.set_title(f"{player_name} - Avg Points by Opponent", fontsize=14, fontweight='bold')
        ax1.set_xlabel('Opponent', fontsize=12)
        ax1.set_ylabel('Average Points', fontsize=12)
        ax1.tick_params(axis='x', rotation=90)
        
        # 2. Detailed history against specific opponent
        opp_games = player_data[player_data['Opp'].str.contains(opponent_team)]
        
        if len(opp_games) > 0:
            # Sort by date
            if 'Data' in opp_games.columns:
                if opp_games['Data'].dtype == object:
                    opp_games['Data'] = pd.to_datetime(opp_games['Data'])
                opp_games = opp_games.sort_values('Data')
                
                # Plot points in games against this opponent
                ax2.plot(
                    opp_games['Data'],
                    opp_games['PTS'],
                    marker='o',
                    markersize=10,
                    linewidth=2,
                    color=self.nba_colors['main_red'],
                    label=f'vs {opponent_team}'
                )
                
                # Add season average line
                season_avg = player_data['PTS'].mean()
                ax2.axhline(
                    y=season_avg,
                    color=self.nba_colors['main_black'],
                    linestyle='--',
                    alpha=0.7,
                    label=f'Season Avg: {season_avg:.1f}'
                )
                
                # Add opponent average line
                opp_avg_val = opp_games['PTS'].mean()
                ax2.axhline(
                    y=opp_avg_val,
                    color=self.nba_colors['main_red'],
                    linestyle='--',
                    alpha=0.7,
                    label=f'vs {opponent_team} Avg: {opp_avg_val:.1f}'
                )
                
                # Add prediction if provided
                if prediction is not None:
                    last_date = opp_games['Data'].max()
                    next_date = last_date + pd.Timedelta(days=100)  # Just for visualization
                    
                    ax2.scatter(
                        next_date,
                        prediction,
                        color=self.nba_colors['main_blue'],
                        s=200,
                        marker='*',
                        label=f'Prediction: {prediction:.1f}'
                    )
                    
                    # Add line from last game to prediction
                    ax2.plot(
                        [opp_games['Data'].iloc[-1], next_date],
                        [opp_games['PTS'].iloc[-1], prediction],
                        color=self.nba_colors['main_blue'],
                        linestyle=':',
                        alpha=0.5
                    )
                
                # Set title and labels
                ax2.set_title(f"{player_name} - History vs {opponent_team}", fontsize=14, fontweight='bold')
                ax2.set_xlabel('Game Date', fontsize=12)
                ax2.set_ylabel('Points', fontsize=12)
                ax2.tick_params(axis='x', rotation=45)
                ax2.legend(loc='upper left')
            
            else:
                # If no date column, just show a simple bar chart of all games
                game_nums = range(1, len(opp_games) + 1)
                ax2.bar(
                    game_nums,
                    opp_games['PTS'],
                    color=self.nba_colors['main_red'],
                    alpha=0.7
                )
                
                # Add average line
                opp_avg_val = opp_games['PTS'].mean()
                ax2.axhline(
                    y=opp_avg_val,
                    color=self.nba_colors['main_black'],
                    linestyle='--',
                    alpha=0.7,
                    label=f'vs {opponent_team} Avg: {opp_avg_val:.1f}'
                )
                
                # Add prediction if provided
                if prediction is not None:
                    ax2.bar(
                        len(opp_games) + 1,
                        prediction,
                        color=self.nba_colors['main_blue'],
                        alpha=1.0,
                        label=f'Prediction: {prediction:.1f}'
                    )
                
                # Set title and labels
                ax2.set_title(f"{player_name} - History vs {opponent_team}", fontsize=14, fontweight='bold')
                ax2.set_xlabel('Game Number', fontsize=12)
                ax2.set_ylabel('Points', fontsize=12)
                ax2.legend(loc='upper left')
        
        else:
            # No games against this opponent, show message
            ax2.text(
                0.5, 0.5,
                f"No previous games against {opponent_team}",
                horizontalalignment='center',
                verticalalignment='center',
                transform=ax2.transAxes,
                fontsize=14
            )
            
            # If prediction provided, show that
            if prediction is not None:
                ax2.text(
                    0.5, 0.4,
                    f"Prediction: {prediction:.1f} points",
                    horizontalalignment='center',
                    verticalalignment='center',
                    transform=ax2.transAxes,
                    fontsize=16,
                    fontweight='bold',
                    color=self.nba_colors['main_blue']
                )
            
            # Set title
            ax2.set_title(f"{player_name} - vs {opponent_team}", fontsize=14, fontweight='bold')
            ax2.axis('off')  # Hide axes since no data
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure if output file specified
        if output_file:
            save_path = os.path.join(self.save_dir, output_file)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            return save_path
        else:
            plt.show()
            return None
    
    def prediction_analysis(self, result, player_data, output_file=None):
        """
        Create a comprehensive visualization of the prediction
        
        Args:
            result: Dictionary with prediction results
            player_data: DataFrame with player's historical data
            output_file: Optional filename to save the visualization
            
        Returns:
            Path to saved visualization file or None
        """
        if isinstance(result, str):
            print(f"Cannot visualize prediction: {result}")
            return None
        
        # Extract key info from result
        player_name = result.get('player', 'Unknown Player')
        opponent_team = result.get('opponent', 'Unknown Team')
        prediction = result.get('predicted_points', 0)
        line = result.get('line')
        recommendation = result.get('recommendation', 'None')
        confidence = result.get('confidence', 0)
        confidence_labels = ['No Confidence', 'Low', 'Medium', 'High']
        
        # Create figure with three subplots
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(2, 2, height_ratios=[1, 1])
        
        # 1. Trend plot (top left)
        ax1 = fig.add_subplot(gs[0, 0])
        
        # Convert date column if needed
        if 'Data' in player_data.columns:
            if player_data['Data'].dtype == object:
                player_data['Data'] = pd.to_datetime(player_data['Data'])
            
            # Sort by date and take last 10 games
            player_data = player_data.sort_values('Data')
            recent_data = player_data.tail(10)
            
            # Calculate rolling averages
            recent_data['PTS_rolling_3'] = recent_data['PTS'].rolling(window=3, min_periods=1).mean()
            
            # Plot actual points
            ax1.scatter(
                range(len(recent_data)),
                recent_data['PTS'],
                color=self.nba_colors['main_orange'],
                s=100,
                alpha=0.7,
                label='Actual Points'
            )
            
            # Connect points with line
            ax1.plot(
                range(len(recent_data)),
                recent_data['PTS'],
                color=self.nba_colors['main_orange'],
                alpha=0.3
            )
            
            # Plot rolling average
            ax1.plot(
                range(len(recent_data)),
                recent_data['PTS_rolling_3'],
                color=self.nba_colors['main_blue'],
                linewidth=2,
                label='3-Game Avg'
            )
            
            # Add season average line
            season_avg = player_data['PTS'].mean()
            ax1.axhline(
                y=season_avg,
                color=self.nba_colors['main_black'],
                linestyle='--',
                alpha=0.7,
                label=f'Season Avg: {season_avg:.1f}'
            )
            
            # Add prediction point
            ax1.scatter(
                len(recent_data),
                prediction,
                color=self.nba_colors['main_blue'],
                s=200,
                marker='*',
                label=f'Prediction: {prediction:.1f}'
            )
            
            # Add line from last game to prediction
            ax1.plot(
                [len(recent_data)-1, len(recent_data)],
                [recent_data['PTS'].iloc[-1], prediction],
                color=self.nba_colors['main_blue'],
                linestyle=':',
                alpha=0.5
            )
            
            # Add betting line if provided
            if line is not None:
                ax1.axhline(
                    y=line,
                    color=self.nba_colors['main_red'],
                    linestyle='-.',
                    alpha=0.7,
                    label=f'Line: {line}'
                )
            
            # Set x-ticks to show opponents
            if 'Opp' in recent_data.columns:
                plt.xticks(
                    range(len(recent_data)),
                    recent_data['Opp'],
                    rotation=45
                )
            
            # Add next opponent
            plt.xticks(
                list(range(len(recent_data))) + [len(recent_data)],
                list(recent_data['Opp'] if 'Opp' in recent_data.columns else ['Game ' + str(i+1) for i in range(len(recent_data))]) + [opponent_team],
                rotation=45
            )
            
            # Set title and labels
            ax1.set_title(f"{player_name} - Recent Points Trend", fontsize=14, fontweight='bold')
            ax1.set_xlabel('Opponent', fontsize=12)
            ax1.set_ylabel('Points', fontsize=12)
            ax1.legend(loc='upper left')
        
        else:
            # No date information, show simplified version
            ax1.text(
                0.5, 0.5,
                "Insufficient date information for trend visualization",
                horizontalalignment='center',
                verticalalignment='center',
                transform=ax1.transAxes,
                fontsize=12
            )
        
        # 2. Matchup analysis (top right)
        ax2 = fig.add_subplot(gs[0, 1])
        
        # Group by opponent and calculate average points
        if 'Opp' in player_data.columns:
            opp_avg = player_data.groupby('Opp')['PTS'].agg(['mean', 'count']).reset_index()
            opp_avg = opp_avg[opp_avg['count'] >= 2]  # Only opponents with at least 2 games
            opp_avg = opp_avg.sort_values('mean', ascending=False)
            
            if len(opp_avg) > 0:
                # Color for the specific opponent
                colors = [self.nba_colors['main_red'] if opponent_team in opp else self.nba_colors['main_blue'] for opp in opp_avg['Opp']]
                
                # Plot average points by opponent
                ax2.bar(
                    range(len(opp_avg)),
                    opp_avg['mean'],
                    color=colors,
                    alpha=0.7
                )
                
                # Set x-ticks to show opponents
                plt.sca(ax2)
                plt.xticks(
                    range(len(opp_avg)),
                    opp_avg['Opp'],
                    rotation=90
                )
                
                # Add prediction marker
                ax2.axhline(
                    y=prediction,
                    color=self.nba_colors['main_blue'],
                    linestyle=':',
                    alpha=0.7,
                    label=f'Prediction: {prediction:.1f}'
                )
                
                # Add line
                if line is not None:
                    ax2.axhline(
                        y=line,
                        color=self.nba_colors['main_red'],
                        linestyle='-.',
                        alpha=0.7,
                        label=f'Line: {line}'
                    )
                
                # Set title and labels
                ax2.set_title(f"{player_name} - Avg Points by Opponent", fontsize=14, fontweight='bold')
                ax2.set_xlabel('Opponent', fontsize=12)
                ax2.set_ylabel('Average Points', fontsize=12)
                ax2.legend(loc='upper right')
            
            else:
                # Not enough opponent data
                ax2.text(
                    0.5, 0.5,
                    "Insufficient opponent data for matchup analysis",
                    horizontalalignment='center',
                    verticalalignment='center',
                    transform=ax2.transAxes,
                    fontsize=12
                )
        
        else:
            # No opponent information
            ax2.text(
                0.5, 0.5,
                "No opponent information available",
                horizontalalignment='center',
                verticalalignment='center',
                transform=ax2.transAxes,
                fontsize=12
            )
        
        # 3. Prediction specifics (bottom left)
        ax3 = fig.add_subplot(gs[1, 0])
        
        # Create a prediction summary visualization
        metrics = ['Predicted', 'Season Avg', 'Recent Avg']
        values = [prediction, result.get('season_avg', 0), result.get('recent_avg', 0)]
        
        # Color bars based on comparison to line
        if line is not None:
            colors = [
                self.nba_colors['main_red'] if val > line else self.nba_colors['main_blue']
                for val in values
            ]
        else:
            colors = [self.nba_colors['main_blue']] * len(metrics)
        
        # Create bar chart of key metrics
        ax3.bar(
            metrics,
            values,
            color=colors,
            alpha=0.7
        )
        
        # Add line
        if line is not None:
            ax3.axhline(
                y=line,
                color=self.nba_colors['main_red'],
                linestyle='-.',
                alpha=0.7,
                label=f'Line: {line}'
            )
        
        # Add value labels
        for i, v in enumerate(values):
            ax3.text(
                i,
                v + 0.5,
                f"{v:.1f}",
                ha='center',
                fontweight='bold'
            )
        
        # Set title and labels
        ax3.set_title(f"{player_name} vs {opponent_team} - Prediction Summary", fontsize=14, fontweight='bold')
        ax3.set_ylabel('Points', fontsize=12)
        
        # Add betting recommendation
        if line is not None and recommendation:
            rec_color = self.nba_colors['main_red'] if recommendation == 'OVER' else self.nba_colors['main_blue']
            conf_text = confidence_labels[min(confidence, 3)]
            
            ax3.text(
                0.5, -0.15,
                f"Recommendation: {recommendation} {line} ({conf_text} Confidence)",
                horizontalalignment='center',
                verticalalignment='center',
                transform=ax3.transAxes,
                fontsize=14,
                fontweight='bold',
                color=rec_color
            )
        
        # 4. Context factors (bottom right)
        ax4 = fig.add_subplot(gs[1, 1])
        
        # Turn off axis
        ax4.axis('off')
        
        # Create a text summary of contextual factors
        context_text = [f"{player_name} vs {opponent_team} - Contextual Factors"]
        
        # Add live game factors if available
        if "live_game_factors" in result:
            live_factors = result["live_game_factors"]
            context_text.append("\nLIVE GAME STATUS:")
            
            if live_factors.get("game_status") == "LIVE":
                context_text.append(f"• Current Points: {live_factors.get('current_pts', 0)}")
                context_text.append(f"• Current Quarter: Q{live_factors.get('current_quarter', 1)}")
                context_text.append(f"• Minutes Played: {live_factors.get('current_mins', 0)}")
                
                if live_factors.get("foul_trouble", False):
                    context_text.append("• ⚠️ Player is in foul trouble")
                
                if live_factors.get("blowout", False):
                    context_text.append("• ⚠️ Game is a blowout")
            
            elif live_factors.get("game_status") == "SCHEDULED":
                context_text.append(f"• Game scheduled today at {live_factors.get('game_time', 'TBD')}")
                context_text.append(f"• Home Game: {'Yes' if live_factors.get('is_home', False) else 'No'}")
        
        # Add injury and team data if available
        if "api_enhanced_data" in result:
            api_data = result["api_enhanced_data"]
            
            context_text.append("\nPLAYER STATUS:")
            
            if api_data.get("is_injured", 0) == 1:
                context_text.append(f"• ⚠️ Injury: {api_data.get('injury_status', 'Unknown')}")
            else:
                context_text.append("• No reported injuries")
            
            if api_data.get("is_back_to_back", 0) == 1:
                context_text.append("• ⚠️ Playing on back-to-back games")
            
            team_injuries = api_data.get("team_injury_count", 0)
            opp_injuries = api_data.get("opp_injury_count", 0)
            
            if team_injuries > 0 or opp_injuries > 0:
                context_text.append("\nTEAM STATUS:")
                if team_injuries > 0:
                    context_text.append(f"• {team_injuries} teammate(s) injured")
                if opp_injuries > 0:
                    context_text.append(f"• {opp_injuries} opponent player(s) injured")
            
            # Add recent performance metrics if available
            if "recent_weighted_pts" in api_data:
                context_text.append("\nRECENT PERFORMANCE:")
                context_text.append(f"• Recency-weighted average: {api_data.get('recent_weighted_pts', 0):.1f} pts")
                
                # Add other metrics if available
                if "recent_weighted_fg_pct" in api_data:
                    context_text.append(f"• Recent FG%: {api_data.get('recent_weighted_fg_pct', 0):.3f}")
                if "recent_weighted_fg3_pct" in api_data:
                    context_text.append(f"• Recent 3P%: {api_data.get('recent_weighted_fg3_pct', 0):.3f}")
        
        # Add recent games info
        if "recent_games" in result and result["recent_games"]:
            context_text.append("\nRECENT GAMES:")
            for i, game in enumerate(result["recent_games"][:3], 1):
                pts = game.get('PTS', 'N/A')
                opp = game.get('Opp', 'Unknown')
                date = game.get('Data', 'Unknown Date')
                if isinstance(date, pd.Timestamp):
                    date = date.strftime('%Y-%m-%d')
                context_text.append(f"• Game {i}: {pts} pts vs {opp} ({date})")
        
        # Display the context text
        ax4.text(
            0.05, 0.95,
            '\n'.join(context_text),
            horizontalalignment='left',
            verticalalignment='top',
            transform=ax4.transAxes,
            fontsize=12
        )
        
        # Add recommendation summary at the bottom
        if line is not None and recommendation:
            recommendation_text = []
            
            # Determine color based on over/under
            rec_color = self.nba_colors['main_red'] if recommendation == 'OVER' else self.nba_colors['main_blue']
            
            # Calculate difference from line
            diff = prediction - line
            diff_text = f"{abs(diff):.1f} points {'above' if diff > 0 else 'below'} the line"
            
            # Create recommendation box
            rect = plt.Rectangle(
                (0.1, 0.05),
                0.8, 0.15,
                facecolor=rec_color,
                alpha=0.2,
                transform=ax4.transAxes
            )
            ax4.add_patch(rect)
            
            # Add recommendation text
            ax4.text(
                0.5, 0.125,
                f"RECOMMENDATION: {recommendation} {line} ({diff_text})",
                horizontalalignment='center',
                verticalalignment='center',
                transform=ax4.transAxes,
                fontsize=14,
                fontweight='bold',
                color=rec_color
            )
        
        # Set title
        ax4.set_title("Contextual Factors", fontsize=14, fontweight='bold')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure if output file specified
        if output_file:
            save_path = os.path.join(self.save_dir, output_file)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            return save_path
        else:
            plt.show()
            return None
    
    def create_pdf_report(self, result, player_data, output_file="prediction_report.pdf"):
        """
        Create a comprehensive PDF report of the prediction
        
        Args:
            result: Dictionary with prediction results
            player_data: DataFrame with player's historical data
            output_file: Filename to save the PDF report
            
        Returns:
            Path to saved PDF file or None
        """
        from matplotlib.backends.backend_pdf import PdfPages
        import matplotlib.pyplot as plt
        
        # Extract key info
        player_name = result.get('player', 'Unknown Player')
        opponent_team = result.get('opponent', 'Unknown Team')
        
        # Create PDF
        pdf_path = os.path.join(self.save_dir, output_file)
        
        with PdfPages(pdf_path) as pdf:
            # 1. Performance trend visualization
            self.player_performance_trend(
                player_data, 
                player_name, 
                result.get('predicted_points'), 
                result.get('line')
            )
            pdf.savefig()
            plt.close()
            
            # 2. Matchup analysis
            self.matchup_analysis(
                player_data,
                opponent_team,
                player_name,
                result.get('predicted_points')
            )
            pdf.savefig()
            plt.close()
            
            # 3. Prediction analysis
            self.prediction_analysis(
                result,
                player_data
            )
            pdf.savefig()
            plt.close()
            
            # 4. Create info page with metadata
            plt.figure(figsize=(8.5, 11))
            plt.axis('off')
            
            # Title
            plt.text(0.5, 0.95, f"NBA Player Points Prediction Report", 
                    ha='center', fontsize=16, fontweight='bold')
            plt.text(0.5, 0.9, f"{player_name} vs {opponent_team}", 
                    ha='center', fontsize=14)
            
            # Date
            plt.text(0.5, 0.85, f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M')}", 
                    ha='center', fontsize=10)
            
            # Prediction details
            plt.text(0.1, 0.75, "Prediction Details:", fontsize=12, fontweight='bold')
            y_pos = 0.7
            for key, value in result.items():
                if key not in ['player', 'opponent', 'recent_games', 'api_enhanced_data', 'live_game_factors']:
                    plt.text(0.1, y_pos, f"{key}:", fontsize=10)
                    plt.text(0.5, y_pos, f"{value}", fontsize=10)
                    y_pos -= 0.03
            
            pdf.savefig()
            plt.close()
        
        print(f"PDF report saved to {pdf_path}")
        return pdf_path