import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from typing import List, Optional
import cv2

class ExerciseSummaryWindow:
    """
    Summary window that displays exercise performance metrics including:
    - Line graph of rep scores over time
    - Overall form correctness percentage
    - Rep count and statistics
    """
    
    def __init__(self, rep_scores: List[float], rep_times: Optional[List[float]] = None):
        self.rep_scores = rep_scores
        self.rep_times = rep_times or list(range(1, len(rep_scores) + 1))
        
        # Create main window
        self.root = tk.Tk()
        self.root.title("Exercise Summary")
        self.root.geometry("800x600")
        self.root.configure(bg='#f0f0f0')
        
        # Calculate statistics
        self.overall_score = np.mean(rep_scores) if rep_scores else 0.0
        self.total_reps = len(rep_scores)
        self.excellent_reps = sum(1 for score in rep_scores if score >= 0.8)
        self.good_reps = sum(1 for score in rep_scores if score >= 0.5)
        
        self._create_widgets()
        self._create_graph()
        
    def _create_widgets(self):
        """Create and layout the main widgets."""
        # Title
        title_label = tk.Label(
            self.root, 
            text="Exercise Performance Summary", 
            font=("Arial", 18, "bold"),
            bg='#f0f0f0',
            fg='#333333'
        )
        title_label.pack(pady=20)
        
        # Statistics frame
        stats_frame = tk.Frame(self.root, bg='#f0f0f0')
        stats_frame.pack(pady=10)
        
        # Overall score
        score_frame = tk.Frame(stats_frame, bg='#ffffff', relief='raised', bd=2)
        score_frame.pack(side='left', padx=10, pady=5)
        
        score_label = tk.Label(
            score_frame,
            text="Overall Form Correctness",
            font=("Arial", 12, "bold"),
            bg='#ffffff',
            fg='#333333'
        )
        score_label.pack(pady=5)
        
        score_value = tk.Label(
            score_frame,
            text=f"{self.overall_score:.1%}",
            font=("Arial", 24, "bold"),
            bg='#ffffff',
            fg='#4CAF50' if self.overall_score >= 0.7 else '#FF9800' if self.overall_score >= 0.5 else '#F44336'
        )
        score_value.pack(pady=5)
        
        # Rep count
        rep_frame = tk.Frame(stats_frame, bg='#ffffff', relief='raised', bd=2)
        rep_frame.pack(side='left', padx=10, pady=5)
        
        rep_label = tk.Label(
            rep_frame,
            text="Total Reps",
            font=("Arial", 12, "bold"),
            bg='#ffffff',
            fg='#333333'
        )
        rep_label.pack(pady=5)
        
        rep_value = tk.Label(
            rep_frame,
            text=str(self.total_reps),
            font=("Arial", 24, "bold"),
            bg='#ffffff',
            fg='#2196F3'
        )
        rep_value.pack(pady=5)
        
        # Quality breakdown
        quality_frame = tk.Frame(stats_frame, bg='#ffffff', relief='raised', bd=2)
        quality_frame.pack(side='left', padx=10, pady=5)
        
        quality_label = tk.Label(
            quality_frame,
            text="Rep Quality",
            font=("Arial", 12, "bold"),
            bg='#ffffff',
            fg='#333333'
        )
        quality_label.pack(pady=5)
        
        quality_text = tk.Label(
            quality_frame,
            text=f"Excellent: {self.excellent_reps}\nGood: {self.good_reps}",
            font=("Arial", 10),
            bg='#ffffff',
            fg='#666666',
            justify='left'
        )
        quality_text.pack(pady=5)
        
        # Graph frame
        self.graph_frame = tk.Frame(self.root, bg='#ffffff', relief='sunken', bd=2)
        self.graph_frame.pack(pady=20, padx=20, fill='both', expand=True)
        
        # Close button
        close_button = tk.Button(
            self.root,
            text="Close",
            command=self._close_and_exit,
            font=("Arial", 12),
            bg='#f44336',
            fg='white',
            relief='raised',
            bd=2,
            padx=20,
            pady=10
        )
        close_button.pack(pady=20)
        
        # Handle window close button (X) to exit application
        self.root.protocol("WM_DELETE_WINDOW", self._close_and_exit)
        
    def _create_graph(self):
        """Create the matplotlib graph showing rep scores over time."""
        if not self.rep_scores:
            # No data to plot
            no_data_label = tk.Label(
                self.graph_frame,
                text="No exercise data available",
                font=("Arial", 14),
                bg='#ffffff',
                fg='#666666'
            )
            no_data_label.pack(expand=True)
            return
        
        # Create matplotlib figure
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.patch.set_facecolor('#ffffff')
        
        # Plot rep scores
        x_values = self.rep_times
        y_values = self.rep_scores
        
        # Create line plot
        line = ax.plot(x_values, y_values, 'b-o', linewidth=2, markersize=6, 
                      markerfacecolor='#2196F3', markeredgecolor='#1976D2')
        
        # Add horizontal lines for thresholds
        ax.axhline(y=0.8, color='#4CAF50', linestyle='--', alpha=0.7, 
                  label='Excellent (0.8)')
        ax.axhline(y=0.5, color='#FF9800', linestyle='--', alpha=0.7, 
                  label='Good (0.5)')
        ax.axhline(y=0.3, color='#F44336', linestyle='--', alpha=0.7, 
                  label='Poor (0.3)')
        
        # Customize the plot
        ax.set_xlabel('Rep Number', fontsize=12, fontweight='bold')
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title('Rep Scores Over Time', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Set y-axis limits
        ax.set_ylim(0, 1.0)
        
        # Add value labels on points
        for i, (x, y) in enumerate(zip(x_values, y_values)):
            ax.annotate(f'{y:.2f}', (x, y), textcoords="offset points", 
                       xytext=(0,10), ha='center', fontsize=8)
        
        # Style the plot
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_facecolor('#f8f9fa')
        
        # Create canvas and add to tkinter
        canvas = FigureCanvasTkAgg(fig, self.graph_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)
        
    def show(self):
        """Display the summary window."""
        self.root.mainloop()
        
    def close(self):
        """Close the summary window."""
        self.root.destroy()
    
    def _close_and_exit(self):
        """Close the summary window and exit the application."""
        self.root.destroy()
        import sys
        sys.exit(0)

def show_exercise_summary(rep_scores: List[float], rep_times: Optional[List[float]] = None):
    """
    Factory function to create and show the exercise summary window.
    
    Args:
        rep_scores: List of rep scores (0-1)
        rep_times: Optional list of timestamps for each rep
    """
    if not rep_scores:
        print("No exercise data to display")
        return
    
    summary_window = ExerciseSummaryWindow(rep_scores, rep_times)
    summary_window.show()

if __name__ == "__main__":
    # Test the summary window
    test_scores = [0.8, 0.6, 0.9, 0.4, 0.7, 0.8, 0.5, 0.9]
    show_exercise_summary(test_scores)
